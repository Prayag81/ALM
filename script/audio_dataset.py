import torch
from torch.utils.data import Dataset, DataLoader , WeightedRandomSampler
import torchaudio.transforms as T
import torchaudio
import random
import librosa
import torch.nn.functional as F
import soundfile as sf
import pandas as pd
import numpy as np
from pathlib import Path
import logging



# ── Mel spectrogram config (matches project brief) ─────────────────────────
MEL_CONFIG = dict(
    sample_rate = 16_000,
    n_fft       = 1024,
    hop_length  = 512,
    n_mels      = 128,
    f_min       = 0,
    f_max       = 8_000,
)


LABEL_MAP = {
    "dog_bark":         "dog",
    "dog":              "dog",
    "engine_idling":    "engine",
    "engine":           "engine",
    "gun_shot":         "gun_shot",
    "gunshot":          "gun_shot",
    "siren":            "siren",
    "alarm":            "siren",
    "scream":           "scream",
    "shouting":         "scream",
    "jackhammer":       "jackhammer",
    "drilling":         "jackhammer",
}
RARE_CLASSES  = {"gun_shot", "scream", "siren"}
COMMON_CLASSES = {"dog", "engine", "children_playing", "street_music", "air_conditioner"}

def normalize_labels(label_str: str) -> str:
    labels = [l.strip().lower() for l in label_str.split(",") if l.strip()]
    labels = [LABEL_MAP.get(l, l) for l in labels]
    return ",".join(sorted(set(labels)))


class AudioTextDataset(Dataset):
    """
    Args:
        metadata_csv : path to unified metadata CSV
        audio_dir    : path to data/processed/ (flat folder of .wav files)
        split        : 'train' | 'val' | 'test' | 'all'
        val_fold     : fold number held out for val/test  (default: 1)
        augment      : apply SpecAugment during training
    """

    def __init__(
        self,
        metadata_csv: str | Path,
        audio_dir:    str | Path,
        split:        str  = 'train',
        val_fold:     int  = 1,
        augment:      bool = False,
    ):
        self.audio_dir = Path(audio_dir)
        self.augment   = augment
        self.split     = split

        df = pd.read_csv(metadata_csv)

        # ── Train / val split by fold ──────────────────────────────────────
        if split == 'train':
            df = df[df['fold'] != val_fold]
        elif split in ('val', 'test'):
            df = df[df['fold'] == val_fold]
        # 'all' → keep everything

        self.df = df.reset_index(drop=True)

        # ── Label → integer mapping ────────────────────────────────────────
        all_labels       = sorted(df['label'].unique())
        self.label2id    = {lbl: i for i, lbl in enumerate(all_labels)}
        self.id2label    = {i: lbl for lbl, i in self.label2id.items()}
        self.num_classes = len(all_labels)

        # ── Mel spectrogram transform ──────────────────────────────────────
        self.mel_transform = T.MelSpectrogram(**MEL_CONFIG)
        self.amp_to_db     = T.AmplitudeToDB(stype='power', top_db=80)

        # ── SpecAugment (only applied when augment=True) ───────────────────
        self.freq_mask = T.FrequencyMasking(freq_mask_param=27)
        self.time_mask = T.TimeMasking(time_mask_param=40)

        print(f"[AudioTextDataset] split={split!r}  samples={len(self.df)}  "
              f"classes={self.num_classes}  val_fold={val_fold}")

    # ── Helpers ────────────────────────────────────────────────────────────

    def _load_audio(self, filepath: Path) -> torch.Tensor:
        data, sr = sf.read(str(filepath), dtype='float32')
        
        # soundfile returns (samples,) for mono or (samples, channels) for stereo
        if data.ndim == 2:
            data = data.mean(axis=1)   # stereo → mono
        
        waveform = torch.from_numpy(data).unsqueeze(0)  # (1, samples)
        
        if sr != MEL_CONFIG['sample_rate']:
            waveform = T.Resample(sr, MEL_CONFIG['sample_rate'])(waveform)
        
        return waveform  # (1, num_samples)

    def _to_melspec(self, waveform):
        mel     = self.mel_transform(waveform)
        log_mel = self.amp_to_db(mel)
        
        # temporary debug — remove after
        # replace the normalisation line with this
        log_mel = (log_mel - log_mel.min()) / (log_mel.max() - log_mel.min())  # [0, 1]
        log_mel = log_mel * 2 - 1  # [0, 1] → [-1, 1]
        return log_mel

    # ── Dataset protocol ───────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]

        # Load audio & compute mel
        audio_path = self.audio_dir / row['file']
        waveform   = self._load_audio(audio_path)
        mel        = self._to_melspec(waveform)

        # SpecAugment (training only)
        if self.augment and self.split == 'train':
            mel = self.freq_mask(mel)
            mel = self.time_mask(mel)

        label_id = self.label2id[row['label']]
        text     = row.get('text_description', row['label'].replace('_', ' '))

        return {
            'mel'      : mel,                          # (1, 128, 157)  float32
            'label_id' : torch.tensor(label_id, dtype=torch.long),
            'text'     : text,                         # raw string for text encoder
            'filename' : row['file'],
        }

    # ── Convenience factory methods ────────────────────────────────────────

    @classmethod
    def get_loaders(
        cls,
        metadata_csv : str | Path,
        audio_dir    : str | Path,
        val_fold     : int  = 1,
        batch_size   : int  = 32,
        num_workers  : int  = 4,
        augment      : bool = True,
    ):
        """Returns (train_loader, val_loader) ready for the training loop."""
        train_ds = cls(metadata_csv, audio_dir, split='train',
                       val_fold=val_fold, augment=augment)
        val_ds   = cls(metadata_csv, audio_dir, split='val',
                       val_fold=val_fold, augment=False)

        # Share the same label mapping so IDs are consistent
        val_ds.label2id    = train_ds.label2id
        val_ds.id2label    = train_ds.id2label
        val_ds.num_classes = train_ds.num_classes

        train_loader = DataLoader(
            train_ds,
            batch_size  = batch_size,
            shuffle     = True,
            num_workers = num_workers,
            pin_memory  = True,
            drop_last   = True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size  = batch_size * 2,
            shuffle     = False,
            num_workers = num_workers,
            pin_memory  = True,
        )
        return train_loader, val_loader, train_ds.label2id


# ── Quick sanity check ─────────────────────────────────────────────────────
if __name__ == '__main__':
    import sys
    ROOT = Path(__file__).resolve().parents[1]
    log_dir = ROOT / "outputs"/"logs"
    logging.basicConfig(
        filename = log_dir / 'audio_dataset.log',
        filemode = 'a',
        format   = '%(asctime)s - %(levelname)s - %(message)s',
        level    = logging.INFO
    )

    logger = logging.getLogger(__name__)

    ds = AudioTextDataset(
        metadata_csv = ROOT / 'data' / 'processed' / 'metadata.csv',
        audio_dir    = ROOT / 'data' / 'processed' / 'audio',
        split        = 'all',
    )

    sample = ds[0]
    logger.info("\n── Sample item ──────────────────────────────────────────────")
    logger.info(f"  mel shape  : {sample['mel'].shape}")
    logger.info(f"  mel range  : [{sample['mel'].min():.3f}, {sample['mel'].max():.3f}]")
    logger.info(f"  label_id   : {sample['label_id']}")
    logger.info(f"  text       : {sample['text']!r}")
    logger.info(f"  filename   : {sample['file']}")
    logger.info(f"\n  num_classes: {ds.num_classes}")
    logger.info(f"  dataset len: {len(ds)}")

# ── Module-level constants (accessible everywhere in the file) ─────────────
URBAN_CLASSES = {
    "jackhammer", "engine", "dog", "children_playing",
    "street_music", "air_conditioner", "siren", "car_horn", "gun_shot",
    "drilling",           # UrbanSound alias
}
RARE_CLASSES   = {"gun_shot", "scream", "siren"}
COMMON_CLASSES = {"dog", "engine", "children_playing", "street_music", "air_conditioner"}


class MultiLabelAudioDataset(Dataset):
    """
    Multi-label audio event detection dataset.

    Expected metadata.csv columns:
        file  – relative path under audio_dir   (or 'filename', auto-detected)
        label – single class OR comma-separated, e.g. "gun_shot,scream"
        fold  – integer fold (UrbanSound8K convention; ESC-50 uses 1-5)

    Key fixes vs. previous version
    ───────────────────────────────
    * self.df is always the *split* subset (was mistakenly the full df)
    * oversampling updates self.df unconditionally inside the train block
    * URBAN_CLASSES / RARE_CLASSES / COMMON_CLASSES are module-level constants
      so _mix_sample can see them without being passed as args
    * file column is auto-detected ('file' or 'filename')
    * label vocabulary is built from the split-subset only (consistent with val/test)
    * compute_pos_weight runs on the raw df counts, NOT the sampler-reweighted loader
    * pos_weight uses sqrt-dampening + clamp(0.5, 8.0) instead of raw ratio + 20.0
    * WeightedRandomSampler weights are capped so ESC-50 9x rows don't dominate
    """

    _mel_transform = None   # class-level cache

    # ─────────────────────────────────────────────────────────────────────────
    # Mel transform (shared across all instances)
    # ─────────────────────────────────────────────────────────────────────────
    @classmethod
    def _get_mel(cls) -> T.MelSpectrogram:
        if cls._mel_transform is None:
            cls._mel_transform = T.MelSpectrogram(
                sample_rate = 24_000,
                n_fft       = 1024,
                hop_length  = 256,
                n_mels      = 128,
                f_min       = 50,
                f_max       = 12_000,
            )
        return cls._mel_transform

    # ─────────────────────────────────────────────────────────────────────────
    # __init__
    # ─────────────────────────────────────────────────────────────────────────
    def __init__(
        self,
        metadata_csv : Path,
        audio_dir    : Path,
        split        : str   = 'train',   # 'train' | 'val' | 'test'
        val_fold     : int   = 9,
        test_fold    : int   = 10,
        label2id     : dict  = None,      # pass from train_ds for val/test
        augment      : bool  = False,
        mix_prob     : float = 0.4,
    ):
        self.audio_dir = Path(audio_dir)
        self.augment   = augment
        self.mix_prob  = mix_prob
        self.mel_tf    = self.__class__._get_mel()

        # ── Load & normalise labels ──────────────────────────────────────────
        df = pd.read_csv(metadata_csv)
        df["label"] = df["label"].apply(normalize_labels)

        # ── Auto-detect the filename column ─────────────────────────────────
        if 'file' in df.columns:
            self._file_col = 'file'
        elif 'filename' in df.columns:
            self._file_col = 'filename'
        else:
            raise ValueError(
                "metadata.csv must have a column named 'file' or 'filename'"
            )

        # ── Split by fold ────────────────────────────────────────────────────
        #    BUG FIX: assign the *filtered* df, not the full df, to self.df
        if split == 'train':
            df = df[~df['fold'].isin([val_fold, test_fold])].reset_index(drop=True)
        elif split == 'val':
            df = df[df['fold'] == val_fold].reset_index(drop=True)
        elif split == 'test':
            df = df[df['fold'] == test_fold].reset_index(drop=True)
        # 'all' → keep everything (useful for diagnostics)

        # ── ESC-50 oversampling (train only, before building vocab) ──────────
        #    BUG FIX: self.df is set *after* oversampling so __len__ is correct
        if split == 'train' and augment:
            esc_mask = ~df['label'].apply(
                lambda x: all(l.strip() in URBAN_CLASSES for l in x.split(','))
            )
            esc_rows = df[esc_mask]

            if len(esc_rows) > 0:
                # Match ESC-50 count to UrbanSound count instead of fixed 9x
                urban_count = len(df[~esc_mask])
                esc_count   = len(esc_rows)
                # How many extra copies do we need so ESC-50 ≈ UrbanSound?
                repeat = max(1, round(urban_count / esc_count) - 1)
                repeat = min(repeat, 15)          # hard cap to avoid explosion
                extra  = pd.concat([esc_rows] * repeat, ignore_index=True)
                df     = pd.concat([df, extra], ignore_index=True).reset_index(drop=True)
                print(
                    f"  ESC-50 oversampled {repeat+1}×: "
                    f"{esc_count} → {esc_count*(repeat+1)} rows  |  "
                    f"total train rows: {len(df)}"
                )

        # ── Assign the final (possibly oversampled) df ───────────────────────
        self.df = df.reset_index(drop=True)

        # ── Build label vocabulary from this split's data ────────────────────
        all_labels = sorted({
            lbl.strip()
            for cell in self.df['label']
            for lbl in str(cell).split(',')
            if lbl.strip()
        })
        if label2id is None:
            self.label2id = {lbl: i for i, lbl in enumerate(all_labels)}
        else:
            self.label2id = label2id

        self.id2label    = {v: k for k, v in self.label2id.items()}
        self.num_classes = len(self.label2id)

        print(
            f"  [{split:5s}] {len(self.df):>6} samples | "
            f"{self.num_classes} classes | augment={augment}"
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Dataset protocol
    # ─────────────────────────────────────────────────────────────────────────
    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row       = self.df.iloc[idx]
        wav       = self._load_wave(row[self._file_col])
        label_vec = self._to_multihot(row['label'])

        if self.augment:
            wav = self._augment_wave(wav, str(row['label']))

        if self.augment and random.random() < self.mix_prob:
            wav, label_vec = self._mix_sample(wav, label_vec, str(row['label']))

        mel = self.mel_tf(wav)
        mel = (mel + 1e-9).log()
        mel = (mel - mel.mean()) / (mel.std() + 1e-6)

        return {
            'mel'      : mel,          # (1, N_MELS, T)  float32
            'label'    : label_vec,    # (num_classes,)  float32 multi-hot
            'filename' : row[self._file_col],
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────────────
    def _to_multihot(self, label_str: str) -> torch.Tensor:
        vec = torch.zeros(self.num_classes, dtype=torch.float32)
        for lbl in str(label_str).split(','):
            lbl = lbl.strip()
            if lbl in self.label2id:
                vec[self.label2id[lbl]] = 1.0
        return vec

    def _load_wave(self, filename: str) -> torch.Tensor:
        path = self.audio_dir / filename
        wav, sr = torchaudio.load(str(path))

        if sr != 24_000:
            wav = torchaudio.functional.resample(wav, sr, 24_000)
        if wav.shape[0] > 1:                       # stereo → mono
            wav = wav.mean(dim=0, keepdim=True)

        n = 120_000                                # 5 s at 24 kHz
        if wav.shape[1] < n:
            wav = F.pad(wav, (0, n - wav.shape[1]))
        else:
            wav = wav[:, :n]

        return wav   # (1, 120_000)

    def _augment_wave(self, wav: torch.Tensor, label_str: str) -> torch.Tensor:
        """Pitch / stretch / noise augmentation, applied only to rare classes."""
        is_rare = any(rc in label_str for rc in RARE_CLASSES)
        if not is_rare:
            return wav

        wav_np     = wav.squeeze(0).numpy()
        aug_choice = random.random()

        if aug_choice < 0.33:
            steps  = random.uniform(-2, 2)
            wav_np = librosa.effects.pitch_shift(wav_np, sr=24_000, n_steps=steps)
        elif aug_choice < 0.66:
            rate   = random.uniform(0.85, 1.15)
            wav_np = librosa.effects.time_stretch(wav_np, rate=rate)
            n      = 120_000
            wav_np = wav_np[:n] if len(wav_np) >= n else np.pad(wav_np, (0, n - len(wav_np)))
        else:
            noise  = np.random.randn(len(wav_np)) * 0.005
            wav_np = wav_np + noise

        return torch.tensor(wav_np, dtype=torch.float32).unsqueeze(0)

    def _mix_sample(
        self,
        wav1      : torch.Tensor,
        label1    : torch.Tensor,
        label_str1: str,
    ):
        """
        Controlled mixing: rare↔common or common↔rare.
        BUG FIX: uses self.df (the split subset) not the full df.
        BUG FIX: RARE_CLASSES / COMMON_CLASSES are now module-level constants.
        """
        is_rare = any(rc in label_str1 for rc in RARE_CLASSES)

        if is_rare:
            mask = self.df['label'].apply(
                lambda x: any(c in x for c in COMMON_CLASSES)
            )
        else:
            mask = self.df['label'].apply(
                lambda x: any(rc in x for rc in RARE_CLASSES)
            )

        candidates = self.df[mask]
        if len(candidates) == 0:
            row2 = self.df.iloc[random.randint(0, len(self.df) - 1)]
        else:
            row2 = candidates.sample(1).iloc[0]

        wav2   = self._load_wave(row2[self._file_col])
        label2 = self._to_multihot(row2['label'])

        alpha  = random.uniform(0.2, 0.4)
        mixed  = wav1 + alpha * wav2
        peak   = mixed.abs().max()
        if peak > 1.0:
            mixed = mixed / peak

        combined_label = (label1 + label2).clamp(0, 1)
        return mixed, combined_label

    # ─────────────────────────────────────────────────────────────────────────
    # pos_weight helper  (call this instead of compute_pos_weight over loader)
    # ─────────────────────────────────────────────────────────────────────────
    def compute_pos_weight(self, device, max_weight: float = 8.0) -> torch.Tensor:
        """
        Compute BCEWithLogitsLoss pos_weight from raw df counts.

        Why NOT from the loader:
            The WeightedRandomSampler distorts effective class frequencies,
            making every class look equally common → all weights hit the cap.

        Uses sqrt-dampening so extreme ratios don't collapse to the same value.
        """
        pos_counts = torch.zeros(self.num_classes)
        total      = len(self.df)

        for label_str in self.df['label']:
            for lbl in str(label_str).split(','):
                lbl = lbl.strip()
                if lbl in self.label2id:
                    pos_counts[self.label2id[lbl]] += 1

        neg_counts = total - pos_counts
        # sqrt dampens extreme imbalance while still upweighting rare classes
        pos_weight = torch.sqrt(neg_counts / (pos_counts + 1e-6)).clamp(0.5, max_weight)

        # ── Diagnostic print ────────────────────────────────────────────────
        print(f"\n  pos_weight diagnostics (max_weight={max_weight}):")
        for i, (lbl, pw) in enumerate(
            sorted(zip(self.label2id.keys(), pos_weight.tolist()), key=lambda x: -x[1])
        ):
            bar = "█" * int(pw * 2)
            print(f"    {lbl:30s}  pw={pw:5.2f}  {bar}")

        return pos_weight.to(device)

    # ─────────────────────────────────────────────────────────────────────────
    # Factory
    # ─────────────────────────────────────────────────────────────────────────
    @classmethod
    def get_loaders(
        cls,
        metadata_csv : Path,
        audio_dir    : Path,
        val_fold     : int   = 9,
        test_fold    : int   = 10,
        batch_size   : int   = 16,
        num_workers  : int   = 4,
        augment      : bool  = True,
    ):
        train_ds = cls(
            metadata_csv, audio_dir,
            split='train', val_fold=val_fold, test_fold=test_fold,
            augment=augment,
        )
        val_ds = cls(
            metadata_csv, audio_dir,
            split='val', val_fold=val_fold, test_fold=test_fold,
            label2id=train_ds.label2id, augment=False,
        )

        # ── WeightedRandomSampler ─────────────────────────────────────────────
        # Count from the raw (oversampled) df
        label_counts: dict[str, int] = {}
        for label_str in train_ds.df['label']:
            for l in str(label_str).split(','):
                l = l.strip()
                if l:
                    label_counts[l] = label_counts.get(l, 0) + 1

        max_count = max(label_counts.values())
        # Floor: no class weight can be more than 10× the weight of the most common class
        min_count = max(min(label_counts.values()), max_count // 10)

        sample_weights = []
        for label_str in train_ds.df['label']:
            labels = [l.strip() for l in str(label_str).split(',') if l.strip()]
            # Inverse-frequency weight, capped to avoid explosion
            w = sum(1.0 / max(label_counts[l], min_count) for l in labels)
            sample_weights.append(w)

        sampler = WeightedRandomSampler(
            weights     = sample_weights,
            num_samples = len(sample_weights),
            replacement = True,
        )

        train_loader = DataLoader(
            train_ds,
            batch_size  = batch_size,
            sampler     = sampler,
            shuffle     = False,           # must be False when sampler is set
            num_workers = num_workers,
            pin_memory  = True,
            drop_last   = True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size  = batch_size * 2,
            shuffle     = False,
            num_workers = num_workers,
            pin_memory  = True,
        )
        return train_loader, val_loader, train_ds.label2id