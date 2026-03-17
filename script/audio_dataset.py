import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio.transforms as T
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
    logger.info(f"  filename   : {sample['filename']}")
    logger.info(f"\n  num_classes: {ds.num_classes}")
    logger.info(f"  dataset len: {len(ds)}")