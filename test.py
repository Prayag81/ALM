"""Builds ALM/scripts/alm_mad_encoder_training.ipynb programmatically."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []


def md(src):
    cells.append(nbf.v4.new_markdown_cell(src))


def code(src):
    cells.append(nbf.v4.new_code_cell(src.strip("\n")))


# ══════════════════════════════════════════════════════════════════════
# Title
# ══════════════════════════════════════════════════════════════════════
md("""\
# ALM — MAD Audio Encoder Training (Single-Label)

This notebook trains the **first single-label MAD audio encoder**.

- Dataset: MAD (`data/processed_mad/`)
- Task: single-label classification, **7 classes** (numeric labels `0`–`6`)
- Audio duration: **10 seconds** (MAD-specific — different from the original 5-second ALM configuration)
- Architecture: `MADAudioEncoder` from `utils/audio_mad_encoder.py`

This notebook is **self-contained** and can be run end-to-end from a fresh kernel, provided the
project structure described in `README.md` exists (`data/processed_mad/metadata.csv` and
`data/processed_mad/audio/`).

> **Not covered here:** multi-label training, `MultiLabelAudioEncoder`, `BCEWithLogitsLoss`, sigmoid
> activations, multi-hot labels, or per-class thresholds. Those belong to the separate multi-label
> experiment (`alm_audio_encoder_multilabel.ipynb`).
""")

# ══════════════════════════════════════════════════════════════════════
# 1. Imports & Setup
# ══════════════════════════════════════════════════════════════════════
md("## 1. Imports & Setup")

code("""
import os
import time
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import librosa
import librosa.display
import soundfile as sf

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from tqdm.auto import tqdm

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support,
)

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

print("Imports OK")
""")

# ══════════════════════════════════════════════════════════════════════
# 2. Reproducibility & Device
# ══════════════════════════════════════════════════════════════════════
md("## 2. Reproducibility & Device")

code("""
# ── Reproducibility ────────────────────────────────────────────────────
SEED = 42

def set_seed(seed: int = SEED) -> None:
    \"\"\"Seed all relevant RNGs for reproducible results.\"\"\"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(SEED)

# ── Device ──────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 60)
print("ALM — MAD Audio Encoder Training")
print("=" * 60)
print(f"Device        : {device}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU           : {torch.cuda.get_device_name(0)}")
    print(f"CUDA version  : {torch.version.cuda}")
print("=" * 60)
""")

# ══════════════════════════════════════════════════════════════════════
# 3. Configuration
# ══════════════════════════════════════════════════════════════════════
md("""## 3. Configuration

**Important:** MAD clips range from ~1–10 seconds, unlike the original 5-second ALM
configuration. This experiment intentionally uses `DURATION = 10.0` so that no MAD clip
loses temporal information through trimming.
""")

code('''
class Config:
    """Central configuration for the MAD single-label encoder training run."""

    # Audio (MAD-specific — do NOT reuse the old 5-second ALM value)
    SAMPLE_RATE: int = 16000
    DURATION: float = 10.0                      # seconds — MAD-specific (NOT 5.0)
    NUM_SAMPLES: int = 160000                   # SAMPLE_RATE * DURATION

    # Mel-spectrogram
    N_MELS: int = 128
    N_FFT: int = 1024
    HOP_LENGTH: int = 512

    # Model
    EMBED_DIM: int = 512
    NUM_CLASSES: int = 7
    DROPOUT: float = 0.3

    # Training
    BATCH_SIZE: int = 16
    EPOCHS: int = 50
    LEARNING_RATE: float = 3e-4
    WEIGHT_DECAY: float = 1e-4
    VAL_SPLIT: float = 0.20
    PATIENCE: int = 8
    NUM_WORKERS: int = 0                        # kept at 0 for Windows compatibility

    # Mixed precision
    AMP: bool = torch.cuda.is_available()

    SEED: int = SEED


CFG = Config()

assert CFG.NUM_SAMPLES == int(CFG.SAMPLE_RATE * CFG.DURATION), "NUM_SAMPLES must match SAMPLE_RATE * DURATION"

print("=" * 60)
print("Configuration")
print("=" * 60)
for key, value in vars(Config).items():
    if not key.startswith("_"):
        print(f"{key:15s}: {value}")
print("=" * 60)
''')

# ══════════════════════════════════════════════════════════════════════
# 4. Project Paths
# ══════════════════════════════════════════════════════════════════════
md("## 4. Project Paths")

code('''
# ── Project root discovery ──────────────────────────────────────────────
# This notebook lives at <PROJECT_ROOT>/scripts/alm_mad_encoder_training.ipynb.
# We resolve PROJECT_ROOT robustly from the notebook's own working directory
# rather than assuming any absolute path.
_cwd = Path.cwd().resolve()
PROJECT_ROOT = _cwd.parent if _cwd.name == "scripts" else _cwd

DATA_DIR = PROJECT_ROOT / "data" / "processed_mad"
AUDIO_DIR = DATA_DIR / "audio"
METADATA_FILE = DATA_DIR / "metadata.csv"

MODEL_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("Project Paths")
print("=" * 60)
print(f"Project Root    : {PROJECT_ROOT}")
print(f"Metadata        : {METADATA_FILE}")
print(f"Audio Directory : {AUDIO_DIR}")
print(f"Model Directory : {MODEL_DIR}")
print(f"Figures Directory: {FIGURES_DIR}")
print("=" * 60)

if not METADATA_FILE.exists():
    raise FileNotFoundError(
        f"MAD metadata file not found at: {METADATA_FILE}\\n"
        "Expected the MAD dataset to already be preprocessed under "
        "'data/processed_mad/' (metadata.csv + audio/)."
    )
''')

# ══════════════════════════════════════════════════════════════════════
# 5. Load and Inspect MAD Metadata
# ══════════════════════════════════════════════════════════════════════
md("## 5. Load and Inspect MAD Metadata")

code('''
metadata = pd.read_csv(METADATA_FILE)

print("=" * 60)
print("MAD Metadata — Raw Inspection")
print("=" * 60)
print(f"Shape   : {metadata.shape}")
print(f"Columns : {list(metadata.columns)}")
print("-" * 60)
print(metadata.head())
print("=" * 60)
''')

code('''
# ── Discover the relevant columns dynamically ───────────────────────────
# We inspect the actual CSV rather than assuming specific column names.

def _find_column(columns, candidates):
    """Return the first column name (case-insensitive) matching any candidate."""
    lower_map = {c.lower(): c for c in columns}
    for candidate in candidates:
        if candidate in lower_map:
            return lower_map[candidate]
    return None


columns = list(metadata.columns)

PATH_COL = _find_column(columns, ["path", "filepath", "file_path", "audio_path", "filename", "file"])
LABEL_COL = _find_column(columns, ["label", "class", "class_id", "target", "y"])
SPLIT_COL = _find_column(columns, ["split", "subset", "set", "partition"])

if PATH_COL is None or LABEL_COL is None:
    raise ValueError(
        f"Could not identify required columns in metadata.csv. "
        f"Found columns: {columns}. Expected an audio path column and a numeric label column."
    )

print("Discovered columns:")
print(f"  Audio path column : '{PATH_COL}'")
print(f"  Label column      : '{LABEL_COL}'")
print(f"  Split column      : '{SPLIT_COL}'" if SPLIT_COL else "  Split column      : NOT FOUND")
''')

code('''
# ── Dataset summary ──────────────────────────────────────────────────────
labels_sorted = sorted(metadata[LABEL_COL].unique().tolist())
num_classes_found = len(labels_sorted)

if num_classes_found != CFG.NUM_CLASSES:
    print(
        f"⚠ WARNING: metadata.csv contains {num_classes_found} unique labels "
        f"({labels_sorted}), but CFG.NUM_CLASSES = {CFG.NUM_CLASSES}."
    )

print("=" * 60)
print("MAD Dataset")
print("=" * 60)
print(f"Total samples : {len(metadata)}")
print(f"Classes       : {num_classes_found}")
print(f"Labels        : {labels_sorted}")
if SPLIT_COL:
    print(f"Splits found  : {sorted(metadata[SPLIT_COL].unique().tolist())}")
print("=" * 60)

class_distribution = metadata[LABEL_COL].value_counts().sort_index()
class_distribution.index = [f"Class {i}" for i in class_distribution.index]
print("\\nClass distribution:")
print(class_distribution.to_string())
''')

code('''
# ── Class distribution figure ────────────────────────────────────────────
plt.style.use("dark_background")
fig, ax = plt.subplots(figsize=(9, 5))

bars = ax.bar(
    class_distribution.index,
    class_distribution.values,
    color="#4FC3F7",
    edgecolor="white",
    linewidth=0.5,
)
ax.set_title("MAD Dataset — Class Distribution", fontsize=13, fontweight="bold")
ax.set_xlabel("Class")
ax.set_ylabel("Number of Samples")
ax.grid(axis="y", alpha=0.3)

for bar, value in zip(bars, class_distribution.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), str(value),
            ha="center", va="bottom", fontsize=9, color="white")

fig.tight_layout()
class_dist_path = FIGURES_DIR / "mad_training_class_distribution.png"
fig.savefig(class_dist_path, dpi=150, facecolor=fig.get_facecolor())
plt.show()

print(f"Saved figure → {class_dist_path}")
''')

# ══════════════════════════════════════════════════════════════════════
# 6. Train / Validation / Test Split
# ══════════════════════════════════════════════════════════════════════
md("""## 6. Train / Validation / Test Split

MAD already provides an official training/test split. The **official test set is kept
completely untouched** until final evaluation (Section 18). Only the official training
partition is split into training/validation (80/20, stratified by label).
""")

code('''
if SPLIT_COL is not None:
    split_values = metadata[SPLIT_COL].astype(str).str.lower()
    train_mask = split_values.str.contains("train")
    test_mask = split_values.str.contains("test")

    official_train_df = metadata[train_mask].reset_index(drop=True)
    official_test_df = metadata[test_mask].reset_index(drop=True)

    if len(official_train_df) == 0 or len(official_test_df) == 0:
        raise ValueError(
            f"Split column '{SPLIT_COL}' was found but training/test partitions could not be "
            f"identified from its values: {sorted(split_values.unique().tolist())}"
        )
else:
    # No explicit split column: derive the split from the audio path, which is expected
    # to contain 'training' or 'test' as described in the project structure.
    path_values = metadata[PATH_COL].astype(str).str.lower()
    train_mask = path_values.str.contains("train")
    test_mask = path_values.str.contains("test")

    official_train_df = metadata[train_mask].reset_index(drop=True)
    official_test_df = metadata[test_mask].reset_index(drop=True)

    if len(official_train_df) == 0 or len(official_test_df) == 0:
        raise ValueError(
            "Could not identify a training/test split from metadata.csv. Neither a split "
            "column nor path-based 'training'/'test' markers were found."
        )

print(f"Original training samples : {len(official_train_df)}")
print(f"Official test samples     : {len(official_test_df)}")
''')

code('''
# ── Stratified train/validation split (test set is NEVER touched here) ──
try:
    train_df, val_df = train_test_split(
        official_train_df,
        test_size=CFG.VAL_SPLIT,
        random_state=CFG.SEED,
        stratify=official_train_df[LABEL_COL],
    )
except ValueError as exc:
    raise ValueError(
        "Stratified train/validation split failed. This usually means a class has too few "
        f"samples to be stratified. Original error: {exc}"
    ) from exc

train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)
test_df = official_test_df.reset_index(drop=True)

print("=" * 60)
print("Train / Validation / Test Split")
print("=" * 60)
print(f"Training samples   : {len(train_df)}")
print(f"Validation samples : {len(val_df)}")
print(f"Test samples        : {len(test_df)}")
print("=" * 60)

for name, split_df in [("Train", train_df), ("Validation", val_df), ("Test", test_df)]:
    dist = split_df[LABEL_COL].value_counts().sort_index()
    print(f"\\n{name} class distribution:")
    print(dist.to_string())

missing_train = set(labels_sorted) - set(train_df[LABEL_COL].unique())
missing_val = set(labels_sorted) - set(val_df[LABEL_COL].unique())
if missing_train or missing_val:
    raise ValueError(
        f"Stratified split produced missing classes. Missing in train: {missing_train}, "
        f"missing in val: {missing_val}."
    )
''')

# ══════════════════════════════════════════════════════════════════════
# 7. Audio Loading & Preprocessing
# ══════════════════════════════════════════════════════════════════════
md("""## 7. Audio Loading & Preprocessing

MAD clips range from ~1–10 seconds, with no clip exceeding 10 seconds. To preserve all
temporal information:

- clips **shorter** than 10s are zero-padded at the end
- clips **exactly** 10s are left unchanged
- clips **longer** than 10s should never occur — this raises an explicit error instead of
  silently trimming audio.
""")

code('''
def load_audio(filepath: Path) -> np.ndarray:
    """
    Load a MAD audio file, convert to mono, resample to CFG.SAMPLE_RATE, and
    zero-pad to exactly CFG.NUM_SAMPLES.

    10 seconds is intentionally used for MAD so that all valid MAD clips can be
    represented without temporal information loss.

    Args:
        filepath: path to the audio file.

    Returns:
        1-D float32 numpy array of length CFG.NUM_SAMPLES.

    Raises:
        FileNotFoundError: if the audio file does not exist.
        ValueError: if the loaded audio exceeds CFG.NUM_SAMPLES (the 10-second limit).
    """
    if not filepath.exists():
        raise FileNotFoundError(f"Audio file not found: {filepath}")

    y, _ = librosa.load(filepath, sr=CFG.SAMPLE_RATE, mono=True)
    y = y.astype(np.float32)

    if len(y) > CFG.NUM_SAMPLES:
        raise ValueError(
            f"Audio sample exceeds the configured 10-second limit "
            f"({len(y)} samples > {CFG.NUM_SAMPLES} samples) for file: {filepath}. "
            "This pipeline intentionally does not trim audio to avoid losing information; "
            "please check this file, as no MAD clip is expected to exceed 10 seconds."
        )

    if len(y) < CFG.NUM_SAMPLES:
        pad_width = CFG.NUM_SAMPLES - len(y)
        y = np.pad(y, (0, pad_width), mode="constant")

    return y


def compute_melspec(y: np.ndarray) -> np.ndarray:
    """
    Compute a log-Mel spectrogram from a waveform.

    Args:
        y: 1-D float32 waveform at CFG.SAMPLE_RATE.

    Returns:
        (N_MELS, ~313) float32 log-Mel spectrogram in dB.
    """
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=CFG.SAMPLE_RATE,
        n_fft=CFG.N_FFT,
        hop_length=CFG.HOP_LENGTH,
        n_mels=CFG.N_MELS,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)
    return log_mel.astype(np.float32)


def preprocess_audio(filepath: Path) -> np.ndarray:
    """Load, pad to 10 seconds, and compute the log-Mel spectrogram for a file."""
    y = load_audio(filepath)
    return compute_melspec(y)


print("Audio preprocessing functions ready.")
print(f"Target sample rate : {CFG.SAMPLE_RATE} Hz")
print(f"Target duration    : {CFG.DURATION} s ({CFG.NUM_SAMPLES} samples)")
''')

code('''
# ── Quick preprocessing sanity check on a real file ─────────────────────
_sample_row = train_df.iloc[0]
_sample_path = AUDIO_DIR / str(_sample_row[PATH_COL])

_mel_sample = preprocess_audio(_sample_path)
print(f"Sample file      : {_sample_path.name}")
print(f"Mel spectrogram  : {_mel_sample.shape}  (expected ~({CFG.N_MELS}, 313))")
''')

# ══════════════════════════════════════════════════════════════════════
# 8. Dataset Class
# ══════════════════════════════════════════════════════════════════════
md("## 8. Dataset Class")

code('''
class MADAudioDataset(Dataset):
    """
    PyTorch Dataset for single-label MAD audio classification.

    Audio is loaded and processed lazily in __getitem__ to avoid loading the
    entire dataset into memory.
    """

    def __init__(self, dataframe: pd.DataFrame, audio_dir: Path, path_col: str, label_col: str):
        self.df = dataframe.reset_index(drop=True)
        self.audio_dir = audio_dir
        self.path_col = path_col
        self.label_col = label_col

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        filepath = self.audio_dir / str(row[self.path_col])

        if not filepath.exists():
            raise FileNotFoundError(
                f"Expected MAD audio file not found at: {filepath} (row index {idx})"
            )

        mel = preprocess_audio(filepath)          # (128, ~313)
        mel_tensor = torch.from_numpy(mel).unsqueeze(0)  # (1, 128, ~313)

        label = torch.tensor(int(row[self.label_col]), dtype=torch.long)

        return {"mel": mel_tensor, "label": label}


print("MADAudioDataset ready.")
''')

# ══════════════════════════════════════════════════════════════════════
# 9. DataLoaders
# ══════════════════════════════════════════════════════════════════════
md("## 9. DataLoaders")

code('''
train_dataset = MADAudioDataset(train_df, AUDIO_DIR, PATH_COL, LABEL_COL)
val_dataset = MADAudioDataset(val_df, AUDIO_DIR, PATH_COL, LABEL_COL)
test_dataset = MADAudioDataset(test_df, AUDIO_DIR, PATH_COL, LABEL_COL)

_pin_memory = torch.cuda.is_available()

train_loader = DataLoader(
    train_dataset, batch_size=CFG.BATCH_SIZE, shuffle=True,
    num_workers=CFG.NUM_WORKERS, pin_memory=_pin_memory,
)
val_loader = DataLoader(
    val_dataset, batch_size=CFG.BATCH_SIZE, shuffle=False,
    num_workers=CFG.NUM_WORKERS, pin_memory=_pin_memory,
)
test_loader = DataLoader(
    test_dataset, batch_size=CFG.BATCH_SIZE, shuffle=False,
    num_workers=CFG.NUM_WORKERS, pin_memory=_pin_memory,
)

print(f"Train batches      : {len(train_loader)}")
print(f"Validation batches : {len(val_loader)}")
print(f"Test batches       : {len(test_loader)}")
print(f"Batch size         : {CFG.BATCH_SIZE}")
''')

code('''
# ── Single batch sanity check ────────────────────────────────────────────
_batch = next(iter(train_loader))
print(f"Mel batch shape  : {_batch['mel'].shape}")
print(f"Label batch shape: {_batch['label'].shape}")
print(f"Label values     : {_batch['label'].tolist()}")
''')

# ══════════════════════════════════════════════════════════════════════
# 10. Model Creation
# ══════════════════════════════════════════════════════════════════════
md("## 10. Model Creation")

code('''
from utils.audio_mad_encoder import MADAudioEncoder

model = MADAudioEncoder(
    embed_dim=CFG.EMBED_DIM,
    num_classes=CFG.NUM_CLASSES,
    dropout=CFG.DROPOUT,
).to(device)

print("=" * 60)
print("MAD Audio Encoder")
print("=" * 60)
print("Architecture : CNN + Projection Head")
print(f"Embedding dim: {CFG.EMBED_DIM}")
print(f"Classes      : {CFG.NUM_CLASSES}")
print(f"Parameters   : {model.count_params():,}")
print(f"Device       : {device}")
print("=" * 60)
''')

code('''
# ── Forward-pass sanity check on a real batch ────────────────────────────
model.eval()
with torch.no_grad():
    _mel = _batch["mel"].to(device)
    _embedding = model.encode(_mel)
    _logits = model(_mel)

print(f"Input shape     : {tuple(_mel.shape)}")
print(f"Embedding shape : {tuple(_embedding.shape)}")
print(f"Logits shape    : {tuple(_logits.shape)}")

_norms = _embedding.norm(dim=-1)
print(f"Embedding norms : mean={_norms.mean().item():.4f}, "
      f"min={_norms.min().item():.4f}, max={_norms.max().item():.4f}")
''')

# ══════════════════════════════════════════════════════════════════════
# 11. Loss, Optimizer & Scheduler
# ══════════════════════════════════════════════════════════════════════
md("""## 11. Loss, Optimizer & Scheduler

This is a **single-label** classification problem: `nn.CrossEntropyLoss` operates on raw
logits and integer class targets. `BCEWithLogitsLoss` and sigmoid activations are not used
anywhere in this notebook.
""")

code('''
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=CFG.LEARNING_RATE,
    weight_decay=CFG.WEIGHT_DECAY,
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG.EPOCHS)

scaler = torch.amp.GradScaler("cuda", enabled=CFG.AMP)

print(f"Criterion : {criterion.__class__.__name__}")
print(f"Optimizer : {optimizer.__class__.__name__} (lr={CFG.LEARNING_RATE}, weight_decay={CFG.WEIGHT_DECAY})")
print(f"Scheduler : {scheduler.__class__.__name__} (T_max={CFG.EPOCHS})")
print(f"AMP       : {CFG.AMP}")
''')

# ══════════════════════════════════════════════════════════════════════
# 12. Training Functions
# ══════════════════════════════════════════════════════════════════════
md("## 12. Training Functions")

code('''
def train_one_epoch(model, loader, optimizer, criterion, scaler, device, epoch, total_epochs) -> tuple[float, float]:
    """Run one training epoch. Returns (avg_loss, accuracy)."""
    model.train()
    running_loss = 0.0
    running_correct = 0
    seen = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs}", leave=False)
    for batch in pbar:
        mel = batch["mel"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=CFG.AMP):
            logits = model(mel)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        running_correct += (logits.argmax(dim=-1) == labels).sum().item()
        seen += batch_size

        pbar.set_postfix(
            loss=f"{running_loss / seen:.4f}",
            acc=f"{running_correct / seen * 100:.2f}%",
            lr=f"{optimizer.param_groups[0]['lr']:.2e}",
        )

    return running_loss / seen, running_correct / seen


@torch.no_grad()
def evaluate(model, loader, criterion, device, desc: str = "Validation") -> tuple[float, float]:
    """Run evaluation over a loader. Returns (avg_loss, accuracy)."""
    model.eval()
    running_loss = 0.0
    running_correct = 0
    seen = 0

    pbar = tqdm(loader, desc=desc, leave=False)
    for batch in pbar:
        mel = batch["mel"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=CFG.AMP):
            logits = model(mel)
            loss = criterion(logits, labels)

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        running_correct += (logits.argmax(dim=-1) == labels).sum().item()
        seen += batch_size

        pbar.set_postfix(
            loss=f"{running_loss / seen:.4f}",
            acc=f"{running_correct / seen * 100:.2f}%",
        )

    return running_loss / seen, running_correct / seen


print("Training and evaluation functions ready.")
''')

# ══════════════════════════════════════════════════════════════════════
# 13. Training Loop
# ══════════════════════════════════════════════════════════════════════
md("## 13. Training Loop")

code('''
history = {
    "train_loss": [],
    "val_loss": [],
    "train_acc": [],
    "val_acc": [],
    "lr": [],
}

BEST_MODEL_PATH = MODEL_DIR / "alm_mad_audio_encoder_best.pth"
FINAL_MODEL_PATH = MODEL_DIR / "alm_mad_audio_encoder_final.pth"

_checkpoint_config = {
    "sample_rate": CFG.SAMPLE_RATE,
    "duration": CFG.DURATION,
    "num_samples": CFG.NUM_SAMPLES,
    "n_mels": CFG.N_MELS,
    "n_fft": CFG.N_FFT,
    "hop_length": CFG.HOP_LENGTH,
    "embed_dim": CFG.EMBED_DIM,
    "num_classes": CFG.NUM_CLASSES,
    "batch_size": CFG.BATCH_SIZE,
    "learning_rate": CFG.LEARNING_RATE,
    "weight_decay": CFG.WEIGHT_DECAY,
    "dropout": CFG.DROPOUT,
}

best_val_loss = float("inf")
best_epoch = 0
epochs_without_improvement = 0

training_start_time = time.time()

for epoch in range(1, CFG.EPOCHS + 1):
    epoch_start = time.time()

    train_loss, train_acc = train_one_epoch(
        model, train_loader, optimizer, criterion, scaler, device, epoch, CFG.EPOCHS
    )
    val_loss, val_acc = evaluate(model, val_loader, criterion, device, desc=f"Validation {epoch}/{CFG.EPOCHS}")

    scheduler.step()
    current_lr = optimizer.param_groups[0]["lr"]

    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["train_acc"].append(train_acc)
    history["val_acc"].append(val_acc)
    history["lr"].append(current_lr)

    epoch_time = time.time() - epoch_start

    improved = val_loss < best_val_loss
    if improved:
        best_val_loss = val_loss
        best_epoch = epoch
        epochs_without_improvement = 0

        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "val_loss": val_loss,
            "val_acc": val_acc,
            "config": _checkpoint_config,
            "label_classes": labels_sorted,
            "dataset": "MAD",
            "single_label": True,
        }, BEST_MODEL_PATH)
    else:
        epochs_without_improvement += 1

    print(
        f"Epoch {epoch:02d}/{CFG.EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc * 100:.2f}% | "
        f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc * 100:.2f}% | "
        f"LR: {current_lr:.2e} | Time: {epoch_time:.1f}s"
        + ("  ✅ Best model saved" if improved else "")
    )
    if improved:
        print(f"  ✅ Best model saved → {BEST_MODEL_PATH}")

    if epochs_without_improvement >= CFG.PATIENCE:
        print(f"\\n⏹ Early stopping triggered at epoch {epoch}")
        break

total_training_time = time.time() - training_start_time

print("\\n" + "=" * 60)
print("Training Complete")
print("=" * 60)
print(f"Best epoch    : {best_epoch}")
print(f"Best val loss : {best_val_loss:.4f}")
print(f"Best val acc  : {history['val_acc'][best_epoch - 1] * 100:.2f}%")
print(f"Training time : {total_training_time / 60:.2f} min")
print(f"Best model    : {BEST_MODEL_PATH}")
print("=" * 60)
''')

# ══════════════════════════════════════════════════════════════════════
# 14. Early Stopping (documented — implemented within the training loop)
# ══════════════════════════════════════════════════════════════════════
md("""## 14. Early Stopping

Early stopping is implemented directly inside the training loop above (Section 13), using
`PATIENCE = 8` consecutive epochs without validation-loss improvement. The cell below simply
reports the outcome for clarity.
""")

code('''
print(f"Patience              : {CFG.PATIENCE}")
print(f"Epochs completed      : {len(history['train_loss'])}")
print(f"Early stopping fired  : {epochs_without_improvement >= CFG.PATIENCE}")
''')

# ══════════════════════════════════════════════════════════════════════
# 15. Save Final Model
# ══════════════════════════════════════════════════════════════════════
md("## 15. Save Final Model")

code('''
torch.save({
    "model_state": model.state_dict(),
    "optimizer_state": optimizer.state_dict(),
    "scheduler_state": scheduler.state_dict(),
    "history": history,
    "config": _checkpoint_config,
    "label_classes": labels_sorted,
    "best_epoch": best_epoch,
    "dataset": "MAD",
    "single_label": True,
}, FINAL_MODEL_PATH)

print(f"✅ Final model saved → {FINAL_MODEL_PATH}")

print("\\nFiles in models directory:")
for f in sorted(MODEL_DIR.glob("*.pth")):
    size_mb = f.stat().st_size / (1024 * 1024)
    print(f"  {f.name:45s} {size_mb:8.2f} MB")
''')

# ══════════════════════════════════════════════════════════════════════
# 16. Training Curves
# ══════════════════════════════════════════════════════════════════════
md("## 16. Training Curves")

code('''
epochs_range = range(1, len(history["train_loss"]) + 1)

fig = plt.figure(figsize=(15, 4.5))
gs = gridspec.GridSpec(1, 3, figure=fig)

# Loss
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(epochs_range, history["train_loss"], label="Train", color="#4FC3F7", linewidth=2)
ax1.plot(epochs_range, history["val_loss"], label="Validation", color="#FF7043", linewidth=2)
ax1.axvline(best_epoch, color="white", linestyle="--", alpha=0.4, label="Best epoch")
ax1.set_title("MAD Audio Encoder — Loss", fontweight="bold")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.legend()
ax1.grid(alpha=0.3)

# Accuracy
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(epochs_range, [a * 100 for a in history["train_acc"]], label="Train", color="#4FC3F7", linewidth=2)
ax2.plot(epochs_range, [a * 100 for a in history["val_acc"]], label="Validation", color="#FF7043", linewidth=2)
ax2.axvline(best_epoch, color="white", linestyle="--", alpha=0.4, label="Best epoch")
ax2.set_title("MAD Audio Encoder — Accuracy", fontweight="bold")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Accuracy (%)")
ax2.legend()
ax2.grid(alpha=0.3)

# Learning rate
ax3 = fig.add_subplot(gs[0, 2])
ax3.plot(epochs_range, history["lr"], color="#AB47BC", linewidth=2)
ax3.set_title("MAD Audio Encoder — Learning Rate", fontweight="bold")
ax3.set_xlabel("Epoch")
ax3.set_ylabel("Learning Rate")
ax3.grid(alpha=0.3)

fig.tight_layout()
curves_path = FIGURES_DIR / "mad_encoder_training_curves.png"
fig.savefig(curves_path, dpi=150, facecolor=fig.get_facecolor())
plt.show()
print(f"Saved figure → {curves_path}")
''')

code('''
# Separate loss figure
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(epochs_range, history["train_loss"], label="Train", color="#4FC3F7", linewidth=2)
ax.plot(epochs_range, history["val_loss"], label="Validation", color="#FF7043", linewidth=2)
ax.set_title("MAD Audio Encoder — Loss", fontweight="bold")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
loss_path = FIGURES_DIR / "mad_encoder_loss.png"
fig.savefig(loss_path, dpi=150, facecolor=fig.get_facecolor())
plt.show()
print(f"Saved figure → {loss_path}")
''')

code('''
# Separate accuracy figure
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(epochs_range, [a * 100 for a in history["train_acc"]], label="Train", color="#4FC3F7", linewidth=2)
ax.plot(epochs_range, [a * 100 for a in history["val_acc"]], label="Validation", color="#FF7043", linewidth=2)
ax.set_title("MAD Audio Encoder — Accuracy", fontweight="bold")
ax.set_xlabel("Epoch")
ax.set_ylabel("Accuracy (%)")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
acc_path = FIGURES_DIR / "mad_encoder_accuracy.png"
fig.savefig(acc_path, dpi=150, facecolor=fig.get_facecolor())
plt.show()
print(f"Saved figure → {acc_path}")
''')

# ══════════════════════════════════════════════════════════════════════
# 17. Load Best Model
# ══════════════════════════════════════════════════════════════════════
md("## 17. Load Best Model")

code('''
best_checkpoint = torch.load(BEST_MODEL_PATH, map_location=device, weights_only=False)

best_model = MADAudioEncoder(
    embed_dim=best_checkpoint["config"]["embed_dim"],
    num_classes=best_checkpoint["config"]["num_classes"],
    dropout=CFG.DROPOUT,
).to(device)
best_model.load_state_dict(best_checkpoint["model_state"])
best_model.eval()

print("✅ Loaded best MAD encoder")
print(f"Epoch              : {best_checkpoint['epoch']}")
print(f"Validation loss     : {best_checkpoint['val_loss']:.4f}")
print(f"Validation accuracy : {best_checkpoint['val_acc'] * 100:.2f}%")
print(f"Embedding dimension : {best_checkpoint['config']['embed_dim']}")
print(f"Number of classes   : {best_checkpoint['config']['num_classes']}")
''')

# ══════════════════════════════════════════════════════════════════════
# 18. Test Evaluation
# ══════════════════════════════════════════════════════════════════════
md("""## 18. Test Evaluation

The official MAD test set has not been used anywhere in training or validation up to this
point. It is evaluated here for the first and only time.
""")

code('''
@torch.no_grad()
def run_test_evaluation(model, loader, criterion, device):
    """Evaluate the model on the held-out test set and return metrics + predictions."""
    model.eval()
    running_loss = 0.0
    seen = 0

    all_preds, all_labels, all_probs = [], [], []

    pbar = tqdm(loader, desc="Testing", leave=True)
    for batch in pbar:
        mel = batch["mel"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        logits = model(mel)
        loss = criterion(logits, labels)

        probs = torch.softmax(logits, dim=-1)
        preds = probs.argmax(dim=-1)

        running_loss += loss.item() * labels.size(0)
        seen += labels.size(0)

        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())
        all_probs.append(probs.cpu())

        pbar.set_postfix(loss=f"{running_loss / seen:.4f}")

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    all_probs = torch.cat(all_probs).numpy()

    test_loss = running_loss / seen
    return test_loss, all_preds, all_labels, all_probs


test_loss, test_preds, test_labels, test_probs = run_test_evaluation(best_model, test_loader, criterion, device)

test_acc = accuracy_score(test_labels, test_preds)
precision, recall, f1_macro, _ = precision_recall_fscore_support(
    test_labels, test_preds, average="macro", zero_division=0
)
_, _, f1_weighted, _ = precision_recall_fscore_support(
    test_labels, test_preds, average="weighted", zero_division=0
)

print("=" * 60)
print("MAD Audio Encoder — Test Results")
print("=" * 60)
print(f"Test Loss      : {test_loss:.4f}")
print(f"Test Accuracy  : {test_acc * 100:.2f}%")
print(f"Macro Precision: {precision:.4f}")
print(f"Macro Recall   : {recall:.4f}")
print(f"Macro F1       : {f1_macro:.4f}")
print(f"Weighted F1    : {f1_weighted:.4f}")
print("=" * 60)

target_names = [f"Class {c}" for c in labels_sorted]
print("\\nClassification Report:")
print(classification_report(test_labels, test_preds, target_names=target_names, zero_division=0))
''')

# ══════════════════════════════════════════════════════════════════════
# 19. Confusion Matrix
# ══════════════════════════════════════════════════════════════════════
md("## 19. Confusion Matrix")

code('''
cm = confusion_matrix(test_labels, test_preds, labels=labels_sorted)
cm_normalized = cm.astype(np.float64) / cm.sum(axis=1, keepdims=True)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax, matrix, title, fmt in [
    (axes[0], cm, "Confusion Matrix — Counts", "d"),
    (axes[1], cm_normalized, "Confusion Matrix — Normalized", ".2f"),
]:
    im = ax.imshow(matrix, cmap="viridis")
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks(range(len(labels_sorted)))
    ax.set_yticks(range(len(labels_sorted)))
    ax.set_xticklabels(target_names, rotation=45, ha="right")
    ax.set_yticklabels(target_names)

    threshold = matrix.max() / 2.0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            text = f"{value:{fmt}}"
            color = "white" if value < threshold else "black"
            ax.text(j, i, text, ha="center", va="center", color=color, fontsize=8)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

fig.tight_layout()
cm_path = FIGURES_DIR / "mad_encoder_confusion_matrix.png"
fig.savefig(cm_path, dpi=150, facecolor=fig.get_facecolor())
plt.show()
print(f"Saved figure → {cm_path}")
''')

# ══════════════════════════════════════════════════════════════════════
# 20. Embedding Sanity Check
# ══════════════════════════════════════════════════════════════════════
md("## 20. Embedding Sanity Check")

code('''
_check_batch = next(iter(test_loader))
with torch.no_grad():
    _mel = _check_batch["mel"].to(device)
    _embeddings = best_model.encode(_mel)

_norms = _embeddings.norm(dim=-1)

print(f"Embedding shape : {tuple(_embeddings.shape)}")
print(f"Mean norm       : {_norms.mean().item():.4f}")
print(f"Min norm        : {_norms.min().item():.4f}")
print(f"Max norm        : {_norms.max().item():.4f}")
''')

# ══════════════════════════════════════════════════════════════════════
# 21. Example Predictions
# ══════════════════════════════════════════════════════════════════════
md("## 21. Example Predictions")

code('''
N_EXAMPLES = 10

example_indices = np.random.RandomState(CFG.SEED).choice(len(test_labels), size=min(N_EXAMPLES, len(test_labels)), replace=False)

print("Example Test Predictions")
print("-" * 60)
for i, idx in enumerate(example_indices, start=1):
    actual = test_labels[idx]
    predicted = test_preds[idx]
    confidence = test_probs[idx, predicted] * 100
    print(f"Sample {i:02d} | Actual: Class {actual} | Predicted: Class {predicted} | Confidence: {confidence:.2f}%")
''')

# ══════════════════════════════════════════════════════════════════════
# 22. Final Summary
# ══════════════════════════════════════════════════════════════════════
md("## 22. Final Summary")

code('''
summary_lines = [
    "╔══════════════════════════════════════════════════════════════╗",
    "║             MAD AUDIO ENCODER — FINAL SUMMARY                 ║",
    "╠══════════════════════════════════════════════════════════════╣",
    f"║ Dataset          : MAD".ljust(65) + "║",
    f"║ Task             : Single-label audio classification".ljust(65) + "║",
    f"║ Classes          : {CFG.NUM_CLASSES}".ljust(65) + "║",
    f"║ Sample rate      : {CFG.SAMPLE_RATE} Hz".ljust(65) + "║",
    f"║ Audio duration   : {CFG.DURATION:.0f} seconds".ljust(65) + "║",
    f"║ Mel bins         : {CFG.N_MELS}".ljust(65) + "║",
    f"║ FFT              : {CFG.N_FFT}".ljust(65) + "║",
    f"║ Hop length       : {CFG.HOP_LENGTH}".ljust(65) + "║",
    f"║ Embedding dim     : {CFG.EMBED_DIM}".ljust(65) + "║",
    f"║ Test accuracy    : {test_acc * 100:.2f}%".ljust(65) + "║",
    f"║ Macro F1         : {f1_macro:.4f}".ljust(65) + "║",
    f"║ Best checkpoint  : {BEST_MODEL_PATH.name}".ljust(65) + "║",
    "╚══════════════════════════════════════════════════════════════╝",
]
print("\\n".join(summary_lines))

print("\\nSaved figures:")
for path in [class_dist_path, curves_path, loss_path, acc_path, cm_path]:
    print(f"  {path}")
''')

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.10"},
}

out_path = "alm_mad_encoder_training.ipynb"
with open(out_path, "w" , encoding ="utf-8") as f:
    nbf.write(nb, f)

print(f"Notebook written to {out_path} with {len(cells)} cells.")