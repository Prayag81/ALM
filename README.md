# 🎧 Audio Language Model (ALM)

A CNN-based audio understanding system that classifies environmental sounds and extracts semantic audio embeddings. Trained on a combined **ESC-50 + UrbanSound8K** dataset (58 classes).

---

## 📁 Project Structure

```
ALM/
├── demo.py                          # ← Inference entry point (this file)
├── requirements.txt
├── AUDIO_ENCODER_WORKFLOW.md        # End-to-end pipeline documentation
│
├── models/
│   ├── alm_audio_encoder_best.pth   # Single-label encoder (best val acc)
│   ├── alm_audio_encoder_final.pth  # Single-label encoder (final epoch)
│   ├── alm_multilabel_best.pth      # Multi-label encoder (best val F1)
│   └── alm_multilabel_final.pth     # Multi-label encoder (final epoch)
│
├── utils/
│   └── audio_encoder.py             # Model definitions: AudioEncoder, MultiLabelAudioEncoder
│
├── script/
│   ├── audio_dataset.py             # PyTorch datasets & MEL transform configs
│   ├── alm_preprocessing.ipynb      # Data preprocessing pipeline
│   ├── alm_audio_encoder_training.ipynb   # Single-label training loop
│   ├── alm_audio_encoder_multilabel.ipynb # Multi-label training & evaluation
│   └── alm_inference_demo.ipynb     # Embedding viz, similarity search
│
├── data/
│   ├── raw/                         # Original ESC-50 / UrbanSound8K files
│   └── processed/
│       ├── audio/                   # Standardized .wav files (alm_XXXXXX.wav)
│       └── metadata.csv             # Unified label + fold index
│
└── outputs/
    ├── logs/
    └── figures/
```

---

## 🏗️ Architecture Overview

### 1. `AudioEncoder` — Single-Label
- **Input**: `(B, 1, 128, T)` log-mel spectrogram at **16 kHz**
- **Backbone**: 4 × DoubleConv blocks: `1 → 32 → 64 → 128 → 256` channels
- **Pooling**: `AdaptiveAvgPool2d((4,4))`
- **Projection Head**: `Linear(4096→1024) → LayerNorm → ReLU → Linear(1024→512)`
- **L2 Normalization** → unit-sphere embedding
- **Classifier**: `Linear(512 → num_classes)`

### 2. `MultiLabelAudioEncoder` — Multi-Label
- **Input**: `(B, 1, 128, T)` log-mel spectrogram at **24 kHz**
- **Backbone**: 3 × DoubleConv blocks with GELU activation
- **Pooling**: `AdaptiveAvgPool2d((4,8))`
- **Projector**: `Linear(4096 → 512) → GELU → Linear(512 → embed_dim)`
- **Transformer Encoder**: 6-layer, 8-head self-attention
- **Classifier**: `Linear(embed_dim → num_classes)` — outputs raw logits (no sigmoid)

---

## 🔧 Data Preprocessing Pipeline

Implemented in `script/alm_preprocessing.ipynb`:

| Step | Operation |
|------|-----------|
| 1 | Convert any format → `.wav` (16-bit PCM) |
| 2 | Stereo → Mono (channel average) |
| 3 | Resample to **24,000 Hz** |
| 4 | Trim / zero-pad to **5.0 seconds** (120,000 samples) |
| 5 | Normalize peak amplitude to `[-1, 1]` |
| 6 | Save as `alm_XXXXXX.wav` in `data/processed/audio/` |
| 7 | Write `metadata.csv` with `file`, `label`, `fold`, `dataset` |

---

## 🚀 Quick Start — Inference (`demo.py`)

### Install dependencies
```bash
py -3.10 -m venv venv
venv/scripts/activate
pip install -r requirements.txt
nvidia-smi # to check for the GPU
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

```

### Single-Label Classification
```bash
# Predict the class of any audio file
python demo.py --audio path/to/sound.wav

# Show top-10 predictions
python demo.py --audio path/to/sound.wav --top-k 10

# Use a specific checkpoint
python demo.py --audio path/to/sound.wav --ckpt models/alm_audio_encoder_final.pth
```

### Multi-Label Event Detection
```bash
# Detect multiple overlapping sound events
python demo.py --audio path/to/sound.wav --mode multilabel

# Mix two audio files and detect events
python demo.py --audio sound1.wav --mix sound2.wav --mix-alpha 0.5 --mode multilabel

# Simulate noisy real-world conditions (SNR in dB)
python demo.py --audio sound.wav --noise-snr 10 --mode multilabel

# Full combined demo: mix + noise
python demo.py --audio sound1.wav --mix sound2.wav --noise-snr 15 --mode multilabel
```

### Force CPU / GPU
```bash
python demo.py --audio sound.wav --device cpu
python demo.py --audio sound.wav --device cuda
```

---

## 📊 Inference Pipeline (What `demo.py` Does)

```
Input .wav / .mp3 / .flac
        │
        ▼
┌─────────────────────────────────────────────┐
│  load_and_standardize()                     │
│  1. Load waveform (torchaudio)              │
│  2. Stereo → Mono                           │
│  3. Resample → 24 kHz                       │
│  4. Trim/Pad → 5 s (120,000 samples)        │
│  5. Peak normalize → [-1, 1]                │
└───────────────────┬─────────────────────────┘
                    │
        ┌───────────┴───────────┐
        │                       │
        ▼                       ▼
  [singlelabel]           [multilabel]
wav_to_mel_singlelabel  wav_to_mel_multilabel
 sr=16k, hop=512         sr=24k, hop=256
 AmplitudeToDB           log(mel + 1e-9)
 min-max → [-1,1]        z-score normalize
        │                       │
        ▼                       ▼
  AudioEncoder         MultiLabelAudioEncoder
  logits → Softmax     logits → Sigmoid
  argmax → class       threshold per-class
        │                       │
        ▼                       ▼
  pred_label             detected_events[]
  confidence             probabilities{}
  embedding (512-d)      embedding (256-d)
```

---

## 🏷️ Supported Classes (58 total)

| ESC-50 (Natural & Environmental) | UrbanSound8K (Urban) |
|----------------------------------|----------------------|
| airplane, breathing, brushing_teeth, can_opening, cat, chainsaw, chirping_birds, church_bells, clapping, clock_alarm, clock_tick, coughing, cow, crackling_fire, crickets, crow, crying_baby, dog, door_wood_creaks, door_wood_knock, drinking_sipping, fireworks, footsteps, frog, glass_breaking, hand_saw, hen, insects, keyboard_typing, laughing, mouse_click, pig, pouring_water, rain, rooster, sea_waves, sheep, sneezing, snoring, thunderstorm, toilet_flush, vacuum_cleaner, washing_machine, water_drops, wind | air_conditioner, car_horn, children_playing, dog_bark, drilling, engine, engine_idling, gun_shot, jackhammer, siren, street_music, train |

---

## 🧪 Multi-Label Detection Thresholds

| Class | Threshold | Reason |
|-------|-----------|--------|
| `gun_shot` | 0.55 | High false-positive risk |
| `siren` | 0.50 | Overlaps with alarm/horn |
| `fireworks` | 0.45 | Acoustically close to gun_shot |
| `train` | 0.50 | High activation on fireworks samples |
| *all others* | 0.35 | Default |

> **Fallback**: if no class exceeds its threshold, the single highest-probability class is returned (`top_1_fallback`).

---

## 📦 Requirements

```
torch
torchaudio
numpy
pandas
librosa
soundfile
scikit-learn  # for evaluation metrics
matplotlib    # for embedding visualization
```

Install all at once:
```bash
pip install -r requirements.txt
```

---

## 🔬 Embedding Mode

Both models output **L2-normalized embeddings** that can be used for:
- **Similarity Search**: cosine similarity between audio fingerprints
- **Vector Database**: store embeddings for retrieval (e.g., FAISS, Qdrant)
- **Downstream Tasks**: zero-shot classification, audio clustering

```python
# The embedding is printed in every run, or extract programmatically:
from demo import load_singlelabel_encoder, load_and_standardize, singlelabel_inference

model, labels = load_singlelabel_encoder("models/alm_audio_encoder_best.pth", "cpu")
wav    = load_and_standardize("sound.wav")
result = singlelabel_inference(model, wav, labels, "cpu")
emb    = result["embedding"]   # np.ndarray, shape (512,), L2-norm ≈ 1.0
```

---

## 📓 Notebooks

| Notebook | Purpose |
|----------|---------|
| `script/alm_preprocessing.ipynb` | Build the unified dataset from ESC-50 + UrbanSound8K |
| `script/alm_audio_encoder_training.ipynb` | Train the single-label `AudioEncoder` |
| `script/alm_audio_encoder_multilabel.ipynb` | Train + evaluate `MultiLabelAudioEncoder` |
| `script/alm_inference_demo.ipynb` | Embedding visualization, similarity search, t-SNE |

---

## 📈 Training Setup

| Setting | Single-Label | Multi-Label |
|---------|-------------|-------------|
| Loss | `CrossEntropyLoss` (label_smoothing=0.1) | `BCEWithLogitsLoss` (pos_weight) |
| Optimizer | `AdamW` | `AdamW` |
| Scheduler | `CosineAnnealingLR` | `CosineAnnealingLR` |
| Precision | FP16 mixed (torch.cuda.amp) | FP16 mixed |
| Augmentation | SpecAugment (freq + time masking) | SpecAugment + pitch shift + mixup |
| Sample Rate | 16,000 Hz | 24,000 Hz |
