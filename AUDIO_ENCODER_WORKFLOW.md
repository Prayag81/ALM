# 🎧 ALM: Audio Encoder Workflow Analysis

This document provides a comprehensive breakdown of the **Audio-Language Model (ALM) Audio Encoder** pipeline, connecting the dots between preprocessing, architecture design, training, and inference.

---

## 🏗️ 1. Data Preprocessing Pipeline
*File: `script/alm_preprocessing.ipynb`*

Before any training happens, raw audio from **ESC-50** and **UrbanSound8K** is standardized into a unified format.

1.  **Format Standardization**: Every file is converted to a 16-bit PCM `.wav` format.
2.  **Mono Conversion**: Stereo channels are averaged to create a single-channel waveform.
3.  **Resampling**: All audio is resampled to a consistent **24,000 Hz** (or 16kHz depending on the specific trainer config).
4.  **Temporal Padding/Trimming**: Audio is fixed at **5.0 seconds**.
    *   Shorter clips are zero-padded at the end.
    *   Longer clips are trimmed to the first 5 seconds.
5.  **Amplitude Normalization**: The waveform is scaled so that its peak amplitude is in the `[-1, 1]` range.
6.  **Metadata Generation**: A `metadata.csv` is created that maps these unified files to labels and cross-validation folds.

---

## 🎨 2. Feature Extraction: Mel Spectrograms
*File: `script/audio_dataset.py`*

The model does not process raw waveforms. Instead, it processes "visual" representations of sound.

1.  **STFT**: A Short-Time Fourier Transform is applied (`n_fft=1024`, `hop_length=512`).
2.  **Mel Scaling**: Linear frequencies are converted to the Mel scale (**128 bins**), concentrating on the frequencies humans are most sensitive to.
3.  **Log-Scaling**: Power values are converted to Decibels (dB) via `AmplitudeToDB`.
4.  **SpecAugment (Training Only)**:
    *   **Frequency Masking**: Randomly "hiding" a horizontal band of frequencies.
    *   **Time Masking**: Randomly "hiding" a vertical band of time frames.
    *   *Purpose*: Forces the model to learn features from partial information, preventing overfitting.

---

## 🧠 3. Model Architecture (`AudioEncoder`)
*File: `utils/audio_encoder.py`*

The architecture is a specialized CNN designed to extract deep features from Mel spectrograms.

### A. The "Double Conv" Backbone
The heart of the encoder is the **ConvBlock**, which uses a high-performance double convolution structure:

**Inside one ConvBlock:**
1.  **Conv2d** (3x3) -> **BatchNorm** -> **ReLU**
2.  **Conv2d** (3x3) -> **BatchNorm** -> **ReLU**
3.  **MaxPool2d** (2x2)
4.  **Dropout**

> [!TIP]
> **Why Double Conv?** Two 3x3 layers have the same receptive field as one 5x5 layer but use fewer parameters and include two non-linear activations (ReLU) instead of one, allowing the network to learn much more complex spatial patterns in the sound "image."

### B. Global Flow
*   **Backbone**: 4 stages of these Double ConvBlocks, increasing channel depth: **1 → 32 → 64 → 128 → 256**.
*   **Pooling**: `AdaptiveAvgPool2d((4, 4))` compresses the feature map to a fixed 256x4x4 size.
*   **Projection Head**: A multi-layer perceptron (Flatten -> Linear -> LayerNorm -> ReLU -> Linear) that outputs the final **256 or 512-dim embedding**.
*   **L2 Normalization**: The embedding is projected onto a unit sphere, enabling effective **Cosine Similarity** calculations.
*   **Classifier**: An optional final Linear layer that maps embeddings to class logits.

---

## 🏋️ 4. Training Strategy
*File: `script/alm_audio_encoder_training.ipynb`*

1.  **Loss Function**: `CrossEntropyLoss` with **Label Smoothing (0.1)** to improve generalization.
2.  **Optimizer**: `AdamW` with weight decay for better regularization.
3.  **Scheduler**: `CosineAnnealingLR` to decay the learning rate gracefully.
4.  **FP16 Mixed Precision**: Uses `torch.cuda.amp` to accelerate training and reduce memory footprint on GPUs with limited VRAM (like an RTX 2050).

---

## 🚀 5. Inference and Output
*File: `script/alm_inference_demo.ipynb`*

For any new audio input, the pipeline performs the following:

1.  **Audio -> Mel**: Standardizes input to match the 5s, 128-mel training distribution.
2.  **Classification**: 
    *   `Encoder(mel)` -> **Logits**
    *   `Softmax(logits)` -> **Confidence Scores** (e.g., "Dog Barking: 92%")
3.  **Embedding Extraction**: 
    *   `Encoder.encode(mel)` -> **L2-Normalized Vector**
    *   This vector acts as a "digital fingerprint" of the sound, which can be stored in a vector database for similarity search or retrieval.

---
