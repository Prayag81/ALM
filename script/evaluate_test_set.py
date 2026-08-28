import os
import sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import classification_report, confusion_matrix
import librosa

# Setup path
PROJECT_ROOT = Path("c:/Users/PRAYAG/Desktop/ALM/alm")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.mad_audio_encoder import MADAudioEncoder

# Config matching training
class CFG:
    SAMPLE_RATE = 16000
    DURATION = 10.0
    NUM_SAMPLES = int(SAMPLE_RATE * DURATION)
    N_MELS = 128
    N_FFT = 1024
    HOP_LENGTH = 512
    EMBED_DIM = 512
    NUM_CLASSES = 7
    DROPOUT = 0.3
    BATCH_SIZE = 16
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_audio(filepath: Path) -> np.ndarray:
    y, _ = librosa.load(filepath, sr=CFG.SAMPLE_RATE, mono=True)
    y = y.astype(np.float32)
    if len(y) > CFG.NUM_SAMPLES:
        y = y[:CFG.NUM_SAMPLES]
    elif len(y) < CFG.NUM_SAMPLES:
        pad_width = CFG.NUM_SAMPLES - len(y)
        y = np.pad(y, (0, pad_width), mode="constant")
    return y

def compute_melspec(y: np.ndarray) -> np.ndarray:
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=CFG.SAMPLE_RATE,
        n_fft=CFG.N_FFT,
        hop_length=CFG.HOP_LENGTH,
        n_mels=CFG.N_MELS,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)
    return log_mel.astype(np.float32)

class MADAudioDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, audio_dir: Path):
        self.df = dataframe.reset_index(drop=True)
        self.audio_dir = audio_dir

    def __len__(self) -> int:
        return len(self.df)

    @staticmethod
    def _normalize_mel(mel_tensor: torch.Tensor) -> torch.Tensor:
        """Min-max normalize log-mel spectrogram to [-1, 1] range."""
        mel_min = mel_tensor.min()
        mel_max = mel_tensor.max()
        if mel_max - mel_min > 0:
            mel_tensor = (mel_tensor - mel_min) / (mel_max - mel_min)  # [0, 1]
            mel_tensor = mel_tensor * 2 - 1                           # [-1, 1]
        return mel_tensor

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        filepath = self.audio_dir / str(row["file"])
        mel = compute_melspec(load_audio(filepath))
        mel_tensor = torch.from_numpy(mel).unsqueeze(0)
        mel_tensor = self._normalize_mel(mel_tensor)
        label = torch.tensor(int(row["label"]), dtype=torch.long)
        return {"mel": mel_tensor, "label": label, "file": row["file"]}

def main():
    metadata_file = PROJECT_ROOT / "data" / "processed_mad" / "metadata.csv"
    audio_dir = PROJECT_ROOT / "data" / "processed_mad" / "audio"
    model_path = PROJECT_ROOT / "models" / "alm_mad_audio_encoder_best.pth"

    df = pd.read_csv(metadata_file)
    test_df = df[df["split"].str.lower().str.contains("test")].reset_index(drop=True)
    print(f"Test samples found: {len(test_df)}")

    dataset = MADAudioDataset(test_df, audio_dir)
    loader = DataLoader(dataset, batch_size=CFG.BATCH_SIZE, shuffle=False, num_workers=0)

    # Initialize model
    model = MADAudioEncoder(
        embed_dim=CFG.EMBED_DIM,
        num_classes=CFG.NUM_CLASSES,
        dropout=CFG.DROPOUT,
    )
    
    # Load weights
    checkpoint = torch.load(model_path, map_location=CFG.DEVICE)
    model.load_state_dict(checkpoint["model_state"])
    model.to(CFG.DEVICE)
    model.eval()

    all_preds = []
    all_labels = []
    all_files = []
    
    print("Evaluating test set...")
    with torch.no_grad():
        for batch in loader:
            mels = batch["mel"].to(CFG.DEVICE)
            labels = batch["label"].to(CFG.DEVICE)
            files = batch["file"]
            
            logits = model(mels)
            preds = logits.argmax(dim=-1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_files.extend(files)

    # Compute metrics
    labels_sorted = sorted(list(set(all_labels)))
    label_map = {
        0: "Communication",
        1: "Gunshot",
        2: "Footsteps",
        3: "Shelling",
        4: "Vehicle",
        5: "Helicopter",
        6: "Fighter"
    }
    target_names = [label_map[i] for i in labels_sorted]

    print("\n" + "=" * 60)
    print("Classification Report:")
    print("=" * 60)
    print(classification_report(all_labels, all_preds, target_names=target_names, digits=4))

    print("\n" + "=" * 60)
    print("Confusion Matrix (Counts):")
    print("=" * 60)
    cm = confusion_matrix(all_labels, all_preds, labels=labels_sorted)
    
    # Print header
    header = f"{'Actual/Pred':<15}"
    for name in target_names:
        header += f"{name:>15}"
    print(header)
    print("-" * len(header))
    
    for i, row in enumerate(cm):
        row_str = f"{target_names[i]:<15}"
        for val in row:
            row_str += f"{val:>15}"
        print(row_str)

    print("\n" + "=" * 60)
    print("Confusion Matrix (Normalized %):")
    print("=" * 60)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print(header)
    print("-" * len(header))
    for i, row in enumerate(cm_norm):
        row_str = f"{target_names[i]:<15}"
        for val in row:
            row_str += f"{val * 100:>14.1f}%"
        print(row_str)

    # Detailed review of top misclassified pairs
    print("\n" + "=" * 60)
    print("Top Misclassifications Breakdown:")
    print("=" * 60)
    misclassified = []
    for i in range(len(all_labels)):
        if all_labels[i] != all_preds[i]:
            misclassified.append((all_files[i], all_labels[i], all_preds[i]))

    mis_df = pd.DataFrame(misclassified, columns=["file", "actual", "predicted"])
    mis_df["actual_name"] = mis_df["actual"].map(label_map)
    mis_df["predicted_name"] = mis_df["predicted"].map(label_map)
    
    summary = mis_df.groupby(["actual_name", "predicted_name"]).size().reset_index(name="count")
    summary = summary.sort_values(by="count", ascending=False)
    print(summary.to_string(index=False))

if __name__ == "__main__":
    main()
