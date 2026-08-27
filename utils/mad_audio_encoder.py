"""
MAD Audio Encoder — Single-Label Architecture
================================================================================

This module defines `MADAudioEncoder`, a CNN-based audio encoder for the
FIRST SINGLE-LABEL MAD (7-class) audio classification experiment.

This file is intentionally independent from `utils/audio_encoder.py` (the
original ALM single-label encoder) and from any multi-label encoder used
elsewhere in the project. Nothing in this module modifies or imports those
files — it simply mirrors the same architectural pattern for the MAD
single-label experiment, whose only meaningful input change is the audio
duration (5s → 10s), which is absorbed transparently via AdaptiveAvgPool2d.

Task: single-label classification (7 classes) — NOT multi-label.
Loss expected to be used with this model: nn.CrossEntropyLoss (raw logits).
"""

from __future__ import annotations

import torch
import torch.nn as nn


# ── ConvBlock ──────────────────────────────────────────────────────────────
class ConvBlock(nn.Module):
    """
    Standard convolutional block used by the CNN backbone:

        Conv2d -> BatchNorm2d -> ReLU -> Conv2d -> BatchNorm2d -> ReLU
        -> MaxPool2d -> Dropout2d

    Two stacked convolutions per block increase representational capacity
    before spatial downsampling via max pooling.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        pool_size: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        padding = kernel_size // 2

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(pool_size)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.dropout(x)
        return x


# ── MADAudioEncoder ────────────────────────────────────────────────────────
class MADAudioEncoder(nn.Module):
    """
    Single-label CNN audio encoder for the MAD dataset (7 classes).

    Pipeline:
        Input log-Mel spectrogram (B, 1, 128, ~313)
            -> 4x ConvBlock (1->32->64->128->256 channels)
            -> AdaptiveAvgPool2d((4, 4))
            -> Flatten (256 * 4 * 4 = 4096)
            -> Linear(4096, 1024) -> LayerNorm -> ReLU -> Dropout
            -> Linear(1024, embed_dim)
            -> L2 normalization  => 512-dim embedding
            -> Linear(embed_dim, num_classes)  => raw classification logits

    The time dimension of the input mel spectrogram is NOT hard-coded: the
    AdaptiveAvgPool2d((4, 4)) layer absorbs any time-frame count, which is
    what allows this exact architecture to move from the original 5-second
    ALM configuration (~157 frames) to the MAD 10-second configuration
    (~313 frames) with zero structural changes.

    Notes:
        - forward() returns RAW LOGITS of shape (B, num_classes). Softmax is
          intentionally NOT applied inside the model; use nn.CrossEntropyLoss
          during training and torch.softmax(logits, dim=-1) only at
          evaluation time to obtain probabilities.
        - encode() returns an L2-normalized (B, embed_dim) embedding,
          independent of the classification head.
    """

    def __init__(self, embed_dim: int = 512, num_classes: int = 7, dropout: float = 0.3) -> None:
        super().__init__()

        self.embed_dim = embed_dim
        self.num_classes = num_classes

        # ── CNN backbone ────────────────────────────────────────────────
        self.block1 = ConvBlock(1, 32, kernel_size=3, pool_size=2, dropout=0.0)
        self.block2 = ConvBlock(32, 64, kernel_size=3, pool_size=2, dropout=0.1)
        self.block3 = ConvBlock(64, 128, kernel_size=3, pool_size=2, dropout=0.1)
        self.block4 = ConvBlock(128, 256, kernel_size=3, pool_size=2, dropout=0.2)

        self.global_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.flatten = nn.Flatten()

        # ── Projection head ─────────────────────────────────────────────
        flattened_dim = 256 * 4 * 4  # 4096
        self.projection = nn.Sequential(
            nn.Linear(flattened_dim, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(1024, embed_dim),
        )

        # ── Classification head ─────────────────────────────────────────
        self.classifier = nn.Linear(embed_dim, num_classes)

        self._init_weights()

    # ── Weight initialization ───────────────────────────────────────────
    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    # ── Backbone forward ─────────────────────────────────────────────────
    def _backbone(self, mel: torch.Tensor) -> torch.Tensor:
        x = self.block1(mel)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.global_pool(x)
        x = self.flatten(x)
        return x

    # ── Public API ───────────────────────────────────────────────────────
    def encode(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Compute a normalized 512-dimensional embedding from a log-Mel
        spectrogram batch.

        Args:
            mel: (B, 1, 128, T) log-Mel spectrogram tensor.

        Returns:
            (B, embed_dim) L2-normalized embedding tensor.
        """
        features = self._backbone(mel)
        embedding = self.projection(features)
        embedding = nn.functional.normalize(embedding, p=2, dim=-1)
        return embedding

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Compute raw classification logits for a batch of log-Mel
        spectrograms.

        Args:
            mel: (B, 1, 128, T) log-Mel spectrogram tensor.

        Returns:
            (B, num_classes) raw logits (no softmax applied).
        """
        embedding = self.encode(mel)
        logits = self.classifier(embedding)
        return logits

    def count_params(self) -> int:
        """Return the number of trainable parameters in the model."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── Architecture sanity check ───────────────────────────────────────────────
def _sanity_check() -> None:
    """
    Quick standalone check confirming the architecture behaves as expected
    for the MAD configuration: 10s audio @ 16kHz, 128 mel bins, ~313 frames.
    """
    model = MADAudioEncoder(embed_dim=512, num_classes=7, dropout=0.3)
    dummy = torch.randn(2, 1, 128, 313)

    logits = model(dummy)
    embedding = model.encode(dummy)
    norms = embedding.norm(dim=-1)

    assert logits.shape == (2, 7), f"Unexpected logits shape: {logits.shape}"
    assert embedding.shape == (2, 512), f"Unexpected embedding shape: {embedding.shape}"
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4), "Embeddings are not unit-normalized"

    print("MADAudioEncoder sanity check passed.")
    print(f"  Logits shape    : {tuple(logits.shape)}")
    print(f"  Embedding shape : {tuple(embedding.shape)}")
    print(f"  Embedding norms : min={norms.min().item():.4f}, max={norms.max().item():.4f}")
    print(f"  Trainable params: {model.count_params():,}")


if __name__ == "__main__":
    _sanity_check()