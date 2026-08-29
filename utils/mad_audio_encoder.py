"""
MAD Audio Encoder — Single-Label Architecture (v2, tuned for 10s clips)
================================================================================

Changes from v1 (the one you trained at 96% train / 88% test):

1. Pooling before the projection head now keeps more temporal resolution.
   v1 used AdaptiveAvgPool2d((4, 4)) — fine for 5s clips (~157 frames,
   ~39:1 compression) but way too aggressive for 10s clips (~313 frames,
   ~78:1 compression). v2 uses AdaptiveAvgPool2d((4, 8)) by default, or an
   optional learned attention pool over time (recommended — see
   `pooling="attention"` below).

2. Later conv blocks pool the frequency axis but NOT the time axis as
   hard, so temporal detail survives longer into the network
   (pool_size=(2, 1) on blocks 3 and 4).

3. Slightly higher dropout in the projection head to compensate for the
   extra capacity from the larger flattened feature map.

4. Classifier loss is expected to use label_smoothing (set in the
   training script, not here — see train_v2 notes at the bottom of this
   file for the exact CrossEntropyLoss call to use).

Nothing else about the public API changed: encode() and forward() have
the same signatures as v1, so it's a drop-in replacement in your training
script (just re-import MADAudioEncoder from here instead of utils).

Task: single-label classification (7 classes) — NOT multi-label.
"""

from __future__ import annotations

import torch
import torch.nn as nn


# ── ConvBlock ──────────────────────────────────────────────────────────────
class ConvBlock(nn.Module):
    """
    Conv2d -> BatchNorm2d -> ReLU -> Conv2d -> BatchNorm2d -> ReLU
    -> MaxPool2d(pool_size) -> Dropout2d

    pool_size can now be a tuple, e.g. (2, 1), to pool frequency without
    pooling time — this is what lets v2 preserve temporal resolution for
    longer (10s) clips.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        pool_size: int | tuple[int, int] = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        padding = kernel_size // 2

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding, bias=False)
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


# ── Attention pooling over time ─────────────────────────────────────────────
class AttentionPool2d(nn.Module):
    """
    Learned attention pooling over the time axis, average pooling over the
    (already-small) frequency axis.

    Input:  (B, C, F, T)
    Output: (B, C, F_out) after also mean-pooling frequency down to F_out,
            flattened for the projection head.

    Why this instead of AdaptiveAvgPool2d: plain average pooling weighs
    every time frame equally, including silence/padding and low-information
    frames. For 10s clips (twice the frames of the 5s setup this
    architecture was originally tuned for), that dilutes the signal a lot.
    Attention pooling lets the model learn which frames actually matter.
    """

    def __init__(self, channels: int, freq_out: int = 4):
        super().__init__()
        self.freq_pool = nn.AdaptiveAvgPool2d((freq_out, None))  # keep time axis untouched
        self.attn = nn.Conv1d(channels * freq_out, 1, kernel_size=1)
        self.freq_out = freq_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, F, T) -> (B, C, freq_out, T)
        x = self.freq_pool(x)
        b, c, f, t = x.shape
        x_flat = x.reshape(b, c * f, t)  # (B, C*freq_out, T)

        attn_logits = self.attn(x_flat)          # (B, 1, T)
        attn_weights = torch.softmax(attn_logits, dim=-1)  # (B, 1, T)

        pooled = (x_flat * attn_weights).sum(dim=-1)  # (B, C*freq_out)
        return pooled


# ── MADAudioEncoder v2 ──────────────────────────────────────────────────────
class MADAudioEncoder(nn.Module):
    """
    Single-label CNN audio encoder for the MAD dataset (7 classes), tuned
    for 10-second inputs.

    Pipeline:
        Input log-Mel spectrogram (B, 1, 128, ~313)
            -> 4x ConvBlock (1->32->64->128->256 channels)
               blocks 1-2 pool (2,2); blocks 3-4 pool (2,1) [time preserved]
            -> pooling head: AdaptiveAvgPool2d((4,8)) [default]
               or AttentionPool2d [pooling="attention", recommended]
            -> Flatten
            -> Linear(-, 1024) -> LayerNorm -> ReLU -> Dropout
            -> Linear(1024, embed_dim)
            -> L2 normalization  => 512-dim embedding
            -> Linear(embed_dim, num_classes)  => raw classification logits

    forward() returns RAW LOGITS — use nn.CrossEntropyLoss(label_smoothing=0.1)
    during training, softmax only at eval time.
    """

    def __init__(
        self,
        embed_dim: int = 512,
        num_classes: int = 7,
        dropout: float = 0.4,
        pooling: str = "attention",   # "attention" (recommended) or "avg"
        time_bins: int = 8,           # only used when pooling="avg"
    ) -> None:
        super().__init__()

        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.pooling_type = pooling

        # ── CNN backbone ────────────────────────────────────────────────
        # blocks 1-2: pool both freq and time as before
        self.block1 = ConvBlock(1, 32, kernel_size=3, pool_size=2, dropout=0.0)
        self.block2 = ConvBlock(32, 64, kernel_size=3, pool_size=2, dropout=0.1)
        # blocks 3-4: pool freq only (2,1) so time resolution survives
        # for the longer 10s / ~313-frame input
        self.block3 = ConvBlock(64, 128, kernel_size=3, pool_size=(2, 1), dropout=0.1)
        self.block4 = ConvBlock(128, 256, kernel_size=3, pool_size=(2, 1), dropout=0.2)

        if pooling == "attention":
            freq_out = 4
            self.pool = AttentionPool2d(channels=256, freq_out=freq_out)
            flattened_dim = 256 * freq_out
        elif pooling == "avg":
            self.pool = nn.AdaptiveAvgPool2d((4, time_bins))
            flattened_dim = 256 * 4 * time_bins
        else:
            raise ValueError(f"Unknown pooling type: {pooling!r} (expected 'attention' or 'avg')")

        self.flatten = nn.Flatten()

        # ── Projection head ─────────────────────────────────────────────
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

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Conv1d)):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def _backbone(self, mel: torch.Tensor) -> torch.Tensor:
        x = self.block1(mel)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.pool(x)
        x = self.flatten(x)
        return x

    def encode(self, mel: torch.Tensor) -> torch.Tensor:
        """(B, 1, 128, T) -> (B, embed_dim) L2-normalized embedding."""
        features = self._backbone(mel)
        embedding = self.projection(features)
        embedding = nn.functional.normalize(embedding, p=2, dim=-1)
        return embedding

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """(B, 1, 128, T) -> (B, num_classes) raw logits (no softmax)."""
        embedding = self.encode(mel)
        logits = self.classifier(embedding)
        return logits

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── Mixup helper (use in your training loop, see notes below) ──────────────
def mixup_batch(mel: torch.Tensor, labels: torch.Tensor, num_classes: int, alpha: float = 0.2):
    """
    Mixup for spectrogram batches. Returns (mixed_mel, mixed_labels_onehot).
    Use with a soft-label loss: -(mixed_labels * log_softmax(logits)).sum(-1).mean()

    Example training-loop usage:
        mel, labels = batch["mel"].to(device), batch["label"].to(device)
        mel_mixed, labels_soft = mixup_batch(mel, labels, num_classes=CFG.NUM_CLASSES)
        logits = model(mel_mixed)
        log_probs = torch.log_softmax(logits, dim=-1)
        loss = -(labels_soft * log_probs).sum(dim=-1).mean()
    """
    batch_size = mel.size(0)
    lam = float(torch.distributions.Beta(alpha, alpha).sample()) if alpha > 0 else 1.0
    perm = torch.randperm(batch_size, device=mel.device)

    mixed_mel = lam * mel + (1 - lam) * mel[perm]

    labels_onehot = torch.nn.functional.one_hot(labels, num_classes=num_classes).float()
    mixed_labels = lam * labels_onehot + (1 - lam) * labels_onehot[perm]

    return mixed_mel, mixed_labels


# ── Architecture sanity check ───────────────────────────────────────────────
def _sanity_check() -> None:
    for pooling in ["attention", "avg"]:
        model = MADAudioEncoder(embed_dim=512, num_classes=7, dropout=0.4, pooling=pooling)
        dummy = torch.randn(2, 1, 128, 313)  # 10s @ 16kHz, 128 mels, hop=512

        logits = model(dummy)
        embedding = model.encode(dummy)
        norms = embedding.norm(dim=-1)

        assert logits.shape == (2, 7), f"[{pooling}] Unexpected logits shape: {logits.shape}"
        assert embedding.shape == (2, 512), f"[{pooling}] Unexpected embedding shape: {embedding.shape}"
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4), f"[{pooling}] Embeddings not unit-normalized"

        print(f"[pooling={pooling}] sanity check passed.")
        print(f"  Logits shape    : {tuple(logits.shape)}")
        print(f"  Embedding shape : {tuple(embedding.shape)}")
        print(f"  Trainable params: {model.count_params():,}")

    # quick mixup sanity check
    mel = torch.randn(4, 1, 128, 313)
    labels = torch.tensor([0, 1, 2, 3])
    mel_mixed, labels_soft = mixup_batch(mel, labels, num_classes=7)
    assert mel_mixed.shape == mel.shape
    assert labels_soft.shape == (4, 7)
    print("mixup_batch sanity check passed.")


if __name__ == "__main__":
    _sanity_check()