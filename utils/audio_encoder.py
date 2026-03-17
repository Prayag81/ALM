"""
models/audio_encoder.py
------------------------
CNN-based audio encoder.
Input  : log-mel spectrogram  (B, 1, 128, 157)
Output : L2-normalised embedding  (B, embed_dim)

Architecture
------------
  4× Conv block  (Conv2d → BN → ReLU → MaxPool)
  Adaptive Average Pool  → flatten
  2-layer projection head  → embed_dim (512)
  L2 normalisation

References
----------
- CLIP paper        : https://arxiv.org/abs/2103.00020
- Audio CNN design  : https://arxiv.org/abs/1608.04363  (EnvNet)
- SpecAugment       : https://arxiv.org/abs/1904.08779
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Building blocks ────────────────────────────────────────────────────────

class ConvBlock(nn.Module):
    """Conv2d → BatchNorm → ReLU → MaxPool."""

    def __init__(self, in_ch: int, out_ch: int,
                 kernel: int = 3, pool: int = 2, dropout: float = 0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel, padding=kernel // 2, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=kernel, padding=kernel // 2, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(pool),
            nn.Dropout2d(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ── Main encoder ───────────────────────────────────────────────────────────

class AudioEncoder(nn.Module):
    """
    Args:
        embed_dim  : output embedding dimension (default 512, matches text encoder)
        dropout    : dropout rate in projection head
    """

    def __init__(self, embed_dim: int = 512, dropout: float = 0.3 , num_classes : int = None):
        super().__init__()

        # ── CNN backbone ───────────────────────────────────────────────────
        # Input: (B, 1, 128, 157)
        self.backbone = nn.Sequential(
            ConvBlock(1,   32,  kernel=3, pool=2, dropout=0.0),   # → (B, 32,  64, 78)
            ConvBlock(32,  64,  kernel=3, pool=2, dropout=0.1),   # → (B, 64,  32, 39)
            ConvBlock(64,  128, kernel=3, pool=2, dropout=0.1),   # → (B, 128, 16, 19)
            ConvBlock(128, 256, kernel=3, pool=2, dropout=0.2),   # → (B, 256,  8,  9)
        )

        # ── Global pooling → flat vector ───────────────────────────────────
        self.pool = nn.AdaptiveAvgPool2d((4, 4))   # (B, 256, 4, 4)

        flat_dim = 256 * 4 * 4

        # ── Projection head (flat → embed_dim) ────────────────────────────
        self.projector = nn.Sequential(
            nn.Linear(flat_dim, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(1024, embed_dim),
        )
        self.num_classes = num_classes
        if self.num_classes is not None:
            self.classifier = nn.Linear(embed_dim ,num_classes)

        self.embed_dim = embed_dim
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mel : (B, 1, n_mels, time_frames)  log-mel spectrogram
        Returns:
            embedding : (B, embed_dim)  L2-normalised
        """
        x = self.backbone(mel)          # (B, 256, 8, 9)
        x = self.pool(x)                # (B, 256, 4, 4)
        x = x.flatten(1)               # (B, 4096)
        x = self.projector(x)           # (B, embed_dim)
        x = F.normalize(x, dim=-1)      # L2 normalise → unit sphere

        if self.num_classes is not None:
            return self.classifier(x)
        
        return x

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── Sanity check ───────────────────────────────────────────────────────────
if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')

    model = AudioEncoder(embed_dim=512).to(device)
    print(f'Trainable params: {model.count_params():,}')

    # Fake batch — same shape the Dataset will produce
    dummy_mel = torch.randn(8, 1, 128, 157).to(device)
    with torch.no_grad():
        emb = model(dummy_mel)

    print(f'Input  shape : {dummy_mel.shape}')
    print(f'Output shape : {emb.shape}')
    print(f'Norm check   : {emb.norm(dim=-1).mean():.4f}')

    # Check all intermediate shapes
    print('\n── Layer-by-layer shapes ────────────────────────────────────')
    x = dummy_mel
    for i, layer in enumerate(model.backbone):
        x = layer(x)
        print(f'  ConvBlock {i+1}: {tuple(x.shape)}')
    x = model.pool(x)
    print(f'  AdaptivePool: {tuple(x.shape)}')
    x = x.flatten(1)
    print(f'  Flatten     : {tuple(x.shape)}')
    x = model.projector(x)
    print(f'  Projector   : {tuple(x.shape)}')