"""
emotion_heads.py — Per-modality 6-class emotion classifiers.

EmotionHeadA: audio path  (input 1536-dim = Wav2Vec 768 + BERT 768)
EmotionHeadB: visual path (input  768-dim = ViT CLS token)

Both return RAW LOGITS. Softmax is applied externally:
  - CrossEntropyLoss receives logits directly.
  - Delta computation applies F.softmax before subtraction.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class EmotionHeadA(nn.Module):
    """Audio emotion head: Linear(1536→256) → GELU → Dropout → Linear(256→6)."""

    def __init__(self, input_dim: int = 1536, n_classes: int = 6, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, n_classes),
        )

    def forward(self, z_at: torch.Tensor) -> torch.Tensor:
        """Args: z_at (B, 1536). Returns logits (B, 6)."""
        return self.net(z_at)


class EmotionHeadB(nn.Module):
    """Visual emotion head: Linear(768→256) → GELU → Dropout → Linear(256→6)."""

    def __init__(self, input_dim: int = 768, n_classes: int = 6, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, n_classes),
        )

    def forward(self, z_v: torch.Tensor) -> torch.Tensor:
        """Args: z_v (B, 768). Returns logits (B, 6)."""
        return self.net(z_v)
