"""
classifier.py — MLP that maps [bilinear_fused ; delta] → P(fake) logit.

Input:  (B, 65542)  = 65536 (bilinear) + 6 (delta = |emotion_A - emotion_B|)
Output: (B, 1)      raw logit — sigmoid applied externally by BCEWithLogitsLoss.

At inference, call torch.sigmoid(logit) to get P(fake) ∈ [0, 1].
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ClassifierMLP(nn.Module):
    def __init__(
        self,
        input_dim: int = 65542,   # 65536 + 6
        hidden1: int = 512,
        hidden2: int = 128,
        dropout: float = 0.4,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x (B, 65542). Returns logit (B, 1)."""
        return self.net(x)
