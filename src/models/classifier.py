"""
classifier.py — MLP that maps [bilinear_fused ; delta] → P(fake) logit.

Input:  (B, 8198)  = 8192 (CBP) + 6 (delta = |emotion_A - emotion_B|)
Output: (B, 1)     raw logit — sigmoid applied externally by BCEWithLogitsLoss.

At inference, call torch.sigmoid(logit) to get P(fake) ∈ [0, 1].
"""
from __future__ import annotations

import torch
import torch.nn as nn


class SqueezeExcitation1D(nn.Module):
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.GELU(),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.fc(x)
        return x * w


class ClassifierMLP(nn.Module):
    def __init__(
        self,
        input_dim: int = 8198,
        hidden1: int = 512,
        hidden2: int = 128,
        dropout: float = 0.4,
    ):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden1)
        self.act1 = nn.GELU()
        self.se   = SqueezeExcitation1D(hidden1, reduction=4)
        self.drop = nn.Dropout(dropout)
        self.fc2  = nn.Linear(hidden1, hidden2)
        self.act2 = nn.GELU()
        self.out  = nn.Linear(hidden2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x (B, input_dim). Returns logit (B, 1)."""
        h1 = self.act1(self.fc1(x))
        h1_se = self.se(h1)
        h2 = self.act2(self.fc2(self.drop(h1_se)))
        return self.out(h2)
