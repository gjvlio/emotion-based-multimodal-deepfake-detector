"""
sarcasm_head.py — Binary sarcasm classifier on Z_at.

Input:  Z_at (B, 1536)
Output: (B, 1) raw logit — sigmoid externally via BCEWithLogitsLoss.
"""
from __future__ import annotations

import torch.nn as nn


class SarcasmHead(nn.Module):
    def __init__(self, input_dim: int = 1536, hidden: int = 256, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, z_at):
        """Args: z_at (B, 1536). Returns logit (B, 1)."""
        return self.net(z_at)
