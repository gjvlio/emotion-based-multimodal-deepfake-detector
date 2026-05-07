"""
bilinear.py — Bilinear fusion of audio-text and visual feature vectors.

Projects both inputs to 256-dim before the outer product to avoid the
768×768 = 590K parameter OOM on RTX 3060.

Input:  Z_at (B, 1536), Z_v (B, 768)
Output: fused (B, 256*256) = (B, 65536)

IMPORTANT: operates on RAW embeddings only — do NOT pass emotion probabilities here.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class BilinearFusion(nn.Module):
    def __init__(
        self,
        z_at_dim: int = 1536,
        z_v_dim: int = 768,
        proj_dim: int = 256,
    ):
        super().__init__()
        self.proj_a = nn.Linear(z_at_dim, proj_dim)
        self.proj_v = nn.Linear(z_v_dim,  proj_dim)
        self.proj_dim = proj_dim

    @property
    def output_dim(self) -> int:
        return self.proj_dim * self.proj_dim  # 65536

    def forward(self, z_at: torch.Tensor, z_v: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_at: (B, 1536) — audio-text embedding
            z_v:  (B,  768) — visual embedding
        Returns:
            fused: (B, 65536) — flattened outer product
        """
        a = self.proj_a(z_at)  # (B, 256)
        v = self.proj_v(z_v)   # (B, 256)
        # outer product: (B, 256, 1) × (B, 1, 256) → (B, 256, 256)
        outer = a.unsqueeze(2) * v.unsqueeze(1)
        return outer.flatten(start_dim=1)  # (B, 65536)
