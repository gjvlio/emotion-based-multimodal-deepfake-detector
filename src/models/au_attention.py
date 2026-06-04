"""
au_attention.py — Dynamic Action Unit attention for microexpression-aware detection.

Learns which FACS Action Units are most discriminative for deepfake detection.
Input:  (B, AU_DIM) raw AU intensity values from py-feat
Output: (B, AU_DIM) attention-weighted AU features
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

AU_DIM = 20   # FACS AUs extracted by py-feat (padded/truncated to this size)


class AUAttention(nn.Module):
    """
    Learnable softmax weights over AU dimensions.
    During training, the model discovers which muscle activations (AUs)
    are most discriminative for emotion-manipulation detection.
    """

    def __init__(self, au_dim: int = AU_DIM):
        super().__init__()
        self.attn = nn.Linear(au_dim, au_dim)

    def forward(self, z_au: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_au: (B, AU_DIM) raw AU intensities
        Returns:
            (B, AU_DIM) attention-weighted AU features
        """
        weights = F.softmax(self.attn(z_au), dim=-1)   # (B, AU_DIM)
        return weights * z_au                            # element-wise scaling
