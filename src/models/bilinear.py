"""
bilinear.py — Compact Bilinear Pooling of audio-text and visual feature vectors.

Implements Tensor Sketch (Fukui et al., 2016): approximates the cross-modal
outer product in output_dim space (default 8192) via Count Sketch + FFT convolution.
Reduces the full 1,179,648-dim outer product (1536 x 768) to 8,192 dims while
preserving cross-modal multiplicative interactions between Z_at and Z_v.

The output is post-processed with signed square-root + L2 normalization
(Fukui et al., 2016). This is essential: without it the raw CBP magnitude is
unbounded (~380 here) and the downstream classifier logits explode into the
hundreds, saturating the sigmoid to exactly 0/1. Normalization keeps the
representation bounded so the model can express calibrated uncertainty.

Input:  Z_at (B, 1536), Z_v (B, 768)
Output: fused (B, output_dim) — default (B, 8192), L2-normalized
"""
from __future__ import annotations

import torch
import torch.nn as nn


class CompactBilinearFusion(nn.Module):
    """
    Compact Bilinear Pooling via Count Sketch + FFT convolution (Fukui et al., 2016).

    Approximates outer_product(Z_at, Z_v) in output_dim space without
    materializing the full d1×d2 matrix. Count Sketch parameters are fixed
    random projections (not learned) seeded deterministically at construction.

    IMPORTANT: operates on RAW embeddings only — do NOT pass emotion probabilities here.
    """

    def __init__(
        self,
        z_at_dim:   int = 1536,
        z_v_dim:    int = 768,
        output_dim: int = 8192,
        seed:       int = 42,
    ):
        super().__init__()
        self.output_dim = output_dim

        rng = torch.Generator()
        rng.manual_seed(seed)

        h_at = torch.randint(0, output_dim, (z_at_dim,), generator=rng)
        s_at = torch.randint(0, 2,          (z_at_dim,), generator=rng).float() * 2 - 1
        h_v  = torch.randint(0, output_dim, (z_v_dim,),  generator=rng)
        s_v  = torch.randint(0, 2,          (z_v_dim,),  generator=rng).float() * 2 - 1

        self.register_buffer("h_at", h_at)
        self.register_buffer("s_at", s_at)
        self.register_buffer("h_v",  h_v)
        self.register_buffer("s_v",  s_v)

    def _sketch(self, x: torch.Tensor, h: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """Count Sketch projection: (B, d) → (B, output_dim)."""
        B = x.shape[0]
        y = torch.zeros(B, self.output_dim, device=x.device, dtype=x.dtype)
        y.scatter_add_(1, h.unsqueeze(0).expand(B, -1), x * s)
        return y

    def forward(self, z_at: torch.Tensor, z_v: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_at: (B, 1536) — audio-text embedding
            z_v:  (B,  768) — visual embedding
        Returns:
            fused: (B, output_dim) — compact bilinear representation
        """
        psi_at = self._sketch(z_at, self.h_at, self.s_at)  # (B, output_dim)
        psi_v  = self._sketch(z_v,  self.h_v,  self.s_v)   # (B, output_dim)

        fft_at = torch.fft.rfft(psi_at)
        fft_v  = torch.fft.rfft(psi_v)
        fused = torch.fft.irfft(fft_at * fft_v, n=self.output_dim)  # (B, output_dim)

        # Signed square-root + L2 normalization (Fukui et al., 2016).
        # Without this the CBP output magnitude is unbounded (norm ~380 here),
        # which drives the classifier logits to the hundreds -> sigmoid saturates
        # to exactly 0/1. These two steps bound the representation so probabilities
        # stay meaningful and the model can express uncertainty.
        fused = torch.sign(fused) * torch.sqrt(torch.abs(fused) + 1e-12)
        fused = torch.nn.functional.normalize(fused, p=2, dim=-1)
        return fused


# Backward-compatible alias
BilinearFusion = CompactBilinearFusion
