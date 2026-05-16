"""
detection_model.py — Full Deepfake Detection Module.

Two forward paths:
  forward(audio_values, input_ids, attention_mask, keyframe_pixels)
      — end-to-end, includes Wav2Vec2 + BERT + ViT backbones (Phase 2)
  forward_from_features(z_at, z_v)
      — bypasses backbones, uses precomputed feature vectors (Phase 1 / inference)

Output: DetectorOutput(logit, emotion_a, emotion_b)
  logit      — (B, 1) raw fake score; apply sigmoid for P(fake)
  emotion_a  — (B, 6) audio emotion logits
  emotion_b  — (B, 6) visual emotion logits
"""
from __future__ import annotations

from typing import Optional
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .emotion_heads import EmotionHeadA, EmotionHeadB
from .bilinear import BilinearFusion
from .classifier import ClassifierMLP


@dataclass
class DetectorOutput:
    logit: torch.Tensor       # (B, 1) — raw, no sigmoid
    emotion_a: torch.Tensor   # (B, 6) — audio emotion logits
    emotion_b: torch.Tensor   # (B, 6) — visual emotion logits


class DeepfakeDetector(nn.Module):
    """
    Full detection model.

    Backbone components (Wav2Vec2, BERT, ViT) are loaded lazily on first call to
    forward() so that forward_from_features() can be used without loading 1+ GB of
    pretrained weights during Phase 1 training.
    """

    Z_AT_DIM = 1536   # 768 (Wav2Vec) + 768 (BERT)
    Z_V_DIM  = 768    # ViT CLS token
    CBP_DIM  = 8192   # Compact Bilinear Pooling output dimension

    def __init__(
        self,
        wav2vec_model: str = "facebook/wav2vec2-base",
        bert_model:    str = "bert-base-uncased",
        vit_model:     str = "google/vit-base-patch16-224",
        n_emotions:    int = 6,
        cbp_dim:       int = 8192,
        dropout_heads: float = 0.3,
        dropout_cls:   float = 0.4,
    ):
        super().__init__()
        self._wav2vec_name = wav2vec_model
        self._bert_name    = bert_model
        self._vit_name     = vit_model

        # Detection components (always present)
        self.emotion_head_a  = EmotionHeadA(self.Z_AT_DIM, n_emotions, dropout_heads)
        self.emotion_head_b  = EmotionHeadB(self.Z_V_DIM,  n_emotions, dropout_heads)
        self.bilinear_fusion = BilinearFusion(self.Z_AT_DIM, self.Z_V_DIM, cbp_dim)
        fused_dim = cbp_dim + n_emotions   # 8192 + 6
        self.classifier = ClassifierMLP(fused_dim, dropout=dropout_cls)

        # Backbones — loaded on demand
        self._wav2vec: Optional[nn.Module] = None
        self._bert:    Optional[nn.Module] = None
        self._vit:     Optional[nn.Module] = None
        self._backbones_loaded = False

    # ── Backbone management ────────────────────────────────────────────────────

    def load_backbones(self) -> None:
        """Instantiate Wav2Vec2, BERT, ViT. Safe to call multiple times."""
        if self._backbones_loaded:
            return
        from transformers import Wav2Vec2Model, BertModel, ViTModel
        self._wav2vec = Wav2Vec2Model.from_pretrained(self._wav2vec_name)
        self._bert    = BertModel.from_pretrained(self._bert_name)
        self._vit     = ViTModel.from_pretrained(self._vit_name)
        self.wav2vec2 = self._wav2vec
        self.bert     = self._bert
        self.vit      = self._vit
        self._backbones_loaded = True

    def freeze_backbones(self) -> None:
        self.load_backbones()
        for m in (self._wav2vec, self._bert, self._vit):
            for p in m.parameters():
                p.requires_grad = False

    def unfreeze_backbones(self) -> None:
        if not self._backbones_loaded:
            raise RuntimeError("Call load_backbones() before unfreeze_backbones().")
        for m in (self._wav2vec, self._bert, self._vit):
            for p in m.parameters():
                p.requires_grad = True

    # ── Core detection logic ───────────────────────────────────────────────────

    def _detect(self, z_at: torch.Tensor, z_v: torch.Tensor) -> DetectorOutput:
        """Shared logic after feature extraction."""
        emo_a = self.emotion_head_a(z_at)  # (B, 6)
        emo_b = self.emotion_head_b(z_v)   # (B, 6)

        fused = self.bilinear_fusion(z_at, z_v)  # (B, 65536)

        delta = torch.abs(
            F.softmax(emo_a, dim=-1) - F.softmax(emo_b, dim=-1)
        )  # (B, 6)

        combined = torch.cat([fused, delta], dim=-1)  # (B, 65542)
        logit = self.classifier(combined)              # (B, 1)

        return DetectorOutput(logit=logit, emotion_a=emo_a, emotion_b=emo_b)

    # ── Phase 1 path (cached features) ────────────────────────────────────────

    def forward_from_features(
        self,
        z_at: torch.Tensor,
        z_v:  torch.Tensor,
    ) -> DetectorOutput:
        """
        Phase 1 forward pass — takes precomputed Z_at (B,1536) and Z_v (B,768).
        Does NOT require backbones to be loaded.
        """
        return self._detect(z_at, z_v)

    # ── Phase 2 path (end-to-end) ─────────────────────────────────────────────

    def forward(
        self,
        audio_values:    torch.Tensor,            # (B, T_audio)
        input_ids:       torch.Tensor,            # (B, seq_len)
        attention_mask:  torch.Tensor,            # (B, seq_len)
        keyframe_pixels: torch.Tensor,            # (B, K, 3, 224, 224)
    ) -> DetectorOutput:
        """
        Phase 2 end-to-end forward pass.
        Runs Wav2Vec2 + BERT for audio-text, ViT for visual.
        Call load_backbones() once before using this path.
        """
        if not self._backbones_loaded:
            raise RuntimeError(
                "Backbones not loaded. Call model.load_backbones() first, "
                "or use forward_from_features() for Phase 1 training."
            )

        # Audio branch — Wav2Vec2
        w2v_out = self._wav2vec(audio_values).last_hidden_state  # (B, T', 768)
        w2v_emb = w2v_out.mean(dim=1)                            # (B, 768)

        # Text branch — BERT on ASR transcript tokens
        bert_out = self._bert(input_ids=input_ids, attention_mask=attention_mask)
        bert_emb = bert_out.last_hidden_state[:, 0, :]           # (B, 768) CLS token

        z_at = torch.cat([w2v_emb, bert_emb], dim=-1)            # (B, 1536)

        # Visual branch — ViT on K keyframes per clip
        B, K, C, H, W = keyframe_pixels.shape
        frames = keyframe_pixels.view(B * K, C, H, W)
        vit_out = self._vit(pixel_values=frames).last_hidden_state[:, 0, :]  # (B*K, 768)
        z_v = vit_out.view(B, K, 768).mean(dim=1)                # (B, 768)

        return self._detect(z_at, z_v)

    # ── Convenience ───────────────────────────────────────────────────────────

    @staticmethod
    def p_fake(logit: torch.Tensor) -> torch.Tensor:
        """Convert raw logit → P(fake) ∈ [0, 1]."""
        return torch.sigmoid(logit)
