"""
detection_model.py — Full Deepfake Detection Module.

Two forward paths:
  forward(audio_values, input_ids, attention_mask, keyframe_pixels)
      — end-to-end, includes Wav2Vec2 + BERT + ViT backbones (Phase 2)
  forward_from_features(z_at, z_v)
      — bypasses backbones, uses precomputed feature vectors (Phase 1 / inference)

Output: DetectorOutput(logit, emotion_a, emotion_b, sarcasm)
  logit      — (B, 1) raw fake score; apply sigmoid for P(fake)
  emotion_a  — (B, 6) audio emotion logits
  emotion_b  — (B, 6) visual emotion logits
  sarcasm    — (B, 1) raw sarcasm score; apply sigmoid for P(sarcastic)
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
from .sarcasm_head import SarcasmHead


@dataclass
class DetectorOutput:
    logit: torch.Tensor       # (B, 1) — raw, no sigmoid
    emotion_a: torch.Tensor   # (B, 6) — audio emotion logits
    emotion_b: torch.Tensor   # (B, 6) — visual emotion logits
    sarcasm: torch.Tensor     # (B, 1) — raw sarcasm logit


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
        classifier_mode: str = "baseline",
    ):
        super().__init__()
        self._wav2vec_name = wav2vec_model
        self._bert_name    = bert_model
        self._vit_name     = vit_model
        self.classifier_mode = classifier_mode

        # Detection components (always present)
        self.emotion_head_a  = EmotionHeadA(self.Z_AT_DIM, n_emotions, dropout_heads)
        self.emotion_head_b  = EmotionHeadB(self.Z_V_DIM,  n_emotions, dropout_heads)
        self.sarcasm_head    = SarcasmHead(self.Z_AT_DIM, dropout=dropout_heads)
        self.bilinear_fusion = BilinearFusion(self.Z_AT_DIM, self.Z_V_DIM, cbp_dim)

        if classifier_mode == "mismatch_only":
            fused_dim = n_emotions + 1
        elif classifier_mode == "emotion_bilinear":
            fused_dim = 36 + n_emotions + 1
        elif self.classifier_mode == "bottleneck":
            self.bilinear_proj = nn.Linear(cbp_dim, 256)
            self.proj_ln = nn.LayerNorm(256)
            fused_dim = 256 + 36 + n_emotions + 1
        elif self.classifier_mode == "high_dropout":
            self.high_dropout = nn.Dropout(0.85)
            fused_dim = cbp_dim + n_emotions + 1
        else:  # baseline
            fused_dim = cbp_dim + n_emotions + 1

        self.classifier = ClassifierMLP(fused_dim, dropout=dropout_cls)

        # Cross-Modal Attention between audio/text and visual keyframes
        self.cross_attn_at = nn.MultiheadAttention(embed_dim=768, num_heads=8, batch_first=True)
        self.cross_attn_v  = nn.MultiheadAttention(embed_dim=768, num_heads=8, batch_first=True)
        self.norm_at       = nn.LayerNorm(768)
        self.norm_v        = nn.LayerNorm(768)

        # Temporal GRU Aggregator over visual keyframe sequence
        self.vit_gru       = nn.GRU(input_size=768, hidden_size=768, num_layers=2, batch_first=True)

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
        device = next(self.classifier.parameters()).device
        from transformers import Wav2Vec2Model, BertModel, ViTModel
        self._wav2vec = Wav2Vec2Model.from_pretrained(self._wav2vec_name).to(device)
        self._wav2vec.gradient_checkpointing_disable()
        self._bert    = BertModel.from_pretrained(self._bert_name).to(device)
        self._vit     = ViTModel.from_pretrained(self._vit_name).to(device)
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

    def unfreeze_top_layers(self, n_layers: int = 2) -> None:
        """Unfreeze only the top N transformer blocks of each backbone.
        Much lower VRAM than full unfreeze — gradients only for last N layers."""
        if not self._backbones_loaded:
            raise RuntimeError("Call load_backbones() before unfreeze_top_layers().")
        # Keep everything frozen first
        for m in (self._wav2vec, self._bert, self._vit):
            for p in m.parameters():
                p.requires_grad = False

        # Wav2Vec2: encoder.layers[-n_layers:]
        for layer in self._wav2vec.encoder.layers[-n_layers:]:
            for p in layer.parameters():
                p.requires_grad = True

        # BERT: encoder.layer[-n_layers:]
        for layer in self._bert.encoder.layer[-n_layers:]:
            for p in layer.parameters():
                p.requires_grad = True
        for p in self._bert.pooler.parameters():
            p.requires_grad = True

        # ViT unfreezing (supports both self._vit.encoder.layer and self._vit.layers architectures)
        vit_layers = None
        vit_encoder = getattr(self._vit, "encoder", None)
        if vit_encoder is None and hasattr(self._vit, "vit"):
            vit_encoder = getattr(self._vit.vit, "encoder", None)
            
        if vit_encoder is not None and hasattr(vit_encoder, "layer"):
            vit_layers = vit_encoder.layer
        elif hasattr(self._vit, "layers"):
            vit_layers = self._vit.layers
            
        if vit_layers is None:
            raise AttributeError("Could not find ViT layers or encoder module. Check transformers library version.")
            
        for layer in vit_layers[-n_layers:]:
            for p in layer.parameters():
                p.requires_grad = True
                
        vit_layernorm = getattr(self._vit, "layernorm", None)
        if vit_layernorm is None and hasattr(self._vit, "vit"):
            vit_layernorm = getattr(self._vit.vit, "layernorm", None)
            
        if vit_layernorm is not None:
            for p in vit_layernorm.parameters():
                p.requires_grad = True

    def enable_gradient_checkpointing(self) -> None:
        """Trade compute for memory — recompute activations on backward pass."""
        if not self._backbones_loaded:
            raise RuntimeError("Call load_backbones() first.")
        self._wav2vec.gradient_checkpointing_enable()
        self._bert.gradient_checkpointing_enable()
        self._vit.gradient_checkpointing_enable()

    # ── Core detection logic ───────────────────────────────────────────────────

    def _detect(self, z_at: torch.Tensor, z_v: torch.Tensor) -> DetectorOutput:
        """Shared logic after feature extraction."""
        emo_a = self.emotion_head_a(z_at)  # (B, 6)
        emo_b = self.emotion_head_b(z_v)   # (B, 6)
        sarc  = self.sarcasm_head(z_at)    # (B, 1)

        fused = self.bilinear_fusion(z_at, z_v)  # (B, 8192)

        prob_a = F.softmax(emo_a, dim=-1)
        prob_b = F.softmax(emo_b, dim=-1)
        delta = torch.abs(prob_a - prob_b)  # (B, 6)

        if self.classifier_mode == "mismatch_only":
            combined = torch.cat([delta, sarc], dim=-1)
        elif self.classifier_mode == "emotion_bilinear":
            outer = torch.bmm(prob_a.unsqueeze(2), prob_b.unsqueeze(1))  # (B, 6, 6)
            fused_emo = outer.view(prob_a.size(0), 36)                   # (B, 36)
            combined = torch.cat([fused_emo, delta, sarc], dim=-1)       # (B, 43)
        elif self.classifier_mode == "bottleneck":
            outer = torch.bmm(prob_a.unsqueeze(2), prob_b.unsqueeze(1))  # (B, 6, 6)
            fused_emo = outer.view(prob_a.size(0), 36)                   # (B, 36)
            fused_proj = F.gelu(self.proj_ln(self.bilinear_proj(fused))) # (B, 256)
            combined = torch.cat([fused_proj, fused_emo, delta, sarc], dim=-1) # (B, 299)
        elif self.classifier_mode == "high_dropout":
            fused_drop = self.high_dropout(fused)
            combined = torch.cat([fused_drop, delta, sarc], dim=-1)
        else:  # baseline
            combined = torch.cat([fused, delta, sarc], dim=-1)

        logit = self.classifier(combined)                    # (B, 1)

        return DetectorOutput(logit=logit, emotion_a=emo_a, emotion_b=emo_b, sarcasm=sarc)

    # ── Phase 1 path (cached features) ────────────────────────────────────────

    def forward_from_features(
        self,
        z_at: torch.Tensor,
        z_v:  torch.Tensor,
    ) -> DetectorOutput:
        """
        Phase 1 forward pass - takes precomputed Z_at (B,1536) and Z_v (B,768).
        Does NOT require backbones to be loaded.
        """
        w2v_emb = z_at[:, :768]
        bert_emb = z_at[:, 768:]
        z_v_seq = z_v.unsqueeze(1).repeat(1, 8, 1)  # Expand to sequence of length 8
        return self._forward_impl(w2v_emb, bert_emb, z_v_seq)

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
        Runs Wav2Vec2 + BERT for audio-text, ViT for visual, followed by Cross-Attention and GRU.
        Call load_backbones() once before using this path.
        """
        if not self._backbones_loaded:
            raise RuntimeError(
                "Backbones not loaded. Call model.load_backbones() first, "
                "or use forward_from_features() for Phase 1 training."
            )

        # Wav2Vec2 conv layers are not FP16-safe — run in float32 outside autocast
        with torch.amp.autocast("cuda", enabled=False):
            w2v_out = self._wav2vec(audio_values.float()).last_hidden_state  # (B, T', 768)
            w2v_emb = w2v_out.mean(dim=1).to(audio_values.dtype)            # (B, 768)

        # Text branch — BERT on ASR transcript tokens
        bert_out = self._bert(input_ids=input_ids, attention_mask=attention_mask)
        bert_emb = bert_out.last_hidden_state[:, 0, :]            # (B, 768) CLS token

        # Visual branch — ViT on K keyframes per clip
        B, K, C, H, W = keyframe_pixels.shape
        frames = keyframe_pixels.view(B * K, C, H, W)
        vit_out = self._vit(pixel_values=frames).last_hidden_state[:, 0, :]  # (B*K, 768)
        z_v_seq = vit_out.view(B, K, 768)                        # (B, K, 768)

        return self._forward_impl(w2v_emb, bert_emb, z_v_seq)

    def _forward_impl(
        self,
        w2v_emb: torch.Tensor,
        bert_emb: torch.Tensor,
        z_v_seq: torch.Tensor,
    ) -> DetectorOutput:
        # 1. Multi-Head Cross-Modal Attention
        audio_text_seq = torch.stack([w2v_emb, bert_emb], dim=1)  # (B, 2, 768)

        # Visual queries audio/text
        z_v_attn, _ = self.cross_attn_v(query=z_v_seq, key=audio_text_seq, value=audio_text_seq)
        z_v_seq = self.norm_v(z_v_seq + z_v_attn)

        # Audio/text queries visual
        at_attn, _ = self.cross_attn_at(query=audio_text_seq, key=z_v_seq, value=z_v_seq)
        audio_text_seq = self.norm_at(audio_text_seq + at_attn)

        # Split back to acoustic and linguistic
        w2v_emb = audio_text_seq[:, 0, :]
        bert_emb = audio_text_seq[:, 1, :]
        z_at = torch.cat([w2v_emb, bert_emb], dim=-1)            # (B, 1536)

        # 2. Temporal GRU Aggregation on Visual Sequence
        gru_out, _ = self.vit_gru(z_v_seq)                       # (B, K, 768)
        z_v = gru_out[:, -1, :]                                  # Take last hidden state (B, 768)

        # 3. Detect
        return self._detect(z_at, z_v)

    # ── Convenience ───────────────────────────────────────────────────────────

    @staticmethod
    def p_fake(logit: torch.Tensor) -> torch.Tensor:
        """Convert raw logit → P(fake) ∈ [0, 1]."""
        return torch.sigmoid(logit)
