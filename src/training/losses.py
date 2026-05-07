"""
losses.py — Multi-task loss for deepfake detection.

L_total = L_BCE(fake_logit, fake_label)
        + lambda_a * L_CE(audio_emotion_logits, audio_emotion_label)
        + lambda_b * L_CE(visual_emotion_logits, visual_emotion_label)

L_BCE uses BCEWithLogitsLoss (stable numerics, expects raw logits).
L_CE  uses CrossEntropyLoss  (expects raw logits, NOT softmax output).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class LossOutput:
    total: torch.Tensor
    bce: torch.Tensor
    emotion_a: torch.Tensor
    emotion_b: torch.Tensor


class MultiTaskLoss(nn.Module):
    def __init__(self, lambda_a: float = 0.5, lambda_b: float = 0.5):
        super().__init__()
        self.lambda_a = lambda_a
        self.lambda_b = lambda_b
        self._bce = nn.BCEWithLogitsLoss()
        self._ce  = nn.CrossEntropyLoss(ignore_index=-1)  # -1 = label unknown / masked

    def forward(
        self,
        fake_logit:          torch.Tensor,   # (B, 1)  raw fake score
        fake_label:          torch.Tensor,   # (B,)    0 or 1
        emotion_a_logits:    torch.Tensor,   # (B, 6)  audio emotion
        emotion_b_logits:    torch.Tensor,   # (B, 6)  visual emotion
        audio_emotion_label: torch.Tensor,   # (B,)    0-5 or -1
        visual_emotion_label: torch.Tensor,  # (B,)    0-5 or -1
    ) -> LossOutput:
        l_bce = self._bce(fake_logit.squeeze(1), fake_label.float())
        l_emo_a = self._ce(emotion_a_logits, audio_emotion_label)
        l_emo_b = self._ce(emotion_b_logits, visual_emotion_label)
        total = l_bce + self.lambda_a * l_emo_a + self.lambda_b * l_emo_b
        return LossOutput(total=total, bce=l_bce, emotion_a=l_emo_a, emotion_b=l_emo_b)
