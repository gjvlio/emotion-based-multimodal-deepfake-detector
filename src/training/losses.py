"""
losses.py — Multi-task loss for deepfake detection.

L_total = L_BCE(fake_logit, fake_label)
        + lambda_a * L_CE(audio_emotion_logits, audio_emotion_label)
        + lambda_b * L_CE(visual_emotion_logits, visual_emotion_label)
        + lambda_s * L_BCE_masked(sarcasm_logit, sarcasm_label)

L_BCE uses BCEWithLogitsLoss (stable numerics, expects raw logits).
L_CE  uses CrossEntropyLoss  (expects raw logits, NOT softmax output).
Sarcasm loss is masked: samples with sarcasm_label == -1 are excluded.
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
    sarcasm: torch.Tensor


class MultiTaskLoss(nn.Module):
    def __init__(
        self,
        lambda_a: float = 0.5,
        lambda_b: float = 0.5,
        lambda_sarcasm: float = 0.3,
    ):
        super().__init__()
        self.lambda_a       = lambda_a
        self.lambda_b       = lambda_b
        self.lambda_sarcasm = lambda_sarcasm
        self._bce     = nn.BCEWithLogitsLoss()
        self._ce      = nn.CrossEntropyLoss(ignore_index=-1)
        self._bce_sum = nn.BCEWithLogitsLoss(reduction="sum")

    def forward(
        self,
        fake_logit:           torch.Tensor,   # (B, 1)  raw fake score
        fake_label:           torch.Tensor,   # (B,)    0 or 1 or -1 (MUStARD)
        emotion_a_logits:     torch.Tensor,   # (B, 6)  audio emotion
        emotion_b_logits:     torch.Tensor,   # (B, 6)  visual emotion
        audio_emotion_label:  torch.Tensor,   # (B,)    0-5 or -1
        visual_emotion_label: torch.Tensor,   # (B,)    0-5 or -1
        sarcasm_logit:        torch.Tensor,   # (B, 1)  raw sarcasm score
        sarcasm_label:        torch.Tensor,   # (B,)    0, 1, or -1 (masked)
    ) -> LossOutput:
        # Detection BCE — mask MUStARD clips (fake_label=-1, no ground truth)
        fake_mask = fake_label != -1
        if fake_mask.any():
            l_bce = self._bce(
                fake_logit.squeeze(1)[fake_mask],
                fake_label[fake_mask].float(),
            )
        else:
            l_bce = fake_logit.new_zeros(1).squeeze()

        # Emotion CE — CrossEntropyLoss(ignore_index=-1) returns nan when ALL masked
        emo_a_valid = (audio_emotion_label != -1).any()
        l_emo_a = self._ce(emotion_a_logits, audio_emotion_label) if emo_a_valid \
                  else fake_logit.new_zeros(1).squeeze()

        emo_b_valid = (visual_emotion_label != -1).any()
        l_emo_b = self._ce(emotion_b_logits, visual_emotion_label) if emo_b_valid \
                  else fake_logit.new_zeros(1).squeeze()

        # Sarcasm BCE — mask non-MUStARD clips (sarcasm_label=-1)
        sarc_mask = sarcasm_label != -1
        if sarc_mask.any():
            l_sarc = self._bce_sum(
                sarcasm_logit.squeeze(1)[sarc_mask],
                sarcasm_label[sarc_mask].float(),
            ) / sarc_mask.sum().float()
        else:
            l_sarc = fake_logit.new_zeros(1).squeeze()

        total = (
            l_bce
            + self.lambda_a * l_emo_a
            + self.lambda_b * l_emo_b
            + self.lambda_sarcasm * l_sarc
        )
        return LossOutput(
            total=total,
            bce=l_bce,
            emotion_a=l_emo_a,
            emotion_b=l_emo_b,
            sarcasm=l_sarc,
        )
