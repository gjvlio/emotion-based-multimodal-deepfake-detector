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
import torch.nn.functional as F


@dataclass
class LossOutput:
    total: torch.Tensor
    bce: torch.Tensor
    emotion_a: torch.Tensor
    emotion_b: torch.Tensor
    sarcasm: torch.Tensor
    domain: torch.Tensor = torch.tensor(0.0)
    margin: torch.Tensor = torch.tensor(0.0)


class MultiTaskLoss(nn.Module):
    def __init__(
        self,
        lambda_a: float = 0.1,
        lambda_b: float = 0.1,
        lambda_sarcasm: float = 0.05,
        lambda_domain: float = 0.1,
        lambda_margin: float = 0.2,
        margin: float = 1.5,
        pos_weight: float | None = 1.3835,
    ):
        super().__init__()
        self.lambda_a       = lambda_a
        self.lambda_b       = lambda_b
        self.lambda_sarcasm = lambda_sarcasm
        self.lambda_domain  = lambda_domain
        self.lambda_margin  = lambda_margin
        self.margin         = margin
        self._pw      = pos_weight
        self._bce     = nn.BCEWithLogitsLoss()
        self._ce      = nn.CrossEntropyLoss(ignore_index=-1)
        self._bce_sum = nn.BCEWithLogitsLoss(reduction="sum")

    def forward(
        self,
        fake_logit:           torch.Tensor,             # (B, 1)  raw fake score
        fake_label:           torch.Tensor,             # (B,)    0 or 1 or -1 (MUStARD)
        emotion_a_logits:     torch.Tensor,             # (B, 6)  audio emotion
        emotion_b_logits:     torch.Tensor,             # (B, 6)  visual emotion
        audio_emotion_label:  torch.Tensor,             # (B,)    0-5 or -1
        visual_emotion_label: torch.Tensor,             # (B,)    0-5 or -1
        sarcasm_logit:        torch.Tensor,             # (B, 1)  raw sarcasm score
        sarcasm_label:        torch.Tensor,             # (B,)    0, 1, or -1 (masked)
        domain_logits:        torch.Tensor | None = None,# (B, 5)  dataset domain logits (DANN)
        domain_label:         torch.Tensor | None = None,# (B,)    0-4 dataset domain index
    ) -> LossOutput:
        # Detection BCE — mask MUStARD clips (fake_label=-1, no ground truth)
        fake_mask = fake_label != -1
        if fake_mask.any():
            logits = fake_logit.squeeze(1)[fake_mask]
            labels = fake_label[fake_mask].float()
            if self._pw is not None:
                pw = torch.tensor([self._pw], device=logits.device, dtype=logits.dtype)
                l_bce = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pw)
            else:
                l_bce = self._bce(logits, labels)
        else:
            l_bce = fake_logit.new_zeros(1).squeeze()

        # Supervised Contrastive Margin Loss
        if self.lambda_margin > 0 and fake_mask.any():
            f_mask = (fake_label == 1) & fake_mask
            r_mask = (fake_label == 0) & fake_mask
            if f_mask.any() and r_mask.any():
                mean_fake = fake_logit.squeeze(1)[f_mask].mean()
                mean_real = fake_logit.squeeze(1)[r_mask].mean()
                l_margin = torch.clamp(self.margin - (mean_fake - mean_real), min=0.0)
            else:
                l_margin = fake_logit.new_zeros(1).squeeze()
        else:
            l_margin = fake_logit.new_zeros(1).squeeze()

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

        # Domain CE (DANN)
        if domain_logits is not None and domain_label is not None and (domain_label != -1).any():
            dom_valid = domain_label != -1
            l_domain = F.cross_entropy(domain_logits[dom_valid], domain_label[dom_valid])
        else:
            l_domain = fake_logit.new_zeros(1).squeeze()

        total = (
            l_bce
            + self.lambda_a * l_emo_a
            + self.lambda_b * l_emo_b
            + self.lambda_sarcasm * l_sarc
            + self.lambda_domain * l_domain
            + self.lambda_margin * l_margin
        )
        return LossOutput(
            total=total,
            bce=l_bce,
            emotion_a=l_emo_a,
            emotion_b=l_emo_b,
            sarcasm=l_sarc,
            domain=l_domain,
            margin=l_margin,
        )
