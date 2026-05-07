"""
ablation.py — Δ-removal ablation study (validates RQ2).

Builds a variant of DeepfakeDetector where the Δ pathway is zeroed out
(classifier receives [fused ; 0_6] instead of [fused ; delta]).
Compares full-model vs no-delta performance on the same test set.
The accuracy drop quantifies Δ's contribution to detection.
"""
from __future__ import annotations

import logging
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.models.detection_model import DeepfakeDetector, DetectorOutput
from src.evaluation.metrics import DetectionMetrics, EvalResults

log = logging.getLogger(__name__)


class DeepfakeDetectorNoDelta(DeepfakeDetector):
    """Identical to DeepfakeDetector but Δ is replaced with zeros."""

    def _detect(self, z_at: torch.Tensor, z_v: torch.Tensor) -> DetectorOutput:
        emo_a  = self.emotion_head_a(z_at)
        emo_b  = self.emotion_head_b(z_v)
        fused  = self.bilinear_fusion(z_at, z_v)
        # Replace delta with zeros — isolates whether delta contributes
        delta  = torch.zeros(z_at.size(0), 6, device=z_at.device)
        combined = torch.cat([fused, delta], dim=-1)
        logit    = self.classifier(combined)
        return DetectorOutput(logit=logit, emotion_a=emo_a, emotion_b=emo_b)


class AblationEvaluator:
    def __init__(self, threshold: float = 0.5, device: str = "cpu"):
        self.threshold = threshold
        self.device    = device
        self.metrics   = DetectionMetrics(threshold)

    @torch.no_grad()
    def _run_inference(
        self, model: DeepfakeDetector, loader: DataLoader
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        model.eval()
        logits, labels, delta_norms = [], [], []
        for batch in loader:
            z_at = batch["z_at"].to(self.device)
            z_v  = batch["z_v"].to(self.device)
            out  = model.forward_from_features(z_at, z_v)
            logits.append(out.logit.cpu())
            labels.append(batch["fake_label"])
            d = torch.abs(
                F.softmax(out.emotion_a.cpu(), dim=-1) -
                F.softmax(out.emotion_b.cpu(), dim=-1)
            ).norm(dim=-1)
            delta_norms.append(d)
        return (
            torch.cat(logits),
            torch.cat(labels),
            torch.cat(delta_norms),
        )

    def run(
        self,
        full_model:       DeepfakeDetector,
        test_loader:      DataLoader,
        checkpoint_path:  str | None = None,
    ) -> dict:
        """
        Returns dict with 'full' and 'no_delta' EvalResults.
        """
        # Build no-delta variant using same weights
        no_delta_model = DeepfakeDetectorNoDelta()
        # Copy detection component weights (not backbones)
        no_delta_model.emotion_head_a.load_state_dict(full_model.emotion_head_a.state_dict())
        no_delta_model.emotion_head_b.load_state_dict(full_model.emotion_head_b.state_dict())
        no_delta_model.bilinear_fusion.load_state_dict(full_model.bilinear_fusion.state_dict())
        no_delta_model.classifier.load_state_dict(full_model.classifier.state_dict())
        no_delta_model = no_delta_model.to(self.device)

        full_model = full_model.to(self.device)

        pipelines = []
        for batch in test_loader:
            pipelines.extend(batch["source_pipeline"])

        logits_full, labels, d_norms = self._run_inference(full_model, test_loader)
        logits_nd, _, _              = self._run_inference(no_delta_model, test_loader)

        results_full = self.metrics.evaluate(logits_full, labels, d_norms, pipelines)
        results_nd   = self.metrics.evaluate(logits_nd,   labels, None,    pipelines)

        drop = results_full.f1 - results_nd.f1
        log.info(f"Ablation — Full F1: {results_full.f1:.4f}  No-Δ F1: {results_nd.f1:.4f}  Drop: {drop:.4f}")
        return {"full": results_full, "no_delta": results_nd, "f1_drop": drop}

