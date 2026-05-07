"""
metrics.py — Detection evaluation metrics for RQ1 and RQ2.

Primary (RQ1 — detection accuracy):
    accuracy, precision, recall, F1, AUC-ROC, confusion matrix,
    per-pipeline breakdown

Secondary (RQ2 — Δ as predictor of P(fake)):
    Pearson/Spearman correlation between ||Δ|| and P(fake)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, confusion_matrix,
)
from scipy.stats import pearsonr, spearmanr

log = logging.getLogger(__name__)


@dataclass
class PipelineMetrics:
    pipeline:  str
    n:         int
    accuracy:  float
    precision: float
    recall:    float
    f1:        float


@dataclass
class EvalResults:
    accuracy:   float
    precision:  float
    recall:     float
    f1:         float
    auc_roc:    float
    confusion:  np.ndarray                         # (2, 2)
    per_pipeline: List[PipelineMetrics] = field(default_factory=list)
    delta_pearson:  Optional[float] = None
    delta_spearman: Optional[float] = None


class DetectionMetrics:
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def evaluate(
        self,
        all_logits:    torch.Tensor,        # (N, 1)
        all_labels:    torch.Tensor,        # (N,)
        all_delta_norms: Optional[torch.Tensor] = None,   # (N,)  ||Δ||
        all_pipelines: Optional[List[str]]  = None,
    ) -> EvalResults:

        probs  = torch.sigmoid(all_logits.squeeze(1)).numpy()
        preds  = (probs >= self.threshold).astype(int)
        labels = all_labels.numpy().astype(int)

        acc        = accuracy_score(labels, preds)
        prec, rec, f1, _ = precision_recall_fscore_support(
            labels, preds, average="binary", zero_division=0
        )
        try:
            auc = roc_auc_score(labels, probs)
        except ValueError:
            auc = float("nan")
        cm = confusion_matrix(labels, preds)

        per_pipeline = []
        if all_pipelines:
            unique = sorted(set(all_pipelines))
            for pipe in unique:
                idx = [i for i, p in enumerate(all_pipelines) if p == pipe]
                p_l = labels[idx]; p_p = preds[idx]
                p_prec, p_rec, p_f1, _ = precision_recall_fscore_support(
                    p_l, p_p, average="binary", zero_division=0
                )
                per_pipeline.append(PipelineMetrics(
                    pipeline=pipe, n=len(idx),
                    accuracy=accuracy_score(p_l, p_p),
                    precision=float(p_prec), recall=float(p_rec), f1=float(p_f1),
                ))

        # RQ2 — Δ correlation
        delta_p = delta_s = None
        if all_delta_norms is not None:
            d = all_delta_norms.numpy()
            try:
                delta_p = float(pearsonr(d, probs).statistic)
                delta_s = float(spearmanr(d, probs).statistic)
            except Exception:
                pass

        return EvalResults(
            accuracy=float(acc), precision=float(prec),
            recall=float(rec), f1=float(f1), auc_roc=float(auc),
            confusion=cm, per_pipeline=per_pipeline,
            delta_pearson=delta_p, delta_spearman=delta_s,
        )

    def report(self, results: EvalResults) -> str:
        lines = [
            "=" * 55,
            "DETECTION EVALUATION RESULTS",
            "=" * 55,
            f"Accuracy  : {results.accuracy:.4f}",
            f"Precision : {results.precision:.4f}",
            f"Recall    : {results.recall:.4f}",
            f"F1        : {results.f1:.4f}",
            f"AUC-ROC   : {results.auc_roc:.4f}",
            f"Confusion :\n{results.confusion}",
        ]
        if results.per_pipeline:
            lines.append("\nPer-pipeline breakdown:")
            for pm in results.per_pipeline:
                lines.append(
                    f"  {pm.pipeline:12s}  n={pm.n:5d}  "
                    f"acc={pm.accuracy:.3f}  f1={pm.f1:.3f}"
                )
        if results.delta_pearson is not None:
            lines.append(
                f"\nRQ2 — ||Δ|| vs P(fake): "
                f"Pearson={results.delta_pearson:.4f}  "
                f"Spearman={results.delta_spearman:.4f}"
            )
        lines.append("=" * 55)
        return "\n".join(lines)
