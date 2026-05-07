"""
ood_eval.py — Out-of-distribution evaluation on benchmark datasets.

Evaluates the trained model on an external benchmark (DFDC, FaceForensics++,
DigiFakeAV, etc.) without training on it. Reports accuracy degradation vs the
in-distribution test set to address the generalization claim.

The benchmark CSV must have columns:
    clip_id, video_path, label  (0=real, 1=fake)
and cached Z_at/Z_v tensors under preprocessed_dir/features/{z_at,z_v}/{clip_id}.pt
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from src.models.detection_model import DeepfakeDetector
from src.evaluation.metrics import DetectionMetrics, EvalResults

log = logging.getLogger(__name__)


class OODDataset(Dataset):
    def __init__(self, csv_path: str | Path, preprocessed_dir: str | Path):
        self.preprocessed_dir = Path(preprocessed_dir)
        df = pd.read_csv(csv_path)
        self._records = []
        for _, row in df.iterrows():
            clip_id  = str(row["clip_id"])
            z_at_p   = self.preprocessed_dir / "features" / "z_at" / f"{clip_id}.pt"
            z_v_p    = self.preprocessed_dir / "features" / "z_v"  / f"{clip_id}.pt"
            if not z_at_p.exists() or not z_v_p.exists():
                continue
            self._records.append({
                "clip_id":    clip_id,
                "z_at_path":  str(z_at_p),
                "z_v_path":   str(z_v_p),
                "fake_label": int(row["label"]),
            })
        log.info(f"OOD dataset: {len(self._records)} samples from {csv_path}")

    def __len__(self): return len(self._records)

    def __getitem__(self, idx):
        r = self._records[idx]
        return {
            "z_at":       torch.load(r["z_at_path"], weights_only=True).float(),
            "z_v":        torch.load(r["z_v_path"],  weights_only=True).float(),
            "fake_label": torch.tensor(r["fake_label"], dtype=torch.long),
            "clip_id":    r["clip_id"],
        }


class OODEvaluator:
    def __init__(
        self,
        model:           DeepfakeDetector,
        preprocessed_dir: str | Path,
        threshold:       float = 0.5,
        batch_size:      int   = 32,
        device:          str   = "cpu",
    ):
        self.model            = model.to(device)
        self.preprocessed_dir = Path(preprocessed_dir)
        self.threshold        = threshold
        self.batch_size       = batch_size
        self.device           = device
        self.metrics          = DetectionMetrics(threshold)

    @torch.no_grad()
    def evaluate(
        self,
        benchmark_csv:      str | Path,
        indist_results:     Optional[EvalResults] = None,
    ) -> EvalResults:
        """
        Run inference on OOD benchmark.
        If indist_results provided, logs degradation stats.
        """
        ds = OODDataset(benchmark_csv, self.preprocessed_dir)
        if len(ds) == 0:
            raise ValueError(f"No preprocessed samples found for {benchmark_csv}.")

        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=False, num_workers=0)
        self.model.eval()
        logits, labels = [], []
        for batch in loader:
            z_at = batch["z_at"].to(self.device)
            z_v  = batch["z_v"].to(self.device)
            out  = self.model.forward_from_features(z_at, z_v)
            logits.append(out.logit.cpu())
            labels.append(batch["fake_label"])

        logits = torch.cat(logits)
        labels = torch.cat(labels)
        results = self.metrics.evaluate(logits, labels)

        log.info(self.metrics.report(results))

        if indist_results:
            drop_acc = indist_results.accuracy - results.accuracy
            drop_f1  = indist_results.f1       - results.f1
            log.info(
                f"OOD degradation — Accuracy: -{drop_acc:.4f}  F1: -{drop_f1:.4f}"
            )

        return results
