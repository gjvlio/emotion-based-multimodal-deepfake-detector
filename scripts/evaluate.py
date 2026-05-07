"""
evaluate.py — Run full evaluation: detection metrics + ablation + OOD check.

Usage (from repo root):
    python scripts/evaluate.py --checkpoint checkpoints/best_phase1.pt
    python scripts/evaluate.py --checkpoint checkpoints/best_phase1.pt --ood_csv path/to/benchmark.csv
    python scripts/evaluate.py --checkpoint checkpoints/best_phase1.pt --ablation --threshold 0.5
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models.detection_model import DeepfakeDetector
from src.training.dataset import DeepfakeDataset
from src.evaluation.metrics import DetectionMetrics
from src.evaluation.ablation import AblationEvaluator
from src.evaluation.ood_eval import OODEvaluator
from src.utils.config import Config
import torch.nn.functional as F

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


@torch.no_grad()
def run_test_inference(model, test_loader, device):
    model.eval()
    all_logits, all_labels, all_deltas, all_pipelines = [], [], [], []
    for batch in test_loader:
        z_at = batch["z_at"].to(device)
        z_v  = batch["z_v"].to(device)
        out  = model.forward_from_features(z_at, z_v)
        all_logits.append(out.logit.cpu())
        all_labels.append(batch["fake_label"])
        delta_norm = torch.abs(
            F.softmax(out.emotion_a.cpu(), dim=-1) -
            F.softmax(out.emotion_b.cpu(), dim=-1)
        ).norm(dim=-1)
        all_deltas.append(delta_norm)
        all_pipelines.extend(batch["source_pipeline"])
    return (
        torch.cat(all_logits),
        torch.cat(all_labels),
        torch.cat(all_deltas),
        all_pipelines,
    )


def main():
    parser = argparse.ArgumentParser(description="Evaluate deepfake detector")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config",     default=None)
    parser.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--threshold",  type=float, default=0.5)
    parser.add_argument("--ablation",   action="store_true", help="Run Δ-removal ablation (RQ2)")
    parser.add_argument("--ood_csv",    default=None, help="CSV for OOD benchmark evaluation")
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config)

    model = DeepfakeDetector(
        wav2vec_model  = cfg.model.wav2vec_model,
        bert_model     = cfg.model.bert_model,
        vit_model      = cfg.model.vit_model,
        n_emotions     = cfg.model.n_emotions,
        proj_dim       = cfg.model.proj_dim,
        dropout_heads  = cfg.model.dropout_heads,
        dropout_cls    = cfg.model.dropout_classifier,
    )
    ckpt = torch.load(args.checkpoint, weights_only=True)
    model.load_state_dict(ckpt["model_state"], strict=False)
    model = model.to(args.device)
    log.info(f"Loaded checkpoint: {args.checkpoint}")

    _, _, test_ds = DeepfakeDataset.stratified_split(
        preprocessed_dir = cfg.paths.preprocessed_dir,
        train_ratio      = cfg.training.train_ratio,
        val_ratio        = cfg.training.val_ratio,
        track1_meta      = cfg.paths.track1_meta,
        track2_meta      = cfg.paths.track2_meta,
        track3_meta      = cfg.paths.track3_meta,
        track4_meta      = cfg.paths.track4_meta,
        meld_real_csv    = cfg.paths.meld_real_csv,
        mosei_real_csv   = cfg.paths.mosei_real_csv,
    )
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)

    # Primary evaluation (RQ1)
    logits, labels, deltas, pipelines = run_test_inference(model, test_loader, args.device)
    metrics = DetectionMetrics(args.threshold)
    results = metrics.evaluate(logits, labels, deltas, pipelines)
    print(metrics.report(results))

    # Strict threshold report
    metrics65 = DetectionMetrics(0.65)
    r65 = metrics65.evaluate(logits, labels)
    log.info(f"At threshold=0.65: Acc={r65.accuracy:.4f}  F1={r65.f1:.4f}  FP_rate={(r65.confusion[0,1]/max(r65.confusion[0].sum(),1)):.4f}")

    # Ablation (RQ2)
    if args.ablation:
        evaluator = AblationEvaluator(args.threshold, args.device)
        ab_results = evaluator.run(model, test_loader)
        print(f"\nAblation — Full F1: {ab_results['full'].f1:.4f}  "
              f"No-Δ F1: {ab_results['no_delta'].f1:.4f}  "
              f"Drop: {ab_results['f1_drop']:.4f}")

    # OOD evaluation
    if args.ood_csv:
        ood_eval = OODEvaluator(model, cfg.paths.preprocessed_dir, args.threshold, device=args.device)
        ood_results = ood_eval.evaluate(args.ood_csv, indist_results=results)
        print(f"OOD F1: {ood_results.f1:.4f}  AUC: {ood_results.auc_roc:.4f}")


if __name__ == "__main__":
    main()
