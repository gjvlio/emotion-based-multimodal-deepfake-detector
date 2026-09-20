"""
eval_acenet_baseline.py — Run the reimplemented ACE-Net baseline (separate repo:
github.com/gjvlio/Baseline_Training) on the IDENTICAL FakeAVCeleb sample used by
evaluate_fakeavceleb.py, so the two frameworks can be compared on the same clips
(paired significance testing — see scripts/compare_frameworks.py).

Requires: ffmpeg on PATH, Baseline_Training repo cloned locally with its own
.venv/checkpoints already set up (see that repo's README).

Usage:
    python scripts/eval_acenet_baseline.py --save_csv results/acenet_fakeavceleb.csv
    python scripts/eval_acenet_baseline.py --baseline_repo D:/Documents/Programming/Baseline_Training
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import logging
import sys
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))       # for evaluate_fakeavceleb import
from evaluate_fakeavceleb import load_clips, SECTION            # reuse IDENTICAL clip sampling

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

DEFAULT_BASELINE_REPO = "D:/Documents/Programming/Baseline_Training"


def _section(title: str) -> None:
    print(f"\n{SECTION}\n  {title}\n{SECTION}")


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    parser = argparse.ArgumentParser(description="Run ACE-Net baseline on the same FakeAVCeleb sample")
    parser.add_argument("--baseline_repo", default=DEFAULT_BASELINE_REPO,
                        help="Path to the Baseline_Training repo (ACE-Net reimplementation)")
    parser.add_argument("--device",   default="cuda")
    parser.add_argument("--n_real",   type=int, default=200)
    parser.add_argument("--n_fake",   type=int, default=800)
    parser.add_argument("--seed",     type=int, default=42)
    parser.add_argument("--no_hard",  action="store_true",
                        help="Must match whatever was used for the DeepSentinel run being compared against")
    parser.add_argument("--save_csv", default="results/acenet_fakeavceleb.csv")
    args = parser.parse_args()

    baseline_root = Path(args.baseline_repo)
    if not baseline_root.exists():
        print(f"ERROR: Baseline_Training repo not found at {baseline_root}")
        return

    _section("ACE-Net baseline - FakeAVCeleb (same sample as DeepSentinel eval)")
    hard = not args.no_hard
    print(f"  Baseline repo : {baseline_root}")
    print(f"  Sample        : {args.n_real} real + {args.n_fake} fake, seed={args.seed}, hard={hard}")
    print(f"  NOTE          : uses the SAME sampling function/seed/args as evaluate_fakeavceleb.py -")
    print(f"                  match --n_real/--n_fake/--seed/--no_hard to that run for a valid pairing.")

    clips = load_clips(args.n_real, args.n_fake, seed=args.seed, hard=hard)
    print(f"  Sampled       : {len(clips)} clips")

    _section("Loading ACE-Net baseline models")
    preprocess_mod = _load_module("acenet_preprocess", baseline_root / "web" / "preprocess.py")
    inference_mod = _load_module("acenet_inference", baseline_root / "web" / "inference.py")
    preprocess_mod.warm_up()
    inference_mod.warm_up(device=args.device)

    _section("Running ACE-Net inference (MP4 -> mel/BERT/keyframes -> P(fake))")
    results = []
    pbar = tqdm(clips, desc="ACE-Net", unit="clip", dynamic_ncols=True)
    for c in pbar:
        try:
            batch = preprocess_mod.preprocess(c["video_path"])
            out = inference_mod.run(batch)
        except Exception as e:
            log.warning(f"ACE-Net failed on {c['clip_id']}: {type(e).__name__}: {e}")
            continue
        score = float(out["fake_prob"])
        results.append({
            "clip_id":    c["clip_id"],
            "fake_label": c["fake_label"],
            "method":     c["method"],
            "type":       c["type"],
            "score":      score,
            "pred":       1 if score >= 0.5 else 0,
        })
        pbar.set_postfix(score=f"{score:.3f}")

    print(f"\n  Evaluated : {len(results)} clips  ({len(clips) - len(results)} failed/skipped)")
    if not results:
        print("  No results — check ffmpeg/checkpoints/paths.")
        return

    from sklearn.metrics import roc_auc_score
    labels = [r["fake_label"] for r in results]
    scores = [r["score"] for r in results]
    acc = sum(1 for r in results if r["pred"] == r["fake_label"]) / len(results)
    auc = roc_auc_score(labels, scores) if len(set(labels)) == 2 else float("nan")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  AUC-ROC  : {auc:.4f}")

    csv_path = Path(args.save_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["clip_id", "fake_label", "method", "type", "score", "pred"])
        writer.writeheader()
        writer.writerows(results)
    print(f"  Per-clip results saved -> {csv_path}  ({len(results)} rows)")
    print(f"\n  Next: python scripts/compare_frameworks.py --a results/fakeavceleb_deepsentinel.csv --b {csv_path}")


if __name__ == "__main__":
    main()
