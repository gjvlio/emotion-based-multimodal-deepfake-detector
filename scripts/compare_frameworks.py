"""
compare_frameworks.py — "Battle of the Frameworks": DeLong's test (AUC) +
paired bootstrap (Accuracy/Precision/Recall/F1) comparing two frameworks'
per-clip scores on the SAME FakeAVCeleb sample.

Both tests are PAIRED — they require the same clips, same labels, on both
sides. That's why evaluate_fakeavceleb.py and eval_acenet_baseline.py must be
run with matching --n_real/--n_fake/--seed/--no_hard.

Inputs are two CSVs with columns: clip_id,fake_label,method,type,score,pred
  - scripts/evaluate_fakeavceleb.py --save_csv ...   (DeepSentinel)
  - scripts/eval_acenet_baseline.py --save_csv ...   (ACE-Net baseline)

Usage:
    python scripts/compare_frameworks.py \
        --a results/fakeavceleb_deepsentinel.csv --name_a DeepSentinel \
        --b results/acenet_fakeavceleb.csv       --name_b ACE-Net
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluation.significance import delong_test, paired_bootstrap_scorecard, print_scorecard


def load_csv(path: Path) -> dict:
    rows = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows[row["clip_id"]] = {
                "fake_label": int(row["fake_label"]),
                "score":      float(row["score"]),
            }
    return rows


def main():
    parser = argparse.ArgumentParser(description="Compare two frameworks' per-clip scores (paired significance tests)")
    parser.add_argument("--a", required=True, help="CSV from framework A (e.g. DeepSentinel)")
    parser.add_argument("--b", required=True, help="CSV from framework B (e.g. ACE-Net baseline)")
    parser.add_argument("--name_a", default="DeepSentinel")
    parser.add_argument("--name_b", default="ACE-Net")
    parser.add_argument("--n_boot", type=int, default=10_000)
    parser.add_argument("--seed",   type=int, default=42)
    args = parser.parse_args()

    rows_a = load_csv(Path(args.a))
    rows_b = load_csv(Path(args.b))

    shared = sorted(set(rows_a) & set(rows_b))
    only_a = set(rows_a) - set(rows_b)
    only_b = set(rows_b) - set(rows_a)
    if only_a or only_b:
        print(f"  WARNING: {len(only_a)} clips only in A, {len(only_b)} clips only in B — "
              f"dropped so the comparison stays paired (same clips on both sides).")
    if not shared:
        print("  ERROR: no clip_ids in common between the two CSVs — nothing to compare.")
        return

    mismatched = [cid for cid in shared if rows_a[cid]["fake_label"] != rows_b[cid]["fake_label"]]
    if mismatched:
        print(f"  ERROR: {len(mismatched)} clips have a DIFFERENT fake_label between the two CSVs "
              f"(e.g. {mismatched[0]}) — the two runs are not scoring the same ground truth. Aborting.")
        return

    y        = [rows_a[cid]["fake_label"] for cid in shared]
    scores_a = [rows_a[cid]["score"] for cid in shared]
    scores_b = [rows_b[cid]["score"] for cid in shared]

    print(f"  Paired clips  : {len(shared)}")
    print(f"  Real/Fake     : {y.count(0)}/{y.count(1)}")

    delong = delong_test(y, scores_a, scores_b)
    bootstrap = paired_bootstrap_scorecard(y, scores_a, scores_b, n_boot=args.n_boot, seed=args.seed)
    print_scorecard(args.name_a, args.name_b, delong, bootstrap)


if __name__ == "__main__":
    main()
