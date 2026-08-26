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
                "method":     row.get("method", "unknown"),
                "type":       row.get("type", "unknown"),
            }
    return rows


def print_method_breakdown(name_a: str, name_b: str, shared: list[str], rows_a: dict, rows_b: dict):
    from collections import defaultdict
    by_method = defaultdict(list)
    for cid in shared:
        method = rows_a[cid].get("method", "unknown")
        by_method[method].append(cid)

    print("\n" + "=" * 78)
    print(f"  PER-MANIPULATION STRESS BREAKDOWN: {name_a} vs {name_b}")
    print("=" * 78)
    print(f"  {'Manipulation Category':<24} {'N':>5}  {name_a + ' Acc':>15}  {name_b + ' Acc':>15}  {'Advantage':>12}")
    print("  " + "-" * 76)

    for method, cids in sorted(by_method.items()):
        y_m = [rows_a[c]["fake_label"] for c in cids]
        p_a = [1 if rows_a[c]["score"] >= 0.50 else 0 for c in cids]
        p_b = [1 if rows_b[c]["score"] >= 0.50 else 0 for c in cids]

        acc_a = sum(p == y for p, y in zip(p_a, y_m)) / max(len(cids), 1) * 100
        acc_b = sum(p == y for p, y in zip(p_b, y_m)) / max(len(cids), 1) * 100
        diff = acc_a - acc_b
        diff_str = f"{diff:+.1f}%" if diff != 0 else "0.0%"
        sign = f"🏆 {name_a}" if diff > 0 else (f"⚠️ {name_b}" if diff < 0 else "Tie")

        print(f"  {method:<24} {len(cids):>5}  {acc_a:>14.2f}%  {acc_b:>14.2f}%  {diff_str:>8} ({sign})")
    print("=" * 78 + "\n")


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
    print_method_breakdown(args.name_a, args.name_b, shared, rows_a, rows_b)


if __name__ == "__main__":
    main()
