"""
sample_by_track.py
==================
Stratified split of swap_pairs.csv into three per-track manifests.

Split rationale (default 20/30/50):
  Track 1 (20%): synthetic audio, original face — simplest fake
  Track 2 (30%): synthetic audio, Wav2Lip lip-reanimated face — medium difficulty
  Track 3 (50%): synthetic audio, SadTalker fully generated face — hardest to detect

Sampling is stratified by actor_id so every actor contributes proportionally
to all three tracks, preserving actor diversity across the dataset.

Usage:
    python scripts/sample_by_track.py \\
        --pairs_csv  data/processed/track1_manifests/swap_pairs.csv \\
        --out_dir    data/processed/track1_manifests \\
        [--t1 0.20]  [--t2 0.30]  [--t3 0.50] \\
        [--seed 42]

Outputs (written to --out_dir):
    track1_pairs.csv  — pairs assigned to Track 1
    track2_pairs.csv  — pairs assigned to Track 2 (Wav2Lip input)
    track3_pairs.csv  — pairs assigned to Track 3 (SadTalker input)

The three CSVs are non-overlapping and their rows sum to the full swap_pairs.csv.
Pass --filter_csv to track2_generate.py / track3_generate.py to use them.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def stratified_split(
    pairs: pd.DataFrame,
    t1: float,
    t2: float,
    t3: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    assert abs(t1 + t2 + t3 - 1.0) < 1e-6, f"Fractions must sum to 1.0, got {t1+t2+t3}"

    rng = np.random.default_rng(seed)
    t1_idx, t2_idx, t3_idx = [], [], []

    for actor_id, group in pairs.groupby("actor_id"):
        idx = np.array(group.index.tolist())
        rng.shuffle(idx)
        n = len(idx)
        n1 = round(n * t1)
        n2 = round(n * t2)
        # t3 takes remainder to absorb rounding error
        t1_idx.extend(idx[:n1].tolist())
        t2_idx.extend(idx[n1 : n1 + n2].tolist())
        t3_idx.extend(idx[n1 + n2 :].tolist())

    return pairs.loc[t1_idx].reset_index(drop=True), \
           pairs.loc[t2_idx].reset_index(drop=True), \
           pairs.loc[t3_idx].reset_index(drop=True)


def print_split_summary(
    pairs: pd.DataFrame,
    t1_pairs: pd.DataFrame,
    t2_pairs: pd.DataFrame,
    t3_pairs: pd.DataFrame,
):
    total = len(pairs)
    print("\n" + "=" * 55)
    print("TRACK SPLIT SUMMARY")
    print("=" * 55)
    print(f"Total pairs:  {total}")
    print(f"Track 1:      {len(t1_pairs):>5}  ({100*len(t1_pairs)/total:.1f}%)")
    print(f"Track 2:      {len(t2_pairs):>5}  ({100*len(t2_pairs)/total:.1f}%)")
    print(f"Track 3:      {len(t3_pairs):>5}  ({100*len(t3_pairs)/total:.1f}%)")
    check = len(t1_pairs) + len(t2_pairs) + len(t3_pairs)
    print(f"Sum check:    {check}  ({'OK' if check == total else 'MISMATCH'})")

    print("\nPer-track actor coverage:")
    for label, df in [("Track 1", t1_pairs), ("Track 2", t2_pairs), ("Track 3", t3_pairs)]:
        print(f"  {label}: {df['actor_id'].nunique()} actors, "
              f"{df['video_emotion'].nunique()} face emotions, "
              f"{df['audio_emotion'].nunique()} audio emotions")
    print("=" * 55)


def main():
    parser = argparse.ArgumentParser(
        description="Stratified 20/30/50 split of swap pairs across Track 1/2/3"
    )
    parser.add_argument("--pairs_csv", required=True,
                        help="swap_pairs.csv from parse_cremad.py")
    parser.add_argument("--out_dir",   required=True,
                        help="Directory to write track1/2/3_pairs.csv")
    parser.add_argument("--t1",  type=float, default=0.20,
                        help="Fraction assigned to Track 1 (default: 0.20)")
    parser.add_argument("--t2",  type=float, default=0.30,
                        help="Fraction assigned to Track 2 (default: 0.30)")
    parser.add_argument("--t3",  type=float, default=0.50,
                        help="Fraction assigned to Track 3 (default: 0.50)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    args = parser.parse_args()

    if abs(args.t1 + args.t2 + args.t3 - 1.0) > 1e-6:
        print(f"Error: --t1 + --t2 + --t3 must equal 1.0 (got {args.t1+args.t2+args.t3})")
        sys.exit(1)

    pairs_path = Path(args.pairs_csv)
    out_dir    = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not pairs_path.exists():
        print(f"Error: pairs CSV not found: {pairs_path}")
        sys.exit(1)

    pairs = pd.read_csv(pairs_path)
    print(f"Loaded {len(pairs)} pairs from {pairs_path}")
    print(f"Split: Track1={args.t1:.0%}  Track2={args.t2:.0%}  Track3={args.t3:.0%}  seed={args.seed}")

    t1_pairs, t2_pairs, t3_pairs = stratified_split(
        pairs, args.t1, args.t2, args.t3, args.seed
    )

    t1_path = out_dir / "track1_pairs.csv"
    t2_path = out_dir / "track2_pairs.csv"
    t3_path = out_dir / "track3_pairs.csv"

    t2_pairs = t2_pairs.copy()
    t2_pairs['output_stem'] = t2_pairs['output_stem'].str.replace('FAKE_T1_', 'FAKE_T2_', regex=False)
    t3_pairs = t3_pairs.copy()
    t3_pairs['output_stem'] = t3_pairs['output_stem'].str.replace('FAKE_T1_', 'FAKE_T3_', regex=False)

    t1_pairs.to_csv(t1_path, index=False)
    t2_pairs.to_csv(t2_path, index=False)
    t3_pairs.to_csv(t3_path, index=False)

    print_split_summary(pairs, t1_pairs, t2_pairs, t3_pairs)

    print(f"\nWritten to {out_dir}/")
    print(f"  {t1_path.name}: {len(t1_pairs)} rows")
    print(f"  {t2_path.name}: {len(t2_pairs)} rows")
    print(f"  {t3_path.name}: {len(t3_pairs)} rows")
    print(f"\nNext steps:")
    print(f"  Track 1: python src/track1/track1_generate.py --pairs_csv {t1_path} ...")
    print(f"  Track 2: python src/track2/track2_generate.py --filter_csv {t2_path} ...")
    print(f"  Track 3: python src/track3/track3_generate.py --filter_csv {t3_path} ...")


if __name__ == "__main__":
    main()
