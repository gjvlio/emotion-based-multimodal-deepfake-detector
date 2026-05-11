"""
sample_meld_mismatch.py
=======================
Generate emotion-mismatch pairs for Track 4 (MuseTalk).

Reads meld_fake_src.csv (the fake-source half from sample_meld.py — disjoint
from the real training pool in meld_real.csv) and pairs every clip with a
donor whose emotion differs.

Pairing rules:
  - video_emotion != audio_emotion  (hard constraint)
  - donor sampled from same fake-source pool (real pool never touched)
  - same clip cannot be both video and audio in the same pair
  - seed fixed for reproducibility

Outputs:
  data/processed/meld_manifests/meld_mismatch_pairs.csv

Usage:
    python scripts/sample_meld_mismatch.py
    python scripts/sample_meld_mismatch.py --seed 99 --out_csv custom_pairs.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def sample_mismatch_pairs(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    df = df.sample(frac=1, random_state=int(rng.integers(1_000_000))).reset_index(drop=True)
    # Use ALL clips as video sources — entire fake pool becomes Track 4 fakes
    video_pool = df
    donor_pool = df

    rows = []
    for _, vrow in video_pool.iterrows():
        candidates = donor_pool[
            (donor_pool['emotion'] != vrow['emotion']) &
            (donor_pool['clip_id']  != vrow['clip_id'])
        ]
        if candidates.empty:
            continue
        donor = candidates.sample(1, random_state=int(rng.integers(1_000_000))).iloc[0]
        rows.append({
            'video_clip':      vrow['video_path'],
            'audio_clip':      donor['video_path'],
            'video_speaker':   vrow['speaker'],
            'audio_speaker':   donor['speaker'],
            'video_emotion':   vrow['emotion'],
            'audio_emotion':   donor['emotion'],
            'video_utterance': vrow['utterance'],
            'audio_utterance': donor['utterance'],
            'video_duration':  vrow['duration'],
            'audio_duration':  donor['duration'],
            'video_clip_id':   vrow['clip_id'],
            'audio_clip_id':   donor['clip_id'],
            'split':           vrow['split'],
            'output_stem':     f"FAKE_T4_{vrow['clip_id']}__AUDIO_{donor['clip_id']}",
            'label':           1,
        })

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Sample emotion-mismatch pairs for Track 4 (MuseTalk on MELD)"
    )
    parser.add_argument(
        '--fake_src_csv',
        default='data/processed/meld_manifests/meld_fake_src.csv',
        help='meld_fake_src.csv — fake-source half produced by sample_meld.py',
    )
    parser.add_argument(
        '--out_csv',
        default='data/processed/meld_manifests/meld_mismatch_pairs.csv',
    )
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    fake_src_csv = Path(args.fake_src_csv)
    if not fake_src_csv.exists():
        print(f"ERROR: {fake_src_csv} not found.", file=sys.stderr)
        print("Run: python scripts/sample_meld.py  (re-saves meld_fake_src.csv)")
        print("Or reconstruct from existing meld_pairs.csv:")
        print("  python scripts/reconstruct_fake_src.py")
        sys.exit(1)

    df = pd.read_csv(fake_src_csv)
    print(f"Loaded {len(df)} fake-source clips from {fake_src_csv}")
    print(f"Emotion distribution:\n{df['emotion'].value_counts().to_string()}")

    rng = np.random.default_rng(args.seed)
    pairs = sample_mismatch_pairs(df, rng)

    mismatch_rate = (pairs['video_emotion'] != pairs['audio_emotion']).mean()
    no_donor = len(df) - len(pairs)
    print(f"\nGenerated {len(pairs)} pairs  (mismatch rate: {mismatch_rate:.1%})")
    if no_donor:
        print(f"  {no_donor} clips skipped — no emotion-mismatched donor available")
    print(f"Emotion pair breakdown:")
    pair_counts = (
        pairs.groupby(['video_emotion', 'audio_emotion'])
        .size()
        .reset_index(name='count')
        .sort_values('count', ascending=False)
    )
    print(pair_counts.to_string(index=False))

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(out_csv, index=False)
    print(f"\nSaved -> {out_csv}")
    print(f"\nReal pool  (meld_real.csv):        {len(pd.read_csv(fake_src_csv.parent / 'meld_real.csv'))} clips  label=0")
    print(f"Fake pairs (meld_mismatch_pairs):  {len(pairs)} clips  label=1")


if __name__ == '__main__':
    main()
