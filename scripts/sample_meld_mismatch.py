"""
sample_meld_mismatch.py
=======================
Generate emotion-mismatch pairs for Track 5.

Reads meld_real.csv and samples 50% of clips as video sources. Each video
clip is paired with a donor clip whose emotion differs — this is the
manipulated signal: face expresses emotion A while voice carries emotion B.

Pairing rules:
  - video_emotion != audio_emotion  (hard constraint)
  - donor sampled proportionally across all other emotions
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

EMOTIONS = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']


def sample_mismatch_pairs(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    df = df.sample(frac=1, random_state=int(rng.integers(1_000_000))).reset_index(drop=True)
    n_video = len(df) // 2
    video_pool = df.iloc[:n_video].copy()
    donor_pool = df  # donors drawn from ALL clips (a clip can donate audio to others)

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
            'output_stem':     f"FAKE_T5_{vrow['clip_id']}__AUDIO_{donor['clip_id']}",
            'label':           1,
        })

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Sample emotion-mismatch pairs for Track 5 (MELD audio-swap)"
    )
    parser.add_argument(
        '--real_csv',
        default='data/processed/meld_manifests/meld_real.csv',
        help='meld_real.csv produced by the MELD preprocessing step',
    )
    parser.add_argument(
        '--out_csv',
        default='data/processed/meld_manifests/meld_mismatch_pairs.csv',
    )
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    real_csv = Path(args.real_csv)
    if not real_csv.exists():
        print(f"ERROR: {real_csv} not found.", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(real_csv)
    print(f"Loaded {len(df)} clips from {real_csv}")
    print(f"Emotion distribution:\n{df['emotion'].value_counts().to_string()}")

    rng = np.random.default_rng(args.seed)
    pairs = sample_mismatch_pairs(df, rng)

    mismatch_rate = (pairs['video_emotion'] != pairs['audio_emotion']).mean()
    print(f"\nGenerated {len(pairs)} pairs  (mismatch rate: {mismatch_rate:.1%})")
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


if __name__ == '__main__':
    main()
