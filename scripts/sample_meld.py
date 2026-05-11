"""
sample_meld.py
==============
Split MELD clips into real (50%) and fake-source (50%) halves, then build
cross-speaker swap pairs for Track 4 (Wav2Lip on real MELD audio).

Split is stratified by speaker so every character contributes proportionally
to both halves.  Pairing rule: video_speaker != audio_speaker.

Usage (run from repo root):
    python scripts/sample_meld.py \\
        --meld_dir  data/raw/MELD/MELD-RAW/MELD.Raw \\
        --out_dir   data/processed/meld_manifests \\
        [--seed 42]

Outputs:
    data/processed/meld_manifests/
        meld_real.csv        -- real clips (label=0)
        meld_pairs.csv       -- cross-speaker swap pairs for Track 4 (label=1)
"""

import argparse
import json
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd


def get_duration(video_path: str) -> float:
    try:
        r = subprocess.run(
            ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_streams', video_path],
            capture_output=True, text=True, timeout=10,
        )
        d = json.loads(r.stdout)
        streams = [s for s in d.get('streams', []) if s.get('codec_type') == 'video']
        if streams:
            return float(streams[0].get('duration', 0))
    except Exception:
        pass
    return 0.0

SPLIT_DIRS = {
    "train": ("train/train_splits",         "train/train_sent_emo.csv"),
    "dev":   ("dev/dev_splits_complete",    "dev/dev_sent_emo.csv"),
    "test":  ("test/output_repeated_splits_test", "test/test_sent_emo.csv"),
}


EMOTION_MAP = {
    "neutral":  "neutral",
    "joy":      "joy",
    "sadness":  "sadness",
    "anger":    "anger",
    "fear":     "fear",
    "disgust":  "disgust",
    "surprise": "neutral",  # mapped to neutral per label spec
}


def load_meld(meld_dir: Path, min_duration: float = 2.5) -> pd.DataFrame:
    rows = []
    skipped_missing = skipped_short = 0
    for split, (vid_subdir, csv_name) in SPLIT_DIRS.items():
        csv_path = meld_dir / csv_name
        vid_dir  = meld_dir / vid_subdir
        if not csv_path.exists():
            print(f"  WARNING: CSV not found: {csv_path}")
            continue
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            d, u = int(row["Dialogue_ID"]), int(row["Utterance_ID"])
            mp4 = vid_dir / f"dia{d}_utt{u}.mp4"
            if not mp4.exists():
                skipped_missing += 1
                continue
            dur = get_duration(str(mp4))
            if dur < min_duration:
                skipped_short += 1
                continue
            emo_raw = str(row["Emotion"]).lower()
            rows.append({
                "split":        split,
                "dialogue_id":  d,
                "utterance_id": u,
                "speaker":      str(row["Speaker"]),
                "emotion":      EMOTION_MAP.get(emo_raw, emo_raw),
                "emotion_raw":  emo_raw,
                "sentiment":    str(row["Sentiment"]).lower(),
                "utterance":    str(row["Utterance"]),
                "duration":     round(dur, 2),
                "video_path":   str(mp4),
                "clip_id":      f"{split}_dia{d}_utt{u}",
            })
    print(f"  Skipped: {skipped_missing} missing files, {skipped_short} clips < {min_duration}s")
    return pd.DataFrame(rows)


def stratified_half_split(df: pd.DataFrame, seed: int):
    """Split 50/50 by speaker, return (real_df, fake_src_df)."""
    rng = np.random.default_rng(seed)
    real_idx, fake_idx = [], []
    for _, group in df.groupby("speaker"):
        idx = np.array(group.index.tolist())
        rng.shuffle(idx)
        mid = len(idx) // 2
        real_idx.extend(idx[:mid].tolist())
        fake_idx.extend(idx[mid:].tolist())
    return df.loc[real_idx].reset_index(drop=True), \
           df.loc[fake_idx].reset_index(drop=True)


def build_pairs(fake_src: pd.DataFrame, seed: int) -> pd.DataFrame:
    """
    For each clip in fake_src, find a donor audio clip from a different speaker.
    Prefer matching emotion for a harder-to-detect fake; fall back to any speaker.
    """
    rng = np.random.default_rng(seed + 1)
    records = []
    fake_src = fake_src.reset_index(drop=True)

    for _, row in fake_src.iterrows():
        # candidates: different speaker, same emotion first
        same_emo = fake_src[
            (fake_src["speaker"] != row["speaker"]) &
            (fake_src["emotion"] == row["emotion"])
        ]
        if len(same_emo) == 0:
            same_emo = fake_src[fake_src["speaker"] != row["speaker"]]
        if len(same_emo) == 0:
            continue
        donor = same_emo.sample(1, random_state=int(rng.integers(0, 2**31))).iloc[0]

        records.append({
            "video_clip":        row["video_path"],
            "audio_clip":        donor["video_path"],
            "video_speaker":     row["speaker"],
            "audio_speaker":     donor["speaker"],
            "video_emotion":     row["emotion"],
            "audio_emotion":     donor["emotion"],
            "video_utterance":   row["utterance"],
            "audio_utterance":   donor["utterance"],
            "video_duration":    row["duration"],
            "audio_duration":    donor["duration"],
            "video_clip_id":     row["clip_id"],
            "audio_clip_id":     donor["clip_id"],
            "split":             row["split"],
            "output_stem":       f"FAKE_T4_{row['clip_id']}__AUDIO_{donor['clip_id']}",
            "label":             1,
        })

    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser(description="MELD 50/50 real/fake split + Track 4 pairs")
    parser.add_argument("--meld_dir", required=True, help="MELD.Raw root directory")
    parser.add_argument("--out_dir",  required=True, help="Output directory for manifests")
    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument("--min_duration", type=float, default=2.5,
                        help="Minimum clip duration in seconds (default 2.5). "
                             "Filters both video and audio sides of each pair.")
    args = parser.parse_args()

    meld_dir = Path(args.meld_dir)
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading MELD clips (min_duration={args.min_duration}s)...")
    df = load_meld(meld_dir, min_duration=args.min_duration)
    print(f"  Found {len(df)} clips across train/dev/test")
    print(f"  Speakers: {sorted(df['speaker'].unique())}")
    print(f"  Emotions: {df['emotion'].value_counts().to_dict()}")

    print("\nSplitting 50/50 by speaker...")
    real_df, fake_src = stratified_half_split(df, args.seed)
    print(f"  Real half:      {len(real_df)} clips")
    print(f"  Fake-src half:  {len(fake_src)} clips")

    print("\nBuilding cross-speaker swap pairs...")
    pairs = build_pairs(fake_src, args.seed)
    print(f"  Pairs built:    {len(pairs)}")

    real_path     = out_dir / "meld_real.csv"
    pairs_path    = out_dir / "meld_pairs.csv"
    fake_src_path = out_dir / "meld_fake_src.csv"

    real_df.to_csv(real_path, index=False)
    pairs.to_csv(pairs_path,  index=False)

    # Save fake-source pool for sample_meld_mismatch.py (Track 4 MuseTalk)
    fake_src.to_csv(fake_src_path, index=False)

    print(f"\n{'='*55}")
    print("MELD SPLIT SUMMARY")
    print(f"{'='*55}")
    print(f"Total clips:       {len(df)}")
    print(f"Real class:        {len(real_df)}  -> {real_path.name}")
    print(f"Fake-source pool:  {len(fake_src)}  -> {fake_src_path.name}")
    print(f"Track 4 pairs:     {len(pairs)}  -> {pairs_path.name}")
    print(f"{'='*55}")
    print(f"\nNext step:")
    print(f"  python src/track4/track4_generate.py \\")
    print(f"      --pairs_csv {pairs_path} \\")
    print(f"      --out_dir   data/synthetic/track4_fakes \\")
    print(f"      --wav2lip_dir tools/Wav2Lip")


if __name__ == "__main__":
    main()
