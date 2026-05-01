"""
parse_cremad.py
===============
Step 1 of the Track 1 pipeline.

Scans your CREMA-D directory, builds a full clip manifest,
and creates the emotion-swap pairing table that the generator
uses to produce fake audio-video pairs.

Usage:
    python parse_cremad.py --cremad_dir /path/to/CREMA-D

Outputs (written to ./track1_manifests/):
    clips.csv          — all genuine clips with metadata
    swap_pairs.csv     — (video_clip, fake_audio_clip) pairing plan
    actor_stats.csv    — per-actor clip counts for sanity checking

CREMA-D filename format:
    {ActorID}_{SentenceKey}_{Emotion}_{Intensity}.{ext}
    Example: 1001_DFA_ANG_XX.wav
"""

import os
import re
import argparse
import pandas as pd
from pathlib import Path
from itertools import permutations
from tqdm import tqdm

# ── CREMA-D constants ──────────────────────────────────────────────────────────

EMOTION_MAP = {
    'ANG': 'angry',
    'DIS': 'disgusted',
    'FEA': 'fearful',
    'HAP': 'happy',
    'NEU': 'neutral',
    'SAD': 'sad',
}

SENTENCE_TEXT = {
    'IEO': "It's eleven o'clock",
    'TIE': "That is exactly what happened",
    'IOM': "I'm on my way to the meeting",
    'IWW': "I wonder what this is about",
    'TAI': "The airplane is almost full",
    'MTI': "Maybe tomorrow it will be cold",
    'IWL': "I would like a new alarm clock",
    'ITH': "I think I have a doctor's appointment",
    'DFA': "Don't forget a jacket",
    'ITS': "I think I've seen this before",
    'TSI': "The surface is slippery",
    'WSI': "We'll stop in a couple of minutes",
}

INTENSITY_MAP = {
    'XX': 'unspecified',
    'LO': 'low',
    'MD': 'medium',
    'HI': 'high',
}

CLIP_PATTERN = re.compile(
    r'^(\d{4})_([A-Z]{2,3})_([A-Z]{2,3})_([A-Z]{2})(\.\w+)$'
)

# ── Parser ─────────────────────────────────────────────────────────────────────

def parse_filename(fname: str) -> dict | None:
    """
    Parse a CREMA-D filename into its metadata components.
    Returns None if the filename doesn't match the expected format.
    """
    m = CLIP_PATTERN.match(fname)
    if not m:
        return None
    actor_id, sentence_key, emotion_code, intensity_code, ext = m.groups()
    if emotion_code not in EMOTION_MAP:
        return None
    if sentence_key not in SENTENCE_TEXT:
        return None
    return {
        'actor_id':      int(actor_id),
        'sentence_key':  sentence_key,
        'sentence_text': SENTENCE_TEXT[sentence_key],
        'emotion_code':  emotion_code,
        'emotion_label': EMOTION_MAP[emotion_code],
        'intensity_code':intensity_code,
        'intensity_label':INTENSITY_MAP.get(intensity_code, intensity_code),
        'ext':           ext.lower(),
        'stem':          fname[:fname.rfind('.')],
    }


def scan_cremad(cremad_dir: Path) -> pd.DataFrame:
    """
    Walk the CREMA-D directory and collect all recognised clip files.
    Looks in AudioWAV/, VideoFlash/, and VideoMP4/ subfolders.
    Returns a DataFrame with one row per (actor, sentence, emotion, intensity).
    """
    # Gather all file paths by stem so we can merge audio + video paths
    stem_to_paths: dict[str, dict] = {}

    search_dirs = {
        'audio': ['AudioWAV', 'Audio', 'audio'],
        'video': ['VideoFlash', 'VideoMP4', 'Video', 'video'],
    }

    print(f"Scanning {cremad_dir} ...")

    for modality, folder_names in search_dirs.items():
        for folder_name in folder_names:
            folder = cremad_dir / folder_name
            if not folder.exists():
                continue
            files = list(folder.iterdir())
            print(f"  Found {len(files)} files in {folder_name}/")
            for f in tqdm(files, desc=f"  Parsing {folder_name}"):
                meta = parse_filename(f.name)
                if meta is None:
                    continue
                stem = meta['stem']
                if stem not in stem_to_paths:
                    stem_to_paths[stem] = meta.copy()
                    stem_to_paths[stem]['audio_path'] = None
                    stem_to_paths[stem]['video_path'] = None
                if modality == 'audio':
                    stem_to_paths[stem]['audio_path'] = str(f)
                else:
                    stem_to_paths[stem]['video_path'] = str(f)

    if not stem_to_paths:
        raise FileNotFoundError(
            f"No CREMA-D clips found in {cremad_dir}.\n"
            f"Expected subfolders: AudioWAV/, VideoFlash/ or VideoMP4/"
        )

    df = pd.DataFrame(list(stem_to_paths.values()))
    df = df.sort_values(['actor_id', 'sentence_key', 'emotion_code', 'intensity_code'])
    df = df.reset_index(drop=True)
    return df


def build_swap_pairs(clips: pd.DataFrame) -> pd.DataFrame:
    """
    Build the emotion-swap pairing table.

    For each genuine clip (video_clip), find another clip from the
    SAME ACTOR and SAME SENTENCE but a DIFFERENT EMOTION, and pair its
    audio as the fake audio track.

    This creates pure emotional mismatch:
        face shows emotion A  ←  video from clip X
        voice says emotion B  ←  audio from clip Y  (same actor, sentence, ≠ emotion)

    Strategy: deterministic emotion rotation so each source emotion maps to
    a unique target emotion, keeping the set balanced across all six emotion classes.
    """
    emotions = list(EMOTION_MAP.keys())
    # Deterministic emotion rotation: ANG→DIS, DIS→FEA, FEA→HAP, HAP→NEU, NEU→SAD, SAD→ANG
    rotation = {e: emotions[(i + 1) % len(emotions)] for i, e in enumerate(emotions)}

    rows = []
    skipped = 0

    groups = clips.groupby(['actor_id', 'sentence_key'])
    for (actor_id, sentence_key), group in tqdm(groups, desc="Building swap pairs"):
        emotion_index = {row['emotion_code']: row for _, row in group.iterrows()}

        for src_emotion, src_row in emotion_index.items():
            # Skip if this clip has no video file — we need a video to swap into
            if pd.isna(src_row.get('video_path')) or src_row['video_path'] is None:
                skipped += 1
                continue

            # Find a target clip (same actor, same sentence, different emotion)
            tgt_emotion = rotation[src_emotion]
            if tgt_emotion not in emotion_index:
                # Fall back to any other available emotion
                available = [e for e in emotion_index if e != src_emotion]
                if not available:
                    skipped += 1
                    continue
                tgt_emotion = available[0]

            tgt_row = emotion_index[tgt_emotion]
            if pd.isna(tgt_row.get('audio_path')) or tgt_row['audio_path'] is None:
                skipped += 1
                continue

            rows.append({
                # Video source (face shows src_emotion)
                'video_stem':        src_row['stem'],
                'video_path':        src_row['video_path'],
                'video_emotion':     src_row['emotion_code'],
                'video_emotion_lbl': src_row['emotion_label'],
                'actor_id':          actor_id,
                'sentence_key':      sentence_key,
                'sentence_text':     src_row['sentence_text'],

                # Audio source (voice says tgt_emotion — the MISMATCH)
                'audio_stem':        tgt_row['stem'],
                'audio_path':        tgt_row['audio_path'],
                'audio_emotion':     tgt_row['emotion_code'],
                'audio_emotion_lbl': tgt_row['emotion_label'],

                # Output
                'output_stem':       f"FAKE_T1_{src_row['stem']}__AUDIO_{tgt_row['stem']}",
                'label':             1,  # 1 = fake
                'status':            'pending',
            })

    print(f"\nSwap pairs created: {len(rows)}")
    if skipped:
        print(f"Skipped (missing audio or video): {skipped}")
    return pd.DataFrame(rows)


def print_summary(clips: pd.DataFrame, pairs: pd.DataFrame):
    print("\n" + "=" * 55)
    print("CREMA-D MANIFEST SUMMARY")
    print("=" * 55)
    print(f"Total clips found:       {len(clips)}")
    print(f"  With audio path:       {clips['audio_path'].notna().sum()}")
    print(f"  With video path:       {clips['video_path'].notna().sum()}")
    print(f"  With both:             {(clips['audio_path'].notna() & clips['video_path'].notna()).sum()}")
    print(f"\nActors:                  {clips['actor_id'].nunique()}")
    print(f"Sentences:               {clips['sentence_key'].nunique()}")
    print(f"Emotions:                {clips['emotion_code'].nunique()}")
    print(f"\nEmotion-swap pairs:      {len(pairs)}")
    print(f"\nEmotion distribution:")
    for code, label in EMOTION_MAP.items():
        n = (clips['emotion_code'] == code).sum()
        print(f"  {code} ({label:12s}):  {n}")
    print("=" * 55)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Parse CREMA-D into Track 1 manifests")
    parser.add_argument('--cremad_dir', type=str, required=True,
                        help='Root directory of your CREMA-D download')
    parser.add_argument('--out_dir', type=str, default='./track1_manifests',
                        help='Where to save the CSV manifests (default: ./track1_manifests)')
    args = parser.parse_args()

    cremad_dir = Path(args.cremad_dir)
    out_dir    = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not cremad_dir.exists():
        raise FileNotFoundError(f"CREMA-D directory not found: {cremad_dir}")

    # 1. Scan all clips
    clips = scan_cremad(cremad_dir)

    # 2. Actor-level stats
    actor_stats = clips.groupby('actor_id').agg(
        total_clips=('stem', 'count'),
        emotions=('emotion_code', 'nunique'),
        sentences=('sentence_key', 'nunique'),
        has_audio=('audio_path', lambda x: x.notna().sum()),
        has_video=('video_path', lambda x: x.notna().sum()),
    ).reset_index()

    # 3. Build swap pairs
    pairs = build_swap_pairs(clips)

    # 4. Save all manifests
    clips_path  = out_dir / 'clips.csv'
    pairs_path  = out_dir / 'swap_pairs.csv'
    actors_path = out_dir / 'actor_stats.csv'

    clips.to_csv(clips_path, index=False)
    pairs.to_csv(pairs_path, index=False)
    actor_stats.to_csv(actors_path, index=False)

    print_summary(clips, pairs)
    print(f"\nManifests saved to: {out_dir}/")
    print(f"  {clips_path.name}:    {len(clips)} rows")
    print(f"  {pairs_path.name}: {len(pairs)} rows")
    print(f"  {actors_path.name}: {len(actor_stats)} rows")
    print(f"\nNext step: python scripts/sample_by_track.py --pairs_csv {pairs_path} --out_dir {out_dir}")


if __name__ == '__main__':
    main()
