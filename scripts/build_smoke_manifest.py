"""
build_smoke_manifest.py
=======================
Builds a small smoke-test dataset manifest for the exp/sarcasm-training branch.

Outputs:
  data/processed/smoke_manifests/smoke_detector.csv   -- fake/real clips for detector
  data/processed/smoke_manifests/smoke_sarcasm.csv    -- MUStARD clips for SarcasmHead
  data/raw/CMU-MOSEI/segments/                        -- segmented CMU-MOSEI utterances

Smoke-test sizes (configurable below):
  N_PER_TRACK  = 150  fake clips per track (Tracks 1-4)
  N_MELD_REAL  = 300  real MELD clips
  N_MOSEI_REAL = 150  real CMU-MOSEI segments (best-effort)

Emotion label columns (audio_emotion_label, visual_emotion_label):
  MELD real       : both = MELD emotion string (neutral/happy/sad/angry/fear/disgust)
  CMU-MOSEI real  : both = dominant Ekman emotion from CSD features
  Track 3 fakes   : visual = CREMA-D video emotion, audio = CREMA-D audio emotion (mismatched)
  Track 1/2/4     : both = "" (UNKNOWN — no per-clip annotation available)
  MUStARD         : both = "" (UNKNOWN — sarcasm dataset, no emotion labels)

Usage:
  python scripts/build_smoke_manifest.py [--seed 42]
"""

import argparse
import csv
import json
import random
import subprocess
from pathlib import Path

import h5py
import numpy as np

# ── Config ──────────────────────────────────────────────────────────────────
N_PER_TRACK  = 150
N_MELD_REAL  = 300
N_MOSEI_REAL = 150
MIN_DUR      = 2.0   # seconds
MAX_DUR      = 8.0   # seconds

REPO_ROOT   = Path(__file__).resolve().parent.parent
OUT_DIR     = REPO_ROOT / "data/processed/smoke_manifests"
SEG_DIR     = REPO_ROOT / "data/raw/CMU-MOSEI/segments"

TRACK_DIRS  = {
    "track1": REPO_ROOT / "data/synthetic/track1_fakes/videos",
    "track2": REPO_ROOT / "data/synthetic/track2_fakes/videos",
    "track3": REPO_ROOT / "data/synthetic/track3_fakes/videos",
    # track4 excluded from smoke: emotion-mismatch fakes share Z_at/Z_v distribution
    # with real MELD clips → conflicting BCE signal at smoke scale.
    # Add back for full training once emotion heads are well-trained.
    # "track4": REPO_ROOT / "data/synthetic/track4_fakes",
}
MELD_REAL_CSV  = REPO_ROOT / "data/processed/meld_manifests/meld_real.csv"
MOSEI_CSD      = REPO_ROOT / "data/raw/CMU-MOSEI/labels/CMU_MOSEI_Labels.csd"
MOSEI_VIDS_DIR = REPO_ROOT / "data/raw/CMU-MOSEI/videos"
MUSTARD_JSON   = REPO_ROOT / "data/raw/MUStARD/repo/data/sarcasm_data.json"
MUSTARD_VIDS   = REPO_ROOT / "data/raw/MUStARD/raw_data/utterances_final"

# CREMA-D emotion codes → string accepted by EMOTION_TO_IDX
_CREMA_EMO = {
    "ANG": "angry", "DIS": "disgust", "FEA": "fear",
    "HAP": "happy", "NEU": "neutral", "SAD": "sad",
}

# CMU-MOSEI feature indices 1-6: happy, sad, anger, surprise, disgust, fear
_MOSEI_EMO_NAMES = ["happy", "sad", "anger", "surprise", "disgust", "fear"]


def get_duration(path: Path) -> float:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True, timeout=15,
    )
    try:
        return float(r.stdout.strip())
    except ValueError:
        return 0.0


def _parse_track3_emotions(stem: str) -> tuple[str, str]:
    """
    Extract (visual_emotion, audio_emotion) from CREMA-D-based track3 filename.
    Format: FAKE_T3_ACTOR_SENT_VEMO_LEVEL__AUDIO_ACTOR_SENT_AEMO_LEVEL_sadtalker
    """
    if "__AUDIO_" not in stem:
        return "", ""
    v_part, a_part = stem.split("__AUDIO_", 1)
    v_tokens = v_part.split("_")   # ['FAKE', 'T3', '1001', 'DFA', 'HAP', 'XX']
    a_tokens = a_part.split("_")   # ['1001', 'DFA', 'NEU', 'XX', 'sadtalker']
    v_code = v_tokens[-2] if len(v_tokens) >= 2 else ""
    a_code = a_tokens[2]  if len(a_tokens) > 2  else ""
    return _CREMA_EMO.get(v_code, ""), _CREMA_EMO.get(a_code, "")


def sample_track_fakes(track_name: str, vid_dir: Path, n: int, rng: random.Random) -> list[dict]:
    mp4s = sorted(vid_dir.glob("*.mp4"))
    sampled = rng.sample(mp4s, min(n, len(mp4s)))
    rows = []
    for p in sampled:
        if track_name == "track3":
            vis_emo, aud_emo = _parse_track3_emotions(p.stem)
        else:
            vis_emo, aud_emo = "", ""
        parts = p.stem.split("_")
        actor = parts[2] if len(parts) > 2 else p.stem  # FAKE_T{N}_{ACTOR}_...
        rows.append({
            "path":                str(p),
            "label":               1,
            "source":              track_name,
            "speaker_id":          f"crema_{actor}",
            "sarcasm_label":       -1,
            "audio_emotion_label":  aud_emo,
            "visual_emotion_label": vis_emo,
        })
    return rows


def sample_meld_real(csv_path: Path, n: int, rng: random.Random) -> list[dict]:
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        all_rows = list(reader)
    sampled = rng.sample(all_rows, min(n, len(all_rows)))
    for row in sampled:
        path = row.get("path") or row.get("video_path") or row.get("filepath") or ""
        emotion = (row.get("emotion") or row.get("emotion_raw") or "").strip()
        speaker = (row.get("speaker") or "").strip().lower().replace(" ", "_") or "unknown"
        rows.append({
            "path":                path,
            "label":               0,
            "source":              "meld_real",
            "speaker_id":          f"meld_{speaker}",
            "sarcasm_label":       -1,
            "audio_emotion_label":  emotion,
            "visual_emotion_label": emotion,
        })
    return rows


def segment_mosei(csd_path: Path, vids_dir: Path, seg_dir: Path,
                  n_target: int, rng: random.Random) -> list[dict]:
    seg_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    f = h5py.File(str(csd_path), "r")
    data = f["All Labels"]["data"]
    vid_ids = list(data.keys())
    rng.shuffle(vid_ids)

    print(f"CMU-MOSEI: scanning {len(vid_ids)} video IDs for downloaded files...")

    for vid_id in vid_ids:
        if len(rows) >= n_target:
            break

        matches = list(vids_dir.glob(f"{vid_id}.*"))
        if not matches:
            continue
        src_vid = matches[0]

        intervals = data[vid_id]["intervals"][:]
        features  = data[vid_id]["features"][:]   # shape (n_segs, 1, 7)

        for i, (start, end) in enumerate(intervals):
            dur = float(end) - float(start)
            if not (MIN_DUR <= dur <= MAX_DUR):
                continue

            # Extract dominant Ekman emotion (indices 1-6; index 0 = sentiment)
            # features[i] shape varies: (1,7) or (7,) — flatten handles both
            emo_scores = np.array(features[i], dtype=float).flatten()[1:]  # [happy,sad,anger,surprise,disgust,fear]
            if len(emo_scores) >= 6 and emo_scores.max() >= 0.1:
                emotion = _MOSEI_EMO_NAMES[int(np.argmax(emo_scores))]
            else:
                emotion = ""

            out_path = seg_dir / f"{vid_id}_{i:03d}.mp4"
            if not out_path.exists():
                result = subprocess.run(
                    [
                        "ffmpeg", "-y", "-loglevel", "error",
                        "-ss", str(start),
                        "-t",  str(dur),
                        "-i",  str(src_vid),
                        "-c:v", "libx264", "-c:a", "aac",
                        "-r", "25",
                        str(out_path),
                    ],
                    capture_output=True, timeout=60,
                )
                if result.returncode != 0:
                    continue

            rows.append({
                "path":                str(out_path),
                "label":               0,
                "source":              "cmumosei_real",
                "speaker_id":          f"mosei_{vid_id}",
                "sarcasm_label":       -1,
                "audio_emotion_label":  emotion,
                "visual_emotion_label": emotion,
            })
            print(f"  Segmented {out_path.name}  ({dur:.1f}s)  emo={emotion or 'unknown'}")

            if len(rows) >= n_target:
                break

    f.close()
    print(f"CMU-MOSEI: got {len(rows)} segments")
    return rows


def build_mustard_sarcasm(json_path: Path, vids_dir: Path) -> list[dict]:
    data = json.load(open(json_path))
    rows = []
    for clip_id, entry in data.items():
        mp4 = vids_dir / f"{clip_id}.mp4"
        if not mp4.exists():
            continue
        rows.append({
            "path":                str(mp4),
            "label":               -1,
            "source":              "mustard",
            "speaker_id":          f"must_{entry.get('speaker','unk').lower()}",
            "sarcasm_label":       1 if entry["sarcasm"] else 0,
            "audio_emotion_label":  "",
            "visual_emotion_label": "",
            "utterance":           entry["utterance"],
        })
    return rows


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        print(f"WARNING: no rows for {path.name}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows -> {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    detector_rows = []

    # Fakes from Tracks 1-4
    for track_name, vid_dir in TRACK_DIRS.items():
        if not vid_dir.exists():
            print(f"SKIP {track_name} — dir not found: {vid_dir}")
            continue
        rows = sample_track_fakes(track_name, vid_dir, N_PER_TRACK, rng)
        print(f"{track_name}: sampled {len(rows)} fake clips")
        detector_rows.extend(rows)

    # MELD real
    if MELD_REAL_CSV.exists():
        rows = sample_meld_real(MELD_REAL_CSV, N_MELD_REAL, rng)
        print(f"MELD real: sampled {len(rows)} clips")
        detector_rows.extend(rows)
    else:
        print(f"SKIP MELD real — CSV not found: {MELD_REAL_CSV}")

    # CMU-MOSEI segments
    if MOSEI_CSD.exists() and MOSEI_VIDS_DIR.exists():
        rows = segment_mosei(MOSEI_CSD, MOSEI_VIDS_DIR, SEG_DIR, N_MOSEI_REAL, rng)
        detector_rows.extend(rows)
    else:
        print("SKIP CMU-MOSEI — CSD or videos dir not found")

    rng.shuffle(detector_rows)
    write_csv(OUT_DIR / "smoke_detector.csv", detector_rows)

    # Emotion coverage summary
    with_emo = sum(1 for r in detector_rows if r.get("audio_emotion_label"))
    print(f"\n  Emotion labels    : {with_emo}/{len(detector_rows)} clips have audio_emotion_label")

    # MUStARD sarcasm
    if MUSTARD_JSON.exists() and MUSTARD_VIDS.exists():
        sarc_rows = build_mustard_sarcasm(MUSTARD_JSON, MUSTARD_VIDS)
        print(f"MUStARD: {len(sarc_rows)} clips")
        rng.shuffle(sarc_rows)
        write_csv(OUT_DIR / "smoke_sarcasm.csv", sarc_rows)
    else:
        print("SKIP MUStARD — JSON or videos not found")

    # Summary
    real  = sum(1 for r in detector_rows if r["label"] == 0)
    fake  = sum(1 for r in detector_rows if r["label"] == 1)
    print(f"\n=== Smoke manifest summary ===")
    print(f"Total detector clips : {len(detector_rows)}")
    print(f"  Real               : {real}")
    print(f"  Fake               : {fake}")
    print(f"  Real:Fake ratio    : {real/max(fake,1):.2f}")


if __name__ == "__main__":
    main()
