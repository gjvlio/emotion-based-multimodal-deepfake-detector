"""
build_preprocessed_manifest.py
==============================
Scans `data/preprocessed/` and matches all cached clips against the project's
source manifests (Track 1-3 CREMA-D fakes, Track 4 MELD fakes, MELD real,
CMU-MOSEI real, MUStARD sarcasm, FakeAVCeleb test clips, and WebApp uploads).

Generates a unified master CSV tracking file:
    data/preprocessed/preprocessed_manifest.csv

Usage:
    python scripts/build_preprocessed_manifest.py
"""
import csv
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
PREPROCESSED_DIR = REPO_ROOT / "data/preprocessed"
OUT_CSV = PREPROCESSED_DIR / "preprocessed_manifest.csv"

# Source manifests
TRACK1_CSV = REPO_ROOT / "data/processed/track1_manifests/track1_pairs.csv"
TRACK2_CSV = REPO_ROOT / "data/processed/track1_manifests/track2_pairs.csv"
TRACK3_CSV = REPO_ROOT / "data/processed/track1_manifests/track3_pairs.csv"
TRACK4_CSV = REPO_ROOT / "data/processed/meld_manifests/meld_mismatch_pairs.csv"
MELD_REAL_CSV = REPO_ROOT / "data/processed/meld_manifests/meld_real.csv"
MOSEI_REAL_CSV = REPO_ROOT / "data/processed/mosei_manifests/mosei_real.csv"
MUSTARD_JSON = REPO_ROOT / "data/raw/MUStARD/repo/data/sarcasm_data.json"
MUSTARD_VIDS = REPO_ROOT / "data/raw/MUStARD/raw_data/utterances_final"
MUSTARD_FALLBACK = REPO_ROOT / "data/processed/smoke_manifests/smoke_sarcasm.csv"

# CREMA-D emotion mapping
CREMA_EMO = {
    "ANG": "angry",
    "DIS": "disgust",
    "FEA": "fear",
    "HAP": "happy",
    "NEU": "neutral",
    "SAD": "sad",
}


def parse_crema_stem(stem: str) -> tuple[str, str, str]:
    """Extract (actor_id, visual_emotion, audio_emotion) from CREMA-D fake filename stem."""
    actor_id = stem.split("_")[2] if len(stem.split("_")) > 2 else "unknown"
    v_emo, a_emo = "", ""

    if "__AUDIO_" in stem:
        v_part, a_part = stem.split("__AUDIO_", 1)
        v_tokens = v_part.split("_")
        a_tokens = a_part.split("_")
        v_code = v_tokens[-2] if len(v_tokens) >= 2 else ""
        a_code = a_tokens[2] if len(a_tokens) > 2 else ""
        v_emo = CREMA_EMO.get(v_code, "")
        a_emo = CREMA_EMO.get(a_code, "")

    return actor_id, v_emo, a_emo


def load_source_metadata() -> dict[str, dict]:
    """Build a lookup map from clip_id (stem) -> metadata dictionary."""
    meta_lookup = {}

    def _add(cid, pipeline, fake_label, spk, a_emo="", v_emo="", sarc=-1, v_path=""):
        if cid not in meta_lookup:
            meta_lookup[cid] = {
                "source_pipeline": pipeline,
                "fake_label": fake_label,
                "speaker_id": spk,
                "audio_emotion": a_emo,
                "visual_emotion": v_emo,
                "sarcasm_label": sarc,
                "video_path": v_path,
            }

    # 1. Tracks 1-3 CSVs
    for track_name, csv_path in [
        ("track1", TRACK1_CSV),
        ("track2", TRACK2_CSV),
        ("track3", TRACK3_CSV),
    ]:
        if csv_path.exists():
            with open(csv_path, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    stem = str(row["output_stem"])
                    actor_id = str(row.get("actor_id", "") or (stem.split("_")[2] if len(stem.split("_")) > 2 else stem))
                    _add(
                        cid=stem,
                        pipeline=track_name,
                        fake_label=int(row.get("label", 1)),
                        spk=f"crema_{actor_id}",
                        a_emo=str(row.get("audio_emotion", "")),
                        v_emo=str(row.get("video_emotion", "")),
                        sarc=-1,
                        v_path=str(row.get("output_path", "")),
                    )

    # 2. Track 4 CSV
    if TRACK4_CSV.exists():
        with open(TRACK4_CSV, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                stem = str(row["output_stem"])
                spk = f"meld_fake_{row.get('video_speaker', 'UNK')}"
                _add(
                    cid=stem,
                    pipeline="track4",
                    fake_label=int(row.get("label", 1)),
                    spk=spk,
                    a_emo=str(row.get("audio_emotion", "")),
                    v_emo=str(row.get("video_emotion", "")),
                    sarc=-1,
                    v_path=str(row.get("output_path", "")),
                )

    # 3. MELD Real (handles both 'train_dia0_utt3' and stripped 'dia0_utt3')
    if MELD_REAL_CSV.exists():
        with open(MELD_REAL_CSV, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                cid = str(row["clip_id"])
                spk = f"meld_{str(row.get('speaker', 'UNK')).replace(' ', '_')}"
                emo = str(row.get("emotion", ""))
                _add(cid, "meld_real", 0, spk, emo, emo, -1, str(row.get("video_path", "")))
                
                # Strip split prefix (e.g. 'train_dia0_utt3' -> 'dia0_utt3')
                if "_" in cid:
                    parts = cid.split("_", 1)
                    if parts[0] in ("train", "dev", "test"):
                        _add(parts[1], "meld_real", 0, spk, emo, emo, -1, str(row.get("video_path", "")))

    # 4. CMU-MOSEI Real
    if MOSEI_REAL_CSV.exists():
        with open(MOSEI_REAL_CSV, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                cid = str(row["clip_id"])
                _add(
                    cid=cid,
                    pipeline="mosei_real",
                    fake_label=0,
                    spk=f"mosei_{cid}",
                    a_emo="",
                    v_emo="",
                    sarc=-1,
                    v_path=str(row.get("video_path", "")),
                )

    # 5. MUStARD
    if MUSTARD_JSON.exists():
        try:
            data = json.loads(MUSTARD_JSON.read_text(encoding="utf-8"))
            for key, entry in data.items():
                sarc = 1 if entry.get("sarcasm", False) else 0
                spk = f"mustard_{str(entry.get('speaker', 'UNK')).replace(' ', '_')}"
                v_p = str(MUSTARD_VIDS / f"{key}.mp4") if MUSTARD_VIDS.exists() else ""
                _add(
                    cid=key,
                    pipeline="mustard",
                    fake_label=-1,
                    spk=spk,
                    a_emo="",
                    v_emo="",
                    sarc=sarc,
                    v_path=v_p,
                )
        except Exception as e:
            log.warning(f"Could not parse MUStARD JSON: {e}")

    if MUSTARD_FALLBACK.exists():
        with open(MUSTARD_FALLBACK, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                v_p = str(row.get("path", ""))
                cid = Path(v_p).stem if v_p else ""
                if cid:
                    _add(
                        cid=cid,
                        pipeline="mustard",
                        fake_label=-1,
                        spk=str(row.get("speaker_id", f"mustard_{cid}")),
                        a_emo="",
                        v_emo="",
                        sarc=int(row.get("sarcasm_label", -1)),
                        v_path=v_p,
                    )

    return meta_lookup


def build_manifest():
    log.info("Collecting cached clip files in data/preprocessed...")

    z_at_dir = PREPROCESSED_DIR / "features/z_at"
    z_v_dir = PREPROCESSED_DIR / "features/z_v"
    audio_dir = PREPROCESSED_DIR / "audio"
    keyframes_dir = PREPROCESSED_DIR / "keyframes"
    transcripts_dir = PREPROCESSED_DIR / "transcripts"

    # Gather all unique clip IDs from z_at, z_v, audio, keyframes
    clip_ids = set()
    if z_at_dir.exists():
        clip_ids.update(p.stem for p in z_at_dir.glob("*.pt"))
    if z_v_dir.exists():
        clip_ids.update(p.stem for p in z_v_dir.glob("*.pt"))
    if audio_dir.exists():
        clip_ids.update(p.stem for p in audio_dir.glob("*.wav"))
    if keyframes_dir.exists():
        for p in keyframes_dir.glob("*_kf.jpg"):
            stem = p.name[:-7] if p.name.endswith("_kf.jpg") else p.stem
            clip_ids.add(stem)

    log.info(f"Found {len(clip_ids)} total unique clip IDs in cache.")

    # Load source metadata lookup
    meta_lookup = load_source_metadata()
    log.info(f"Loaded explicit metadata lookup for {len(meta_lookup)} manifest items.")

    rows = []
    pipeline_counts = {}

    for cid in sorted(clip_ids):
        z_at_file = z_at_dir / f"{cid}.pt"
        z_v_file = z_v_dir / f"{cid}.pt"
        audio_file = audio_dir / f"{cid}.wav"
        kf_file = keyframes_dir / f"{cid}_kf.jpg"
        txt_json = transcripts_dir / f"{cid}.json"
        txt_txt = transcripts_dir / f"{cid}.txt"

        if cid in meta_lookup:
            meta = meta_lookup[cid]
        else:
            # Pattern matching fallback for generated clips and external datasets
            if cid.startswith("FAKE_T1_"):
                actor_id, v_emo, a_emo = parse_crema_stem(cid)
                meta = {
                    "source_pipeline": "track1",
                    "fake_label": 1,
                    "speaker_id": f"crema_{actor_id}",
                    "audio_emotion": a_emo,
                    "visual_emotion": v_emo,
                    "sarcasm_label": -1,
                    "video_path": "",
                }
            elif cid.startswith("FAKE_T2_"):
                actor_id, v_emo, a_emo = parse_crema_stem(cid)
                meta = {
                    "source_pipeline": "track2",
                    "fake_label": 1,
                    "speaker_id": f"crema_{actor_id}",
                    "audio_emotion": a_emo,
                    "visual_emotion": v_emo,
                    "sarcasm_label": -1,
                    "video_path": "",
                }
            elif cid.startswith("FAKE_T3_"):
                actor_id, v_emo, a_emo = parse_crema_stem(cid)
                meta = {
                    "source_pipeline": "track3",
                    "fake_label": 1,
                    "speaker_id": f"crema_{actor_id}",
                    "audio_emotion": a_emo,
                    "visual_emotion": v_emo,
                    "sarcasm_label": -1,
                    "video_path": "",
                }
            elif cid.startswith("FAKE_T4_"):
                meta = {
                    "source_pipeline": "track4",
                    "fake_label": 1,
                    "speaker_id": f"meld_fake_{cid}",
                    "audio_emotion": "",
                    "visual_emotion": "",
                    "sarcasm_label": -1,
                    "video_path": "",
                }
            elif cid.startswith("fav_") or cid.startswith("id"):
                meta = {
                    "source_pipeline": "fakeavceleb",
                    "fake_label": 0 if ("real" in cid.lower() or cid.count("_") == 1) else 1,
                    "speaker_id": f"fakeavceleb_{cid.split('_')[1] if '_' in cid else cid}",
                    "audio_emotion": "",
                    "visual_emotion": "",
                    "sarcasm_label": -1,
                    "video_path": "",
                }
            elif cid.startswith("upload_"):
                meta = {
                    "source_pipeline": "webapp_upload",
                    "fake_label": -1,
                    "speaker_id": "user_upload",
                    "audio_emotion": "",
                    "visual_emotion": "",
                    "sarcasm_label": -1,
                    "video_path": "",
                }
            else:
                meta = {
                    "source_pipeline": "unknown",
                    "fake_label": -1,
                    "speaker_id": "unknown",
                    "audio_emotion": "",
                    "visual_emotion": "",
                    "sarcasm_label": -1,
                    "video_path": "",
                }

        pipe = meta["source_pipeline"]
        pipeline_counts[pipe] = pipeline_counts.get(pipe, 0) + 1

        rows.append(
            {
                "clip_id": cid,
                "source_pipeline": pipe,
                "fake_label": meta["fake_label"],
                "speaker_id": meta["speaker_id"],
                "audio_emotion": meta["audio_emotion"],
                "visual_emotion": meta["visual_emotion"],
                "sarcasm_label": meta["sarcasm_label"],
                "has_z_at": int(z_at_file.exists()),
                "has_z_v": int(z_v_file.exists()),
                "has_audio": int(audio_file.exists()),
                "has_keyframe": int(kf_file.exists()),
                "has_transcript": int(txt_json.exists() or txt_txt.exists()),
                "video_path": meta["video_path"],
                "z_at_path": str(z_at_file.relative_to(REPO_ROOT)) if z_at_file.exists() else "",
                "z_v_path": str(z_v_file.relative_to(REPO_ROOT)) if z_v_file.exists() else "",
            }
        )

    # Write CSV
    fieldnames = [
        "clip_id",
        "source_pipeline",
        "fake_label",
        "speaker_id",
        "audio_emotion",
        "visual_emotion",
        "sarcasm_label",
        "has_z_at",
        "has_z_v",
        "has_audio",
        "has_keyframe",
        "has_transcript",
        "video_path",
        "z_at_path",
        "z_v_path",
    ]

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    log.info(f"Successfully generated preprocessed master manifest at: {OUT_CSV}")
    log.info("Breakdown by source pipeline:")
    for pipe, count in sorted(pipeline_counts.items()):
        log.info(f"  - {pipe:<15}: {count:,} clips")


if __name__ == "__main__":
    build_manifest()
