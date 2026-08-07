"""
validate_training_prep.py
=========================
1. Validates preprocessing completeness across source manifests (Tracks 1-3, MELD real,
   CMU-MOSEI real, MUStARD), excluding Track 4.
2. Verifies that feature files exist AND are non-empty (>0 bytes).
3. Handles stem suffix variations (_styletts, _rvc, _sadtalker).
4. Strictly EXCLUDES test benchmark clips (fakeavceleb), webapp upload test clips
   (webapp_upload), and track4.
5. Outputs the clean, exclusive training dataset turnover manifest:
     data/processed/training_turnover_manifest.csv

Usage:
    python scripts/validate_training_prep.py
"""
import csv
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
PREPROCESSED_DIR = REPO_ROOT / "data/preprocessed"
Z_AT_DIR = PREPROCESSED_DIR / "features/z_at"
Z_V_DIR = PREPROCESSED_DIR / "features/z_v"

OUT_MANIFEST_CSV = REPO_ROOT / "data/processed/training_turnover_manifest.csv"
PREPROCESSED_TURNOVER_CSV = PREPROCESSED_DIR / "training_turnover_manifest.csv"

SOURCE_MANIFESTS = {
    "track1": REPO_ROOT / "data/processed/track1_manifests/track1_pairs.csv",
    "track2": REPO_ROOT / "data/processed/track1_manifests/track2_pairs.csv",
    "track3": REPO_ROOT / "data/processed/track1_manifests/track3_pairs.csv",
    "meld_real": REPO_ROOT / "data/processed/meld_manifests/meld_real.csv",
    "mosei_real": REPO_ROOT / "data/processed/mosei_manifests/mosei_real.csv",
    "mustard_json": REPO_ROOT / "data/raw/MUStARD/repo/data/sarcasm_data.json",
    "mustard_fallback": REPO_ROOT / "data/processed/smoke_manifests/smoke_sarcasm.csv",
}

CREMA_EMO = {
    "ANG": "angry", "DIS": "disgust", "FEA": "fear",
    "HAP": "happy", "NEU": "neutral", "SAD": "sad",
}


def find_valid_cached_stem(base_stem: str, ready_stems: set[str]) -> str | None:
    """Find the exact cached file stem matching base_stem (handling suffixes)."""
    if base_stem in ready_stems:
        return base_stem
    for suffix in ["_styletts", "_rvc", "_sadtalker", "_wav2lip", "_musetalk"]:
        candidate = f"{base_stem}{suffix}"
        if candidate in ready_stems:
            return candidate
    return None


def run_validation_and_turnover():
    log.info("PREPROCESSING VALIDATION & TRAINING TURNOVER PREPARATION")

    # Only accept files that exist AND have size > 0 bytes
    z_at_stems = set(p.stem for p in Z_AT_DIR.glob("*.pt") if p.stat().st_size > 0) if Z_AT_DIR.exists() else set()
    z_v_stems = set(p.stem for p in Z_V_DIR.glob("*.pt") if p.stat().st_size > 0) if Z_V_DIR.exists() else set()
    ready_stems = z_at_stems.intersection(z_v_stems)

    log.info(f"Feature Cache Index: {len(z_at_stems):,} z_at, {len(z_v_stems):,} z_v -> {len(ready_stems):,} valid non-empty preprocessed pairs.")
    training_records = []

    def _check_clip(cid, pipe, fake_lbl, spk, a_emo="", v_emo="", sarc=-1, v_path=""):
        matched_stem = find_valid_cached_stem(cid, ready_stems)
        if matched_stem:
            z_at = Z_AT_DIR / f"{matched_stem}.pt"
            z_v = Z_V_DIR / f"{matched_stem}.pt"
            return {
                "clip_id": matched_stem, "source_pipeline": pipe, "fake_label": fake_lbl,
                "speaker_id": spk, "audio_emotion": a_emo, "visual_emotion": v_emo,
                "sarcasm_label": sarc, "has_z_at": 1, "has_z_v": 1, "is_ready_for_training": 1,
                "video_path": v_path, "z_at_path": str(z_at.relative_to(REPO_ROOT)),
                "z_v_path": str(z_v.relative_to(REPO_ROOT)),
            }
        return {"clip_id": cid, "source_pipeline": pipe, "fake_label": fake_lbl, "speaker_id": spk,
                "audio_emotion": a_emo, "visual_emotion": v_emo, "sarcasm_label": sarc,
                "has_z_at": 0, "has_z_v": 0, "is_ready_for_training": 0, "video_path": v_path,
                "z_at_path": "", "z_v_path": ""}

    for trk_name in ["track1", "track2", "track3"]:
        csv_path = SOURCE_MANIFESTS[trk_name]
        if csv_path.exists():
            with open(csv_path, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    stem = str(row["output_stem"])
                    actor_id = str(row.get("actor_id", "") or (stem.split("_")[2] if len(stem.split("_")) > 2 else stem))
                    rec = _check_clip(stem, trk_name, int(row.get("label", 1)), f"crema_{actor_id}",
                                     row.get("audio_emotion", ""), row.get("video_emotion", ""), -1, row.get("output_path", ""))
                    if rec["is_ready_for_training"]:
                        training_records.append(rec)

    meld_csv = SOURCE_MANIFESTS["meld_real"]
    if meld_csv.exists():
        with open(meld_csv, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                cid = str(row["clip_id"])
                spk = f"meld_{str(row.get('speaker', 'UNK')).replace(' ', '_')}"
                emo = str(row.get("emotion", ""))
                target_cid = cid
                if cid not in z_at_stems and "_" in cid:
                    parts = cid.split("_", 1)
                    if parts[0] in ("train", "dev", "test"):
                        target_cid = parts[1]
                rec = _check_clip(target_cid, "meld_real", 0, spk, emo, emo, -1, row.get("video_path", ""))
                if rec["is_ready_for_training"]:
                    training_records.append(rec)

    mosei_csv = SOURCE_MANIFESTS["mosei_real"]
    if mosei_csv.exists():
        with open(mosei_csv, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                cid = str(row["clip_id"])
                rec = _check_clip(cid, "mosei_real", 0, f"mosei_{cid}", "", "", -1, row.get("video_path", ""))
                if rec["is_ready_for_training"]:
                    training_records.append(rec)

    mustard_json = SOURCE_MANIFESTS["mustard_json"]
    if mustard_json.exists():
        try:
            data = json.loads(mustard_json.read_text(encoding="utf-8"))
            mustard_vids = REPO_ROOT / "data/raw/MUStARD/raw_data/utterances_final"
            for key, entry in data.items():
                sarc = 1 if entry.get("sarcasm", False) else 0
                spk = f"mustard_{str(entry.get('speaker', 'UNK')).replace(' ', '_')}"
                rec = _check_clip(key, "mustard", -1, spk, "", "", sarc, str(mustard_vids / f"{key}.mp4"))
                if rec["is_ready_for_training"]:
                    training_records.append(rec)
        except Exception as e:
            log.warning(f"Error reading MUStARD JSON: {e}")

    # Deduplicate & Filter strict exclusions
    unique_training = {}
    for rec in training_records:
        cid = rec["clip_id"]
        if cid.startswith("fav_") or cid.startswith("id") or cid.startswith("upload_") or cid.startswith("FAKE_T4_"):
            continue
        if cid not in unique_training:
            unique_training[cid] = rec

    final_training_records = list(unique_training.values())

    fieldnames = ["clip_id", "source_pipeline", "fake_label", "speaker_id", "audio_emotion", "visual_emotion",
                  "sarcasm_label", "has_z_at", "has_z_v", "is_ready_for_training", "video_path", "z_at_path", "z_v_path"]

    OUT_MANIFEST_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_MANIFEST_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(final_training_records)

    with open(PREPROCESSED_TURNOVER_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(final_training_records)

    log.info(f"Successfully exported exclusive training turnover manifest with {len(final_training_records):,} clips.")

if __name__ == "__main__":
    run_validation_and_turnover()
