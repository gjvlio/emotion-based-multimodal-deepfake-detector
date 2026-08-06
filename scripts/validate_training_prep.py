"""
validate_training_prep.py
=========================
1. Validates preprocessing completeness across source manifests (Tracks 1-3, MELD real,
   CMU-MOSEI real, MUStARD), excluding Track 4.
2. Handles stem suffix variations (e.g. `_styletts`, `_rvc`, `_sadtalker`) created during preprocessing.
3. Strictly EXCLUDES test benchmark clips (`fakeavceleb`), webapp upload test clips
   (`webapp_upload`), and `track4`.
4. Outputs the clean, exclusive training dataset turnover manifest:
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

# Source manifests to validate for training
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


def find_cached_stem(base_stem: str, z_at_stems: set[str]) -> str | None:
    """Find the exact cached file stem matching base_stem (handling _styletts/_rvc/_sadtalker suffixes)."""
    if base_stem in z_at_stems:
        return base_stem
    for suffix in ["_styletts", "_rvc", "_sadtalker", "_wav2lip", "_musetalk"]:
        candidate = f"{base_stem}{suffix}"
        if candidate in z_at_stems:
            return candidate
    return None


def run_validation_and_turnover():
    log.info("=" * 68)
    log.info("PREPROCESSING VALIDATION & TRAINING TURNOVER PREPARATION")
    log.info("=" * 68)

    # Index existing cache stems
    z_at_stems = set(p.stem for p in Z_AT_DIR.glob("*.pt")) if Z_AT_DIR.exists() else set()
    z_v_stems = set(p.stem for p in Z_V_DIR.glob("*.pt")) if Z_V_DIR.exists() else set()
    ready_stems = z_at_stems.intersection(z_v_stems)

    log.info(f"Loaded feature cache index: {len(z_at_stems):,} z_at, {len(z_v_stems):,} z_v -> {len(ready_stems):,} fully preprocessed clips.")
    log.info("\nPhase 1: Validating preprocessing completeness for training sources (excluding Track 4)...")

    audit_results = {}
    training_records = []

    def _check_clip(cid, pipe, fake_lbl, spk, a_emo="", v_emo="", sarc=-1, v_path=""):
        matched_stem = find_cached_stem(cid, ready_stems)
        if matched_stem:
            z_at = Z_AT_DIR / f"{matched_stem}.pt"
            z_v = Z_V_DIR / f"{matched_stem}.pt"
            return {
                "clip_id": matched_stem,
                "source_pipeline": pipe,
                "fake_label": fake_lbl,
                "speaker_id": spk,
                "audio_emotion": a_emo,
                "visual_emotion": v_emo,
                "sarcasm_label": sarc,
                "has_z_at": 1,
                "has_z_v": 1,
                "is_ready_for_training": 1,
                "video_path": v_path,
                "z_at_path": str(z_at.relative_to(REPO_ROOT)),
                "z_v_path": str(z_v.relative_to(REPO_ROOT)),
            }
        else:
            return {
                "clip_id": cid,
                "source_pipeline": pipe,
                "fake_label": fake_lbl,
                "speaker_id": spk,
                "audio_emotion": a_emo,
                "visual_emotion": v_emo,
                "sarcasm_label": sarc,
                "has_z_at": int(cid in z_at_stems),
                "has_z_v": int(cid in z_v_stems),
                "is_ready_for_training": 0,
                "video_path": v_path,
                "z_at_path": "",
                "z_v_path": "",
            }

    # Audit Track 1, Track 2, Track 3
    for trk_name in ["track1", "track2", "track3"]:
        csv_path = SOURCE_MANIFESTS[trk_name]
        total_source = 0
        ready_cnt = 0
        missing_cnt = 0
        if csv_path.exists():
            with open(csv_path, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    stem = str(row["output_stem"])
                    total_source += 1
                    actor_id = str(row.get("actor_id", "") or (stem.split("_")[2] if len(stem.split("_")) > 2 else stem))
                    rec = _check_clip(
                        cid=stem,
                        pipe=trk_name,
                        fake_lbl=int(row.get("label", 1)),
                        spk=f"crema_{actor_id}",
                        a_emo=str(row.get("audio_emotion", "")),
                        v_emo=str(row.get("video_emotion", "")),
                        sarc=-1,
                        v_path=str(row.get("output_path", "")),
                    )
                    if rec["is_ready_for_training"]:
                        ready_cnt += 1
                        training_records.append(rec)
                    else:
                        missing_cnt += 1
        audit_results[trk_name] = {"source_manifest_total": total_source, "ready_in_cache": ready_cnt, "missing": missing_cnt}

    # Audit MELD Real
    meld_csv = SOURCE_MANIFESTS["meld_real"]
    total_source = 0
    ready_cnt = 0
    missing_cnt = 0
    if meld_csv.exists():
        with open(meld_csv, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                cid = str(row["clip_id"])
                total_source += 1
                spk = f"meld_{str(row.get('speaker', 'UNK')).replace(' ', '_')}"
                emo = str(row.get("emotion", ""))

                target_cid = cid
                if cid not in z_at_stems and "_" in cid:
                    parts = cid.split("_", 1)
                    if parts[0] in ("train", "dev", "test"):
                        target_cid = parts[1]

                rec = _check_clip(
                    cid=target_cid,
                    pipe="meld_real",
                    fake_lbl=0,
                    spk=spk,
                    a_emo=emo,
                    v_emo=emo,
                    sarc=-1,
                    v_path=str(row.get("video_path", "")),
                )
                if rec["is_ready_for_training"]:
                    ready_cnt += 1
                    training_records.append(rec)
                else:
                    missing_cnt += 1
    audit_results["meld_real"] = {"source_manifest_total": total_source, "ready_in_cache": ready_cnt, "missing": missing_cnt}

    # Audit CMU-MOSEI Real
    mosei_csv = SOURCE_MANIFESTS["mosei_real"]
    total_source = 0
    ready_cnt = 0
    missing_cnt = 0
    if mosei_csv.exists():
        with open(mosei_csv, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                cid = str(row["clip_id"])
                total_source += 1
                rec = _check_clip(
                    cid=cid,
                    pipe="mosei_real",
                    fake_lbl=0,
                    spk=f"mosei_{cid}",
                    a_emo="",
                    v_emo="",
                    sarc=-1,
                    v_path=str(row.get("video_path", "")),
                )
                if rec["is_ready_for_training"]:
                    ready_cnt += 1
                    training_records.append(rec)
                else:
                    missing_cnt += 1
    audit_results["mosei_real"] = {"source_manifest_total": total_source, "ready_in_cache": ready_cnt, "missing": missing_cnt}

    # Audit MUStARD Sarcasm
    mustard_json = SOURCE_MANIFESTS["mustard_json"]
    total_source = 0
    ready_cnt = 0
    missing_cnt = 0
    if mustard_json.exists():
        try:
            data = json.loads(mustard_json.read_text(encoding="utf-8"))
            mustard_vids = REPO_ROOT / "data/raw/MUStARD/raw_data/utterances_final"
            for key, entry in data.items():
                total_source += 1
                sarc = 1 if entry.get("sarcasm", False) else 0
                spk = f"mustard_{str(entry.get('speaker', 'UNK')).replace(' ', '_')}"
                rec = _check_clip(
                    cid=key,
                    pipe="mustard",
                    fake_lbl=-1,
                    spk=spk,
                    a_emo="",
                    v_emo="",
                    sarc=sarc,
                    v_path=str(mustard_vids / f"{key}.mp4"),
                )
                if rec["is_ready_for_training"]:
                    ready_cnt += 1
                    training_records.append(rec)
                else:
                    missing_cnt += 1
        except Exception as e:
            log.warning(f"Error reading MUStARD JSON: {e}")
    audit_results["mustard"] = {"source_manifest_total": total_source, "ready_in_cache": ready_cnt, "missing": missing_cnt}

    # Print Validation Report
    log.info("\nPREPROCESSING COMPLETENESS REPORT:")
    log.info(f"{'Source Pipeline':<18} | {'Manifest Total':<14} | {'Ready in Cache':<14} | {'Missing':<10} | {'Coverage':<10}")
    log.info("-" * 72)
    grand_manifest_total = 0
    grand_ready_total = 0
    grand_missing_total = 0

    for pipe, res in audit_results.items():
        tot = res["source_manifest_total"]
        rdy = res["ready_in_cache"]
        msg = res["missing"]
        cov = (rdy / tot * 100) if tot > 0 else 0.0
        log.info(f"{pipe:<18} | {tot:<14,} | {rdy:<14,} | {msg:<10,} | {cov:>8.1f}%")
        grand_manifest_total += tot
        grand_ready_total += rdy
        grand_missing_total += msg

    log.info("-" * 72)
    grand_cov = (grand_ready_total / grand_manifest_total * 100) if grand_manifest_total > 0 else 0.0
    log.info(f"{'TOTAL (excl Track4)':<18} | {grand_manifest_total:<14,} | {grand_ready_total:<14,} | {grand_missing_total:<10,} | {grand_cov:>8.1f}%\n")

    # 2. Verify Strict Exclusions
    log.info("\nPhase 2: Verifying strict exclusions for training...")

    fakeavceleb_cached = [s for s in ready_stems if s.startswith("fav_") or s.startswith("id")]
    webapp_cached = [s for s in ready_stems if s.startswith("upload_")]
    track4_cached = [s for s in ready_stems if s.startswith("FAKE_T4_")]

    log.info(f"  - fakeavceleb clips in cache  : {len(fakeavceleb_cached):,} (EXCLUDED from training manifest)")
    log.info(f"  - webapp_upload clips in cache: {len(webapp_cached):,} (EXCLUDED from training manifest)")
    log.info(f"  - track4 clips in cache       : {len(track4_cached):,} (EXCLUDED from training manifest as requested)")

    # Deduplicate training records by clip_id
    unique_training = {}
    for rec in training_records:
        cid = rec["clip_id"]
        if cid.startswith("fav_") or cid.startswith("id") or cid.startswith("upload_") or cid.startswith("FAKE_T4_"):
            continue
        if cid not in unique_training:
            unique_training[cid] = rec

    final_training_records = list(unique_training.values())

    # 3. Write Turnover Manifest CSVs
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
        "is_ready_for_training",
        "video_path",
        "z_at_path",
        "z_v_path",
    ]

    OUT_MANIFEST_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_MANIFEST_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(final_training_records)

    with open(PREPROCESSED_TURNOVER_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(final_training_records)

    # 4. Turnover Summary Report
    log.info("\n" + "=" * 68)
    log.info("EXCLUSIVE TRAINING TURNOVER SUMMARY REPORT")
    log.info("=" * 68)
    log.info(f"Manifest written to: {OUT_MANIFEST_CSV}")
    log.info(f"Manifest mirror   : {PREPROCESSED_TURNOVER_CSV}")
    log.info(f"Total Exclusive Training Clips: {len(final_training_records):,}\n")

    breakdown = {}
    fake_counts = {"real (0)": 0, "fake (1)": 0, "sarcasm_only (-1)": 0}
    speakers = set()

    for rec in final_training_records:
        pipe = rec["source_pipeline"]
        lbl = rec["fake_label"]
        spk = rec["speaker_id"]

        breakdown[pipe] = breakdown.get(pipe, 0) + 1
        speakers.add(spk)

        if lbl == 0:
            fake_counts["real (0)"] += 1
        elif lbl == 1:
            fake_counts["fake (1)"] += 1
        else:
            fake_counts["sarcasm_only (-1)"] += 1

    log.info("Breakdown by Source Pipeline:")
    for pipe, cnt in sorted(breakdown.items()):
        log.info(f"  - {pipe:<15}: {cnt:,} clips")

    log.info("\nLabel Distribution:")
    for lbl_str, cnt in fake_counts.items():
        log.info(f"  - {lbl_str:<20}: {cnt:,} clips")

    log.info(f"\nUnique Speakers Count: {len(speakers):,} speakers (ready for speaker-stratified split)")
    log.info("=" * 68)


if __name__ == "__main__":
    run_validation_and_turnover()
