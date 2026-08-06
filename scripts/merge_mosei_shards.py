"""
merge_mosei_shards.py
=====================
Safely extracts and merges CMU-MOSEI feature shards (`mosei_features_shard*.zip`)
into `data/preprocessed/features/` without destroying existing preprocessed files.

Generates a safety report:
    data/preprocessed/mosei_shard_extraction_report.csv
And updates:
    data/preprocessed/preprocessed_manifest.csv

Usage:
    python scripts/merge_mosei_shards.py
"""
import csv
import logging
import zipfile
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
PREPROCESSED_DIR = REPO_ROOT / "data/preprocessed"
FEATURES_DIR = PREPROCESSED_DIR / "features"
REPORT_CSV = PREPROCESSED_DIR / "mosei_shard_extraction_report.csv"


def merge_shards():
    zips = sorted(PREPROCESSED_DIR.glob("mosei_features_shard*.zip"))
    if not zips:
        log.warning("No mosei_features_shard*.zip files found in data/preprocessed/")
        return

    log.info(f"Found {len(zips)} shard zip files: {[z.name for z in zips]}")

    target_z_at = FEATURES_DIR / "z_at"
    target_z_v = FEATURES_DIR / "z_v"
    target_z_at.mkdir(parents=True, exist_ok=True)
    target_z_v.mkdir(parents=True, exist_ok=True)

    extracted_records = []
    total_extracted = 0
    total_skipped_existing = 0

    for z_path in zips:
        log.info(f"Processing archive {z_path.name}...")
        with zipfile.ZipFile(z_path, "r") as zf:
            for member in zf.namelist():
                if member.endswith("/"):
                    continue

                # Expected path in zip: 'z_at/clip.pt' or 'z_v/clip.pt' or 'features/z_at/clip.pt'
                p = Path(member)
                if "z_at" in p.parts:
                    sub_dir = target_z_at
                    feature_type = "z_at"
                elif "z_v" in p.parts:
                    sub_dir = target_z_v
                    feature_type = "z_v"
                else:
                    log.warning(f"Unrecognized member path structure: {member}")
                    continue

                out_file = sub_dir / p.name
                already_existed = out_file.exists()

                if already_existed:
                    status = "skipped_existing"
                    total_skipped_existing += 1
                else:
                    # Safely extract single file to sub_dir
                    with zf.open(member) as src, open(out_file, "wb") as dst:
                        dst.write(src.read())
                    status = "extracted_new"
                    total_extracted += 1

                extracted_records.append(
                    {
                        "shard_zip": z_path.name,
                        "clip_id": p.stem,
                        "feature_type": feature_type,
                        "member_path": member,
                        "output_path": str(out_file.relative_to(REPO_ROOT)),
                        "status": status,
                        "size_bytes": out_file.stat().st_size if out_file.exists() else 0,
                    }
                )

    log.info(f"Extraction finished: {total_extracted:,} new feature files extracted, {total_skipped_existing:,} skipped existing.")

    # Save extraction report CSV
    fieldnames = ["shard_zip", "clip_id", "feature_type", "member_path", "output_path", "status", "size_bytes"]
    with open(REPORT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(extracted_records)
    log.info(f"Extraction report saved to: {REPORT_CSV}")

    # Rebuild master preprocessed manifest CSV
    log.info("Updating preprocessed_manifest.csv...")
    from build_preprocessed_manifest import build_manifest

    build_manifest()


if __name__ == "__main__":
    merge_shards()
