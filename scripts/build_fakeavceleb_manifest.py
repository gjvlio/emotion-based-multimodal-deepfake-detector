"""
Builds data/preprocessed/fakeavceleb_cached_manifest.csv from meta_data.csv and cached feature tensors.
Allows evaluating FakeAVCeleb on precomputed features without needing raw MP4 files.
"""

import csv
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
META_CSV  = REPO_ROOT / "data/raw/FakeAVCeleb_v1.2/meta_data.csv"
OUT_CSV   = REPO_ROOT / "data/preprocessed/fakeavceleb_cached_manifest.csv"
Z_AT_DIR  = REPO_ROOT / "data/preprocessed/features/z_at"
Z_V_DIR   = REPO_ROOT / "data/preprocessed/features/z_v"
REAL_TYPE = "RealVideo-RealAudio"


def main():
    if not META_CSV.exists():
        log.error(f"meta_data.csv not found at: {META_CSV}")
        return

    records = []
    missing_features = 0

    with open(META_CSV, newline="", encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            cat      = row.get("type",   "").strip()
            race     = row.get("race",   "").strip()
            gender   = row.get("gender", "").strip()
            source   = row.get("source", "").strip()
            filename = row.get("path",   "").strip()
            method   = row.get("method", "real").strip()

            if not all([cat, race, gender, source, filename]):
                continue

            stem = Path(filename).stem
            clip_id = f"fav_{source}_{stem}"
            z_at_p = Z_AT_DIR / f"{clip_id}.pt"
            z_v_p  = Z_V_DIR / f"{clip_id}.pt"

            z_at_exists = z_at_p.exists()
            z_v_exists  = z_v_p.exists()

            if not (z_at_exists and z_v_exists):
                missing_features += 1

            records.append({
                "clip_id":     clip_id,
                "fake_label":  0 if cat == REAL_TYPE else 1,
                "method":      method,
                "type":        cat,
                "speaker_id":  source,
                "z_at_exists": 1 if z_at_exists else 0,
                "z_v_exists":  1 if z_v_exists else 0,
            })

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["clip_id", "fake_label", "method", "type", "speaker_id", "z_at_exists", "z_v_exists"]

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    cached_cnt = sum(1 for r in records if r["z_at_exists"] and r["z_v_exists"])
    log.info(f"Exported {len(records):,} FakeAVCeleb manifest records -> {OUT_CSV}")
    log.info(f"Cached feature pairs (z_at & z_v): {cached_cnt:,} / {len(records):,} clips ready for 0-second evaluation.")


if __name__ == "__main__":
    main()
