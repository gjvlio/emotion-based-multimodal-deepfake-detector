"""
zip_preprocessed.py
===================
Zips all files in `data/preprocessed/` into `data/preprocessed/preprocessed_all.zip`.
Excludes existing `.zip` files to avoid self-archiving or duplicate archives.

Usage:
    python scripts/zip_preprocessed.py
"""
import logging
import zipfile
from pathlib import Path
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
PREPROCESSED_DIR = REPO_ROOT / "data/preprocessed"
OUT_ZIP = PREPROCESSED_DIR / "preprocessed_all.zip"


def create_zip():
    log.info("=" * 68)
    log.info("PREPROCESSED DIRECTORY COMPRESSION (ZIP ARCHIVE)")
    log.info("=" * 68)
    log.info(f"Target Zip Output: {OUT_ZIP}")

    # Gather all files in data/preprocessed excluding .zip files
    all_files = [
        f for f in PREPROCESSED_DIR.rglob("*")
        if f.is_file() and not f.name.endswith(".zip") and f.resolve() != OUT_ZIP.resolve()
    ]

    total_files = len(all_files)
    total_bytes = sum(f.stat().st_size for f in all_files)
    log.info(f"Found {total_files:,} files ({total_bytes / (1024 * 1024):.2f} MB) to compress...")

    t0 = datetime.now()
    added_files = 0
    written_bytes = 0

    # Write zip with compression
    with zipfile.ZipFile(OUT_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for idx, file_path in enumerate(all_files, 1):
            arcname = file_path.relative_to(PREPROCESSED_DIR)
            zf.write(file_path, arcname=arcname)
            added_files += 1
            written_bytes += file_path.stat().st_size

            if idx % 10000 == 0 or idx == total_files:
                dt = (datetime.now() - t0).total_seconds()
                log.info(f"  Progress: {idx:,} / {total_files:,} files ({idx/total_files*100:.1f}%) in {dt:.1f}s")

    zip_size_mb = OUT_ZIP.stat().st_size / (1024 * 1024)
    dt_total = (datetime.now() - t0).total_seconds()

    log.info("\n" + "=" * 68)
    log.info("ZIP COMPRESSION SUMMARY REPORT")
    log.info("=" * 68)
    log.info(f"Zip Location        : {OUT_ZIP}")
    log.info(f"Files Archived      : {added_files:,} files")
    log.info(f"Uncompressed Size   : {written_bytes / (1024 * 1024):.2f} MB")
    log.info(f"Compressed Zip Size : {zip_size_mb:.2f} MB")
    log.info(f"Compression Ratio   : {(1 - zip_size_mb / (written_bytes / (1024 * 1024))) * 100:.1f}% space saved")
    log.info(f"Total Time Taken    : {dt_total:.1f} seconds")
    log.info("=" * 68)


if __name__ == "__main__":
    create_zip()
