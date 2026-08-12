"""
check_fakeavceleb_cache.py — Cache Status & Readiness Verifier for FakeAVCeleb

Scans local SSD and Google Drive for preprocessed feature tensors (z_at and z_v).
Displays detailed category breakdowns and automatically syncs feature tensors from
Google Drive to local SSD for fast evaluation.
"""

import os
import sys
import csv
import shutil
from pathlib import Path
from collections import defaultdict

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

MANIFEST_CSV = REPO_ROOT / "data/preprocessed/fakeavceleb_cached_manifest.csv"
META_CSV     = REPO_ROOT / "data/raw/FakeAVCeleb_v1.2/meta_data.csv"

LOCAL_Z_AT_DIR = REPO_ROOT / "data/preprocessed/features/z_at"
LOCAL_Z_V_DIR  = REPO_ROOT / "data/preprocessed/features/z_v"

DRIVE_BASE     = Path("/content/drive/MyDrive/THESIS_MOTHERFILE")
DRIVE_Z_AT_DIR = DRIVE_BASE / "preprocessed/features/z_at"
DRIVE_Z_V_DIR  = DRIVE_BASE / "preprocessed/features/z_v"


def check_cache(sync_from_drive: bool = True):
    print("=" * 68)
    print("  FAKEAVCELEB PREPROCESSED FEATURE CACHE VERIFIER")
    print("=" * 68)

    # 1. Determine metadata source
    csv_file = MANIFEST_CSV if MANIFEST_CSV.exists() else META_CSV
    if not csv_file.exists():
        print(f"ERROR: Neither manifest ({MANIFEST_CSV}) nor metadata ({META_CSV}) found.")
        return

    print(f"  Manifest Source: {csv_file.relative_to(REPO_ROOT)}")

    # 2. Check Drive Availability & Sync if requested
    drive_available = DRIVE_Z_AT_DIR.exists() and DRIVE_Z_V_DIR.exists()
    if drive_available:
        drive_at_count = len(list(DRIVE_Z_AT_DIR.glob("*.pt")))
        drive_v_count  = len(list(DRIVE_Z_V_DIR.glob("*.pt")))
        print(f"  Google Drive   : {DRIVE_BASE} (z_at: {drive_at_count:,}, z_v: {drive_v_count:,})")

        if sync_from_drive:
            LOCAL_Z_AT_DIR.mkdir(parents=True, exist_ok=True)
            LOCAL_Z_V_DIR.mkdir(parents=True, exist_ok=True)
            synced = 0
            for pt_file in DRIVE_Z_AT_DIR.glob("*.pt"):
                local_at = LOCAL_Z_AT_DIR / pt_file.name
                local_v  = LOCAL_Z_V_DIR / pt_file.name
                drive_v  = DRIVE_Z_V_DIR / pt_file.name
                if not local_at.exists():
                    shutil.copy2(pt_file, local_at)
                    synced += 1
                if drive_v.exists() and not local_v.exists():
                    shutil.copy2(drive_v, local_v)
            if synced > 0:
                print(f"  [AUTO-SYNC]    Synced {synced:,} feature pairs from Google Drive to local SSD!")
    else:
        print("  Google Drive   : Not mounted / Drive feature folder not found.")

    # 3. Scan Local Tensors
    LOCAL_Z_AT_DIR.mkdir(parents=True, exist_ok=True)
    LOCAL_Z_V_DIR.mkdir(parents=True, exist_ok=True)

    local_at_files = set(p.stem for p in LOCAL_Z_AT_DIR.glob("*.pt"))
    local_v_files  = set(p.stem for p in LOCAL_Z_V_DIR.glob("*.pt"))
    valid_pairs    = local_at_files.intersection(local_v_files)

    print(f"  Local SSD      : {LOCAL_Z_AT_DIR.parent}")
    print(f"    - z_at tensors: {len(local_at_files):,}")
    print(f"    - z_v tensors : {len(local_v_files):,}")
    print(f"    - Valid Pairs : {len(valid_pairs):,}")

    # 4. Read Manifest and Analyze Coverage by Category & Method
    total_clips = 0
    cat_counts   = defaultdict(int)
    cat_cached   = defaultdict(int)
    method_counts = defaultdict(int)
    method_cached = defaultdict(int)

    with open(csv_file, newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cat      = row.get("type", "").strip()
            source   = row.get("source", row.get("speaker_id", "")).strip()
            filename = row.get("path", "").strip()
            method   = row.get("method", "real").strip()
            clip_id  = row.get("clip_id", "").strip() or f"fav_{source}_{Path(filename).stem}"

            total_clips += 1
            cat_counts[cat] += 1
            method_counts[method] += 1

            if clip_id in valid_pairs:
                cat_cached[cat] += 1
                method_cached[method] += 1

    print("\n" + "=" * 68)
    print("  CATEGORY BREAKDOWN & CACHE COVERAGE")
    print("=" * 68)
    print(f"  {'Category / Type':<25} {'Total Clips':>12} {'Cached Pairs':>14} {'Coverage':>10}")
    print("  " + "-" * 64)
    for cat in sorted(cat_counts.keys()):
        tot = cat_counts[cat]
        cac = cat_cached[cat]
        pct = (cac / tot) * 100 if tot > 0 else 0.0
        print(f"  {cat:<25} {tot:>12,} {cac:>14,} {pct:>9.1f}%")

    print("\n" + "=" * 68)
    print("  DEEPFAKE METHOD BREAKDOWN")
    print("=" * 68)
    print(f"  {'Method':<25} {'Total Clips':>12} {'Cached Pairs':>14} {'Coverage':>10}")
    print("  " + "-" * 64)
    for method in sorted(method_counts.keys()):
        tot = method_counts[method]
        cac = method_cached[method]
        pct = (cac / tot) * 100 if tot > 0 else 0.0
        print(f"  {method:<25} {tot:>12,} {cac:>14,} {pct:>9.1f}%")

    # 5. Benchmark Readiness Assessment
    print("\n" + "=" * 68)
    print("  EVALUATION BENCHMARK READINESS")
    print("=" * 68)
    
    # Check 1,000-clip benchmark pool
    ready_1k = len(valid_pairs) >= 1000
    print(f"  Standard 1,000-Clip Benchmark : {'READY (Instant 1-sec bypass)' if ready_1k else f'IN PROGRESS ({len(valid_pairs):,}/1,000 cached)'}")
    
    # Check 10,000-clip benchmark pool
    ready_10k = len(valid_pairs) >= 10000
    print(f"  Full 10,000-Clip Benchmark     : {'READY (Instant cached mode)' if ready_10k else f'IN PROGRESS ({len(valid_pairs):,}/10,000 cached)'}")
    
    print("=" * 68 + "\n")


if __name__ == "__main__":
    check_cache(sync_from_drive=True)
