"""
generate_fakeavceleb_eval_manifest.py

Scans local SSD for all physically available FakeAVCeleb MP4 videos,
verifies their existence, applies Hard Mode stratification (40% Compound, 40% Wav2Lip, 20% Single-mod),
and generates data/manifests/fakeavceleb_eval_500_500.csv with 100% guaranteed on-disk files.
"""
from __future__ import annotations

import csv
import os
import random
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

_HARD_METHODS = {"faceswap-wav2lip", "fsgan-wav2lip"}
_MED_METHODS  = {"wav2lip"}

def main(n_real: int = 500, n_fake: int = 500, seed: int = 42):
    random.seed(seed)
    
    print("=" * 60)
    print("SCANNING FAKEAVCELEB FOR VERIFIED ON-DISK MEDIA")
    print("=" * 60)
    
    # 1. Index all MP4 files on disk
    mp4_files = []
    scan_roots = [
        REPO_ROOT / "data/raw/FakeAVCeleb_v1.2",
        REPO_ROOT / "data/FakeAVCeleb_v1.2",
        REPO_ROOT / "data/raw/FakeAVCeleb",
        REPO_ROOT / "data/FakeAVCeleb",
        REPO_ROOT / "data/raw",
        REPO_ROOT / "data",
        Path("/content/thesis/data/raw/FakeAVCeleb_v1.2"),
        Path("/content/thesis/data/raw"),
        Path("/content/thesis/data")
    ]
    
    seen_paths = set()
    for root in scan_roots:
        if root.exists():
            for dp, _, fns in os.walk(root):
                if "drive" in dp or ".git" in dp:
                    continue
                for fn in fns:
                    if fn.lower().endswith(".mp4"):
                        p = Path(dp) / fn
                        rel_str = str(p.resolve())
                        if rel_str not in seen_paths:
                            seen_paths.add(rel_str)
                            mp4_files.append(p)
                            
    print(f"  [Disk Scan] Discovered {len(mp4_files):,} physical MP4 video files.")

    # 2. Build metadata mappings
    meta_candidates = [
        REPO_ROOT / "data/raw/FakeAVCeleb_v1.2/meta_data.csv",
        REPO_ROOT / "data/FakeAVCeleb_v1.2/meta_data.csv",
        REPO_ROOT / "data/raw/meta_data.csv",
        REPO_ROOT / "data/meta_data.csv",
        Path("/content/thesis/data/raw/FakeAVCeleb_v1.2/meta_data.csv"),
    ]
    
    meta_path = next((m for m in meta_candidates if m.exists()), None)
    if not meta_path:
        print("  [ERROR] meta_data.csv not found!")
        return

    print(f"  [Metadata] Reading: {meta_path}")

    # Build fast lookup hashmap of available MP4 files
    file_map = {}
    for p in mp4_files:
        file_map[p.name] = p
        parts = p.parts
        if len(parts) >= 2:
            file_map[f"{parts[-2]}/{parts[-1]}"] = p
        if len(parts) >= 3:
            file_map[f"{parts[-3]}/{parts[-2]}/{parts[-1]}"] = p
        if len(parts) >= 4:
            file_map[f"{parts[-4]}/{parts[-3]}/{parts[-2]}/{parts[-1]}"] = p
        if len(parts) >= 5:
            file_map[f"{parts[-5]}/{parts[-4]}/{parts[-3]}/{parts[-2]}/{parts[-1]}"] = p

    real_pool = []
    fake_by_tier = {"hard": [], "med": [], "easy": []}

    with open(meta_path, newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cat      = row.get("type", "").strip()
            source   = row.get("source", row.get("speaker_id", "")).strip()
            fname    = row.get("path", "").strip()
            method   = row.get("method", "real").strip().lower()
            race     = row.get("race", "").strip()
            gender   = row.get("gender", "").strip()
            
            # Find physical video file
            v_path = file_map.get(fname) or file_map.get(f"{source}/{fname}") or file_map.get(f"{cat}/{race}/{gender}/{source}/{fname}")
            if v_path is None or not v_path.is_file():
                continue

            # Format relative path
            try:
                rel_v = str(v_path.relative_to(REPO_ROOT / "data/raw/FakeAVCeleb_v1.2")).replace("\\", "/")
            except Exception:
                try:
                    rel_v = str(v_path.relative_to(REPO_ROOT / "data")).replace("\\", "/")
                except Exception:
                    rel_v = f"{cat}/{race}/{gender}/{source}/{fname}"

            clip_id = f"fav_{source}_{Path(fname).stem}"
            is_real = (cat == "RealVideo-RealAudio" or method == "real")
            
            entry = {
                "clip_id": clip_id,
                "fake_label": 0 if is_real else 1,
                "method": method,
                "type": cat,
                "speaker_id": source,
                "rel_video_path": rel_v,
            }

            if is_real:
                real_pool.append(entry)
            elif method in _HARD_METHODS:
                fake_by_tier["hard"].append(entry)
            elif method in _MED_METHODS:
                fake_by_tier["med"].append(entry)
            else:
                fake_by_tier["easy"].append(entry)

    print(f"\n  Verified On-Disk Media:")
    print(f"    Genuine Real Videos : {len(real_pool):,}")
    print(f"    Compound Fakes      : {len(fake_by_tier['hard']):,}")
    print(f"    Wav2Lip Fakes       : {len(fake_by_tier['med']):,}")
    print(f"    Single-Mod Fakes    : {len(fake_by_tier['easy']):,}")

    # Shuffle pools
    random.shuffle(real_pool)
    for k in fake_by_tier:
        random.shuffle(fake_by_tier[k])

    sampled_real = real_pool[:min(n_real, len(real_pool))]
    
    # Hard mode sampling
    n_h = min(int(n_fake * 0.40), len(fake_by_tier["hard"]))
    n_m = min(int(n_fake * 0.40), len(fake_by_tier["med"]))
    n_e = min(n_fake - n_h - n_m, len(fake_by_tier["easy"]))
    
    sampled_fake = fake_by_tier["hard"][:n_h] + fake_by_tier["med"][:n_m] + fake_by_tier["easy"][:n_e]
    if len(sampled_fake) < n_fake:
        used = set(c["clip_id"] for c in sampled_fake)
        remaining = [c for tier in fake_by_tier.values() for c in tier if c["clip_id"] not in used]
        random.shuffle(remaining)
        sampled_fake += remaining[:n_fake - len(sampled_fake)]

    final_dataset = sampled_real + sampled_fake
    random.shuffle(final_dataset)

    out_csv = REPO_ROOT / "data/manifests/fakeavceleb_eval_500_500.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["clip_id", "fake_label", "method", "type", "speaker_id", "rel_video_path"])
        writer.writeheader()
        writer.writerows(final_dataset)

    print(f"\n" + "=" * 60)
    print(f"MANIFEST GENERATED SUCCESSFULLY: {out_csv}")
    print(f"  Total Clips : {len(final_dataset)} (Real: {len(sampled_real)}, Fake: {len(sampled_fake)})")
    print("=" * 60)

if __name__ == "__main__":
    main()
