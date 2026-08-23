"""
generate_fakeavceleb_eval_manifest.py

Scans local SSD for all physically available FakeAVCeleb MP4 videos directly from disk,
parses categories from path hierarchy, applies Hard Mode stratification
(40% Compound, 40% Wav2Lip, 20% Single-modality), and generates
data/manifests/fakeavceleb_eval_500_500.csv with 100% guaranteed on-disk files.
"""
from __future__ import annotations

import csv
import os
import random
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

_HARD_METHODS = {"faceswap-wav2lip", "fsgan-wav2lip"}
_MED_METHODS  = {"wav2lip"}

def parse_method_and_type(p: Path) -> tuple[str, str, int, str]:
    """
    Directly infers (type, method, fake_label, speaker_id) from physical file path.
    """
    path_str = str(p).replace("\\", "/")
    
    # Speaker extraction
    speaker_id = p.parent.name if p.parent.name.startswith("id") else "fav_spk"
    
    # 1. Real Video Check
    if "RealVideo-RealAudio" in path_str or "RealAudio-RealVideo" in path_str:
        return "RealVideo-RealAudio", "real", 0, speaker_id

    # 2. Compound Fakes (Faceswap + Wav2Lip or FSGAN + Wav2Lip)
    fname_lower = p.name.lower()
    if ("faceswap" in fname_lower or "fs" in fname_lower or "face" in fname_lower) and ("wav2lip" in fname_lower or "wavtolip" in fname_lower):
        if "fsgan" in fname_lower:
            return "FakeVideo-FakeAudio", "fsgan-wav2lip", 1, speaker_id
        return "FakeVideo-FakeAudio", "faceswap-wav2lip", 1, speaker_id

    # 3. Wav2Lip Fakes
    if "wav2lip" in fname_lower or "wavtolip" in fname_lower:
        cat = "FakeVideo-FakeAudio" if "FakeVideo-FakeAudio" in path_str else "FakeVideo-RealAudio"
        return cat, "wav2lip", 1, speaker_id

    # 4. RTVC Audio Fakes
    if "rtvc" in fname_lower or "RealVideo-FakeAudio" in path_str:
        return "RealVideo-FakeAudio", "rtvc", 1, speaker_id

    # 5. FSGAN Video Fakes
    if "fsgan" in fname_lower or "FakeVideo-RealAudio" in path_str:
        return "FakeVideo-RealAudio", "fsgan", 1, speaker_id

    # 6. Generic Faceswap Video Fakes
    return "FakeVideo-RealAudio", "faceswap", 1, speaker_id


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
        Path("/content/thesis/data/FakeAVCeleb_v1.2"),
        Path("/content/thesis/data/raw"),
        Path("/content/thesis/data")
    ]
    
    seen_paths = set()
    for root in scan_roots:
        if root.exists():
            for dp, dirnames, fns in os.walk(root):
                if "drive" in dp or ".git" in dp:
                    dirnames.clear()
                    continue
                for fn in fns:
                    if fn.lower().endswith(".mp4"):
                        p = Path(dp) / fn
                        resolved = p.resolve()
                        if str(resolved) not in seen_paths:
                            seen_paths.add(str(resolved))
                            mp4_files.append(p)
                            
    print(f"  [Disk Scan] Discovered {len(mp4_files):,} physical MP4 video files.")

    real_pool = []
    fake_by_tier = {"hard": [], "med": [], "easy": []}

    for p in mp4_files:
        cat, method, label, spk = parse_method_and_type(p)
        
        # Build clean relative path
        rel_v = ""
        for base in [REPO_ROOT / "data/raw/FakeAVCeleb_v1.2", REPO_ROOT / "data/FakeAVCeleb_v1.2", REPO_ROOT / "data/raw", REPO_ROOT / "data", Path("/content/thesis/data/raw/FakeAVCeleb_v1.2"), Path("/content/thesis/data/raw"), Path("/content/thesis/data")]:
            if base.exists():
                try:
                    rel_v = str(p.relative_to(base)).replace("\\", "/")
                    break
                except Exception:
                    pass
        if not rel_v:
            rel_v = str(p).replace("\\", "/")

        clip_id = f"fav_{spk}_{p.stem}"
        entry = {
            "clip_id": clip_id,
            "fake_label": label,
            "method": method,
            "type": cat,
            "speaker_id": spk,
            "rel_video_path": rel_v,
        }

        if label == 0:
            real_pool.append(entry)
        elif method in _HARD_METHODS:
            fake_by_tier["hard"].append(entry)
        elif method in _MED_METHODS:
            fake_by_tier["med"].append(entry)
        else:
            fake_by_tier["easy"].append(entry)

    print(f"\n  Verified On-Disk Inventory:")
    print(f"    Genuine Real Videos : {len(real_pool):,}")
    print(f"    Compound Fakes      : {len(fake_by_tier['hard']):,}")
    print(f"    Wav2Lip Fakes       : {len(fake_by_tier['med']):,}")
    print(f"    Single-Mod Fakes    : {len(fake_by_tier['easy']):,}")

    if not real_pool:
        print("  [ERROR] No real video files found on disk!")
        return

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
