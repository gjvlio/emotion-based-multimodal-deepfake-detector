"""
smoke_test_musetalk.py
======================
Smoke test MuseTalk lip-sync on MELD emotion-mismatch pairs.
MuseTalk uses ~3-4 GB VRAM — comfortable on RTX 4050 (6 GB).

Setup (do once):
    cd tools
    git clone https://github.com/TMElyralab/MuseTalk MuseTalk
    cd MuseTalk
    pip install -r requirements.txt
    # Download model weights per their README:
    #   models/musetalk/musetalk.json + pytorch_model.bin
    #   models/dwpose/dw-ll_ucoco_384.pth
    #   models/face-parse-bisent/resnet18.pth + 79999_iter.pth
    #   models/whisper/tiny.pt
    #   models/sd-vae-ft-mse/  (HuggingFace diffusers VAE)

Usage:
    python scripts/smoke_test_musetalk.py --n_clips 3
    python scripts/smoke_test_musetalk.py --musetalk_dir tools/MuseTalk --n_clips 5
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import pandas as pd
import yaml

REQUIRED_CHECKPOINTS = [
    "models/musetalk/pytorch_model.bin",
    "models/musetalk/musetalk.json",
    "models/dwpose/dw-ll_ucoco_384.pth",
    "models/face-parse-bisent/79999_iter.pth",
    "models/whisper/pytorch_model.bin",
    "models/sd-vae-ft-mse/config.json",
]


# ── helpers ────────────────────────────────────────────────────────────────────

def run_ffmpeg(cmd: list[str]) -> bool:
    r = subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"] + cmd,
        capture_output=True, text=True, timeout=60,
    )
    return r.returncode == 0


def get_duration(video_path: str) -> float | None:
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", video_path],
            capture_output=True, text=True, timeout=15,
        )
        return float(r.stdout.strip())
    except Exception:
        return None


def extract_audio(video_path: str, wav_path: str,
                  max_duration: float | None = None) -> bool:
    cmd = ["-i", video_path, "-vn",
           "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1"]
    if max_duration is not None:
        cmd += ["-t", str(max_duration)]
    cmd.append(wav_path)
    return run_ffmpeg(cmd)


def poll_vram(stop: threading.Event, out: list):
    peak = 0
    while not stop.is_set():
        try:
            r = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5,
            )
            val = int(r.stdout.strip().splitlines()[0])
            if val > peak:
                peak = val
        except Exception:
            pass
        time.sleep(0.5)
    out.append(peak)


def check_setup(musetalk_dir: Path) -> bool:
    ok = True
    inference = musetalk_dir / "scripts" / "inference.py"
    if not inference.exists():
        print(f"  MISSING  scripts/inference.py — clone MuseTalk to {musetalk_dir}")
        print("           git clone https://github.com/TMElyralab/MuseTalk tools/MuseTalk")
        return False
    print(f"  OK       scripts/inference.py")

    for rel in REQUIRED_CHECKPOINTS:
        p = musetalk_dir / rel
        if p.exists():
            print(f"  OK       {rel}")
        else:
            print(f"  MISSING  {rel}")
            ok = False
    return ok


def write_inference_config(config_path: str, video_path: str,
                            audio_path: str, bbox_shift: int = 0):
    cfg = {"task_0": {
        "video_path": os.path.abspath(video_path),
        "audio_path": os.path.abspath(audio_path),
        "bbox_shift":  bbox_shift,
    }}
    with open(config_path, "w") as f:
        yaml.dump(cfg, f)


def run_musetalk(musetalk_dir: Path, config_path: str,
                 result_dir: str, batch_size: int = 4) -> tuple[bool, str]:
    cmd = [
        sys.executable, "-m", "scripts.inference",
        "--inference_config",  os.path.abspath(config_path),
        "--result_dir",        os.path.abspath(result_dir),
        "--batch_size",        str(batch_size),
        "--version",           "v1",
        "--unet_model_path",   "models/musetalk/pytorch_model.bin",
        "--unet_config",       "models/musetalk/musetalk.json",
        "--vae_type",          "sd-vae-ft-mse",
    ]
    env = os.environ.copy()
    env["TORCHDYNAMO_DISABLE"] = "1"
    r = subprocess.run(
        cmd,
        cwd=str(musetalk_dir.resolve()),
        capture_output=True, text=True, timeout=600,
        env=env,
    )
    if r.returncode != 0:
        return False, (r.stderr or r.stdout)[-500:].strip()
    return True, ""


def find_output_video(result_dir: str, stem: str) -> str | None:
    """MuseTalk writes to result_dir/task_0/*.mp4 — find it."""
    import glob
    pattern = os.path.join(result_dir, "**", "*.mp4")
    hits = glob.glob(pattern, recursive=True)
    return hits[0] if hits else None


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Smoke test MuseTalk on MELD emotion-mismatch pairs"
    )
    parser.add_argument("--musetalk_dir", default="tools/MuseTalk")
    parser.add_argument("--pairs_csv",
                        default="data/processed/meld_manifests/meld_mismatch_pairs.csv")
    parser.add_argument("--n_clips",   type=int, default=3)
    parser.add_argument("--out_dir",   default="data/synthetic/musetalk_smoketest")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="MuseTalk inference batch size (lower = less VRAM, default 4)")
    parser.add_argument("--bbox_shift", type=int, default=0,
                        help="Vertical lip-region shift in pixels (tune if lips misaligned)")
    args = parser.parse_args()

    musetalk_dir = Path(args.musetalk_dir)
    out_dir      = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    wav_dir = out_dir / "wav_tmp"
    wav_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("MuseTalk smoke test  (~3-4 GB VRAM, safe on RTX 4050)")
    print("=" * 60)

    print("\n[1/3] Checking setup...")
    if not check_setup(musetalk_dir):
        sys.exit(1)
    print("      All required files present.\n")

    pairs_csv = Path(args.pairs_csv)
    if not pairs_csv.exists():
        print(f"Pairs CSV not found: {pairs_csv}")
        print("Run: python scripts/sample_meld_mismatch.py")
        sys.exit(1)

    pairs = pd.read_csv(pairs_csv).head(args.n_clips)
    print(f"[2/3] Running {len(pairs)} clips  (batch_size={args.batch_size})...\n")

    records = []

    for i, (_, row) in enumerate(pairs.iterrows(), 1):
        stem      = row["output_stem"]
        face_mp4  = row["video_clip"]
        audio_src = row["audio_clip"]
        donor_wav = str(wav_dir / f"{stem}_donor.wav")

        print(f"  [{i}/{len(pairs)}] {row['video_emotion']} face | {row['audio_emotion']} audio")

        missing = False
        for path, label in [(face_mp4, "video_clip"), (audio_src, "audio_clip")]:
            if not os.path.exists(path):
                print(f"    SKIP  {label} not found: {path}")
                missing = True
        if missing:
            records.append({"stem": stem, "status": "skip", "error": "file missing"})
            continue

        face_dur = get_duration(face_mp4)
        if not extract_audio(audio_src, donor_wav, max_duration=face_dur):
            print("    FAIL  donor audio extraction")
            records.append({"stem": stem, "status": "fail", "error": "audio extraction"})
            continue

        with tempfile.TemporaryDirectory() as tmp:
            config_path = os.path.join(tmp, "config.yaml")
            result_dir  = os.path.join(tmp, "results")
            os.makedirs(result_dir)
            write_inference_config(config_path, face_mp4, donor_wav, args.bbox_shift)

            stop, vram_out = threading.Event(), []
            vt = threading.Thread(target=poll_vram, args=(stop, vram_out), daemon=True)
            vt.start()
            t0 = time.time()

            ok, err = run_musetalk(musetalk_dir, config_path, result_dir, args.batch_size)

            elapsed = time.time() - t0
            stop.set(); vt.join(timeout=2)
            peak_vram = vram_out[0] if vram_out else -1

            if ok:
                src = find_output_video(result_dir, stem)
                if src:
                    import shutil
                    dst = str(out_dir / f"{stem}_musetalk.mp4")
                    shutil.copy(src, dst)
                    size_mb = os.path.getsize(dst) / 1_048_576
                    print(f"    PASS  {elapsed:.1f}s  VRAM {peak_vram} MiB  {size_mb:.1f} MB")
                    records.append({"stem": stem, "status": "pass",
                                    "elapsed_s": round(elapsed, 1),
                                    "peak_vram_mib": peak_vram,
                                    "output_mb": round(size_mb, 1)})
                else:
                    print(f"    FAIL  inference ok but no output mp4 found")
                    records.append({"stem": stem, "status": "fail",
                                    "error": "no output mp4", "elapsed_s": round(elapsed, 1)})
            else:
                oom = "out of memory" in err.lower() or "cuda out" in err.lower()
                tag = "OOM " if oom else "FAIL"
                print(f"    {tag}  {elapsed:.1f}s  VRAM {peak_vram} MiB")
                if err:
                    print(f"    {err[:280]}")
                records.append({"stem": stem,
                                "status": "oom" if oom else "fail",
                                "elapsed_s": round(elapsed, 1),
                                "peak_vram_mib": peak_vram,
                                "error": err[:280]})
        print()

    passed  = [r for r in records if r["status"] == "pass"]
    failed  = [r for r in records if r["status"] in ("fail", "oom")]

    print("=" * 60)
    print(f"[3/3] {len(passed)} pass  {len(failed)} fail  {len(records)-len(passed)-len(failed)} skip")

    if passed:
        avg_t    = sum(r["elapsed_s"]     for r in passed) / len(passed)
        avg_vram = sum(r["peak_vram_mib"] for r in passed) / len(passed)
        eta_h    = 1193 * avg_t / 3600
        print(f"      avg time  : {avg_t:.1f}s/clip")
        print(f"      avg VRAM  : {avg_vram:.0f} MiB")
        print(f"      ETA full  : ~{eta_h:.1f} hrs for 1193 clips")
        if avg_vram > 5500:
            print(f"\n  WARNING: VRAM {avg_vram:.0f} MiB close to 6 GB limit.")
            print(f"  Lower --batch_size (try 2) if full run hits OOM.")

    if any(r["status"] == "oom" for r in records):
        print(f"\n  OOM hit. Lower --batch_size (try 2 or 1).")

    report = out_dir / "smoketest_report.json"
    with open(report, "w") as f:
        json.dump(records, f, indent=2)
    print(f"\n  Report: {report}")
    print("=" * 60)


if __name__ == "__main__":
    main()
