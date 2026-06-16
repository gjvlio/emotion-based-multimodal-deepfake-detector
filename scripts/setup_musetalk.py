"""
setup_musetalk.py
=================
One-shot MuseTalk setup: clone repo, install deps, download all model weights.

Usage:
    python scripts/setup_musetalk.py
    python scripts/setup_musetalk.py --tools_dir tools --skip_clone
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

# ── model download specs ───────────────────────────────────────────────────────

HF_FILES = [
    # (repo_id, filename, local_rel_path)
    ("TMElyralab/MuseTalk", "musetalk.json",       "models/musetalk/musetalk.json"),
    ("TMElyralab/MuseTalk", "pytorch_model.bin",   "models/musetalk/pytorch_model.bin"),
    ("yzd-v/DWPose",        "dw-ll_ucoco_384.pth", "models/dwpose/dw-ll_ucoco_384.pth"),
]

HF_REPOS = [
    # (repo_id, local_subdir)
    ("stabilityai/sd-vae-ft-mse", "models/sd-vae-ft-mse"),
]

REQUIRED_AFTER = [
    "models/musetalk/pytorch_model.bin",
    "models/musetalk/musetalk.json",
    "models/dwpose/dw-ll_ucoco_384.pth",
    "models/face-parse-bisent/79999_iter.pth",
    "models/whisper/tiny.pt",
    "models/sd-vae-ft-mse/config.json",
]


def run(cmd: list[str], cwd: str | None = None, desc: str = "") -> bool:
    print(f"  $ {' '.join(cmd[:6])}{'...' if len(cmd) > 6 else ''}")
    r = subprocess.run(cmd, cwd=cwd)
    if r.returncode != 0:
        print(f"  FAILED: {desc or cmd[0]}")
    return r.returncode == 0


def pip(*packages: str) -> bool:
    return run([sys.executable, "-m", "pip", "install", "-q", *packages],
               desc=f"pip install {' '.join(packages)}")


def download_hf_file(repo_id: str, filename: str, local_path: Path) -> bool:
    if local_path.exists():
        print(f"  EXISTS   {local_path.relative_to(local_path.parents[2])}")
        return True
    local_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from huggingface_hub import hf_hub_download
        src = hf_hub_download(repo_id=repo_id, filename=filename)
        import shutil
        shutil.copy(src, local_path)
        print(f"  OK       {local_path.relative_to(local_path.parents[2])}")
        return True
    except Exception as e:
        print(f"  FAILED   {repo_id}/{filename}: {e}")
        return False


def download_hf_repo(repo_id: str, local_dir: Path) -> bool:
    sentinel = local_dir / "config.json"
    if sentinel.exists():
        print(f"  EXISTS   {local_dir.name}/")
        return True
    local_dir.mkdir(parents=True, exist_ok=True)
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=repo_id, local_dir=str(local_dir),
                          ignore_patterns=["*.msgpack", "*.h5", "flax_model*"])
        print(f"  OK       {local_dir.name}/")
        return True
    except Exception as e:
        print(f"  FAILED   {repo_id}: {e}")
        return False


def download_whisper(local_path: Path) -> bool:
    if local_path.exists():
        print(f"  EXISTS   models/whisper/tiny.pt")
        return True
    local_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import whisper
        model = whisper.load_model("tiny", download_root=str(local_path.parent))
        # whisper saves as tiny.pt inside download_root
        if local_path.exists():
            print(f"  OK       models/whisper/tiny.pt")
            return True
        # some versions use different filename
        candidates = list(local_path.parent.glob("*.pt"))
        if candidates:
            import shutil
            shutil.move(str(candidates[0]), str(local_path))
            print(f"  OK       models/whisper/tiny.pt (moved from {candidates[0].name})")
            return True
        print(f"  FAILED   whisper download: file not found after load")
        return False
    except Exception as e:
        print(f"  FAILED   whisper: {e}")
        return False


def download_face_parse(musetalk_dir: Path) -> bool:
    """
    BiSeNet face parsing weights.
    Tries HuggingFace mirror first; falls back to direct URL.
    """
    out_dir = musetalk_dir / "models" / "face-parse-bisent"
    out_dir.mkdir(parents=True, exist_ok=True)
    ok = True

    files = {
        "79999_iter.pth": [
            # HF mirrors — try in order
            ("nicehuster/face-parsing-pytorch", "79999_iter.pth"),
        ],
        "resnet18.pth": [
            ("nicehuster/face-parsing-pytorch", "resnet18.pth"),
        ],
    }

    for fname, mirrors in files.items():
        dest = out_dir / fname
        if dest.exists():
            print(f"  EXISTS   models/face-parse-bisent/{fname}")
            continue
        downloaded = False
        for repo_id, hf_fname in mirrors:
            try:
                from huggingface_hub import hf_hub_download
                src = hf_hub_download(repo_id=repo_id, filename=hf_fname)
                import shutil
                shutil.copy(src, dest)
                print(f"  OK       models/face-parse-bisent/{fname}")
                downloaded = True
                break
            except Exception:
                continue
        if not downloaded:
            print(f"  FAILED   models/face-parse-bisent/{fname}")
            print(f"           Manual: download from https://huggingface.co/nicehuster/face-parsing-pytorch")
            ok = False

    return ok


def verify(musetalk_dir: Path) -> bool:
    print("\n[Final check]")
    all_ok = True
    for rel in REQUIRED_AFTER:
        p = musetalk_dir / rel
        if p.exists():
            print(f"  OK       {rel}")
        else:
            print(f"  MISSING  {rel}")
            all_ok = False
    return all_ok


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="One-shot MuseTalk setup")
    parser.add_argument("--tools_dir",  default="tools")
    parser.add_argument("--skip_clone", action="store_true",
                        help="Skip git clone (repo already exists)")
    args = parser.parse_args()

    tools_dir    = Path(args.tools_dir)
    musetalk_dir = tools_dir / "MuseTalk"

    print("=" * 60)
    print("MuseTalk setup")
    print("=" * 60)

    # ── 1. Clone ───────────────────────────────────────────────────────────────
    if not args.skip_clone:
        print("\n[1/5] Cloning MuseTalk...")
        if musetalk_dir.exists():
            print(f"  EXISTS   {musetalk_dir} — skipping clone")
        else:
            tools_dir.mkdir(parents=True, exist_ok=True)
            ok = run(
                ["git", "clone", "https://github.com/TMElyralab/MuseTalk",
                 str(musetalk_dir)],
                desc="git clone MuseTalk",
            )
            if not ok:
                print("Clone failed. Check network and retry.")
                sys.exit(1)
    else:
        print("\n[1/5] Clone skipped.")

    if not (musetalk_dir / "scripts" / "inference.py").exists():
        print(f"  ERROR  scripts/inference.py not found in {musetalk_dir}")
        print("         Clone may be incomplete. Delete and re-run without --skip_clone.")
        sys.exit(1)

    # ── 2. Install Python deps ─────────────────────────────────────────────────
    print("\n[2/5] Installing Python dependencies...")
    req = musetalk_dir / "requirements.txt"
    if req.exists():
        run([sys.executable, "-m", "pip", "install", "-q", "-r", str(req)],
            desc="pip install requirements.txt")
    else:
        print("  WARNING  requirements.txt not found — install manually if inference fails")

    print("  Installing huggingface_hub and openai-whisper...")
    pip("huggingface_hub", "openai-whisper")

    # ── 3. Download HuggingFace individual files ───────────────────────────────
    print("\n[3/5] Downloading model weights from HuggingFace...")
    for repo_id, filename, local_rel in HF_FILES:
        download_hf_file(repo_id, filename, musetalk_dir / local_rel)

    for repo_id, local_subdir in HF_REPOS:
        download_hf_repo(repo_id, musetalk_dir / local_subdir)

    # ── 4. Whisper + face parse ────────────────────────────────────────────────
    print("\n[4/5] Downloading Whisper + face-parse weights...")
    download_whisper(musetalk_dir / "models" / "whisper" / "tiny.pt")
    download_face_parse(musetalk_dir)

    # ── 5. Verify ──────────────────────────────────────────────────────────────
    print("\n[5/5] Verifying installation...")
    ok = verify(musetalk_dir)

    print("\n" + "=" * 60)
    if ok:
        print("Setup complete. Run smoke test:")
        print("  python scripts/smoke_test_musetalk.py --n_clips 3")
    else:
        print("Some files missing — check FAILED lines above.")
        print("Re-run this script after fixing network/auth issues.")
    print("=" * 60)


if __name__ == "__main__":
    main()
