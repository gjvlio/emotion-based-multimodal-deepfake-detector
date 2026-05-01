"""
Track 3 — Per-Actor Hallo LoRA Fine-Tuning

Fine-tunes Hallo's ReferenceNet (identity encoder) with a per-actor LoRA
adapter so that generated face videos closely resemble the actual CREMA-D actor.

Without fine-tuning, Hallo generates a plausible face from the reference image
but identity drift is noticeable. With a LoRA trained on 10–20 actor frames,
the generated face matches the actor's skin tone, facial structure, and features
well enough to be convincing.

Pipeline per actor:
  1. Load the actor's fine-tuning frames from data/processed/actor_portraits/
  2. Run LoRA training on Hallo's ReferenceNet for N steps
  3. Save the LoRA weights to tools/Hallo/lora/actor_XXXX.safetensors
  4. Log result to training_log.csv

Optimisations for 91 actors:
  - LoRA rank 16 — low memory, fast convergence, small adapter files (~4 MB each)
  - Mixed precision (fp16) throughout
  - Gradient checkpointing to fit within 8 GB VRAM
  - Resume support — skips actors with existing LoRA weights

Usage:
  python src/track3/finetune_hallo.py \
    --portraits_dir  data/processed/actor_portraits \
    --hallo_dir      tools/Hallo \
    --out_dir        tools/Hallo/lora \
    [--steps         200]    # training steps per actor (default: 200)
    [--rank          16]     # LoRA rank (default: 16)
    [--resume]               # skip actors that already have a .safetensors file
"""

import argparse
import csv
import logging
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

LOG_FILE = "finetune_log.csv"
LOG_COLS  = ["actor_id", "trained", "steps", "rank", "lora_path", "error", "timestamp"]


# ---------------------------------------------------------------------------
# LoRA training launcher
# ---------------------------------------------------------------------------

def train_actor_lora(
    actor_id: str,
    ft_frames_dir: Path,
    hallo_dir: Path,
    out_dir: Path,
    steps: int,
    rank: int,
) -> tuple[bool, str]:
    """
    Launch Hallo's LoRA fine-tuning script for a single actor.

    Hallo exposes a train_stage1.py / train_stage2.py interface; we call the
    stage-2 reference-net trainer which targets the identity encoder.

    Returns (success: bool, error_msg: str).
    """
    lora_path = out_dir / f"actor_{actor_id}.safetensors"
    train_script = hallo_dir / "scripts" / "train_stage2.py"

    if not train_script.exists():
        return False, f"Hallo train script not found: {train_script}"

    frame_paths = sorted(ft_frames_dir.glob("*.png"))
    if not frame_paths:
        return False, f"No fine-tuning frames found in {ft_frames_dir}"

    # Build image list file expected by Hallo's dataloader
    img_list_path = out_dir / f"actor_{actor_id}_imgs.txt"
    img_list_path.write_text("\n".join(str(p) for p in frame_paths))

    cmd = [
        sys.executable, str(train_script),
        "--image_list",    str(img_list_path),
        "--output_path",   str(lora_path),
        "--lora_rank",     str(rank),
        "--max_train_steps", str(steps),
        "--mixed_precision", "fp16",
        "--gradient_checkpointing",
        "--pretrained_model_path", str(hallo_dir / "pretrained_models"),
    ]

    try:
        result = subprocess.run(
            cmd,
            cwd=str(hallo_dir),
            capture_output=True, text=True, timeout=3600,
        )
        if result.returncode != 0:
            return False, result.stderr[-1000:]
        return True, ""
    except subprocess.TimeoutExpired:
        return False, "Training timed out after 60 min"
    except Exception as e:
        return False, str(e)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Per-actor Hallo LoRA fine-tuning for Track 3."
    )
    parser.add_argument("--portraits_dir", required=True,
                        help="Directory produced by extract_actor_frames.py")
    parser.add_argument("--hallo_dir",     required=True,
                        help="Path to cloned Hallo repository")
    parser.add_argument("--out_dir",       required=True,
                        help="Output directory for per-actor LoRA .safetensors files")
    parser.add_argument("--steps",  type=int, default=200,
                        help="LoRA training steps per actor (default: 200)")
    parser.add_argument("--rank",   type=int, default=16,
                        help="LoRA rank (default: 16, lower = faster/smaller)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip actors that already have a .safetensors file")
    args = parser.parse_args()

    portraits_dir = Path(args.portraits_dir)
    hallo_dir     = Path(args.hallo_dir)
    out_dir       = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Read manifest produced by extract_actor_frames.py
    manifest_path = portraits_dir / "actor_portraits_manifest.csv"
    if not manifest_path.exists():
        log.error(f"Manifest not found at {manifest_path}. Run extract_actor_frames.py first.")
        sys.exit(1)

    with open(manifest_path, newline="", encoding="utf-8") as f:
        actors = list(csv.DictReader(f))

    log.info(f"Found {len(actors)} actors in manifest. Steps/actor={args.steps}, rank={args.rank}")

    # Resume: skip actors already trained
    if args.resume:
        actors = [
            a for a in actors
            if not (out_dir / f"actor_{a['actor_id']}.safetensors").exists()
        ]
        log.info(f"After resume filter: {len(actors)} actors remaining.")

    log_path = out_dir / LOG_FILE
    log_exists = log_path.exists()
    log_f   = open(log_path, "a", newline="", encoding="utf-8")
    writer  = csv.DictWriter(log_f, fieldnames=LOG_COLS)
    if not log_exists:
        writer.writeheader()

    succeeded = 0
    failed    = 0

    for i, actor in enumerate(actors, 1):
        actor_id   = actor["actor_id"]
        ft_dir     = Path(actor["finetune_frames"])
        lora_path  = out_dir / f"actor_{actor_id}.safetensors"

        log.info(f"[{i}/{len(actors)}] Fine-tuning actor {actor_id} ...")

        if not ft_dir.exists():
            msg = f"Fine-tuning frames directory missing: {ft_dir}"
            log.warning(f"  {msg}")
            writer.writerow({
                "actor_id": actor_id, "trained": False,
                "steps": 0, "rank": args.rank,
                "lora_path": "", "error": msg,
                "timestamp": datetime.now().isoformat(),
            })
            failed += 1
            continue

        ok, err = train_actor_lora(
            actor_id, ft_dir, hallo_dir, out_dir, args.steps, args.rank
        )

        writer.writerow({
            "actor_id":  actor_id,
            "trained":   ok,
            "steps":     args.steps if ok else 0,
            "rank":      args.rank,
            "lora_path": str(lora_path) if ok else "",
            "error":     err,
            "timestamp": datetime.now().isoformat(),
        })
        log_f.flush()

        if ok:
            succeeded += 1
            log.info(f"  Saved LoRA → {lora_path}")
        else:
            failed += 1
            log.error(f"  Failed: {err[:200]}")

    log_f.close()
    log.info(f"\nDone. Succeeded: {succeeded}  Failed: {failed}  Total: {len(actors)}")


if __name__ == "__main__":
    main()
