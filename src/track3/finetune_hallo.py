"""
Track 3 — Per-Actor Hallo Fine-Tuning

Fine-tunes Hallo's audio and motion modules on per-actor CREMA-D video clips
so the generated face better matches the actor's identity and mannerisms.

Pipeline per actor:
  1. Select N short video clips for the actor from CREMA-D (NEU preferred)
  2. Convert FLV → MP4 (Hallo data_preprocess.py only handles .mp4)
  3. Run Hallo's data_preprocess.py step 1 — extract face masks + embeddings
  4. Run Hallo's data_preprocess.py step 2 — extract audio embeddings
  5. Run extract_meta_info_stage2.py — build the training JSON metadata file
  6. Write a per-actor stage2 config YAML (reduced steps, actor output dir)
  7. Run train_stage2.py --config <actor_config.yaml>

Prerequisite: the pretrained Hallo model must supply individual component
weights in pretrained_models/hallo/ (reference_unet.pth, denoising_unet.pth,
face_locator.pth, imageproj.pth). The full net.pth from HuggingFace is a
combined checkpoint. Run src/track3/extract_hallo_components.py once to
split it into the per-component files that train_stage2.py expects.

Note: Fine-tuning ~3 actors on an RTX 3060 takes ~1-2 hours. Start with
test actors (1001-1003) to validate, then scale to all 91.

Usage:
  python src/track3/finetune_hallo.py \
    --cremad_dir      data/raw/CREMA-D \
    --hallo_dir       tools/Hallo \
    --out_dir         data/processed/hallo_finetune \
    [--actors         1001 1002 1003]   # specific actors; default = all
    [--steps          200]              # training steps per actor (default: 200)
    [--clips_per_actor 5]               # CREMA-D clips per actor (default: 5)
    [--resume]                          # skip actors with existing checkpoint
"""

import argparse
import csv
import logging
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

LOG_FILE = "finetune_log.csv"
LOG_COLS = ["actor_id", "trained", "steps", "ckpt_dir", "error", "timestamp"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_actor_clips(video_dir: Path, actor_id: str,
                    n: int, prefer_emotion: str = "NEU") -> list[Path]:
    all_clips = sorted(video_dir.glob(f"{actor_id}_*.flv"))
    preferred = [p for p in all_clips if f"_{prefer_emotion}_" in p.name]
    others    = [p for p in all_clips if f"_{prefer_emotion}_" not in p.name]
    return (preferred + others)[:n]


def convert_flv_to_mp4(flv_path: Path, mp4_dir: Path) -> Path | None:
    mp4_path = mp4_dir / (flv_path.stem + ".mp4")
    if mp4_path.exists():
        return mp4_path
    r = subprocess.run(
        ["ffmpeg", "-y", "-i", str(flv_path),
         "-c:v", "libx264", "-c:a", "aac", "-ar", "16000", str(mp4_path)],
        capture_output=True, text=True,
    )
    return mp4_path if r.returncode == 0 else None


def run_data_preprocess(hallo_dir: Path, video_dir: Path,
                        out_dir: Path, step: int) -> bool:
    script = hallo_dir / "scripts" / "data_preprocess.py"
    r = subprocess.run(
        [sys.executable, str(script),
         "-i", str(video_dir), "-o", str(out_dir), "-s", str(step)],
        cwd=str(hallo_dir),
        capture_output=True, text=True, timeout=3600,
    )
    if r.returncode != 0:
        log.error(f"  data_preprocess step {step} failed:\n{r.stderr[-800:]}")
    return r.returncode == 0


def build_meta_json(hallo_dir: Path, data_dir: Path, meta_json_path: Path) -> bool:
    script = hallo_dir / "scripts" / "extract_meta_info_stage2.py"
    r = subprocess.run(
        [sys.executable, str(script),
         "--root_path",      str(data_dir),
         "--dataset_name",   "actor_ft",
         "--meta_info_name", meta_json_path.stem],
        cwd=str(hallo_dir),
        capture_output=True, text=True, timeout=600,
    )
    if r.returncode != 0:
        log.error(f"  extract_meta_info_stage2 failed:\n{r.stderr[-800:]}")
        return False
    candidate = data_dir / (meta_json_path.stem + ".json")
    if candidate.exists() and candidate != meta_json_path:
        shutil.move(str(candidate), str(meta_json_path))
    return meta_json_path.exists()


def write_actor_config(hallo_dir: Path, meta_json_path: Path,
                       actor_ckpt_dir: Path, steps: int) -> Path:
    base_yaml = hallo_dir / "configs" / "train" / "stage2.yaml"
    with open(base_yaml) as f:
        cfg = yaml.safe_load(f)

    cfg["data"]["train_meta_paths"]         = [str(meta_json_path)]
    cfg["data"]["train_bs"]                 = 1
    cfg["solver"]["max_train_steps"]        = steps
    cfg["solver"]["mixed_precision"]        = "fp16"
    cfg["solver"]["gradient_checkpointing"] = True
    cfg["solver"]["use_8bit_adam"]          = True
    cfg["output_dir"]                       = str(actor_ckpt_dir)
    cfg["exp_name"]                         = "actor_ft"
    cfg["checkpointing_steps"]              = max(steps, 100)
    cfg["stage1_ckpt_dir"]                  = "./pretrained_models/hallo"

    actor_cfg_path = actor_ckpt_dir / "train_config.yaml"
    actor_cfg_path.parent.mkdir(parents=True, exist_ok=True)
    with open(actor_cfg_path, "w") as f:
        yaml.dump(cfg, f)
    return actor_cfg_path


def run_training(hallo_dir: Path, actor_cfg_path: Path) -> tuple[bool, str]:
    script = hallo_dir / "scripts" / "train_stage2.py"
    try:
        r = subprocess.run(
            [sys.executable, str(script), "--config", str(actor_cfg_path)],
            cwd=str(hallo_dir),
            capture_output=True, text=True, timeout=7200,
        )
        if r.returncode != 0:
            return False, r.stderr[-1000:]
        return True, ""
    except subprocess.TimeoutExpired:
        return False, "Training timed out (2 h)"
    except Exception as e:
        return False, str(e)


# ---------------------------------------------------------------------------
# Per-actor orchestration
# ---------------------------------------------------------------------------

def finetune_actor(actor_id: str, args, hallo_dir: Path,
                   video_dir: Path, out_dir: Path) -> tuple[bool, str]:
    actor_work = out_dir / f"actor_{actor_id}"
    mp4_dir    = actor_work / "mp4s"
    proc_dir   = actor_work / "processed"
    meta_json  = actor_work / "meta_stage2.json"
    actor_ckpt = actor_work / "checkpoint"
    mp4_dir.mkdir(parents=True, exist_ok=True)
    proc_dir.mkdir(parents=True, exist_ok=True)

    clips = get_actor_clips(video_dir, actor_id, args.clips_per_actor)
    if not clips:
        return False, f"No video clips found for actor {actor_id}"

    log.info(f"  Converting {len(clips)} FLV clips to MP4 ...")
    ok_mp4s = [convert_flv_to_mp4(c, mp4_dir) for c in clips]
    ok_mp4s = [p for p in ok_mp4s if p]
    if not ok_mp4s:
        return False, "All FLV→MP4 conversions failed"

    log.info("  data_preprocess step 1 (face masks + embeddings) ...")
    if not run_data_preprocess(hallo_dir, mp4_dir, proc_dir, step=1):
        return False, "data_preprocess step 1 failed"

    log.info("  data_preprocess step 2 (audio embeddings) ...")
    if not run_data_preprocess(hallo_dir, mp4_dir, proc_dir, step=2):
        return False, "data_preprocess step 2 failed"

    log.info("  Building training metadata JSON ...")
    if not build_meta_json(hallo_dir, proc_dir, meta_json):
        return False, "Metadata JSON creation failed"

    log.info(f"  Writing per-actor config (steps={args.steps}) ...")
    actor_cfg = write_actor_config(hallo_dir, meta_json, actor_ckpt, args.steps)

    log.info("  Launching train_stage2.py ...")
    return run_training(hallo_dir, actor_cfg)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Per-actor Hallo fine-tuning for Track 3."
    )
    parser.add_argument("--cremad_dir",       required=True)
    parser.add_argument("--hallo_dir",        required=True)
    parser.add_argument("--out_dir",          required=True)
    parser.add_argument("--actors",           nargs="*",
                        help="Actor IDs to fine-tune (default: all in CREMA-D)")
    parser.add_argument("--steps",            type=int, default=200)
    parser.add_argument("--clips_per_actor",  type=int, default=5)
    parser.add_argument("--resume",           action="store_true")
    args = parser.parse_args()

    hallo_dir = Path(args.hallo_dir)
    video_dir = Path(args.cremad_dir) / "VideoFlash"
    out_dir   = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Verify component weights exist for train_stage2.py
    stage1_ckpt = hallo_dir / "pretrained_models" / "hallo"
    required = ["reference_unet.pth", "denoising_unet.pth",
                "face_locator.pth", "imageproj.pth"]
    missing = [f for f in required if not (stage1_ckpt / f).exists()]
    if missing:
        log.error(
            f"Missing component weights in {stage1_ckpt}: {missing}\n"
            "Run: python src/track3/extract_hallo_components.py "
            "--net_pth tools/Hallo/pretrained_models/hallo/net.pth "
            "--out_dir tools/Hallo/pretrained_models/hallo"
        )
        sys.exit(1)

    if args.actors:
        actor_ids = args.actors
    else:
        all_clips = sorted(video_dir.glob("*.flv"))
        actor_ids = sorted({p.stem.split("_")[0] for p in all_clips})
    log.info(f"Actors to fine-tune: {len(actor_ids)}")

    if args.resume:
        actor_ids = [
            a for a in actor_ids
            if not (out_dir / f"actor_{a}" / "checkpoint").exists()
        ]
        log.info(f"After resume filter: {len(actor_ids)} remaining.")

    log_path   = out_dir / LOG_FILE
    log_exists = log_path.exists()
    log_f      = open(log_path, "a", newline="", encoding="utf-8")
    writer     = csv.DictWriter(log_f, fieldnames=LOG_COLS)
    if not log_exists:
        writer.writeheader()

    succeeded = 0
    failed    = 0

    for i, actor_id in enumerate(actor_ids, 1):
        log.info(f"[{i}/{len(actor_ids)}] Fine-tuning actor {actor_id} ...")
        ok, err = finetune_actor(actor_id, args, hallo_dir, video_dir, out_dir)

        ckpt_dir = str(out_dir / f"actor_{actor_id}" / "checkpoint") if ok else ""
        writer.writerow({
            "actor_id":  actor_id,
            "trained":   ok,
            "steps":     args.steps if ok else 0,
            "ckpt_dir":  ckpt_dir,
            "error":     err,
            "timestamp": datetime.now().isoformat(),
        })
        log_f.flush()

        if ok:
            succeeded += 1
            log.info(f"  Checkpoint → {ckpt_dir}")
        else:
            failed += 1
            log.error(f"  Failed: {err[:200]}")

    log_f.close()
    log.info(f"\nDone. Succeeded: {succeeded}  Failed: {failed}  Total: {len(actor_ids)}")


if __name__ == "__main__":
    main()
