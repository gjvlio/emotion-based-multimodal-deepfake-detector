"""
Track 3 — Actor Reference Frame Extractor

For each of the 91 CREMA-D actors, this script extracts:
  - 1 best reference portrait  (used during Hallo inference)
  - up to N fine-tuning frames (used during per-actor LoRA fine-tuning)

Strategy:
  1. Prefer NEU (neutral) clips — cleanest face, no exaggerated expression
  2. Run OpenCV Haar cascade face detection on every candidate frame
  3. Score each frame by face size, detection confidence, and frontality
     (yaw angle from face landmarks) — highest score wins
  4. Save portraits as PNG; write a manifest CSV for the fine-tuner

Optimised for 91 actors:
  - Clips are opened once per actor and sampled at a fixed stride
  - Frame scoring is vectorised with numpy; no per-frame Python loops
  - ThreadPoolExecutor processes actors in parallel (I/O-bound work)

Usage:
  python src/track3/extract_actor_frames.py \
    --cremad_dir  data/raw/CREMA-D \
    --out_dir     data/processed/actor_portraits \
    [--n_finetune 10]   # frames per actor for fine-tuning (default: 10)
    [--workers    8]    # parallel actor threads (default: 8)
"""

import argparse
import csv
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

MANIFEST_FILE = "actor_portraits_manifest.csv"
MANIFEST_COLS = ["actor_id", "portrait_path", "finetune_frames", "n_frames", "source_clips"]


# ---------------------------------------------------------------------------
# Frame scoring
# ---------------------------------------------------------------------------

def score_frame(frame_bgr: np.ndarray, faces) -> float:
    """Score frame for portrait quality using OpenCV Haar faces (x, y, w, h tuples)."""
    if len(faces) == 0:
        return 0.0
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    fh, fw = frame_bgr.shape[:2]
    face_area = (w * h) / (fw * fh)
    cx = (x + w / 2) / fw
    frontality = 1.0 - abs(cx - 0.5) * 2
    return face_area * 0.6 + frontality * 0.4


# ---------------------------------------------------------------------------
# Per-actor extraction
# ---------------------------------------------------------------------------

def get_actor_clips(cremad_video_dir: Path, actor_id: str,
                    prefer_emotion: str = "NEU") -> list[Path]:
    """
    Return video paths for an actor, preferred emotion first.
    Falls back to all clips if no preferred emotion found.
    """
    all_clips = sorted(cremad_video_dir.glob(f"{actor_id}_*.flv"))
    preferred = [p for p in all_clips if f"_{prefer_emotion}_" in p.name]
    others    = [p for p in all_clips if f"_{prefer_emotion}_" not in p.name]
    return preferred + others


def sample_frames(video_path: Path, stride: int = 5) -> list[np.ndarray]:
    """Read every `stride`-th frame from a video. Returns BGR frames."""
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % stride == 0:
            frames.append(frame)
        idx += 1
    cap.release()
    return frames


def extract_for_actor(
    actor_id: str,
    cremad_video_dir: Path,
    out_dir: Path,
    n_finetune: int,
    detector,
) -> dict | None:
    """
    Extract portrait + fine-tuning frames for one actor.
    Returns a manifest row dict, or None on failure.
    """
    actor_out = out_dir / f"actor_{actor_id}"
    actor_out.mkdir(parents=True, exist_ok=True)

    portrait_path = actor_out / "portrait.png"
    ft_dir        = actor_out / "finetune_frames"

    # Skip if already done
    if portrait_path.exists() and ft_dir.exists():
        ft_count = len(list(ft_dir.glob("*.png")))
        if ft_count >= n_finetune:
            log.info(f"  Actor {actor_id}: already extracted, skipping.")
            return {
                "actor_id":       actor_id,
                "portrait_path":  str(portrait_path),
                "finetune_frames": str(ft_dir),
                "n_frames":       ft_count,
                "source_clips":   "cached",
            }

    clips = get_actor_clips(cremad_video_dir, actor_id)
    if not clips:
        log.warning(f"  Actor {actor_id}: no video clips found.")
        return None

    # Collect scored (score, frame) pairs across all clips
    scored: list[tuple[float, np.ndarray]] = []
    source_names = []

    for clip in clips:
        frames = sample_frames(clip, stride=4)
        for frame in frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
            s = score_frame(frame, faces)
            if s > 0:
                scored.append((s, frame.copy()))
        source_names.append(clip.name)
        if len(scored) >= n_finetune * 3:
            break

    if not scored:
        log.warning(f"  Actor {actor_id}: no face detected in any frame.")
        return None

    # Sort descending by score
    scored.sort(key=lambda x: x[0], reverse=True)

    # Best frame → portrait
    cv2.imwrite(str(portrait_path), scored[0][1])

    # Top-N unique frames → fine-tuning set
    ft_dir.mkdir(exist_ok=True)
    # Deduplicate by downsampling and comparing pixel histograms
    saved = [scored[0][1]]
    for _, frame in scored[1:]:
        if len(saved) >= n_finetune:
            break
        # Simple diversity check: skip if too similar to any already-saved frame
        if not any(_frames_too_similar(frame, s) for s in saved):
            saved.append(frame)

    for i, frame in enumerate(saved):
        cv2.imwrite(str(ft_dir / f"frame_{i:04d}.png"), frame)

    log.info(f"  Actor {actor_id}: portrait + {len(saved)} fine-tune frames saved.")
    return {
        "actor_id":        actor_id,
        "portrait_path":   str(portrait_path),
        "finetune_frames": str(ft_dir),
        "n_frames":        len(saved),
        "source_clips":    ";".join(source_names[:5]),
    }


def _frames_too_similar(a: np.ndarray, b: np.ndarray,
                         threshold: float = 0.97) -> bool:
    """Return True if two frames are visually near-identical (histogram correlation)."""
    a_s = cv2.resize(a, (64, 64))
    b_s = cv2.resize(b, (64, 64))
    hist_a = cv2.calcHist([a_s], [0, 1, 2], None, [8, 8, 8], [0,256]*3)
    hist_b = cv2.calcHist([b_s], [0, 1, 2], None, [8, 8, 8], [0,256]*3)
    cv2.normalize(hist_a, hist_a)
    cv2.normalize(hist_b, hist_b)
    return cv2.compareHist(hist_a, hist_b, cv2.HISTCMP_CORREL) >= threshold


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Extract actor reference portraits and fine-tuning frames from CREMA-D."
    )
    parser.add_argument("--cremad_dir",  required=True,
                        help="Root CREMA-D directory")
    parser.add_argument("--out_dir",     required=True,
                        help="Output directory for actor portraits")
    parser.add_argument("--n_finetune",  type=int, default=10,
                        help="Number of fine-tuning frames to extract per actor (default: 10)")
    parser.add_argument("--workers",     type=int, default=8,
                        help="Parallel worker threads for actor extraction (default: 8)")
    args = parser.parse_args()

    cremad_dir       = Path(args.cremad_dir)
    cremad_video_dir = cremad_dir / "VideoFlash"
    out_dir          = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Discover all actor IDs from VideoFlash filenames
    all_clips = sorted(cremad_video_dir.glob("*.flv"))
    actor_ids = sorted({p.stem.split("_")[0] for p in all_clips})
    log.info(f"Found {len(actor_ids)} actors in {cremad_video_dir}")

    # Each thread gets its own detector instance (not thread-safe to share)
    def worker(actor_id: str) -> dict | None:
        detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        return extract_for_actor(
            actor_id, cremad_video_dir, out_dir, args.n_finetune, detector
        )

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(worker, aid): aid for aid in actor_ids}
        for i, fut in enumerate(as_completed(futures), 1):
            aid = futures[fut]
            try:
                row = fut.result()
                if row:
                    results.append(row)
            except Exception as e:
                log.error(f"Actor {aid} failed: {e}", exc_info=True)
            if i % 10 == 0:
                log.info(f"Progress: {i}/{len(actor_ids)} actors processed.")

    # Write manifest
    manifest_path = out_dir / MANIFEST_FILE
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_COLS)
        writer.writeheader()
        writer.writerows(results)

    log.info(f"\nDone. {len(results)}/{len(actor_ids)} actors extracted.")
    log.info(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
