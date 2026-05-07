"""
visual.py — Visual feature extraction: face detection + ViT keyframe embedding.

extract_frames(video_path, fps)          → list of BGR numpy arrays
detect_and_align_faces(frames, detector) → list of (aligned_frame, score)
get_z_v(video_path, ...)                 → (768,) tensor from mean-pooled ViT
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

from .filters import coarse_has_face, sharpness_score, select_keyframes, frames_to_pil

log = logging.getLogger(__name__)

_vit_model     = None
_vit_processor = None


def _load_vit(model_name: str = "google/vit-base-patch16-224") -> Tuple:
    global _vit_model, _vit_processor
    if _vit_model is None:
        from transformers import ViTModel, ViTImageProcessor
        log.info(f"Loading ViT: {model_name}")
        _vit_processor = ViTImageProcessor.from_pretrained(model_name)
        _vit_model     = ViTModel.from_pretrained(model_name)
        _vit_model.eval()
    return _vit_model, _vit_processor


# ── Frame extraction ───────────────────────────────────────────────────────────

def extract_frames(
    video_path: str | Path,
    target_fps: float = 3.0,
) -> List[np.ndarray]:
    """
    Read video and sample frames at target_fps.
    Returns list of BGR numpy arrays (H, W, 3).
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        log.warning(f"Cannot open video: {video_path}")
        return []

    native_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    interval   = max(1, int(round(native_fps / target_fps)))
    frames, idx = [], 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % interval == 0:
            frames.append(frame)
        idx += 1
    cap.release()
    return frames


# ── Face detection & alignment ─────────────────────────────────────────────────

def _retinaface_detect(frames: List[np.ndarray]) -> List[Tuple[np.ndarray, float]]:
    """Use RetinaFace to crop + align face. Returns (cropped_frame, score) pairs."""
    try:
        from retinaface import RetinaFace
    except ImportError:
        log.warning("retinaface not installed — falling back to Haar cascade crop.")
        return _haar_fallback(frames)

    results = []
    for frame in frames:
        try:
            detections = RetinaFace.detect_faces(frame)
            if not detections:
                continue
            best = max(detections.values(), key=lambda d: d["score"])
            x1, y1, x2, y2 = best["facial_area"]
            crop = frame[max(0, y1):y2, max(0, x1):x2]
            if crop.size == 0:
                continue
            score = best["score"] * sharpness_score(crop)
            results.append((crop, score))
        except Exception as e:
            log.debug(f"RetinaFace error: {e}")
    return results


def _haar_fallback(frames: List[np.ndarray]) -> List[Tuple[np.ndarray, float]]:
    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    results = []
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, 1.1, 4)
        if len(faces) == 0:
            continue
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        crop  = frame[y:y+h, x:x+w]
        score = sharpness_score(crop)
        results.append((crop, score))
    return results


def detect_and_align_faces(
    frames: List[np.ndarray],
    detector: str = "retinaface",
) -> List[Tuple[np.ndarray, float]]:
    """
    Coarse-filter then run face detector.
    Returns list of (face_crop_bgr, quality_score).
    """
    candidates = [f for f in frames if coarse_has_face(f)]
    if not candidates:
        candidates = frames  # skip coarse filter if nothing passes

    if detector == "retinaface":
        return _retinaface_detect(candidates)
    return _haar_fallback(candidates)


# ── ViT embedding ──────────────────────────────────────────────────────────────

def get_z_v(
    video_path: str | Path,
    vit_model_name: str = "google/vit-base-patch16-224",
    detector: str = "retinaface",
    n_keyframes: int = 8,
    frame_size:  int = 224,
    device:      str = "cpu",
) -> torch.Tensor:
    """
    Full visual pipeline: extract frames → detect faces → select keyframes → ViT.
    Returns mean-pooled CLS token embedding: (768,) float32.
    Falls back to zero vector if no faces detected.
    """
    model, processor = _load_vit(vit_model_name)
    model = model.to(device)

    frames = extract_frames(video_path)
    if not frames:
        log.warning(f"No frames extracted from {video_path}")
        return torch.zeros(768)

    face_results = detect_and_align_faces(frames, detector)
    if not face_results:
        log.warning(f"No faces detected in {video_path} — using raw frames.")
        face_results = [(f, sharpness_score(f)) for f in frames]

    crops  = [r[0] for r in face_results]
    scores = [r[1] for r in face_results]
    keyframes = select_keyframes(crops, scores, k=n_keyframes)
    pil_imgs  = frames_to_pil(keyframes, size=frame_size)

    inputs = processor(images=pil_imgs, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        out = model(**inputs)
    cls_tokens = out.last_hidden_state[:, 0, :]   # (K, 768)
    z_v = cls_tokens.mean(dim=0).cpu()            # (768,)
    return z_v
