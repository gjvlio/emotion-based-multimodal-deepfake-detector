"""
filters.py — Face quality filtering for keyframe selection.

Coarse filter:  OpenCV Haar cascade to quickly drop frame with no face.
Fine filter:    Select top-K frames by (RetinaFace confidence × sharpness score).
"""
from __future__ import annotations

import logging
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image

log = logging.getLogger(__name__)

_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
_cascade: cv2.CascadeClassifier | None = None


def _get_cascade() -> cv2.CascadeClassifier:
    global _cascade
    if _cascade is None:
        _cascade = cv2.CascadeClassifier(_CASCADE_PATH)
    return _cascade


def coarse_has_face(frame: np.ndarray) -> bool:
    """True if Haar cascade finds at least one face in the BGR frame."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = _get_cascade().detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
    return len(faces) > 0


def sharpness_score(frame: np.ndarray) -> float:
    """Laplacian variance — higher = sharper."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def select_keyframes(
    frames: List[np.ndarray],
    scores: List[float],
    k: int = 8,
    score_threshold: float = 0.0,
) -> List[np.ndarray]:
    """
    Return top-K frames ranked by (face confidence × sharpness).
    Frames with score < score_threshold are discarded before ranking.
    Falls back to all frames if nothing passes the threshold.
    """
    if not frames:
        return []
    pairs = list(zip(scores, frames))
    if score_threshold > 0.0:
        filtered = [(s, f) for s, f in pairs if s >= score_threshold]
        if not filtered:
            filtered = pairs   # fallback: threshold too strict, use all
    else:
        filtered = pairs
    ranked = sorted(filtered, key=lambda x: x[0], reverse=True)
    selected = [f for _, f in ranked[:k]]
    while len(selected) < k:
        selected.append(selected[-1])
    return selected[:k]


def frames_to_pil(frames: List[np.ndarray], size: int = 224) -> List[Image.Image]:
    """Convert BGR numpy frames to PIL Images resized to (size × size)."""
    result = []
    for f in frames:
        rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb).resize((size, size), Image.BILINEAR)
        result.append(img)
    return result
