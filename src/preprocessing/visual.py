"""
visual.py — Visual feature extraction: face detection + ViT keyframe embedding.

extract_frames(video_path, fps)          → list of BGR numpy arrays (25 fps default)
optical_flow_gate(frames, threshold)     → motion-filtered frame subset
detect_and_align_faces(frames, detector) → list of (aligned_frame, score)
get_z_v(video_path, ...)                 → (768,) tensor from AU-saliency guided Top-8 ViT

Keyframe scoring: score = insightface_conf × sharpness × AU_saliency
AU_saliency = sum of FACS AU intensities (py-feat). Prioritizes frames with active
facial muscle movement (microexpression-relevant) over merely sharp/confident frames.
Falls back to conf × sharpness if py-feat unavailable.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

from .filters import coarse_has_face, sharpness_score, select_keyframes, frames_to_pil

log = logging.getLogger(__name__)

_vit_model       = None
_vit_processor   = None
_feat_detector   = None
_insightface_app = None

# Check once at import — avoid per-frame spam
try:
    import feat as _feat_pkg  # noqa: F401
    _FEAT_AVAILABLE = True
except ImportError:
    _FEAT_AVAILABLE = False
    log.warning("py-feat not installed — AU saliency unavailable. Keyframe scoring: conf × sharpness only. Install: pip install feat")

try:
    import insightface as _insightface_pkg  # noqa: F401
    _INSIGHTFACE_AVAILABLE = True
except ImportError:
    _INSIGHTFACE_AVAILABLE = False
    log.warning("insightface not installed — face detection will use Haar cascade fallback. Install: pip install insightface")


def _load_vit(model_name: str = "google/vit-base-patch16-224") -> Tuple:
    global _vit_model, _vit_processor
    if _vit_model is None:
        from transformers import ViTModel, ViTImageProcessor
        log.info(f"Loading ViT: {model_name}")
        _vit_processor = ViTImageProcessor.from_pretrained(model_name)
        _vit_model     = ViTModel.from_pretrained(model_name)
        _vit_model.eval()
    return _vit_model, _vit_processor


def _load_feat_detector():
    global _feat_detector
    if _feat_detector is None:
        from feat import Detector
        log.info("Loading py-feat AU Detector")
        _feat_detector = Detector(au_model="xgb", device="cpu")
    return _feat_detector


def _au_saliency(crop: np.ndarray) -> float:
    """
    Sum of FACS AU intensities for one face crop.
    Higher = more facial muscle activity = more expression-relevant.
    Returns 1.0 on failure so score degrades to conf × sharpness.
    """
    if not _FEAT_AVAILABLE:
        return 1.0
    try:
        det = _load_feat_detector()
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        result = det.detect_image(pil)
        if result is not None and not result.empty:
            au_cols = [c for c in result.columns if c.startswith("AU")]
            if au_cols:
                return float(result[au_cols].values[0].sum())
    except Exception as e:
        log.debug(f"AU saliency error: {e}")
    return 1.0


# ── Frame extraction ───────────────────────────────────────────────────────────

def extract_frames(
    video_path: str | Path,
    target_fps: float = 25.0,
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


# ── Optical flow motion gate ───────────────────────────────────────────────────

def optical_flow_gate(
    frames: List[np.ndarray],
    motion_threshold: float = 0.3,
) -> List[np.ndarray]:
    """
    Keep frames where mean optical flow magnitude >= motion_threshold.
    Retains first frame unconditionally. Falls back to all frames if
    nothing passes (fully static clip).
    """
    if len(frames) < 2:
        return frames

    gated = [frames[0]]
    prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)

    for frame in frames[1:]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, gray, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
        )
        magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2).mean()
        if magnitude >= motion_threshold:
            gated.append(frame)
        prev_gray = gray

    return gated if len(gated) > 1 else frames


# ── Face detection & alignment ─────────────────────────────────────────────────

def _load_insightface_app():
    global _insightface_app
    if _insightface_app is None:
        from insightface.app import FaceAnalysis
        _insightface_app = FaceAnalysis(
            name="buffalo_s",
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        _insightface_app.prepare(ctx_id=0, det_size=(640, 640))
    return _insightface_app


def _insightface_detect(
    frames: List[np.ndarray],
    confidence_threshold: float = 0.7,
) -> List[Tuple[np.ndarray, float]]:
    """
    insightface (ONNX RetinaFace) detection with AU-saliency weighted scoring.
    score = conf × sharpness × AU_saliency
    Only keeps detections with conf >= confidence_threshold.
    """
    if not _INSIGHTFACE_AVAILABLE:
        return _haar_fallback(frames)
    try:
        app = _load_insightface_app()
    except Exception as e:
        log.warning(f"insightface load failed: {e} — falling back to Haar cascade")
        return _haar_fallback(frames)

    results = []
    for frame in frames:
        try:
            faces = app.get(frame)
            if not faces:
                continue
            best = max(faces, key=lambda f: f.det_score)
            if best.det_score < confidence_threshold:
                continue
            x1, y1, x2, y2 = best.bbox.astype(int)
            crop = frame[max(0, y1):y2, max(0, x1):x2]
            if crop.size == 0:
                continue
            au_sal = _au_saliency(crop)
            score  = float(best.det_score) * sharpness_score(crop) * au_sal
            results.append((crop, score))
        except Exception as e:
            log.debug(f"insightface error: {e}")
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
        crop    = frame[y:y+h, x:x+w]
        au_sal  = _au_saliency(crop)
        score   = sharpness_score(crop) * au_sal
        results.append((crop, score))
    return results


def detect_and_align_faces(
    frames: List[np.ndarray],
    detector: str = "retinaface",
    confidence_threshold: float = 0.7,
) -> List[Tuple[np.ndarray, float]]:
    """
    Coarse-filter then run face detector.
    Returns list of (face_crop_bgr, AU-saliency-weighted quality_score).
    """
    candidates = [f for f in frames if coarse_has_face(f)]
    if not candidates:
        candidates = frames

    if detector == "retinaface":
        return _insightface_detect(candidates, confidence_threshold)
    return _haar_fallback(candidates)


# ── ViT embedding ──────────────────────────────────────────────────────────────

def get_z_v(
    video_path: str | Path,
    vit_model_name:       str   = "google/vit-base-patch16-224",
    detector:             str   = "retinaface",
    n_keyframes:          int   = 8,
    frame_size:           int   = 224,
    target_fps:           float = 25.0,
    motion_threshold:     float = 0.3,
    confidence_threshold: float = 0.7,
    device:               str   = "cpu",
) -> torch.Tensor:
    """
    Full visual pipeline: extract → optical flow gate → detect (conf≥0.7)
    → AU-saliency weighted Top-8 keyframes → ViT.
    Returns mean-pooled CLS token: (768,) float32.
    """
    model, processor = _load_vit(vit_model_name)
    model = model.to(device)

    frames = extract_frames(video_path, target_fps)
    if not frames:
        log.warning(f"No frames extracted from {video_path}")
        return torch.zeros(768)

    gated_frames = optical_flow_gate(frames, motion_threshold)

    face_results = detect_and_align_faces(gated_frames, detector, confidence_threshold)

    # Fallback 1: relax confidence on gated frames
    if not face_results:
        face_results = detect_and_align_faces(gated_frames, detector, 0.0)

    # Fallback 2: all frames, no confidence gate
    if not face_results:
        log.warning(f"No faces in motion-gated frames for {video_path} — using all frames.")
        face_results = detect_and_align_faces(frames, detector, 0.0)

    # Fallback 3: raw sharpness on frames
    if not face_results:
        face_results = [(f, sharpness_score(f)) for f in frames]

    crops     = [r[0] for r in face_results]
    scores    = [r[1] for r in face_results]
    keyframes = select_keyframes(crops, scores, k=n_keyframes)
    pil_imgs  = frames_to_pil(keyframes, size=frame_size)

    inputs = processor(images=pil_imgs, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        out = model(**inputs)
    cls_tokens = out.last_hidden_state[:, 0, :]   # (K, 768)
    z_v = cls_tokens.mean(dim=0).cpu()            # (768,)
    return z_v
