"""
config.py — Web service settings.

All paths are resolved relative to the repository root so the app runs the same
regardless of the working directory. Override any value with an environment
variable of the same name (upper-cased), e.g. DEEPSENTINEL_DEVICE=cuda.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

REPO_ROOT = Path(__file__).resolve().parents[1]


def _env(name: str, default: str) -> str:
    return os.environ.get(f"DEEPSENTINEL_{name}", default)


def _default_device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


@dataclass
class Settings:
    # Directory the trainer writes checkpoints to.
    checkpoint_dir: Path = field(
        default_factory=lambda: REPO_ROOT / _env("CHECKPOINT_DIR", "checkpoints")
    )
    # Preference order — the FIRST file that exists is served. Adapted Phase 2
    # is preferred over Phase 2/1 whenever it becomes available.
    checkpoint_priority: List[str] = field(
        default_factory=lambda: [
            "best_phase2_adapted.pt",
            "best_phase2_bottleneck.pt",
            "best_phase2.pt",
            "best_phase1_bottleneck.pt",
            "best_phase1.pt",
        ]
    )
    # Seconds between background checks for a newer checkpoint. The active
    # checkpoint is ALSO re-checked on every request, so traffic alone keeps the
    # model current; the watcher just covers idle periods.
    watch_interval_sec: float = float(_env("WATCH_INTERVAL_SEC", "15"))

    # ── Inference ──────────────────────────────────────────────────────────────
    device: str = _env("DEVICE", _default_device())
    # Preload all preprocessing models at startup so the first /detect pays only
    # inference cost (no cold weight-loading). Set to "0" to disable.
    warmup_on_start: bool = _env("WARMUP", "1") == "1"
    preprocess_cache_dir: Path = field(
        default_factory=lambda: REPO_ROOT / _env("PREPROCESS_CACHE_DIR", "data/preprocessed")
    )
    # Uploaded videos are stored here. Cleanup is intentionally left to the user
    # (no auto-delete) — manage this directory yourself.
    upload_dir: Path = field(
        default_factory=lambda: REPO_ROOT / _env("UPLOAD_DIR", "webapp/uploads")
    )

    # ── Backbone model names (must match training) ─────────────────────────────
    wav2vec_model: str = "facebook/wav2vec2-base"
    bert_model: str = "bert-base-uncased"
    whisper_model: str = "openai/whisper-base"
    vit_model: str = "google/vit-base-patch16-224"

    decision_threshold: float = float(_env("DECISION_THRESHOLD", "0.44"))  # Separation threshold

    # Video duration bounds for uploaded clips (in seconds)
    max_upload_duration_sec: float = float(_env("MAX_UPLOAD_DURATION_SEC", "600.0"))  # up to 10 minutes
    min_duration_sec: float = float(_env("MIN_DURATION_SEC", "3.0"))                  # crop slice min
    max_duration_sec: float = float(_env("MAX_DURATION_SEC", "20.0"))                 # crop slice max

    # Post-hoc temperature scaling (Guo et al., 2017): logit /= T before sigmoid.
    # Calibrated temperature scaling maps raw model margins into decisive, well-spread confidence percentages.
    temperature: float = float(_env("TEMPERATURE", "0.65"))


settings = Settings()

# Emotion class index → label (must match training label space in dataset.py).
EMOTIONS = ["neutral", "happy", "sad", "angry", "fear", "disgust"]
