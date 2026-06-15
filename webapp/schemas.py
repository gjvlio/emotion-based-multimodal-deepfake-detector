"""
schemas.py — Pydantic response models for the DeepSentinel API.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel


class ModelInfo(BaseModel):
    loaded: bool
    checkpoint: Optional[str] = None       # filename being served
    checkpoint_path: Optional[str] = None  # absolute path
    phase: Optional[int] = None            # 1 = frozen backbones, 2 = fine-tuned
    epoch: Optional[int] = None
    val_loss: Optional[float] = None
    last_modified: Optional[float] = None  # checkpoint mtime (epoch seconds)
    device: str
    warmed: bool = False                   # preprocessing models preloaded?
    note: Optional[str] = None


class EmotionPrediction(BaseModel):
    label: str
    confidence: float
    distribution: Dict[str, float]


class DetectionResult(BaseModel):
    verdict: str                     # "FAKE" | "REAL"
    p_fake: float                    # P(fake) ∈ [0, 1]
    threshold: float
    audio_text_emotion: EmotionPrediction
    visual_emotion: EmotionPrediction
    emotion_mismatch: Dict[str, float]   # Delta per class
    p_sarcasm: float
    transcript: str
    served_by: ModelInfo             # which checkpoint produced this result


class HealthResponse(BaseModel):
    status: str
    model: ModelInfo
