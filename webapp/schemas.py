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


class ForensicInterpretation(BaseModel):
    state_id: str                          # Unique state code (e.g., STATE_REAL_HARMONY, STATE_FAKE_EMOTION_DESYNC)
    state_tag: Optional[str] = "ANALYSIS COMPLETE"  # Category badge tag
    headline: str                          # Concise verdict headline
    summary: str                           # Plain-language explanation for general audience
    voice_face_analysis: str               # Breakdown of Voice vs Face alignment
    sarcasm_analysis: str                  # Rhetorical context and role of sarcasm filtering
    technical_rationale: Optional[str] = None  # Friendly AI reasoning rationale
    forensic_rationale: Optional[str] = None   # Alias for backward compatibility


class ValidationErrorDetail(BaseModel):
    code: str
    title: str
    message: str
    suggestion: str
    details: Dict[str, float | int | str] = {}


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
    forensic_interpretation: Optional[ForensicInterpretation] = None


class HealthResponse(BaseModel):
    status: str
    model: ModelInfo
