"""
config.py — Loads default.yaml and exposes a typed Config object.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import yaml

log = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class ModelConfig:
    cbp_dim: int = 8192
    n_emotions: int = 6
    dropout_heads: float = 0.3
    dropout_classifier: float = 0.4
    wav2vec_model: str = "facebook/wav2vec2-base"
    bert_model: str = "bert-base-uncased"
    vit_model: str = "google/vit-base-patch16-224"
    whisper_model: str = "openai/whisper-base"


@dataclass
class PreprocessingConfig:
    audio_sample_rate:    int   = 16000
    max_audio_seconds:    int   = 30
    n_keyframes:          int   = 8
    frame_size:           int   = 224
    face_detector:        str   = "retinaface"
    target_fps:           float = 25.0
    motion_threshold:     float = 0.3
    confidence_threshold: float = 0.7


@dataclass
class PhaseConfig:
    lr: float = 1e-3
    weight_decay: float = 1e-4
    max_epochs: int = 10
    patience: int = 5


@dataclass
class TrainingConfig:
    batch_size: int = 8
    fp16: bool = True
    phase1: PhaseConfig = field(default_factory=lambda: PhaseConfig(lr=1e-3, max_epochs=10))
    phase2: PhaseConfig = field(default_factory=lambda: PhaseConfig(lr=1e-5, max_epochs=20))
    lambda_a: float = 0.5
    lambda_b: float = 0.5
    train_ratio: float = 0.80
    val_ratio: float = 0.10
    test_ratio: float = 0.10


@dataclass
class PathsConfig:
    data_root: str = "data"
    preprocessed_dir: str = "data/preprocessed"
    meld_real_csv: str = "data/processed/meld_manifests/meld_real.csv"
    mosei_real_csv: str = "data/processed/mosei_manifests/mosei_real.csv"
    track1_meta: str = "data/synthetic/track1_fakes/metadata.csv"
    track2_meta: str = "data/synthetic/track2_fakes/metadata.csv"
    track3_meta: str = "data/synthetic/track3_fakes/metadata.csv"
    track4_meta: str = "data/synthetic/track4_fakes/metadata.csv"
    checkpoints_dir: str = "checkpoints"
    logs_dir: str = "logs"

    def abs(self, rel: str) -> Path:
        return _PROJECT_ROOT / rel


@dataclass
class EvaluationConfig:
    threshold: float = 0.5
    threshold_strict: float = 0.65
    ood_datasets: List[str] = field(default_factory=list)


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    @classmethod
    def from_yaml(cls, path: str | Path | None = None) -> "Config":
        if path is None:
            path = _PROJECT_ROOT / "configs" / "default.yaml"
        path = Path(path)
        if not path.exists():
            log.warning(f"Config file not found at {path}. Using defaults.")
            return cls()
        with open(path) as f:
            raw = yaml.safe_load(f)
        cfg = cls()
        _apply(cfg.model,          raw.get("model", {}))
        _apply(cfg.preprocessing,  raw.get("preprocessing", {}))
        _apply(cfg.evaluation,     raw.get("evaluation", {}))
        _apply(cfg.paths,          raw.get("paths", {}))
        tr = raw.get("training", {})
        _apply(cfg.training, {k: v for k, v in tr.items() if k not in ("phase1", "phase2")})
        if "phase1" in tr:
            _apply(cfg.training.phase1, tr["phase1"])
        if "phase2" in tr:
            _apply(cfg.training.phase2, tr["phase2"])
        return cfg


def _apply(obj, d: dict) -> None:
    for k, v in d.items():
        if hasattr(obj, k):
            setattr(obj, k, v)
