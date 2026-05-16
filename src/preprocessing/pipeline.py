"""
pipeline.py — Preprocessing orchestrator: raw clip → (Z_at, Z_v) + metadata.

Processes a single video clip and caches:
    data/preprocessed/
        audio/{clip_id}.wav          16kHz mono WAV
        transcripts/{clip_id}.txt    Whisper ASR output
        features/z_at/{clip_id}.pt   (1536,) tensor
        features/z_v/{clip_id}.pt    (768,)  tensor

Re-running on an already-processed clip is a no-op (cached files respected).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import torch

from .audio import extract_audio_to_wav, transcribe, get_z_at
from .visual import get_z_v

log = logging.getLogger(__name__)


@dataclass
class ClipFeatures:
    clip_id:    str
    z_at:       torch.Tensor    # (1536,)
    z_v:        torch.Tensor    # (768,)
    transcript: str


class PreprocessingPipeline:
    """
    Orchestrates audio + visual preprocessing for one clip at a time.
    All outputs are cached — safe to interrupt and resume.
    """

    def __init__(
        self,
        cache_dir:            str | Path = "data/preprocessed",
        wav2vec_model:        str   = "facebook/wav2vec2-base",
        bert_model:           str   = "bert-base-uncased",
        whisper_model:        str   = "openai/whisper-base",
        vit_model:            str   = "google/vit-base-patch16-224",
        face_detector:        str   = "retinaface",
        n_keyframes:          int   = 8,
        frame_size:           int   = 224,
        max_audio_sec:        int   = 30,
        target_fps:           float = 25.0,
        motion_threshold:     float = 0.3,
        confidence_threshold: float = 0.7,
        device:               str   = "cpu",
    ):
        self.cache_dir            = Path(cache_dir)
        self.wav2vec_model        = wav2vec_model
        self.bert_model           = bert_model
        self.whisper_model        = whisper_model
        self.vit_model            = vit_model
        self.face_detector        = face_detector
        self.n_keyframes          = n_keyframes
        self.frame_size           = frame_size
        self.max_audio_sec        = max_audio_sec
        self.target_fps           = target_fps
        self.motion_threshold     = motion_threshold
        self.confidence_threshold = confidence_threshold
        self.device               = device

        (self.cache_dir / "audio").mkdir(parents=True, exist_ok=True)
        (self.cache_dir / "transcripts").mkdir(parents=True, exist_ok=True)
        (self.cache_dir / "features" / "z_at").mkdir(parents=True, exist_ok=True)
        (self.cache_dir / "features" / "z_v").mkdir(parents=True, exist_ok=True)

    # ── Cache paths ────────────────────────────────────────────────────────────

    def _wav_path(self, clip_id: str) -> Path:
        return self.cache_dir / "audio" / f"{clip_id}.wav"

    def _txt_path(self, clip_id: str) -> Path:
        return self.cache_dir / "transcripts" / f"{clip_id}.txt"

    def _z_at_path(self, clip_id: str) -> Path:
        return self.cache_dir / "features" / "z_at" / f"{clip_id}.pt"

    def _z_v_path(self, clip_id: str) -> Path:
        return self.cache_dir / "features" / "z_v" / f"{clip_id}.pt"

    def is_cached(self, clip_id: str) -> bool:
        return self._z_at_path(clip_id).exists() and self._z_v_path(clip_id).exists()

    # ── Processing ─────────────────────────────────────────────────────────────

    def process(
        self,
        clip_id:    str,
        video_path: str | Path,
        force:      bool = False,
    ) -> Optional[ClipFeatures]:
        """
        Process one clip. Returns ClipFeatures or None on failure.
        Skips already-cached clips unless force=True.
        """
        video_path = Path(video_path)
        if not video_path.exists():
            log.error(f"Video not found: {video_path}")
            return None

        if not force and self.is_cached(clip_id):
            return self._load_cached(clip_id)

        # Step 1: extract audio WAV
        wav = self._wav_path(clip_id)
        if not wav.exists() or force:
            ok = extract_audio_to_wav(video_path, wav)
            if not ok or not wav.exists():
                log.warning(f"Audio extraction failed: {clip_id}")
                if video_path.suffix.lower() == ".wav":
                    import shutil
                    shutil.copy2(video_path, wav)
                else:
                    return None

        # Step 2: ASR transcription
        txt_file = self._txt_path(clip_id)
        if not txt_file.exists() or force:
            transcript = transcribe(wav, self.whisper_model)
            txt_file.write_text(transcript, encoding="utf-8")
        else:
            transcript = txt_file.read_text(encoding="utf-8").strip()

        # Step 3: Z_at — acoustic + linguistic
        z_at_path = self._z_at_path(clip_id)
        if not z_at_path.exists() or force:
            try:
                z_at = get_z_at(
                    wav, transcript,
                    self.wav2vec_model, self.bert_model,
                    self.device, self.max_audio_sec,
                )
                torch.save(z_at, z_at_path)
            except Exception as e:
                log.error(f"Z_at extraction failed for {clip_id}: {e}")
                return None
        else:
            z_at = torch.load(z_at_path, weights_only=True)

        # Step 4: Z_v — visual keyframe ViT (AU-saliency guided selection)
        z_v_path = self._z_v_path(clip_id)
        if not z_v_path.exists() or force:
            try:
                z_v = get_z_v(
                    video_path,
                    vit_model_name=self.vit_model,
                    detector=self.face_detector,
                    n_keyframes=self.n_keyframes,
                    frame_size=self.frame_size,
                    target_fps=self.target_fps,
                    motion_threshold=self.motion_threshold,
                    confidence_threshold=self.confidence_threshold,
                    device=self.device,
                )
                torch.save(z_v, z_v_path)
            except Exception as e:
                log.error(f"Z_v extraction failed for {clip_id}: {e}")
                return None
        else:
            z_v = torch.load(z_v_path, weights_only=True)

        return ClipFeatures(clip_id=clip_id, z_at=z_at, z_v=z_v, transcript=transcript)

    def _load_cached(self, clip_id: str) -> ClipFeatures:
        z_at = torch.load(self._z_at_path(clip_id), weights_only=True)
        z_v  = torch.load(self._z_v_path(clip_id),  weights_only=True)
        txt  = self._txt_path(clip_id)
        transcript = txt.read_text(encoding="utf-8").strip() if txt.exists() else ""
        return ClipFeatures(clip_id=clip_id, z_at=z_at, z_v=z_v, transcript=transcript)

    def load_features(self, clip_id: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fast load of cached (Z_at, Z_v) for training."""
        return (
            torch.load(self._z_at_path(clip_id), weights_only=True),
            torch.load(self._z_v_path(clip_id),  weights_only=True),
        )
