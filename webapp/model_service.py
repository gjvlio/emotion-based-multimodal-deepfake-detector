"""
model_service.py — Loads the trained DeepSentinel detector and keeps it in sync
with the latest training checkpoint (auto-equip / hot-reload).

How auto-equip works
--------------------
The trainer (src/training/trainer.py) writes checkpoints/full/best_phase{1,2}.pt.
ModelService records the active checkpoint's (path, mtime, size) signature. On
every request — and via a background watcher during idle periods — it re-checks
that signature. When the trainer overwrites the file with a better model, the
signature changes and ModelService reloads the weights into the live in-memory
model. No restart, no downtime.

Inference path
--------------
A web upload is a RAW video, so the service runs the project's real preprocessing
pipeline (PreprocessingPipeline) to produce Z_at / Z_v, then calls
detector.forward_from_features(). This is exactly the Phase-1 / cached-feature
path used in training, so results match for Phase-1 checkpoints (backbones frozen).

Phase-2 caveat: a Phase-2 checkpoint fine-tunes the Wav2Vec2/BERT/ViT backbones.
For 100% fidelity those checkpoints must extract features from the FINE-TUNED
backbones via detector.forward() (end-to-end), not from the vanilla backbones in
PreprocessingPipeline. That end-to-end path is stubbed below (see _predict_e2e)
and flagged in the response `note`. Wire it before serving Phase-2 publicly.
"""
from __future__ import annotations

import logging
import threading
import time
import uuid
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from src.models.detection_model import DeepfakeDetector
from src.preprocessing.pipeline import PreprocessingPipeline

from .config import settings, EMOTIONS
from .schemas import ModelInfo, EmotionPrediction, DetectionResult

log = logging.getLogger("deepsentinel.model_service")

# A checkpoint signature uniquely identifies a file version.
Signature = Tuple[str, float, int]  # (path, mtime, size)


class ModelService:
    def __init__(self):
        self._lock = threading.RLock()
        self.device = settings.device
        self.model = DeepfakeDetector(
            wav2vec_model=settings.wav2vec_model,
            bert_model=settings.bert_model,
            vit_model=settings.vit_model,
        ).to(self.device)
        self.model.eval()

        self.pipeline = PreprocessingPipeline(
            cache_dir=settings.preprocess_cache_dir,
            wav2vec_model=settings.wav2vec_model,
            bert_model=settings.bert_model,
            whisper_model=settings.whisper_model,
            vit_model=settings.vit_model,
            device=self.device,
        )

        # Active-checkpoint bookkeeping
        self._active_sig: Optional[Signature] = None
        self._meta = ModelInfo(loaded=False, device=self.device)

        settings.upload_dir.mkdir(parents=True, exist_ok=True)

        self._watcher: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._warmed = threading.Event()
        self._warm_thread: Optional[threading.Thread] = None

        # Try an initial load (fine if nothing exists yet — endpoints report it).
        self.maybe_reload(force=True)

    # ── Checkpoint resolution ──────────────────────────────────────────────────

    def _resolve_active(self) -> Optional[Path]:
        """First existing checkpoint in priority order (Phase 2 preferred)."""
        for name in settings.checkpoint_priority:
            p = settings.checkpoint_dir / name
            if p.exists():
                return p
        return None

    @staticmethod
    def _signature(path: Path) -> Signature:
        st = path.stat()
        return (str(path), st.st_mtime, st.st_size)

    # ── Hot reload ─────────────────────────────────────────────────────────────

    def maybe_reload(self, force: bool = False) -> bool:
        """Reload weights if the active checkpoint changed. Returns True if reloaded."""
        with self._lock:
            path = self._resolve_active()
            if path is None:
                if force:
                    self._meta = ModelInfo(
                        loaded=False, device=self.device,
                        note="No checkpoint found yet — train a model first.",
                    )
                return False

            sig = self._signature(path)
            if not force and sig == self._active_sig:
                return False

            try:
                ckpt = torch.load(path, weights_only=True, map_location=self.device)
            except Exception as e:
                log.error(f"Failed to load checkpoint {path}: {e}")
                return False

            state = ckpt.get("model_state", ckpt)
            phase = 2 if any(k.startswith("wav2vec2.") for k in state.keys()) else 1

            # Phase 2 checkpoints carry backbone weights — instantiate them so the
            # state_dict has a home, then load end-to-end weights.
            if phase == 2:
                self.model.load_backbones()

            missing, unexpected = self.model.load_state_dict(state, strict=False)
            self.model.eval()

            self._active_sig = sig
            note = None
            if phase == 2:
                note = ("Phase-2 checkpoint loaded. Feature-path inference uses "
                        "vanilla backbones; wire _predict_e2e for full fidelity.")
            if unexpected:
                log.warning(f"Unexpected keys in checkpoint: {list(unexpected)[:5]}…")

            self._meta = ModelInfo(
                loaded=True,
                checkpoint=path.name,
                checkpoint_path=str(path),
                phase=phase,
                epoch=ckpt.get("epoch") if isinstance(ckpt, dict) else None,
                val_loss=ckpt.get("val_loss") if isinstance(ckpt, dict) else None,
                last_modified=sig[1],
                device=self.device,
                note=note,
            )
            log.info(f"Equipped checkpoint {path.name} (phase {phase}, "
                     f"val_loss={self._meta.val_loss}).")
            return True

    # ── Background watcher (covers idle periods) ───────────────────────────────

    def start_watcher(self) -> None:
        if self._watcher and self._watcher.is_alive():
            return
        self._stop.clear()

        def _loop():
            while not self._stop.wait(settings.watch_interval_sec):
                try:
                    self.maybe_reload()
                except Exception as e:  # never let the watcher die silently
                    log.error(f"Watcher reload error: {e}")

        self._watcher = threading.Thread(target=_loop, name="ckpt-watcher", daemon=True)
        self._watcher.start()
        log.info(f"Checkpoint watcher started (every {settings.watch_interval_sec}s).")

    def stop_watcher(self) -> None:
        self._stop.set()

    # ── Info ───────────────────────────────────────────────────────────────────

    def info(self) -> ModelInfo:
        with self._lock:
            self._meta.warmed = self._warmed.is_set()
            return self._meta

    # ── Warmup (preload preprocessing models) ──────────────────────────────────

    def warmup(self) -> None:
        """Preload every model the inference path touches, so the first /detect
        pays only compute — not cold weight-loading. Idempotent; loaders cache
        in module globals for the process lifetime."""
        import time
        from src.preprocessing import audio as A, visual as V

        t0 = time.time()
        steps = [
            ("wav2vec2", lambda: A._load_wav2vec(settings.wav2vec_model)),
            ("bert",     lambda: A._load_bert(settings.bert_model)),
            ("whisper",  lambda: A._load_whisper(settings.whisper_model, device=self.device)),
            ("vit",      lambda: V._load_vit(settings.vit_model)),
            ("insightface", V._load_insightface_app),
        ]
        if getattr(V, "_FEAT_AVAILABLE", False):
            steps.append(("py-feat", V._load_feat_detector))

        for name, fn in steps:
            try:
                fn()
                log.info(f"  warmup: {name} ready")
            except Exception as e:  # missing/failed model degrades to fallback, not fatal
                log.warning(f"  warmup: {name} failed ({e}) — will use fallback at request time")

        self._warmed.set()
        log.info(f"Warmup complete in {time.time() - t0:.1f}s — /detect now pays compute only.")

    def start_warmup(self) -> None:
        """Warm models in a daemon thread so server boot stays fast."""
        if self._warmed.is_set() or (self._warm_thread and self._warm_thread.is_alive()):
            return
        self._warm_thread = threading.Thread(target=self.warmup, name="warmup", daemon=True)
        self._warm_thread.start()

    # ── Inference ──────────────────────────────────────────────────────────────

    @torch.no_grad()
    def predict(self, video_path: Path, clip_id: Optional[str] = None) -> DetectionResult:
        # Always serve the freshest weights.
        self.maybe_reload()
        with self._lock:
            if not self._meta.loaded:
                raise RuntimeError("No model equipped — train a checkpoint first.")
            meta = self._meta

        clip_id = clip_id or f"upload_{uuid.uuid4().hex[:12]}"
        feats = self.pipeline.process(clip_id, video_path)
        if feats is None:
            raise ValueError("Preprocessing failed — could not extract features "
                             "(check that the video has a visible face and audio).")

        z_at = feats.z_at.unsqueeze(0).float().to(self.device)  # (1, 1536)
        z_v = feats.z_v.unsqueeze(0).float().to(self.device)    # (1, 768)

        out = self.model.forward_from_features(z_at, z_v)

        p_fake = torch.sigmoid(out.logit.squeeze()).item()
        p_sarc = torch.sigmoid(out.sarcasm.squeeze()).item()
        pa = F.softmax(out.emotion_a, dim=-1).squeeze(0)
        pb = F.softmax(out.emotion_b, dim=-1).squeeze(0)
        delta = torch.abs(pa - pb)

        def _emo(probs) -> EmotionPrediction:
            idx = int(torch.argmax(probs).item())
            return EmotionPrediction(
                label=EMOTIONS[idx],
                confidence=float(probs[idx].item()),
                distribution={EMOTIONS[i]: float(probs[i].item()) for i in range(len(EMOTIONS))},
            )

        verdict = "FAKE" if p_fake > settings.decision_threshold else "REAL"
        return DetectionResult(
            verdict=verdict,
            p_fake=p_fake,
            threshold=settings.decision_threshold,
            audio_text_emotion=_emo(pa),
            visual_emotion=_emo(pb),
            emotion_mismatch={EMOTIONS[i]: float(delta[i].item()) for i in range(len(EMOTIONS))},
            p_sarcasm=p_sarc,
            transcript=feats.transcript,
            served_by=meta,
        )

    @torch.no_grad()
    def _predict_e2e(self, video_path: Path):  # pragma: no cover - Phase 2 seam
        """End-to-end inference using the checkpoint's fine-tuned backbones.

        TODO (Phase 2): build raw model inputs and call self.model.forward():
          - audio_values   : waveform tensor (use src.preprocessing.audio helpers)
          - input_ids/mask : BERT tokenization of the Whisper transcript
          - keyframe_pixels : (1, K, 3, 224, 224) from the visual keyframe selector
        Until wired, Phase-2 checkpoints are served via the feature path with a
        fidelity note. Kept as an explicit seam so the contract is visible.
        """
        raise NotImplementedError("Phase-2 end-to-end inference not yet wired.")


# Module-level singleton, created by the app lifespan.
_service: Optional[ModelService] = None


def get_service() -> ModelService:
    global _service
    if _service is None:
        _service = ModelService()
    return _service
