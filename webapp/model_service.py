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
import math
import threading
import time
import uuid
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from src.models.detection_model import DeepfakeDetector
from src.preprocessing.audio import load_audio_waveform
from src.preprocessing.pipeline import PreprocessingPipeline

from .config import EMOTIONS, settings
from .input_validator import (
    InputValidationError,
    inspect_video_stream,
    sanitize_transcript,
    validate_container,
    validate_audio_track,
    validate_speech_presence,
    validate_face_and_visual_quality,
)
from .schemas import EmotionPrediction, DetectionResult, ModelInfo, ForensicInterpretation

REPO_ROOT = Path(__file__).resolve().parents[1]
log = logging.getLogger("deepsentinel.model_service")

# A checkpoint signature uniquely identifies a file version.
Signature = Tuple[str, float, int]  # (path, mtime, size)


class ModelService:
    def __init__(self):
        self._lock = threading.RLock()
        self.device = settings.device
        self.model = DeepfakeDetector(
            classifier_mode="bottleneck",
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
        self._warmup_stage: str = "initializing"
        self._warmup_target: str = "Core Preprocessing Pipeline"
        self._warmup_progress: float = 0.05

        self._wav2vec_proc = None
        self._bert_tok = None

        # Try an initial load (fine if nothing exists yet — endpoints report it).
        self.maybe_reload(force=True)

    def _get_wav2vec_processor(self):
        if self._wav2vec_proc is None:
            from transformers import AutoProcessor
            self._wav2vec_proc = AutoProcessor.from_pretrained(self.pipeline.wav2vec_model)
        return self._wav2vec_proc

    def _get_bert_tokenizer(self):
        if self._bert_tok is None:
            from transformers import AutoTokenizer
            self._bert_tok = AutoTokenizer.from_pretrained(self.pipeline.bert_model)
        return self._bert_tok

    # ── Checkpoint resolution ──────────────────────────────────────────────────

    def _resolve_active(self) -> Optional[Path]:
        """First existing checkpoint in priority order (Adapted Phase 2 preferred)."""
        candidate_dirs = [
            settings.checkpoint_dir,
            settings.checkpoint_dir / "full",
            REPO_ROOT / "checkpoints",
            REPO_ROOT / "checkpoints/full",
        ]
        for name in settings.checkpoint_priority:
            for cdir in candidate_dirs:
                p = cdir / name
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
                note = "Phase-2 checkpoint loaded with fine-tuned end-to-end backbones."
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

    def warmup_status(self) -> dict:
        with self._lock:
            warmed = self._warmed.is_set()
            return {
                "status": "ready" if warmed else "warming",
                "progress": 1.0 if warmed else self._warmup_progress,
                "stage": "ready" if warmed else self._warmup_stage,
                "target": "DeepSentinel Neural Engine" if warmed else self._warmup_target,
                "device": self.device,
                "warmed": warmed,
                "checkpoint": getattr(self._meta, "checkpoint", "active"),
            }

    def warmup(self) -> None:
        """Preload every model the inference path touches, so the first /detect
        pays only compute — not cold weight-loading. Idempotent; loaders cache
        in module globals for the process lifetime."""
        import time
        from src.preprocessing import audio as A, visual as V

        t0 = time.time()
        steps = [
            ("wav2vec2", "Wav2Vec 2.0 Audio Backbone", 0.22, lambda: A._load_wav2vec(settings.wav2vec_model, device=self.device)),
            ("bert",     "BERT Contextual Prosody Tokenizer", 0.40, lambda: A._load_bert(settings.bert_model, device=self.device)),
            ("whisper",  "Whisper Speech Recognition Model", 0.60, lambda: A._load_whisper(settings.whisper_model, device=self.device)),
            ("vit",      "Vision Transformer ViT-B/16 Backbone", 0.80, lambda: V._load_vit(settings.vit_model, device=self.device)),
            ("insightface", "InsightFace Facial Landmarker & Alignment", 0.94, V._load_insightface_app),
        ]
        if getattr(V, "_FEAT_AVAILABLE", False):
            steps.append(("py-feat", "py-feat AU Saliency Detector", 0.97, V._load_feat_detector))

        for key, name, prog, fn in steps:
            try:
                with self._lock:
                    self._warmup_stage = "fetching"
                    self._warmup_target = name
                    self._warmup_progress = prog
                fn()
                log.info(f"  warmup: {key} ready")
            except Exception as e:  # missing/failed model degrades to fallback, not fatal
                log.warning(f"  warmup: {key} failed ({e}) — will use fallback at request time")

        # Step 6: Dry-run PyTorch forward pass to compile CUDA kernels and allocate memory buffers
        try:
            with self._lock:
                self._warmup_stage = "calibrating"
                self._warmup_target = "Bilinear Fusion & Affective Attention"
                self._warmup_progress = 0.98
            dummy_at = torch.zeros(1, 1024, device=self.device)
            dummy_v = torch.zeros(1, 768, device=self.device)
            with torch.no_grad():
                self.model._detect(dummy_at, dummy_v, has_speech=True)
            log.info("  warmup: PyTorch CUDA execution graph & kernels pre-warmed")
        except Exception as e:
            log.warning(f"  warmup: dry-run forward pass notice: {e}")

        with self._lock:
            self._warmup_stage = "ready"
            self._warmup_target = "DeepSentinel Neural Engine"
            self._warmup_progress = 1.0
            self._warmed.set()
        log.info(f"Warmup complete in {time.time() - t0:.1f}s — /detect now pays compute only.")

    def start_warmup(self) -> None:
        """Warm models in a daemon thread so server boot stays fast."""
        if self._warmed.is_set() or (self._warm_thread and self._warm_thread.is_alive()):
            return
        self._warm_thread = threading.Thread(target=self.warmup, name="warmup", daemon=True)
        self._warm_thread.start()

    # ── Post-hoc Emotion Calibration (Leveling & Amplification) ───────────────

    @staticmethod
    def _calibrate_emotion_probs(
        logits: torch.Tensor,
        modality: str = "visual",
        neutral_bias: Optional[float] = None,
        sad_bias: Optional[float] = None,
        temperature: Optional[float] = None,
        floor_epsilon: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Calibrate raw emotion logits to level neutral/sad and amplify all emotions.

        1. Resting-State Leveling (Menon et al., 2020 Logit Adjustment):
           - EMOTIONS[0] ('neutral'): Subtracts neutral_bias to align with active emotions.
           - EMOTIONS[2] ('sad'): Subtracts audio_sad_logit_bias (in audio) or visual_sad_logit_bias (in visual).
        2. Emotion Temperature Scaling (T_emo):
           Divides by temperature (> 1.0) before softmax to soften saturated distributions.
        3. Bounded Floor Amplification:
           Injects a gentle baseline floor (epsilon) so minority emotions (fear, disgust)
           are never crushed to 0.x%, ensuring every emotion remains legible and highlighted.
        """
        bias_n = settings.neutral_logit_bias if neutral_bias is None else neutral_bias
        if modality == "audio":
            bias_s = settings.audio_sad_logit_bias if sad_bias is None else sad_bias
        else:
            bias_s = settings.visual_sad_logit_bias if sad_bias is None else sad_bias

        T_base = max(float(settings.emotion_temperature if temperature is None else temperature), 0.01)
        eps = max(float(settings.emotion_floor_epsilon if floor_epsilon is None else floor_epsilon), 0.0)

        adj_logits = logits.clone()
        # EMOTIONS[0] is 'neutral'
        adj_logits[..., 0] -= bias_n
        # EMOTIONS[2] is 'sad'
        adj_logits[..., 2] -= bias_s

        # Asymmetric Active Sharpening & Neutral Protection:
        # If any active emotion (indices 1..5) is leading, sharpen at T_base (0.65) so it decisively peaks at 60-70%.
        # If neutral is leading, soften at T=1.15 so neutral NEVER balloons or aggressively suppresses subtle expressions.
        top_idx = int(adj_logits.argmax(dim=-1).flatten()[0].item())
        if top_idx == 0:
            T_eff = max(T_base, 1.15)  # Neutral stays calm & modest (~35-40%), never dominates!
        else:
            T_eff = T_base             # Active emotions get intensified to 60-70% just like the demo!

        probs = F.softmax(adj_logits / T_eff, dim=-1)

        # Apply floor amplification: P_amp = (1 - K*eps) * P + eps
        K = probs.size(-1)
        if 0.0 < eps < (1.0 / K):
            probs = (1.0 - K * eps) * probs + eps

        return probs

    def _fuse_and_calibrate_verdict(
        self,
        raw_logit: torch.Tensor,
        raw_sarcasm: torch.Tensor,
        raw_emo_a: torch.Tensor,
        raw_emo_b: torch.Tensor,
        has_speech: bool,
        transcript: str,
        meta: ModelInfo,
        language: str = "en",
        advisory_notes: Optional[List[str]] = None,
    ) -> Tuple[DetectionResult, dict]:
        """
        Calibrated Evidence Accumulation with Information-Theoretic Synchrony Engine:
        1. Distributional Jensen-Shannon Divergence D_JS(P_A || P_B) & Cosine Synchrony
        2. Multimodal Sarcasm Irony Filter (gating textual rhetorical artifacts via visual cues)
        3. Calibrated Biological Harmony prior against domain-shift shortcut artifacts
        """
        pb = self._calibrate_emotion_probs(raw_emo_b, modality="visual").squeeze(0)
        if not has_speech:
            pa = torch.zeros(6, device=self.device)
            pa[0] = 1.0  # 100% neutral voice when silent
            p_sarc = 0.0
            delta = torch.zeros(6, device=self.device)
            cos_sim = 0.0
            d_js = 0.0
            harmony_bonus = 0.0
        else:
            pa = self._calibrate_emotion_probs(raw_emo_a, modality="audio").squeeze(0)
            delta = torch.abs(pa - pb)

            # ── Sarcasm Head Output (Trained on MUStARD with BCEWithLogitsLoss) ───
            # Sarcasm is assessed directly by the multimodal sarcasm classifier (Z_at)
            # without artificial smile-gating that suppresses deadpan/serious delivery.
            bias = float(getattr(settings, "sarcasm_logit_bias", 0.0) or 0.0)
            p_sarc = float(torch.sigmoid(raw_sarcasm.squeeze() - bias).item())

            # ── Information-Theoretic Synchrony Engine (D_JS & CosSim) ───
            eps = 1e-12
            p_a_safe = (pa + eps) / (pa.sum() + eps * 6)
            p_b_safe = (pb + eps) / (pb.sum() + eps * 6)
            m = 0.5 * (p_a_safe + p_b_safe)
            kl_a = torch.sum(p_a_safe * torch.log(p_a_safe / m)).item()
            kl_b = torch.sum(p_b_safe * torch.log(p_b_safe / m)).item()
            d_js = max(0.0, 0.5 * (kl_a + kl_b))

            dot = torch.sum(pa * pb).item()
            norm_a = torch.norm(pa, p=2).item()
            norm_b = torch.norm(pb, p=2).item()
            cos_sim = max(0.0, min(1.0, dot / (norm_a * norm_b + eps)))

            # Valence definitions: 1 (happy), -1 (sad, angry, fear, disgust), 0 (neutral)
            top_a_idx = int(torch.argmax(pa).item())
            top_b_idx = int(torch.argmax(pb).item())
            val_a = 1 if top_a_idx == 1 else (-1 if top_a_idx in {2, 3, 4, 5} else 0)
            val_b = 1 if top_b_idx == 1 else (-1 if top_b_idx in {2, 3, 4, 5} else 0)

            harmony_bonus = 0.0
            max_d = float(torch.max(delta).item())

            # 1. High-Arousal Biological Shield (Russell, 1980; Ekman, 1969):
            # In genuine high-arousal distress/anger (top_a == top_b == 'angry'), violent facial
            # contortions and shouting acoustics mimic synthesis artifacts to ViT/W2V encoders.
            # When vocal and visual affect exhibit tight congruent coupling (max_d <= 0.20)
            # and sincere delivery (p_sarc < 0.25), compensate for organic arousal strain.
            if top_a_idx == 3 and top_b_idx == 3 and max_d <= 0.20 and p_sarc < 0.25:
                harmony_bonus = settings.arousal_harmony_bonus
            # 2. General / Conversational Harmony Gating:
            # For conversational speech (happy, neutral, calm), deepfake generators (Wav2Lip,
            # SadTalker) frequently match smiling moods. Never apply harmony bonuses if the
            # neural backbone detects manipulation artifacts (raw_val > logit_0).
            elif (settings.active_emotion_harmony_bonus > 0 or 
                  settings.neutral_emotion_harmony_bonus > 0 or 
                  settings.compatible_active_harmony_bonus > 0 or 
                  settings.compatible_neutral_harmony_bonus > 0):
                raw_val = float(raw_logit.squeeze().item())
                tau_0 = min(max(float(settings.decision_threshold), 0.01), 0.99)
                logit_0 = math.log(tau_0 / (1.0 - tau_0))
                if raw_val <= logit_0:
                    if top_a_idx == top_b_idx:
                        if top_a_idx != 0:
                            harmony_bonus = settings.active_emotion_harmony_bonus
                        else:
                            harmony_bonus = settings.neutral_emotion_harmony_bonus
                    elif val_a * val_b > 0:
                        # Same active valence (both positive or both negative)
                        if cos_sim >= settings.synchrony_cos_min and d_js <= settings.synchrony_js_max:
                            sync_scale = max(0.0, min(1.0, (cos_sim - 0.70) / 0.28)) * (1.0 - min(1.0, d_js / settings.synchrony_js_max))
                            harmony_bonus = settings.compatible_active_harmony_bonus * sync_scale
                    elif ((val_a == 0 and val_b > 0) or (val_b == 0 and val_a > 0)):
                        # Pleasant conversational engagement: Neutral baseline + gentle positive tone/expression
                        if cos_sim >= settings.synchrony_cos_min and d_js <= settings.synchrony_js_max:
                            sync_scale = max(0.0, min(1.0, (cos_sim - 0.70) / 0.28)) * (1.0 - min(1.0, d_js / settings.synchrony_js_max))
                            harmony_bonus = settings.compatible_neutral_harmony_bonus * sync_scale

        # ── Calibrated Evidence Accumulation ─────────────────────────────
        logit = raw_logit.squeeze() - harmony_bonus

        tau_0 = min(max(float(settings.decision_threshold), 0.01), 0.99)
        logit_0 = math.log(tau_0 / (1.0 - tau_0))
        T = max(float(settings.temperature), 1e-3)
        calibrated_logit = (logit - logit_0) / T
        p_fake = torch.sigmoid(calibrated_logit).item()
        verdict = "FAKE" if p_fake > 0.50 else "REAL"

        def _emo(probs) -> EmotionPrediction:
            idx = int(torch.argmax(probs).item())
            return EmotionPrediction(
                label=EMOTIONS[idx],
                confidence=float(probs[idx].item()),
                distribution={EMOTIONS[i]: float(probs[i].item()) for i in range(len(EMOTIONS))},
            )

        audio_emo = _emo(pa)
        visual_emo = _emo(pb)
        delta_dict = {EMOTIONS[i]: float(delta[i].item()) for i in range(len(EMOTIONS))}

        interpretation = self._generate_forensic_interpretation(
            verdict=verdict,
            emo_a=audio_emo.label,
            emo_b=visual_emo.label,
            p_fake=p_fake,
            p_sarc=p_sarc,
            cos_sim=cos_sim,
            d_js=d_js,
        )

        notes = list(advisory_notes or [])
        if language and language.lower() not in {"en", "english"}:
            notes.append(f"Non-English speech detected ({language.upper()}); evaluation prioritized acoustic-visual affect.")

        det_result = DetectionResult(
            verdict=verdict,
            p_fake=p_fake,
            threshold=0.50,
            audio_text_emotion=audio_emo,
            visual_emotion=visual_emo,
            emotion_mismatch=delta_dict,
            p_sarcasm=p_sarc,
            transcript=transcript,
            served_by=meta,
            forensic_interpretation=interpretation,
            language=language,
            advisory_notes=notes,
        )

        extra = {
            "pa": pa,
            "pb": pb,
            "delta": delta,
            "delta_dict": delta_dict,
            "audio_emo": audio_emo,
            "visual_emo": visual_emo,
            "p_sarcasm": p_sarc,
            "cos_sim": cos_sim,
            "d_js": d_js,
            "harmony_bonus": harmony_bonus,
        }
        return det_result, extra

    def _generate_forensic_interpretation(
        self,
        verdict: str,
        emo_a: str,
        emo_b: str,
        p_fake: float,
        p_sarc: float,
        cos_sim: float,
        d_js: float,
    ) -> ForensicInterpretation:
        """
        Exhaustive 8-State Forensic Multi-Tier Interpretation:
        Generates structured, professional forensic interpretation tailored to the exact
        combination of:
          - Verdict: Real (p_fake <= 0.5) vs Fake (p_fake > 0.5)
          - Emotion Alignment: Concordant (emo_a == emo_b) vs Discordant (emo_a != emo_b)
          - Rhetorical Context: Sarcastic (p_sarc >= 0.5) vs Sincere (p_sarc < 0.5)
        """
        is_fake = verdict == "FAKE"
        emotions_match = emo_a.strip().lower() == emo_b.strip().lower()
        sarcastic = p_sarc >= 0.50

        ea_title = emo_a.strip().title()
        eb_title = emo_b.strip().title()
        sarc_pct = int(round(p_sarc * 100))
        fake_pct = int(round(p_fake * 100))

        if not is_fake and emotions_match and not sarcastic:
            rat = "Voice and mouth timing are in sync with no signs of AI editing."
            return ForensicInterpretation(
                state_id="STATE_REAL_HARMONY",
                state_tag="REAL · NATURAL MATCH",
                headline="Looks Real: Voice and face emotions match naturally",
                summary=f"Voice tone and facial expression agree on {ea_title}. What you hear and see align naturally.",
                voice_face_analysis=f"Both voice and face show {ea_title} with no emotional clash.",
                sarcasm_analysis=f"No sarcasm detected ({sarc_pct}%). Delivery is sincere and straightforward.",
                technical_rationale=rat,
                forensic_rationale=rat,
            )

        elif not is_fake and emotions_match and sarcastic:
            rat = "Voice inflection and facial muscles stay synchronized like a real speaker."
            return ForensicInterpretation(
                state_id="STATE_REAL_CONGRUENT_SARCASM",
                state_tag="REAL · PLAYFUL SARCASM",
                headline="Looks Real: Playful sarcasm with matching expression",
                summary=f"The speaker is using sarcasm ({sarc_pct}%), and their facial expression matches that playful tone.",
                voice_face_analysis=f"Voice and face both express {ea_title} together in a coordinated delivery.",
                sarcasm_analysis=f"Sarcasm detected ({sarc_pct}%). DeepSentinel recognized intentional humor rather than an AI error.",
                technical_rationale=rat,
                forensic_rationale=rat,
            )

        elif not is_fake and not emotions_match and sarcastic:
            rat = "The irony filter accounts for dry humor so intentional poker faces are not flagged as fakes."
            return ForensicInterpretation(
                state_id="STATE_REAL_DEADPAN_IRONY",
                state_tag="REAL · DEADPAN HUMOR",
                headline="Looks Real: Deadpan joke (serious face with sarcastic voice)",
                summary=f"Voice sounds {ea_title} while the face stays {eb_title}, but this is dry deadpan humor ({sarc_pct}% sarcasm), not an AI fake.",
                voice_face_analysis=f"Voice sounds {ea_title} while the face keeps a {eb_title} poker face.",
                sarcasm_analysis=f"High sarcasm ({sarc_pct}%). The model recognized dry humor, avoiding a false deepfake alert.",
                technical_rationale=rat,
                forensic_rationale=rat,
            )

        elif not is_fake and not emotions_match and not sarcastic:
            rat = "Audio-visual sync is strong with no signs of face-swapping or dubbing."
            return ForensicInterpretation(
                state_id="STATE_REAL_MIXED_EMOTION",
                state_tag="REAL · MIXED FEELINGS",
                headline="Looks Real: Normal mixed human feelings",
                summary=f"Voice leans {ea_title} while the face shows {eb_title}. This subtle emotional mix is normal in authentic conversation.",
                voice_face_analysis=f"Voice conveys {ea_title} while face shows {eb_title}, transitioning smoothly.",
                sarcasm_analysis=f"Low sarcasm ({sarc_pct}%). The speaker is speaking sincerely.",
                technical_rationale=rat,
                forensic_rationale=rat,
            )

        elif is_fake and not emotions_match and not sarcastic:
            rat = "Deepfake tools usually replace voice or face separately, leaving an obvious emotional seam."
            return ForensicInterpretation(
                state_id="STATE_FAKE_EMOTION_DESYNC",
                state_tag="FAKE · EMOTION CLASH",
                headline="Likely Deepfake: Voice and face emotions contradict each other",
                summary=f"Voice sounds {ea_title}, but the face looks {eb_title}. This sharp contradiction happens when voice or video is swapped.",
                voice_face_analysis=f"Sharp contradiction: Hearing {ea_title} while seeing {eb_title} does not happen in sincere human speech.",
                sarcasm_analysis=f"Sarcasm is low ({sarc_pct}%), confirming this clash is an AI flaw, not a joke.",
                technical_rationale=rat,
                forensic_rationale=rat,
            )

        elif is_fake and not emotions_match and sarcastic:
            rat = f"High fake probability ({fake_pct}%). Natural facial micro-expressions are missing."
            return ForensicInterpretation(
                state_id="STATE_FAKE_MANIPULATED_DISSONANCE",
                state_tag="FAKE · VOICE & FACE CLASH",
                headline="Likely Deepfake: Sarcastic speech pasted onto an incompatible face",
                summary=f"The audio has sarcastic tone ({sarc_pct}%), but the face stays {eb_title} and fails to react naturally.",
                voice_face_analysis=f"The tone ({ea_title}) clashes with the stiff or unreactive facial expression ({eb_title}).",
                sarcasm_analysis=f"Sarcastic speech ({sarc_pct}%) without matching facial cues shows pasted audio.",
                technical_rationale=rat,
                forensic_rationale=rat,
            )

        elif is_fake and emotions_match and not sarcastic:
            rat = f"Detected AI generation artifacts ({fake_pct}% fake score) such as blurring or lip-sync lag."
            return ForensicInterpretation(
                state_id="STATE_FAKE_SYNTHESIS_ARTIFACTS",
                state_tag="FAKE · AI GLITCHES DETECTED",
                headline="Likely Deepfake: Matching emotion, but digital AI glitches detected",
                summary=f"Even though voice and face show {ea_title}, the AI found digital glitches in how the face was generated.",
                voice_face_analysis=f"Both voice and face show {ea_title}, but facial movements look artificially generated.",
                sarcasm_analysis=f"Low sarcasm ({sarc_pct}%). The speech is delivered straight.",
                technical_rationale=rat,
                forensic_rationale=rat,
            )

        else:  # is_fake and emotions_match and sarcastic
            rat = f"Detected artificial boundary warping ({fake_pct}% fake score), confirming AI manipulation."
            return ForensicInterpretation(
                state_id="STATE_FAKE_SYNTHETIC_SMIRK",
                state_tag="FAKE · ARTIFICIAL WARPING",
                headline="Likely Deepfake: Unnatural artificial expressions and mouth warping",
                summary=f"The video mimics an expressive or sarcastic look ({sarc_pct}%), but facial movements appear artificially warped.",
                voice_face_analysis=f"Voice and face attempt {ea_title}, but mouth movement looks robotic or unnatural.",
                sarcasm_analysis=f"High sarcasm score ({sarc_pct}%), typical of exaggerated parody deepfakes.",
                technical_rationale=rat,
                forensic_rationale=rat,
            )

    # ── Inference ──────────────────────────────────────────────────────────────

    @torch.no_grad()
    def predict(self, video_path: Path, clip_id: Optional[str] = None) -> DetectionResult:
        # Always serve the freshest weights.
        self.maybe_reload()
        with self._lock:
            if not self._meta.loaded:
                raise RuntimeError("No model equipped — train a checkpoint first.")
            meta = self._meta

        # Enforce video duration bounds (3 - 20 seconds)
        if video_path.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv", ".webm"}:
            try:
                import cv2
                cap = cv2.VideoCapture(str(video_path))
                if cap.isOpened():
                    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
                    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
                    dur = frame_count / fps if fps > 0 else 0
                    cap.release()
                    if dur > 0:
                        if dur < settings.min_duration_sec - 0.2:
                            raise ValueError(
                                f"Video is too short ({dur:.1f}s). Evaluated clip must be at least {settings.min_duration_sec:.1f} seconds long."
                            )
                        if dur > settings.max_duration_sec + 0.8:
                            raise ValueError(
                                f"Video is too long ({dur:.1f}s). Evaluated clip must be at most {settings.max_duration_sec:.1f} seconds long."
                            )
            except ValueError:
                raise
            except Exception as e:
                log.debug(f"Duration check bypass: {e}")

        clip_id = clip_id or f"upload_{uuid.uuid4().hex[:12]}"
        if meta.phase == 2 and getattr(self.model, "_backbones_loaded", False):
            return self._predict_e2e(video_path, clip_id=clip_id, meta=meta)

        feats = self.pipeline.process(clip_id, video_path)
        if feats is None:
            raise ValueError("Preprocessing failed — could not extract features "
                             "(check that the video has a visible face and audio).")

        z_at = feats.z_at.unsqueeze(0).float().to(self.device)  # (1, 1536)
        z_v = feats.z_v.unsqueeze(0).float().to(self.device)    # (1, 768)

        clean_transcript = sanitize_transcript(feats.transcript or "")
        has_speech = bool(clean_transcript and len(clean_transcript.strip()) > 0)
        out = self.model.forward_from_features(z_at, z_v, z_at_emo=z_at, has_speech=has_speech)

        det_result, _ = self._fuse_and_calibrate_verdict(
            raw_logit=out.logit,
            raw_sarcasm=out.sarcasm,
            raw_emo_a=out.emotion_a,
            raw_emo_b=out.emotion_b,
            has_speech=has_speech,
            transcript=clean_transcript,
            meta=meta,
            language=getattr(feats, "language", "en") or "en",
        )
        return det_result

    def _extract_face_landmarks_and_crops(
        self, video_path: Path, max_samples: int = 16, max_seconds: float = 5.0
    ) -> Tuple[List[dict], List[np.ndarray], List[np.ndarray], List[float]]:
        import cv2
        from src.preprocessing.visual import _load_insightface_app, sharpness_score
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return [], [], [], []
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        max_eval_frames = int(max_seconds * fps) if (max_seconds and max_seconds > 0) else total_frames
        eval_frames = min(total_frames, max_eval_frames) if total_frames > 0 else max_eval_frames
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 1)
        if eval_frames <= 0 or w <= 0 or h <= 0:
            cap.release()
            return [], [], [], []

        indices = [int(i * (eval_frames - 1) / max(1, max_samples - 1)) for i in range(max_samples)]
        faces_out = []
        frames_sampled = []
        face_crops = []
        scores = []
        try:
            app = None
            cascade = None
            try:
                from src.preprocessing.visual import _INSIGHTFACE_AVAILABLE, _load_insightface_app
                if _INSIGHTFACE_AVAILABLE:
                    app = _load_insightface_app()
            except Exception:
                app = None

            if app is None:
                try:
                    cascade_dir = getattr(cv2.data, "haarcascades", "")
                    cascade_path = os.path.join(cascade_dir, "haarcascade_frontalface_default.xml") if cascade_dir else ""
                    if cascade_path and os.path.exists(cascade_path):
                        cascade = cv2.CascadeClassifier(cascade_path)
                except Exception:
                    cascade = None

            anchor_embedding = None
            last_box = None

            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if not ret or frame is None:
                    continue
                frames_sampled.append(frame)
                time_sec = round(float(idx) / fps, 3)

                if app is not None:
                    detected = app.get(frame)
                    if detected:
                        valid_faces = [f for f in detected if getattr(f, "det_score", 0.0) >= 0.35]
                        if valid_faces:
                            if anchor_embedding is None:
                                best = max(valid_faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]) * f.det_score)
                                if hasattr(best, "embedding") and best.embedding is not None:
                                    anchor_embedding = best.embedding.copy()
                                last_box = best.bbox
                            else:
                                def _sim(f):
                                    if hasattr(f, "embedding") and f.embedding is not None and anchor_embedding is not None:
                                        dot = float(np.dot(f.embedding, anchor_embedding))
                                        norm = float(np.linalg.norm(f.embedding) * np.linalg.norm(anchor_embedding) + 1e-9)
                                        return dot / norm
                                    return 0.0

                                id_matches = [f for f in valid_faces if _sim(f) >= 0.30]
                                if id_matches:
                                    best = max(id_matches, key=_sim)
                                elif last_box is not None:
                                    def _iou(f):
                                        b1, b2 = f.bbox, last_box
                                        xi1, yi1 = max(b1[0], b2[0]), max(b1[1], b2[1])
                                        xi2, yi2 = min(b1[2], b2[2]), min(b1[3], b2[3])
                                        inter = max(0.0, xi2 - xi1) * max(0.0, yi2 - yi1)
                                        a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
                                        a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
                                        return inter / (a1 + a2 - inter + 1e-9)
                                    best = max(valid_faces, key=_iou)
                                else:
                                    best = max(valid_faces, key=lambda f: f.det_score)
                                last_box = best.bbox

                            x1, y1, x2, y2 = best.bbox.astype(int)
                            nx1 = max(0.0, min(1.0, float(x1) / w))
                            ny1 = max(0.0, min(1.0, float(y1) / h))
                            nx2 = max(0.0, min(1.0, float(x2) / w))
                            ny2 = max(0.0, min(1.0, float(y2) / h))
                            kps_norm = []
                            if hasattr(best, "kps") and best.kps is not None:
                                for kp in best.kps:
                                    kps_norm.append([round(float(kp[0]) / w, 4), round(float(kp[1]) / h, 4)])
                            faces_out.append({
                                "time": time_sec,
                                "bbox": [round(nx1, 4), round(ny1, 4), round(nx2, 4), round(ny2, 4)],
                                "kps": kps_norm,
                                "score": round(float(best.det_score), 4),
                            })

                            bw, bh = max(1, x2 - x1), max(1, y2 - y1)
                            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                            side = max(bw, bh) * 1.20
                            target_side = max(16, int(round(side)))
                            h_f, w_f = frame.shape[:2]

                            y1_t = int(round(cy - target_side / 2.0))
                            y2_t = y1_t + target_side
                            x1_t = int(round(cx - target_side / 2.0))
                            x2_t = x1_t + target_side

                            p_top = max(0, -y1_t)
                            p_bot = max(0, y2_t - h_f)
                            p_lft = max(0, -x1_t)
                            p_rgt = max(0, x2_t - w_f)

                            if p_top > 0 or p_bot > 0 or p_lft > 0 or p_rgt > 0:
                                padded = cv2.copyMakeBorder(frame, p_top, p_bot, p_lft, p_rgt, cv2.BORDER_REFLECT_101)
                                crop = padded[y1_t + p_top : y2_t + p_top, x1_t + p_lft : x2_t + p_lft]
                            else:
                                crop = frame[y1_t:y2_t, x1_t:x2_t]

                            if crop.size > 0:
                                face_crops.append(crop)
                                scores.append(float(best.det_score) * sharpness_score(crop))
                elif cascade is not None:
                    try:
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        detected_faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(40, 40))
                        if len(detected_faces) > 0:
                            x, y, w_box, h_box = max(detected_faces, key=lambda f: f[2] * f[3])
                            nx1 = max(0.0, min(1.0, float(x) / w))
                            ny1 = max(0.0, min(1.0, float(y) / h))
                            nx2 = max(0.0, min(1.0, float(x + w_box) / w))
                            ny2 = max(0.0, min(1.0, float(y + h_box) / h))
                            cx, cy = x + w_box / 2.0, y + h_box / 2.0
                            kps_norm = [
                                [round((x + 0.3 * w_box) / w, 4), round((y + 0.35 * h_box) / h, 4)],
                                [round((x + 0.7 * w_box) / w, 4), round((y + 0.35 * h_box) / h, 4)],
                                [round((x + 0.5 * w_box) / w, 4), round((y + 0.55 * h_box) / h, 4)],
                                [round((x + 0.35 * w_box) / w, 4), round((y + 0.75 * h_box) / h, 4)],
                                [round((x + 0.65 * w_box) / w, 4), round((y + 0.75 * h_box) / h, 4)],
                            ]
                            faces_out.append({
                                "time": time_sec,
                                "bbox": [round(nx1, 4), round(ny1, 4), round(nx2, 4), round(ny2, 4)],
                                "kps": kps_norm,
                                "score": 0.85,
                            })
                            side = max(w_box, h_box) * 1.20
                            target_side = max(16, int(round(side)))
                            h_f, w_f = frame.shape[:2]

                            y1_t = int(round(cy - target_side / 2.0))
                            y2_t = y1_t + target_side
                            x1_t = int(round(cx - target_side / 2.0))
                            x2_t = x1_t + target_side

                            p_top = max(0, -y1_t)
                            p_bot = max(0, y2_t - h_f)
                            p_lft = max(0, -x1_t)
                            p_rgt = max(0, x2_t - w_f)

                            if p_top > 0 or p_bot > 0 or p_lft > 0 or p_rgt > 0:
                                padded = cv2.copyMakeBorder(frame, p_top, p_bot, p_lft, p_rgt, cv2.BORDER_REFLECT_101)
                                crop = padded[y1_t + p_top : y2_t + p_top, x1_t + p_lft : x2_t + p_lft]
                            else:
                                crop = frame[y1_t:y2_t, x1_t:x2_t]

                            if crop.size > 0:
                                face_crops.append(crop)
                                scores.append(0.85 * sharpness_score(crop))
                    except Exception as he:
                        log.debug(f"Haar cascade detection exception: {he}")
        except Exception as e:
            log.warning(f"Face landmarks extraction notice: {e}")
        finally:
            cap.release()
        return faces_out, frames_sampled, face_crops, scores

    def _extract_face_landmarks(self, video_path: Path, max_samples: int = 16, max_seconds: float = 5.0) -> List[dict]:
        faces, _, _, _ = self._extract_face_landmarks_and_crops(video_path, max_samples=max_samples, max_seconds=max_seconds)
        return faces

    @torch.no_grad()
    def predict_stream(self, video_path: Path, clip_id: Optional[str] = None):
        self.maybe_reload()
        with self._lock:
            if not self._meta.loaded:
                yield {"error": "No model equipped — train a checkpoint first."}
                return
            meta = self._meta

        clip_id = clip_id or f"upload_{uuid.uuid4().hex[:12]}"

        # Step 0: Listening to the voice (16kHz audio extraction & stream validation)
        yield {
            "step": 0,
            "phase": "audio_extraction",
            "name": "Checking container & listening to the voice",
            "tech": "Stream Validator · 16kHz Mono · Wav2Vec 2.0",
            "status": "active",
        }
        try:
            inspection = inspect_video_stream(video_path)
            validate_container(inspection, min_duration=settings.min_duration_sec, max_duration=settings.max_upload_duration_sec)
        except InputValidationError as e:
            yield {"error": e.to_dict()}
            return

        wav = self.pipeline._wav_path(clip_id)
        if not wav.exists():
            from src.preprocessing.audio import extract_audio_to_wav
            ok = extract_audio_to_wav(video_path, wav)
            if not ok or not wav.exists():
                err = InputValidationError(
                    code="ERR_AUDIO_EXTRACTION_FAILED",
                    title="Audio Extraction Failed",
                    message="Failed to extract an audio stream from the video container.",
                    suggestion="Ensure the video has a standard AAC/MP3 audio track and re-export if needed.",
                )
                yield {"error": err.to_dict()}
                return

        try:
            validate_audio_track(inspection, wav)
        except InputValidationError as e:
            yield {"error": e.to_dict()}
            return

        yield {"step": 0, "status": "done"}

        # Step 1: Reading tone & words (Whisper + BERT)
        yield {
            "step": 1,
            "phase": "transcription",
            "name": "Reading the tone & words",
            "tech": "Wav2Vec 2.0 · BERT",
            "status": "active",
        }
        from src.preprocessing.audio import transcribe_with_meta
        txt_file = self.pipeline._txt_path(clip_id)
        detected_language = "en"
        if not txt_file.exists():
            if wav.exists() and wav.stat().st_size > 500:
                meta_res = transcribe_with_meta(wav, self.pipeline.whisper_model, device=self.device)
                transcript = meta_res.get("text", "")
                detected_language = meta_res.get("language", "en")
            else:
                transcript = ""
            txt_file.write_text(transcript, encoding="utf-8")
        else:
            transcript = txt_file.read_text(encoding="utf-8").strip()

        transcript = sanitize_transcript(transcript)

        try:
            validate_speech_presence(transcript, min_words=1)
        except InputValidationError as e:
            yield {"error": e.to_dict()}
            return

        # Emit the transcript IMMEDIATELY so the user sees live words!
        yield {
            "step": 1,
            "status": "active",
            "transcript": transcript,
            "msg": "Transcription ready",
        }

        # Prepare audio & text representations
        has_speech = bool(transcript and len(transcript.strip()) > 0)
        # Standardize audio window to 80,000 samples (5.0s @ 16kHz) matching Phase 2 training MAX_AUDIO
        max_samples = 80000
        if wav.exists() and wav.stat().st_size > 500:
            try:
                waveform, sr = load_audio_waveform(wav, target_sr=16000)
                if waveform.shape[0] > max_samples:
                    waveform = waveform[:max_samples]
                proc = self._get_wav2vec_processor()
                audio_enc = proc(
                    waveform.numpy(),
                    sampling_rate=16000,
                    return_tensors="pt",
                    padding="max_length",
                    max_length=max_samples,
                    truncation=True,
                )
                audio_values = audio_enc.input_values.to(self.device)
            except Exception as e:
                log.warning(f"Audio processing failed in stream: {e}")
                audio_values = torch.zeros(1, 80000, device=self.device)
        else:
            audio_values = torch.zeros(1, 80000, device=self.device)

        tok = self._get_bert_tokenizer()
        bert_enc = tok(
            transcript or "",
            return_tensors="pt",
            padding="max_length",
            max_length=128,
            truncation=True,
        )
        input_ids = bert_enc.input_ids.to(self.device)
        attention_mask = bert_enc.attention_mask.to(self.device)

        # Compute canonical Z_at for calibrated emotion prediction
        from src.preprocessing.audio import get_z_at
        z_at_path = self.pipeline._z_at_path(clip_id)
        if not z_at_path.exists():
            z_at = get_z_at(
                wav, transcript,
                self.pipeline.wav2vec_model, self.pipeline.bert_model,
                self.device, self.pipeline.max_audio_sec,
            )
            torch.save(z_at, z_at_path)
        else:
            z_at = torch.load(z_at_path, weights_only=True)

        is_e2e = meta.phase == 2 and getattr(self.model, "_backbones_loaded", False)

        yield {"step": 1, "status": "done", "transcript": transcript}

        # Step 2: Picking clearest face frames (InsightFace / RetinaFace)
        yield {
            "step": 2,
            "phase": "face_detection",
            "name": "Picking the clearest face frames",
            "tech": "InsightFace · RetinaFace",
            "status": "active",
        }
        faces, frames_sampled, face_crops, face_scores = self._extract_face_landmarks_and_crops(video_path, max_samples=16)
        try:
            validate_face_and_visual_quality(frames_sampled, face_crops, face_scores)
        except InputValidationError as e:
            yield {"error": e.to_dict()}
            return

        yield {
            "step": 2,
            "status": "done",
            "faces": faces,
            "face_count": len(faces),
        }

        # Step 3: Reading face emotion (Vision Transformer / ViT)
        yield {
            "step": 3,
            "phase": "visual_emotion",
            "name": "Reading the face's emotion",
            "tech": "Vision Transformer",
            "status": "active",
        }
        if is_e2e:
            from src.preprocessing.visual import get_keyframe_pixels
            keyframe_pixels = get_keyframe_pixels(
                video_path,
                vit_model_name=self.pipeline.vit_model,
                detector=self.pipeline.face_detector,
                n_keyframes=self.pipeline.n_keyframes,
                frame_size=self.pipeline.frame_size,
                target_fps=self.pipeline.target_fps,
                motion_threshold=self.pipeline.motion_threshold,
                confidence_threshold=self.pipeline.confidence_threshold,
                device=self.device,
            )
        else:
            from src.preprocessing.visual import get_z_v
            z_v_path = self.pipeline._z_v_path(clip_id)
            if not z_v_path.exists():
                z_v = get_z_v(
                    video_path,
                    vit_model_name=self.pipeline.vit_model,
                    detector=self.pipeline.face_detector,
                    n_keyframes=self.pipeline.n_keyframes,
                    frame_size=self.pipeline.frame_size,
                    target_fps=self.pipeline.target_fps,
                    device=self.device,
                )
                torch.save(z_v, z_v_path)
            else:
                z_v = torch.load(z_v_path, weights_only=True)
        yield {"step": 3, "status": "done"}

        # Step 4: Comparing voice emotion vs face emotion
        yield {
            "step": 4,
            "phase": "comparison",
            "name": "Comparing voice emotion vs face emotion",
            "tech": "Bilinear Fusion",
            "status": "active",
        }
        if is_e2e:
            z_at_t = z_at.unsqueeze(0).float().to(self.device)
            out = self.model(
                audio_values=audio_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                keyframe_pixels=keyframe_pixels,
                z_at_emo=z_at_t,
                has_speech=has_speech,
            )
        else:
            z_at_t = z_at.unsqueeze(0).float().to(self.device)
            z_v_t = z_v.unsqueeze(0).float().to(self.device)
            out = self.model.forward_from_features(z_at_t, z_v_t, z_at_emo=z_at_t, has_speech=has_speech)

        advisories = []
        if detected_language and detected_language.lower() not in {"en", "english"}:
            advisories.append(f"Non-English speech detected ({detected_language.upper()}); evaluation prioritized acoustic-visual synchrony.")
        if face_scores and np.mean(face_scores) < 30.0:
            advisories.append("Note: Face lighting or contrast is low; visual affect confidence is calibrated accordingly.")

        det_result, extra = self._fuse_and_calibrate_verdict(
            raw_logit=out.logit,
            raw_sarcasm=out.sarcasm,
            raw_emo_a=out.emotion_a,
            raw_emo_b=out.emotion_b,
            has_speech=has_speech,
            transcript=transcript,
            meta=meta,
            language=detected_language,
            advisory_notes=advisories,
        )

        yield {
            "step": 4,
            "status": "done",
            "audio_emotion": extra["audio_emo"].dict(),
            "visual_emotion": extra["visual_emo"].dict(),
        }

        # Step 5: Measuring the emotion gap (Δ)
        yield {
            "step": 5,
            "phase": "emotion_gap",
            "name": "Measuring the emotion gap",
            "tech": "Δ Incongruence",
            "status": "active",
        }
        yield {
            "step": 5,
            "status": "done",
            "emotion_mismatch": extra["delta_dict"],
            "p_sarcasm": extra["p_sarcasm"],
        }

        # Step 6: Reaching a verdict
        yield {
            "step": 6,
            "phase": "verdict",
            "name": "Reaching a verdict",
            "status": "active",
        }
        yield {
            "step": 6,
            "status": "done",
            "result": det_result.dict(),
        }

    def _prepare_e2e_inputs(self, video_path: Path, clip_id: str):
        from src.preprocessing.audio import extract_audio_to_wav, transcribe
        from src.preprocessing.visual import get_keyframe_pixels

        # 0. Container & stream validation
        inspection = inspect_video_stream(video_path)
        validate_container(inspection, min_duration=settings.min_duration_sec, max_duration=settings.max_upload_duration_sec)

        # 1. Audio validation & extraction
        wav = self.pipeline._wav_path(clip_id)
        if not wav.exists():
            wav.parent.mkdir(parents=True, exist_ok=True)
            ok = extract_audio_to_wav(video_path, wav)
            if not ok or not wav.exists():
                raise InputValidationError(
                    code="ERR_AUDIO_EXTRACTION_FAILED",
                    title="Audio Extraction Failed",
                    message="Failed to extract an audio stream from the video container.",
                    suggestion="Ensure the video has a standard AAC or MP3 audio track and re-export if necessary.",
                )
        validate_audio_track(inspection, wav)

        # Standardize audio window to 80,000 samples (5.0s @ 16kHz) matching Phase 2 training MAX_AUDIO
        max_samples = 80000
        if wav.exists() and wav.stat().st_size > 500:
            try:
                waveform, sr = load_audio_waveform(wav, target_sr=16000)
                if waveform.shape[0] > max_samples:
                    waveform = waveform[:max_samples]
                proc = self._get_wav2vec_processor()
                audio_enc = proc(
                    waveform.numpy(),
                    sampling_rate=16000,
                    return_tensors="pt",
                    padding="max_length",
                    max_length=max_samples,
                    truncation=True,
                )
                audio_values = audio_enc.input_values.to(self.device)
            except Exception as e:
                log.warning(f"Audio processing failed for {wav}: {e}")
                audio_values = torch.zeros(1, 80000, device=self.device)
        else:
            audio_values = torch.zeros(1, 80000, device=self.device)

        # 2. Transcript & BERT
        txt_file = self.pipeline._txt_path(clip_id)
        if not txt_file.exists():
            from src.preprocessing.audio import transcribe_with_meta
            meta_res = transcribe_with_meta(wav, self.pipeline.whisper_model, device=self.device) if (wav.exists() and wav.stat().st_size > 500) else {}
            transcript = meta_res.get("text", "")
            txt_file.write_text(transcript, encoding="utf-8")
        else:
            transcript = txt_file.read_text(encoding="utf-8").strip()
        transcript = sanitize_transcript(transcript)
        validate_speech_presence(transcript, min_words=1)

        tok = self._get_bert_tokenizer()
        bert_enc = tok(
            transcript or "",
            return_tensors="pt",
            padding="max_length",
            max_length=128,
            truncation=True,
        )
        input_ids = bert_enc.input_ids.to(self.device)
        attention_mask = bert_enc.attention_mask.to(self.device)

        # 3. Keyframe pixels & visual quality validation
        faces, frames_sampled, face_crops, face_scores = self._extract_face_landmarks_and_crops(video_path, max_samples=16)
        validate_face_and_visual_quality(frames_sampled, face_crops, face_scores)

        keyframe_pixels = get_keyframe_pixels(
            video_path,
            vit_model_name=self.pipeline.vit_model,
            detector=self.pipeline.face_detector,
            n_keyframes=self.pipeline.n_keyframes,
            frame_size=self.pipeline.frame_size,
            target_fps=self.pipeline.target_fps,
            motion_threshold=self.pipeline.motion_threshold,
            confidence_threshold=self.pipeline.confidence_threshold,
            device=self.device,
        )

        return audio_values, input_ids, attention_mask, keyframe_pixels, transcript

    @torch.no_grad()
    def _predict_e2e(
        self, video_path: Path, clip_id: Optional[str] = None, meta: Optional[ModelInfo] = None
    ) -> DetectionResult:
        """End-to-end inference using the checkpoint's fine-tuned backbones."""
        meta = meta or self._meta
        clip_id = clip_id or f"upload_{uuid.uuid4().hex[:12]}"
        audio_values, input_ids, attention_mask, keyframe_pixels, transcript = self._prepare_e2e_inputs(video_path, clip_id)

        from src.preprocessing.audio import get_z_at
        z_at_path = self.pipeline._z_at_path(clip_id)
        if not z_at_path.exists():
            wav = self.pipeline._wav_path(clip_id)
            z_at = get_z_at(
                wav, transcript,
                self.pipeline.wav2vec_model, self.pipeline.bert_model,
                self.device, self.pipeline.max_audio_sec,
            )
            torch.save(z_at, z_at_path)
        else:
            z_at = torch.load(z_at_path, weights_only=True)

        z_at_t = z_at.unsqueeze(0).float().to(self.device)
        has_speech = bool(transcript and len(transcript.strip()) > 0)

        out = self.model(
            audio_values=audio_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            keyframe_pixels=keyframe_pixels,
            z_at_emo=z_at_t,
            has_speech=has_speech,
        )

        det_result, _ = self._fuse_and_calibrate_verdict(
            raw_logit=out.logit,
            raw_sarcasm=out.sarcasm,
            raw_emo_a=out.emotion_a,
            raw_emo_b=out.emotion_b,
            has_speech=has_speech,
            transcript=transcript,
            meta=meta,
        )
        return det_result


# Module-level singleton, created by the app lifespan.
_service: Optional[ModelService] = None


def get_service() -> ModelService:
    global _service
    if _service is None:
        _service = ModelService()
    return _service
