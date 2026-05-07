"""
audio.py — Audio feature extraction: Wav2Vec2 acoustic + Whisper→BERT linguistic.

extract_audio_to_wav(video_path, out_wav)  → writes 16kHz mono WAV via ffmpeg
transcribe(audio_path)                      → str transcript via Whisper
get_acoustic_embedding(wav_path, model)     → (768,) tensor via Wav2Vec2
get_linguistic_embedding(text, model, tok)  → (768,) tensor via BERT CLS
get_z_at(wav_path, transcript, ...)         → (1536,) = concat(acoustic, linguistic)
"""
from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

log = logging.getLogger(__name__)

# Shared model instances (loaded once per process)
_wav2vec_model  = None
_wav2vec_proc   = None
_bert_model     = None
_bert_tokenizer = None
_whisper_model  = None


def _load_wav2vec(model_name: str = "facebook/wav2vec2-base") -> Tuple:
    global _wav2vec_model, _wav2vec_proc
    if _wav2vec_model is None:
        from transformers import Wav2Vec2Model, Wav2Vec2Processor
        log.info(f"Loading Wav2Vec2: {model_name}")
        _wav2vec_proc  = Wav2Vec2Processor.from_pretrained(model_name)
        _wav2vec_model = Wav2Vec2Model.from_pretrained(model_name)
        _wav2vec_model.eval()
    return _wav2vec_model, _wav2vec_proc


def _load_bert(model_name: str = "bert-base-uncased") -> Tuple:
    global _bert_model, _bert_tokenizer
    if _bert_model is None:
        from transformers import BertModel, BertTokenizer
        log.info(f"Loading BERT: {model_name}")
        _bert_tokenizer = BertTokenizer.from_pretrained(model_name)
        _bert_model     = BertModel.from_pretrained(model_name)
        _bert_model.eval()
    return _bert_model, _bert_tokenizer


def _load_whisper(model_name: str = "openai/whisper-base") -> object:
    global _whisper_model
    if _whisper_model is None:
        import whisper
        log.info(f"Loading Whisper: {model_name.split('/')[-1]}")
        size = model_name.split("/")[-1].replace("whisper-", "")
        _whisper_model = whisper.load_model(size)
    return _whisper_model


# ── ffmpeg audio extraction ────────────────────────────────────────────────────

def extract_audio_to_wav(
    video_path: str | Path,
    out_wav: str | Path,
    sample_rate: int = 16000,
) -> bool:
    """Extract mono 16kHz WAV from video. Returns True on success."""
    result = subprocess.run(
        [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(video_path), "-vn",
            "-acodec", "pcm_s16le", "-ar", str(sample_rate), "-ac", "1",
            str(out_wav),
        ],
        capture_output=True, timeout=120,
    )
    return result.returncode == 0


# ── Acoustic embedding (Wav2Vec2) ──────────────────────────────────────────────

def get_acoustic_embedding(
    wav_path: str | Path,
    model_name: str = "facebook/wav2vec2-base",
    device: str = "cpu",
    max_seconds: int = 30,
) -> torch.Tensor:
    """
    Load WAV, run Wav2Vec2, mean-pool temporal dim.
    Returns (768,) float32 tensor.
    """
    import torchaudio
    model, processor = _load_wav2vec(model_name)
    model = model.to(device)

    waveform, sr = torchaudio.load(str(wav_path))
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)
    waveform = waveform.mean(dim=0)  # mono

    max_samples = max_seconds * 16000
    if waveform.shape[0] > max_samples:
        waveform = waveform[:max_samples]

    inputs = processor(
        waveform.numpy(), sampling_rate=16000, return_tensors="pt", padding=True
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        out = model(**inputs)
    emb = out.last_hidden_state.mean(dim=1).squeeze(0).cpu()  # (768,)
    return emb


# ── ASR transcription (Whisper) ────────────────────────────────────────────────

def transcribe(
    wav_path: str | Path,
    model_name: str = "openai/whisper-base",
) -> str:
    """Transcribe WAV file using Whisper. Returns text string."""
    try:
        wm = _load_whisper(model_name)
        result = wm.transcribe(str(wav_path), fp16=False)
        return result.get("text", "").strip()
    except Exception as e:
        log.warning(f"Whisper transcription failed for {wav_path}: {e}")
        return ""


# ── Linguistic embedding (BERT) ────────────────────────────────────────────────

def get_linguistic_embedding(
    text: str,
    model_name: str = "bert-base-uncased",
    device: str = "cpu",
) -> torch.Tensor:
    """
    Tokenize text, run BERT, return CLS token embedding.
    Returns (768,) float32 tensor. Empty text returns zero vector.
    """
    if not text:
        return torch.zeros(768)

    model, tokenizer = _load_bert(model_name)
    model = model.to(device)

    enc = tokenizer(
        text, return_tensors="pt",
        truncation=True, max_length=512, padding=True,
    )
    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():
        out = model(**enc)
    cls = out.last_hidden_state[:, 0, :].squeeze(0).cpu()  # (768,)
    return cls


# ── Z_at composite embedding ───────────────────────────────────────────────────

def get_z_at(
    wav_path: str | Path,
    transcript: Optional[str] = None,
    wav2vec_model: str = "facebook/wav2vec2-base",
    bert_model:    str = "bert-base-uncased",
    device:        str = "cpu",
    max_seconds:   int = 30,
) -> torch.Tensor:
    """
    Compute Z_at = concat(acoustic_emb, linguistic_emb) → (1536,).
    If transcript is None, runs Whisper first.
    """
    if transcript is None:
        transcript = transcribe(wav_path)

    acoustic   = get_acoustic_embedding(wav_path, wav2vec_model, device, max_seconds)
    linguistic = get_linguistic_embedding(transcript, bert_model, device)
    return torch.cat([acoustic, linguistic], dim=0)  # (1536,)
