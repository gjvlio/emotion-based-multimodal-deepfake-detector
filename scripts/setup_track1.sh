#!/bin/bash
# ============================================================
# Track 1 Setup Script — Audio Tampering Pipeline
# Run this once before anything else.
# ============================================================
set -e

echo "=== [1/5] Installing PyTorch (CUDA 11.8) ==="
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118 --quiet

echo "=== [2/5] Installing audio processing libraries ==="
pip install \
    librosa \
    soundfile \
    numpy \
    pandas \
    tqdm \
    scipy \
    --quiet

echo "=== [3/5] Installing speaker verification (x-vector filter) ==="
pip install speechbrain --quiet

echo "=== [4/5] Installing StyleTTS 2 (Method B - SOTA synthesis) ==="
# styletts2 pip package wraps the official model with easy emotion control
pip install styletts2 --quiet

echo "=== [5/5] Installing RVC python bindings (Method B - speaker transfer) ==="
# rvc-python is broken on Python 3.11 (omegaconf==2.0.6 invalid metadata on pip>=24.1)
# infer-rvc-python is the maintained replacement with the same API
pip install infer-rvc-python --quiet

echo ""
echo "=== Verifying ffmpeg ==="
ffmpeg -version 2>/dev/null | head -1

echo ""
echo "=== All done. Run parse_cremad.py next. ==="
