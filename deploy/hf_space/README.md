---
title: DeepSentinel Neural Engine
emoji: 🛡️
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
---

# DeepSentinel Neural Backend (GPU / CPU Inference Service)

This private Hugging Face Space hosts the multimodal emotion-aware deepfake detection backend for DeepSentinel.

### Endpoints
- `POST /detect`: Synchronous video deepfake detection.
- `POST /detect/stream`: Real-time Server-Sent Events (SSE) telemetry stream.
- `GET /health`: Model status and equipped checkpoint metadata.
- `GET /warmup/status`: Warmup progress and neural engine readiness.
- `GET /gradio`: Status dashboard.
