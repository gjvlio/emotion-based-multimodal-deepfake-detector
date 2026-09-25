---
title: DeepSentinel Neural Engine
emoji: 🛡️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# DeepSentinel Neural Backend (GPU Inference Service)

This Hugging Face Space serves the multimodal emotion-aware deepfake detection backend for DeepSentinel.

### Endpoints
- `POST /detect`: Synchronous video deepfake detection.
- `POST /detect/stream`: Real-time Server-Sent Events (SSE) telemetry stream.
- `GET /health`: Model status and equipped checkpoint metadata.
- `GET /warmup/status`: Warmup progress and neural engine readiness.
