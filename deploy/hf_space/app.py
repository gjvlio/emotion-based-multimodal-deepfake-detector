"""
app.py — Hugging Face Space Entrypoint for DeepSentinel
Runs on free Gradio SDK without requiring Docker or paid billing.
Exposes full FastAPI endpoints (/detect, /detect/stream, /health, /warmup/status)
while displaying a live service status dashboard on port 7860.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Hugging Face Spaces have a writable /tmp directory
os.environ.setdefault("DEEPSENTINEL_UPLOAD_DIR", "/tmp/deepsentinel_uploads")
os.environ.setdefault("DEEPSENTINEL_WARMUP", "1")

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

Path(os.environ["DEEPSENTINEL_UPLOAD_DIR"]).mkdir(parents=True, exist_ok=True)

import gradio as gr
from webapp.main import app as fastapi_app

# Gradio interactive status dashboard
with gr.Blocks(title="DeepSentinel Neural Engine") as demo:
    gr.Markdown(
        """
        # 🛡️ DeepSentinel — Neural Inference Engine
        ### Status: 🟢 Operational & Listening for Requests
        
        This private Space runs the deep learning inference pipeline (Whisper, ArcFace, Swin Transformer, Wav2Vec 2.0).
        It communicates directly with the DeepSentinel Vercel frontend.

        #### Active Endpoints:
        - `POST /detect/stream` — Real-time Server-Sent Events (SSE) telemetry stream
        - `POST /detect` — Synchronous full-video forensic analysis
        - `GET /health` — Service readiness and checkpoint status
        - `GET /warmup/status` — Model warmup progress tracker
        """
    )

# Mount Gradio onto the existing FastAPI application
app = gr.mount_gradio_app(fastapi_app, demo, path="/gradio")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
