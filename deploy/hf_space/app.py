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

# If the 1.16 GB checkpoint is not inside the repo, fetch it from private HF Model Hub
MODEL_REPO = os.environ.get("DEEPSENTINEL_MODEL_REPO", "gjrvlio/deepsentinel-weights")
CKPT_FILE = Path("checkpoints/best_phase2_adapted.pt")

if not CKPT_FILE.exists():
    token = (
        os.environ.get("HF_TOKEN")
        or os.environ.get("hf_token")
        or os.environ.get("HF_ACCESS_TOKEN")
        or os.environ.get("HUGGINGFACE_TOKEN")
        or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        or os.environ.get("HF_READ_TOKEN")
        or os.environ.get("TOKEN")
    )
    if token:
        masked = token[:4] + "..." + token[-4:] if len(token) > 8 else "***"
        print(f"[DeepSentinel] Auth token detected ({masked}). Downloading weights from {MODEL_REPO}...")
    else:
        print("[DeepSentinel] WARNING: No HF_TOKEN detected in environment!")
        print("[DeepSentinel] 'gjrvlio/deepsentinel-weights' is private. Please add 'HF_TOKEN' as a Secret in Space Settings > Variables and secrets.")

    try:
        from huggingface_hub import hf_hub_download
        print(f"[DeepSentinel] Fetching model weights from {MODEL_REPO} ({CKPT_FILE.name})...")
        CKPT_FILE.parent.mkdir(parents=True, exist_ok=True)
        downloaded = hf_hub_download(
            repo_id=MODEL_REPO,
            filename=CKPT_FILE.name,
            local_dir=str(CKPT_FILE.parent),
            repo_type="model",
            token=token if token else None,
        )
        print(f"[DeepSentinel] Weights ready at {downloaded} ({os.path.getsize(downloaded):,} bytes).")
    except Exception as e:
        print(f"[DeepSentinel] Hub download notice: {e}")

try:
    import spaces
except ImportError:
    class _MockSpaces:
        @staticmethod
        def GPU(func=None, duration=60):
            if func is None:
                return lambda f: f
            return func
    spaces = _MockSpaces()

import gradio as gr
from webapp.main import app as fastapi_app

@spaces.GPU(duration=120)
def predict_video_gpu(video_file):
    """ZeroGPU probe handler allowing direct forensic inference on Nvidia A10G."""
    if not video_file:
        return {"status": "error", "message": "No video uploaded."}
    from webapp.main import _service
    svc = _service()
    if not svc:
        return {"status": "error", "message": "Neural engine not loaded."}
    return svc.predict(Path(video_file))

# Gradio interactive status dashboard
with gr.Blocks(title="DeepSentinel Neural Engine") as demo:
    gr.Markdown(
        """
        # 🛡️ DeepSentinel — Neural Inference Engine
        ### Status: 🟢 Operational & Listening for Requests
        
        This Space runs the multimodal deep learning inference pipeline (Whisper, ArcFace, Swin Transformer, Wav2Vec 2.0).
        It communicates directly with the DeepSentinel Vercel frontend.
        """
    )
    with gr.Tab("Active API Endpoints"):
        gr.Markdown(
            """
            #### Registered Endpoints:
            - `POST /detect/stream` — Real-time Server-Sent Events (SSE) telemetry stream
            - `POST /detect` — Synchronous full-video forensic analysis
            - `GET /health` — Service readiness and checkpoint status
            - `GET /warmup/status` — Model warmup progress tracker
            """
        )
    with gr.Tab("Forensic Probe (A10G GPU)"):
        with gr.Row():
            video_input = gr.Video(label="Upload Video for Test Analysis")
            probe_output = gr.JSON(label="Forensic Report")
        probe_btn = gr.Button("Analyze Video with GPU", variant="primary")
        probe_btn.click(fn=predict_video_gpu, inputs=[video_input], outputs=[probe_output])

# Mount Gradio onto the root of the application so Space health checks pass
app = gr.mount_gradio_app(fastapi_app, demo, path="/")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)


