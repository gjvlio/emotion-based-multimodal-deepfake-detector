"""
app.py — Hugging Face Space Entrypoint for DeepSentinel
Runs on free Gradio SDK without requiring Docker or paid billing.
Exposes full FastAPI endpoints (/detect, /detect/stream, /health, /warmup/status)
while displaying a live service status dashboard on port 7860.
"""
# 1. ZeroGPU must be imported before any other packages
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

import os
import sys
from pathlib import Path

# Force CPU device and disable boot warmup so ZeroGPU CUDA emulation is not tripped at startup
os.environ["DEEPSENTINEL_DEVICE"] = "cpu"
os.environ["DEEPSENTINEL_WARMUP"] = "0"
os.environ.setdefault("DEEPSENTINEL_UPLOAD_DIR", "/tmp/deepsentinel_uploads")

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


import gradio as gr

# Patch Gradio 4.44.0 / Pydantic 2.11+ incompatibility where bool additionalProperties crashes get_type
gr.Blocks.get_api_info = lambda *args, **kwargs: {"named_endpoints": {}, "unnamed_endpoints": {}}

try:
    import gradio_client.utils as gc_utils
    import gradio.blocks as gr_blocks
    def _safe_get_type(schema):
        if isinstance(schema, bool) or not isinstance(schema, dict):
            return "Any"
        return "Any" if "const" not in schema else schema.get("type", "Any")
    gc_utils.get_type = _safe_get_type
    if hasattr(gr_blocks, "client_utils"):
        gr_blocks.client_utils.get_type = _safe_get_type
        gr_blocks.client_utils.json_schema_to_python_type = lambda *args, **kwargs: "Any"
except Exception as e:
    print(f"[DeepSentinel] gradio schema patch notice: {e}")

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
        probe_btn.click(fn=predict_video_gpu, inputs=[video_input], outputs=[probe_output], api_name=False)

# Mount Gradio onto the root of the application so Space health checks pass
app = gr.mount_gradio_app(fastapi_app, demo, path="/")

# Pre-populate api_info on the mounted Gradio App so it never executes the broken
# Pydantic 2.11 boolean schema generator during GET /
safe_info = {"named_endpoints": {}, "unnamed_endpoints": {}, "_safe": True}
for route in app.routes:
    if hasattr(route, "app") and hasattr(route.app, "api_info"):
        route.app.api_info = safe_info
        route.app.all_app_info = safe_info

# Explicit ZeroGPU startup dispatch: ZeroGPU normally triggers client.startup_report()
# through gr.Blocks.launch(). Since gr.mount_gradio_app bypasses launch(), we invoke
# the startup hook directly to notify the ZeroGPU orchestrator of our @spaces.GPU targets.
_zerogpu_notified = False

def _report_zerogpu():
    global _zerogpu_notified
    if _zerogpu_notified:
        return
    try:
        from spaces.zero import client as zero_client
        from spaces.zero import torch as zero_torch
        from spaces.zero import decorator as zero_decorator
        zero_torch.pack()
        if len(zero_decorator.decorated_cache) > 0:
            print(f"[DeepSentinel] ZeroGPU targets registered: {len(zero_decorator.decorated_cache)}")
            zero_client.startup_report()
            _zerogpu_notified = True
            print("[DeepSentinel] ZeroGPU startup report transmitted successfully!")
    except Exception as e:
        print(f"[DeepSentinel] ZeroGPU startup hook notice: {e}")

_report_zerogpu()

@fastapi_app.on_event("startup")
def _on_fastapi_startup():
    _report_zerogpu()

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)



