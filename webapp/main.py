"""
main.py — DeepSentinel FastAPI service.

Run:
    uvicorn webapp.main:app --reload --port 8000
    # or: python -m webapp   (see __main__.py)

Endpoints:
    GET  /              — liveness ping
    GET  /health        — service status + currently equipped model
    GET  /model/info    — equipped checkpoint metadata
    POST /model/reload  — force a checkpoint re-check (normally automatic)
    POST /detect        — upload a video, get a real/fake verdict

No UI yet — this is the model-serving backend. Frontend comes later.
"""
from __future__ import annotations

import logging
import shutil
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .config import settings
from .model_service import get_service
from .schemas import DetectionResult, HealthResponse, ModelInfo

STATIC_DIR = Path(__file__).parent / "static"

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("deepsentinel.api")

ALLOWED_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".wav"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Build the model service, start the auto-equip watcher, and preload models.
    svc = get_service()
    svc.start_watcher()
    if settings.warmup_on_start:
        svc.start_warmup()  # background — boot stays fast, /detect warms behind it
    log.info("DeepSentinel service ready (models warming in background).")
    yield
    svc.stop_watcher()


app = FastAPI(
    title="DeepSentinel API",
    description="Multimodal emotion-aware deepfake detector. "
                "Auto-equips the latest training checkpoint.",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health():
    svc = get_service()
    svc.maybe_reload()
    meta = svc.info()
    return HealthResponse(status="ok" if meta.loaded else "no_model", model=meta)


@app.get("/model/info", response_model=ModelInfo)
def model_info():
    svc = get_service()
    svc.maybe_reload()
    return svc.info()


@app.post("/model/reload", response_model=ModelInfo)
def model_reload():
    svc = get_service()
    reloaded = svc.maybe_reload(force=True)
    meta = svc.info()
    meta.note = (meta.note or "") + (" [reloaded]" if reloaded else " [no change]")
    return meta


@app.post("/detect", response_model=DetectionResult)
async def detect(file: UploadFile = File(...)):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{suffix}'. Allowed: {sorted(ALLOWED_SUFFIXES)}",
        )

    # Persist upload (no auto-delete — user manages webapp/uploads/).
    dest = settings.upload_dir / file.filename
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    svc = get_service()
    try:
        return svc.predict(dest)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:  # noqa: BLE001
        log.exception("Detection failed")
        raise HTTPException(status_code=500, detail=f"Detection error: {e}")


# ── Frontend (SPA) ─────────────────────────────────────────────────────────────
# Static assets (css/js/img) under /static. The single-page app shell is served
# for every client-side route so deep links and refreshes work.
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Client-side routes handled by the SPA shell (History API navigation).
SPA_PATHS = {
    "/", "/upload", "/analyzing", "/results",
    "/about", "/about/thesis", "/about/researchers",
}


@app.get("/{full_path:path}", include_in_schema=False)
def spa_shell(full_path: str):
    """Serve the SPA shell for known view routes; 404 otherwise."""
    if ("/" + full_path) in SPA_PATHS or full_path == "":
        return FileResponse(STATIC_DIR / "index.html")
    raise HTTPException(status_code=404, detail="Not found")
