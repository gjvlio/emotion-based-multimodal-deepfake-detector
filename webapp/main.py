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
from .schemas import DetectionResult, HealthResponse, ModelInfo

STATIC_DIR = Path(__file__).parent / "static"

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("deepsentinel.api")

ALLOWED_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".wav"}

# The model backend (torch/transformers/...) is OPTIONAL. A lightweight checkout
# with only FastAPI installed still serves the full UI and the /demo flow — only
# the live /detect endpoint needs the ML stack. So the heavy imports are lazy and
# tolerated: if they fail, the app runs in demo/static-only mode.
_svc_cache = None
_svc_tried = False


def _service():
    global _svc_cache, _svc_tried
    if not _svc_tried:
        _svc_tried = True
        try:
            from .model_service import get_service
            _svc_cache = get_service()
        except Exception as e:  # noqa: BLE001 — torch / ML deps not installed
            log.warning(f"Model backend unavailable ({type(e).__name__}: {e}). "
                        f"Running in demo/static mode — /detect disabled; UI + /demo work.")
            _svc_cache = None
    return _svc_cache


def _demo_info(note: str) -> ModelInfo:
    return ModelInfo(loaded=False, device="n/a", note=note)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Build the model service if the ML stack is present; otherwise demo/static mode.
    svc = _service()
    if svc:
        svc.start_watcher()
        if settings.warmup_on_start:
            svc.start_warmup()  # background — boot stays fast, /detect warms behind it
        log.info("DeepSentinel service ready (models warming in background).")
    else:
        log.info("DeepSentinel running in DEMO/STATIC mode (no model backend installed).")
    yield
    if svc:
        svc.stop_watcher()


app = FastAPI(
    title="DeepSentinel API",
    description="Multimodal emotion-aware deepfake detector. "
                "Auto-equips the latest training checkpoint.",
    version="0.1.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def no_cache(request, call_next):
    """Never cache anything — guarantees the browser always gets the latest UI."""
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


@app.get("/health", response_model=HealthResponse)
def health():
    svc = _service()
    if not svc:
        return HealthResponse(status="demo_only", model=_demo_info("Demo/static mode — ML backend not installed."))
    svc.maybe_reload()
    meta = svc.info()
    return HealthResponse(status="ok" if meta.loaded else "no_model", model=meta)


@app.get("/model/info", response_model=ModelInfo)
def model_info():
    svc = _service()
    if not svc:
        return _demo_info("Demo/static mode — ML backend not installed.")
    svc.maybe_reload()
    return svc.info()


@app.post("/model/reload", response_model=ModelInfo)
def model_reload():
    svc = _service()
    if not svc:
        return _demo_info("Demo/static mode — nothing to reload.")
    reloaded = svc.maybe_reload(force=True)
    meta = svc.info()
    meta.note = (meta.note or "") + (" [reloaded]" if reloaded else " [no change]")
    return meta


@app.post("/detect", response_model=DetectionResult)
async def detect(file: UploadFile = File(...)):
    svc = _service()
    if not svc:
        raise HTTPException(
            status_code=503,
            detail="Live detection needs the ML stack (torch/transformers), not installed on this "
                   "checkout. Use Demo mode (/demo) for the hardcoded walkthrough.",
        )

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
    "/demo", "/demo/upload", "/demo/analyzing", "/demo/results",
    "/demo/about", "/demo/about/thesis", "/demo/about/researchers",
}


@app.get("/{full_path:path}", include_in_schema=False)
def spa_shell(full_path: str):
    """Serve the SPA shell for known view routes; 404 otherwise.
    The optional /demo prefix maps onto the same views (hardcoded demo mode)."""
    p = "/" + full_path
    if p == "/demo" or p.startswith("/demo/"):
        p = p[len("/demo"):] or "/"
    if p in SPA_PATHS or full_path == "":
        # never cache the shell so updated css/js are always picked up
        return FileResponse(STATIC_DIR / "index.html", headers={"Cache-Control": "no-store"})
    raise HTTPException(status_code=404, detail="Not found")
