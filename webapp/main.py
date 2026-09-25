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

import asyncio
import json
import logging
import re
import shutil
import subprocess
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .config import settings
from .input_validator import InputValidationError
from .schemas import DetectionResult, HealthResponse, ModelInfo

STATIC_DIR = Path(__file__).parent / "static"

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("deepsentinel.api")

ALLOWED_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".wav"}


def _cleanup_old_uploads(max_files: int = 50, max_age_hours: float = 2.0) -> None:
    """Prune stale uploaded clips to prevent server disk exhaustion."""
    try:
        if not settings.upload_dir.exists():
            return
        now = time.time()
        files = sorted(settings.upload_dir.glob("*"), key=lambda p: p.stat().st_mtime)
        # Delete if older than max_age_hours
        for p in files:
            if p.is_file() and (now - p.stat().st_mtime) > (max_age_hours * 3600):
                p.unlink(missing_ok=True)
        # If count still exceeds max_files, prune oldest
        remaining = sorted(settings.upload_dir.glob("*"), key=lambda p: p.stat().st_mtime)
        if len(remaining) > max_files:
            for p in remaining[: len(remaining) - max_files]:
                if p.is_file():
                    p.unlink(missing_ok=True)
    except Exception as e:
        log.debug(f"Upload cleanup notice: {e}")

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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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


@app.get("/warmup/status")
def warmup_status():
    svc = _service()
    if not svc:
        return {
            "status": "ready",
            "progress": 1.0,
            "stage": "ready",
            "target": "Demo Static Mode",
            "device": "demo",
            "warmed": True,
            "checkpoint": "demo-mode",
        }
    return svc.warmup_status()


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


def _prepare_clip(file: UploadFile, start_time: float, end_time: Optional[float]) -> Path:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise InputValidationError(
            code="ERR_UNSUPPORTED_FORMAT",
            title="Unsupported Video Format",
            message=f"The file extension '{suffix}' is not supported.",
            suggestion=f"Please upload a supported video format: {', '.join(sorted(ALLOWED_SUFFIXES))}.",
            details={"suffix": suffix},
        )

    # Periodic cleanup of old uploads
    _cleanup_old_uploads()

    # Sanitize and create collision-resistant unique filename
    raw_name = Path(file.filename or "upload.mp4").name
    clean_name = re.sub(r"[^a-zA-Z0-9_.-]", "_", raw_name)
    unique_name = f"{uuid.uuid4().hex[:8]}_{clean_name}"

    # Persist upload
    dest = settings.upload_dir / unique_name
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    file_bytes = dest.stat().st_size
    if file_bytes < 1000:
        raise InputValidationError(
            code="ERR_EMPTY_FILE",
            title="Empty or Corrupted File",
            message="The uploaded video is empty or smaller than 1 KB.",
            suggestion="Please check that the video file is valid and re-export if needed.",
            details={"bytes": file_bytes},
        )
    if file_bytes > 500 * 1024 * 1024:
        raise InputValidationError(
            code="ERR_FILE_TOO_LARGE",
            title="File Exceeds 500MB Limit",
            message=f"The uploaded video ({file_bytes / (1024 * 1024):.1f} MB) exceeds the 500 MB limit.",
            suggestion="Please compress the video or select a smaller clip under 500 MB.",
            details={"bytes": file_bytes, "max_bytes": 500 * 1024 * 1024},
        )

    # Pre-check overall video duration
    total_dur = None
    try:
        import cv2
        cap = cv2.VideoCapture(str(dest))
        if cap.isOpened():
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
            total_dur = frame_count / fps if fps > 0 else 0
            cap.release()
            if total_dur > settings.max_upload_duration_sec:
                raise InputValidationError(
                    code="ERR_DURATION_TOO_LONG",
                    title="Video Exceeds Time Limit",
                    message=f"Video duration ({total_dur / 60.0:.1f} min) exceeds the 10-minute maximum limit.",
                    suggestion="Please upload or trim a video under 10 minutes.",
                    details={"duration_sec": total_dur, "max_allowed": settings.max_upload_duration_sec},
                )
    except InputValidationError:
        raise
    except Exception as e:
        log.debug(f"Duration inspection bypassed: {e}")

    # Clip extraction / trimming logic
    clip_to_eval = dest
    t_start = max(0.0, float(start_time or 0.0))
    t_end = float(end_time) if end_time is not None and float(end_time) > 0 else None

    # If a sub-clip is requested or video is longer than max_duration_sec (20s)
    if (t_start > 0.05) or (t_end is not None) or (total_dur and total_dur > settings.max_duration_sec + 0.5):
        if t_end is None:
            t_end = t_start + settings.max_duration_sec

        if t_end <= t_start:
            raise InputValidationError(
                code="ERR_INVALID_CROP_RANGE",
                title="Invalid Selection Range",
                message="Start time must be before end time.",
                suggestion="Please drag the timeline handles to select a valid forward time window.",
                details={"start_time": t_start, "end_time": t_end},
            )

        slice_dur = t_end - t_start
        if slice_dur < settings.min_duration_sec - 0.2:
            raise InputValidationError(
                code="ERR_CROP_TOO_SHORT",
                title="Selected Clip Is Too Short",
                message=f"Selected clip ({slice_dur:.1f}s) is shorter than the minimum {settings.min_duration_sec:.1f}s required.",
                suggestion=f"Please drag the timeline scrubber to select at least {settings.min_duration_sec:.0f} seconds.",
                details={"duration": slice_dur, "min_required": settings.min_duration_sec},
            )
        if slice_dur > settings.max_duration_sec + 0.5:
            raise InputValidationError(
                code="ERR_CROP_TOO_LONG",
                title="Selected Clip Is Too Long",
                message=f"Selected clip ({slice_dur:.1f}s) exceeds the maximum {settings.max_duration_sec:.1f}s allowed.",
                suggestion=f"Please drag the timeline scrubber to select at most {settings.max_duration_sec:.0f} seconds.",
                details={"duration": slice_dur, "max_allowed": settings.max_duration_sec},
            )

        trimmed_name = f"trim_{int(t_start * 100)}_{int(t_end * 100)}_{unique_name}"
        trimmed_dest = settings.upload_dir / trimmed_name

        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-ss", f"{t_start:.3f}",
            "-to", f"{t_end:.3f}",
            "-i", str(dest),
            "-c:v", "libx264", "-preset", "ultrafast",
            "-c:a", "aac",
            "-avoid_negative_ts", "make_zero",
            str(trimmed_dest),
        ]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            if r.returncode == 0 and trimmed_dest.exists() and trimmed_dest.stat().st_size > 1000:
                clip_to_eval = trimmed_dest
            else:
                log.warning(f"ffmpeg trim notice: {r.stderr}; using source file")
        except Exception as e:
            log.warning(f"ffmpeg slicing exception ({e}); using source file")

    # If untrimmed non-mp4 (e.g. webm, mov, mkv), remux to CFR H.264 mp4 for reliable cv2 seeking
    if clip_to_eval == dest and dest.suffix.lower() != ".mp4":
        norm_name = f"norm_{dest.stem}_{unique_name}.mp4"
        norm_dest = settings.upload_dir / norm_name
        n_cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(dest),
            "-c:v", "libx264", "-preset", "ultrafast",
            "-c:a", "aac",
            str(norm_dest),
        ]
        try:
            r = subprocess.run(n_cmd, capture_output=True, text=True, timeout=30)
            if r.returncode == 0 and norm_dest.exists() and norm_dest.stat().st_size > 1000:
                clip_to_eval = norm_dest
        except Exception as e:
            log.warning(f"ffmpeg format normalization notice ({e}); using source file")

    return clip_to_eval


# Global GPU semaphore to serialize neural inference and prevent CUDA OOM under concurrency
_gpu_semaphore = asyncio.Semaphore(1)


@app.post("/detect", response_model=DetectionResult)
async def detect(
    file: UploadFile = File(...),
    start_time: float = Form(0.0),
    end_time: Optional[float] = Form(None),
):
    svc = _service()
    if not svc:
        raise HTTPException(
            status_code=503,
            detail="Live detection needs the ML stack (torch/transformers). Use Demo mode (/demo) for walkthrough.",
        )
    async with _gpu_semaphore:
        try:
            clip_to_eval = _prepare_clip(file, start_time, end_time)
            return svc.predict(clip_to_eval)
        except InputValidationError as e:
            raise HTTPException(status_code=422, detail=e.to_dict())
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))
        except RuntimeError as e:
            raise HTTPException(status_code=503, detail=str(e))
        except Exception as e:
            log.exception("Detection failed")
            raise HTTPException(status_code=500, detail=f"Detection error: {e}")


@app.post("/detect/stream")
async def detect_stream(
    file: UploadFile = File(...),
    start_time: float = Form(0.0),
    end_time: Optional[float] = Form(None),
):
    svc = _service()
    if not svc:
        raise HTTPException(
            status_code=503,
            detail="Live detection needs the ML stack (torch/transformers).",
        )

    try:
        clip_to_eval = _prepare_clip(file, start_time, end_time)
    except InputValidationError as e:
        def err_generator():
            yield f"data: {json.dumps({'error': e.to_dict()})}\n\n"
        return StreamingResponse(err_generator(), media_type="text/event-stream")
    except Exception as e:
        def err_generator():
            err_dict = {
                "code": "ERR_PREPARE_CLIP_FAILED",
                "title": "Clip Preparation Error",
                "message": str(e),
                "suggestion": "Please check your video format and try again.",
            }
            yield f"data: {json.dumps({'error': err_dict})}\n\n"
        return StreamingResponse(err_generator(), media_type="text/event-stream")

    async def event_generator():
        async with _gpu_semaphore:
            try:
                for event in svc.predict_stream(clip_to_eval):
                    yield f"data: {json.dumps(event)}\n\n"
                    await asyncio.sleep(0.005)
            except InputValidationError as e:
                yield f"data: {json.dumps({'error': e.to_dict()})}\n\n"
            except Exception as e:
                log.exception("Stream detection error")
                err_dict = {
                    "code": "ERR_INTERNAL",
                    "title": "Pipeline Processing Error",
                    "message": str(e),
                    "suggestion": "Please check the clip formatting or try another video.",
                }
                yield f"data: {json.dumps({'error': err_dict})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


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
