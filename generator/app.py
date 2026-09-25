"""
generator/app.py
================
FastAPI application for the Deepfake Generator Companion Tool.
Provides interactive UI, webcam recording ingestion, deepfake generation,
video streaming, and live detection integration.
"""

import os
import sys
import shutil
import logging
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import urllib.request
import urllib.parse
import json

from generator.service import (
    generator_service,
    probe_media,
    REPO_ROOT,
    OUTPUTS_DIR,
    TEMP_DIR,
)

logger = logging.getLogger("generator.app")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

app = FastAPI(
    title="Deepfake Generator Companion",
    description="Thesis Defense Deepfake Synthesis & Demonstration Tool",
    version="1.0.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directories
STATIC_DIR = Path(__file__).resolve().parent / "static"
TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"

STATIC_DIR.mkdir(parents=True, exist_ok=True)
TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)

# Mount static and outputs
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Render the Deepfake Generator Dashboard."""
    presets = generator_service.get_presets()
    has_wav2lip = generator_service.checkpoint_path.exists()
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "presets": presets,
            "has_wav2lip": has_wav2lip,
            "wav2lip_model": generator_service.checkpoint_path.name if has_wav2lip else "Not Found",
        },
    )


@app.get("/api/presets")
async def get_presets():
    """List preset sample clips available for demo testing."""
    return {"presets": generator_service.get_presets()}


@app.get("/outputs/{filename}")
async def get_output_video(filename: str):
    """Stream generated deepfake video with Range header support."""
    file_path = OUTPUTS_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Generated video not found")
    return FileResponse(path=str(file_path), media_type="video/mp4", filename=filename)


@app.post("/api/generate")
async def generate_deepfake(
    face_file: Optional[UploadFile] = File(None),
    face_preset: Optional[str] = Form(None),
    donor_file: Optional[UploadFile] = File(None),
    donor_preset: Optional[str] = Form(None),
    mode: str = Form("wav2lip"),
    resize_factor: int = Form(2),
    pad_bottom: int = Form(10),
    nosmooth: bool = Form(False),
):
    """
    Generate deepfake from user face and donor material.
    Supports file uploads, browser webcam blobs, and presets.
    """
    upload_temp_files = []
    try:
        # 1. Resolve Face Material
        if face_file and face_file.filename:
            # Save uploaded face (can be webm or mp4)
            ext = Path(face_file.filename).suffix or ".mp4"
            face_path = TEMP_DIR / f"input_face_{os.urandom(6).hex()}{ext}"
            with open(face_path, "wb") as f:
                content = await face_file.read()
                f.write(content)
            upload_temp_files.append(face_path)
        elif face_preset:
            face_path = Path(face_preset)
            if not face_path.exists():
                raise HTTPException(status_code=400, detail=f"Face preset not found: {face_preset}")
        else:
            raise HTTPException(status_code=400, detail="Missing original face video (upload or select preset).")

        # 2. Resolve Donor / Tampering Material
        if donor_file and donor_file.filename:
            ext = Path(donor_file.filename).suffix or ".mp4"
            donor_path = TEMP_DIR / f"input_donor_{os.urandom(6).hex()}{ext}"
            with open(donor_path, "wb") as f:
                content = await donor_file.read()
                f.write(content)
            upload_temp_files.append(donor_path)
        elif donor_preset:
            donor_path = Path(donor_preset)
            if not donor_path.exists():
                raise HTTPException(status_code=400, detail=f"Donor preset not found: {donor_preset}")
        else:
            raise HTTPException(status_code=400, detail="Missing donor tampering material (upload or select preset).")

        # 3. Execute Selected Mode
        if mode == "wav2lip":
            pads = [0, pad_bottom, 0, 0]
            result = generator_service.generate_wav2lip(
                face_path=face_path,
                donor_path=donor_path,
                resize_factor=resize_factor,
                pads=pads,
                nosmooth=nosmooth,
            )
        elif mode == "audio_swap":
            result = generator_service.generate_audio_swap(
                face_path=face_path,
                donor_path=donor_path,
            )
        else:
            raise HTTPException(status_code=400, detail=f"Unknown generation mode: {mode}")

        if not result.get("success"):
            return JSONResponse(status_code=500, content=result)

        # Include input probe info for visualization
        result["face_info"] = probe_media(face_path)
        result["donor_info"] = probe_media(donor_path)
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error in /api/generate")
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})
    finally:
        # Clean up temporary uploaded files
        for p in upload_temp_files:
            try:
                if p.exists():
                    p.unlink()
            except Exception:
                pass


@app.post("/api/send_to_detector")
async def send_to_detector(filename: str = Form(...)):
    """
    Integration endpoint: Submits the generated deepfake directly to
    DeepSentinel (http://localhost:8000/detect) to show live detection score.
    """
    file_path = OUTPUTS_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    detector_url = "http://localhost:8000/detect"
    try:
        # Perform multipart upload to DeepSentinel
        boundary = "----DeepSentinelBoundary" + os.urandom(8).hex()
        with open(file_path, "rb") as f:
            file_data = f.read()

        body = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
            f"Content-Type: video/mp4\r\n\r\n"
        ).encode("utf-8") + file_data + f"\r\n--{boundary}--\r\n".encode("utf-8")

        req = urllib.request.Request(
            detector_url,
            data=body,
            headers={
                "Content-Type": f"multipart/form-data; boundary={boundary}",
                "Content-Length": str(len(body)),
            },
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return {
                "detector_active": True,
                "result": data,
                "deepfake_detected": data.get("prediction", "").upper() == "FAKE",
                "confidence": data.get("confidence", 0),
                "label": data.get("prediction", "UNKNOWN"),
            }

    except urllib.error.URLError:
        return {
            "detector_active": False,
            "message": "DeepSentinel detector is not running on http://localhost:8000. Launch it via 'python -m webapp' to test live.",
        }
    except Exception as e:
        return {
            "detector_active": False,
            "error": str(e),
            "message": f"Detector error: {str(e)}",
        }
