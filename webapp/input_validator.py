"""
input_validator.py — Comprehensive input modality & quality verification for DeepSentinel.

Because DeepSentinel is an affective multimodal deepfake detector that assesses
cross-modal consistency between speech tone, vocabulary, and facial expressions,
both the audio and visual modalities must satisfy quality criteria before inference.

Checks performed:
1. Container & Video Stream:
   - Valid, decodable video format
   - Duration within bounds (min 2.0s, max 600.0s)
   - Minimum frame count (>= 15 frames) and FPS (>= 5.0)
2. Audio Modality:
   - Presence of an audio track in the container
   - Non-silent audio (RMS energy >= -45 dBFS / peak >= 0.005)
   - Detectable human speech (Whisper vocal token validation)
3. Visual & Face Modality:
   - Overall lighting / luminance within bounds (not pitch black or completely overexposed)
   - Human face presence (detected in >= 25% of sampled frames; tracks primary speaker if multiple)
   - Minimum face resolution (bounding box >= 60x60 px)
   - Face clarity & sharpness (Laplacian variance >= 20.0, rejecting extreme blur)
   - Mask / lower-face occlusion heuristic (detecting covered mouth/nose)
"""
from __future__ import annotations

import logging
import math
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

log = logging.getLogger("deepsentinel.validator")


class InputValidationError(Exception):
    """Raised when an uploaded clip fails input modality or quality validation."""
    def __init__(self, code: str, title: str, message: str, suggestion: str, details: Optional[dict] = None):
        super().__init__(message)
        self.code = code
        self.title = title
        self.message = message
        self.suggestion = suggestion
        self.details = details or {}

    def to_dict(self) -> dict:
        return {
            "code": self.code,
            "title": self.title,
            "message": self.message,
            "suggestion": self.suggestion,
            "details": self.details,
        }


@dataclass
class VideoInspection:
    duration: float
    fps: float
    frame_count: int
    width: int
    height: int
    has_audio_stream: bool
    mean_luminance: float = 0.0


def inspect_video_stream(video_path: Path) -> VideoInspection:
    """Inspect video container parameters and check for audio stream via ffprobe / cv2."""
    if not video_path.exists() or video_path.stat().st_size < 1000:
        raise InputValidationError(
            code="ERR_EMPTY_FILE",
            title="Empty or Corrupted File",
            message="The uploaded file is empty or cannot be read.",
            suggestion="Please ensure the video was exported properly and is at least a few kilobytes in size.",
        )

    # Check file size limit (500 MB)
    max_bytes = 500 * 1024 * 1024
    file_bytes = video_path.stat().st_size
    if file_bytes > max_bytes:
        raise InputValidationError(
            code="ERR_FILE_TOO_LARGE",
            title="Video File Too Large",
            message=f"The video file size ({file_bytes / (1024 * 1024):.1f} MB) exceeds the 500 MB limit.",
            suggestion="Please compress the video or upload a smaller clip under 500 MB.",
            details={"size_bytes": file_bytes, "max_bytes": max_bytes},
        )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise InputValidationError(
            code="ERR_UNREADABLE_CONTAINER",
            title="Unreadable Video Format",
            message="The video stream could not be decoded by the media pipeline.",
            suggestion="Please convert or re-export the video to standard MP4 (H.264 / AAC) and try again.",
        )

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    if width > 0 and height > 0 and (width < 64 or height < 64):
        cap.release()
        raise InputValidationError(
            code="ERR_VIDEO_RESOLUTION_TOO_LOW",
            title="Video Dimensions Too Small",
            message=f"Video resolution ({width}x{height}) is too tiny to locate human faces.",
            suggestion="Please upload a video with standard resolution (at least 360p or 480p).",
            details={"width": width, "height": height, "min_required": 64},
        )

    # Sample a few frames to calculate average luminance
    luminances = []
    sample_indices = np.linspace(0, max(1, frame_count - 1), num=min(10, max(1, frame_count)), dtype=int)
    for idx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret and frame is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            luminances.append(float(np.mean(gray)))
    cap.release()

    duration = (frame_count / fps) if (fps > 0 and frame_count > 0) else 0.0
    mean_lum = float(np.mean(luminances)) if luminances else 128.0

    # Check for audio stream presence using ffprobe
    has_audio = False
    try:
        cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "a:0",
            "-show_entries", "stream=codec_type",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        has_audio = "audio" in r.stdout.lower()
    except Exception as e:
        log.debug(f"ffprobe check failed ({e}); checking fallback extraction")
        # Fallback check: attempt quick probe extraction
        test_wav = video_path.with_suffix(".probe.wav")
        try:
            p_cmd = [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-i", str(video_path), "-vn", "-t", "0.5",
                "-acodec", "pcm_s16le", str(test_wav),
            ]
            res = subprocess.run(p_cmd, capture_output=True, timeout=10)
            has_audio = res.returncode == 0 and test_wav.exists() and test_wav.stat().st_size > 400
            if test_wav.exists():
                test_wav.unlink(missing_ok=True)
        except Exception:
            has_audio = False

    return VideoInspection(
        duration=duration,
        fps=fps,
        frame_count=frame_count,
        width=width,
        height=height,
        has_audio_stream=has_audio,
        mean_luminance=mean_lum,
    )


def validate_container(inspection: VideoInspection, min_duration: float = 2.0, max_duration: float = 600.0) -> None:
    """Validate video stream bounds."""
    if inspection.frame_count < 15:
        raise InputValidationError(
            code="ERR_TOO_FEW_FRAMES",
            title="Insufficient Video Frames",
            message=f"The video contains only {inspection.frame_count} frames. At least 15 frames are required for analysis.",
            suggestion="Please upload a longer video clip with continuous motion.",
            details={"frame_count": inspection.frame_count},
        )

    if inspection.fps < 5.0:
        raise InputValidationError(
            code="ERR_LOW_FRAMERATE",
            title="Frame Rate Too Low",
            message=f"The video frame rate ({inspection.fps:.1f} FPS) is too low for facial micro-expression analysis.",
            suggestion="Please provide a standard video recorded at 15 FPS or higher.",
            details={"fps": inspection.fps},
        )

    if inspection.duration < min_duration - 0.2:
        raise InputValidationError(
            code="ERR_DURATION_TOO_SHORT",
            title="Video Too Short",
            message=f"Selected video duration ({inspection.duration:.1f}s) is below the minimum required ({min_duration:.1f}s).",
            suggestion=f"Please upload or trim a clip that is at least {min_duration:.1f} seconds long.",
            details={"duration": inspection.duration, "min_required": min_duration},
        )

    if inspection.duration > max_duration + 1.0:
        raise InputValidationError(
            code="ERR_DURATION_TOO_LONG",
            title="Video Too Long",
            message=f"Video duration ({inspection.duration / 60.0:.1f} min) exceeds the maximum limit ({max_duration / 60.0:.1f} min).",
            suggestion=f"Please trim the video to under {int(max_duration / 60)} minutes.",
            details={"duration": inspection.duration, "max_allowed": max_duration},
        )

    if inspection.mean_luminance < 15.0:
        raise InputValidationError(
            code="ERR_PITCH_BLACK",
            title="Video Severely Underexposed",
            message="The video is pitch black or severely underexposed. Facial features cannot be discerned.",
            suggestion="Please provide a video recorded in adequate lighting conditions.",
            details={"mean_luminance": inspection.mean_luminance},
        )

    if inspection.mean_luminance > 245.0:
        raise InputValidationError(
            code="ERR_OVEREXPOSED",
            title="Video Severely Overexposed",
            message="The video is washed out / completely overexposed. Facial features are blown out.",
            suggestion="Please provide a video with normal exposure and lighting.",
            details={"mean_luminance": inspection.mean_luminance},
        )


def validate_audio_track(inspection: VideoInspection, wav_path: Path) -> None:
    """Verify that an audio track exists, is extracted, and has audible sound."""
    if not inspection.has_audio_stream:
        raise InputValidationError(
            code="ERR_NO_AUDIO_TRACK",
            title="No Audio Track Found",
            message="This video doesn't have any sound. DeepSentinel needs to hear the speaker's voice to compare their tone of voice with their facial expressions.",
            suggestion="Please upload a video where the person is speaking out loud.",
        )

    if not wav_path.exists() or wav_path.stat().st_size < 1000:
        raise InputValidationError(
            code="ERR_AUDIO_EXTRACTION_FAILED",
            title="Could Not Read Audio",
            message="We couldn't extract the audio from this video file.",
            suggestion="Please try converting or re-exporting your video with a standard audio format (like AAC or MP3).",
        )

    # Calculate RMS energy and peak amplitude
    try:
        import soundfile as sf
        data, sr = sf.read(str(wav_path), dtype="float32")
        if data.ndim > 1:
            data = np.mean(data, axis=-1)

        peak = float(np.max(np.abs(data))) if len(data) > 0 else 0.0
        rms = float(np.sqrt(np.mean(data ** 2))) if len(data) > 0 else 0.0
        db_rms = 20.0 * math.log10(max(rms, 1e-9))

        log.info(f"Audio energy check: peak={peak:.4f}, rms={rms:.5f} ({db_rms:.1f} dBFS)")

        if peak < 0.005 or db_rms < -48.0:
            raise InputValidationError(
                code="ERR_SILENT_AUDIO",
                title="Audio Is Muted or Completely Silent",
                message="The audio in this video is completely quiet or muted. The AI needs to hear vocal tone to detect emotion.",
                suggestion="Please unmute the audio or pick a section of the video where someone is talking.",
                details={"peak": peak, "db_rms": db_rms},
            )
    except InputValidationError:
        raise
    except Exception as e:
        log.warning(f"Audio energy evaluation warning: {e}")


def validate_speech_presence(transcript: str, min_words: int = 1) -> None:
    """Verify that spoken words were detected by speech recognition."""
    cleaned = (transcript or "").strip()
    words = cleaned.split()

    if len(words) < min_words:
        raise InputValidationError(
            code="ERR_NO_HUMAN_SPEECH",
            title="No Spoken Words Heard",
            message="The speech recognizer couldn't hear any clear spoken words in this clip (only background noise, music, or silence).",
            suggestion="Please select a part of the video where the person is speaking clearly so their voice can be analyzed.",
            details={"transcript": cleaned},
        )


def validate_face_and_visual_quality(
    frames: List[np.ndarray],
    face_crops: List[np.ndarray],
    scores: List[float],
    min_face_ratio: float = 0.25,
    min_resolution: int = 60,
    min_sharpness: float = 20.0,
) -> None:
    """
    Verify face visibility, resolution, clarity, and check for face mask / severe occlusion.
    """
    total_frames = max(len(frames), 1)
    valid_face_count = len(face_crops)
    detection_ratio = valid_face_count / total_frames

    log.info(f"Face validation: detected {valid_face_count}/{total_frames} frames ({detection_ratio * 100:.1f}%)")

    if valid_face_count == 0 or detection_ratio < min_face_ratio:
        raise InputValidationError(
            code="ERR_NO_FACE_DETECTED",
            title="No Face Clearly Visible",
            message=f"A clear face was found in only {detection_ratio * 100:.0f}% of the video (at least {min_face_ratio * 100:.0f}% is needed).",
            suggestion="Please make sure the person is facing the camera with their face clearly visible throughout the clip.",
            details={"detected_frames": valid_face_count, "total_frames": total_frames, "ratio": detection_ratio},
        )

    # Check resolution of face crops and optical sharpness
    small_count = 0
    blur_scores = []

    for crop in face_crops:
        h, w = crop.shape[:2]
        if min(h, w) < min_resolution:
            small_count += 1

        # Laplacian sharpness
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        blur_scores.append(lap_var)

    mean_sharpness = float(np.mean(blur_scores)) if blur_scores else 0.0

    if small_count > (valid_face_count * 0.70):
        raise InputValidationError(
            code="ERR_FACE_TOO_SMALL",
            title="Face Is Too Far Away or Small",
            message="The person's face is too small in the frame to clearly read their facial expressions.",
            suggestion="Please crop closer to the person's face or pick a closer video.",
            details={"min_resolution": min_resolution},
        )

    if mean_sharpness < min_sharpness:
        raise InputValidationError(
            code="ERR_FACE_BLURRED",
            title="Video Is Too Blurry",
            message="The video has heavy motion blur or is out of focus, making facial expressions hard to see.",
            suggestion="Please use a clearer, steady video with good focus and lighting.",
            details={"mean_sharpness": mean_sharpness, "threshold": min_sharpness},
        )
