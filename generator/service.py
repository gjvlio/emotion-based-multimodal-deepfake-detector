"""
generator/service.py
====================
Deepfake synthesis engine for the Thesis Defense Companion Tool.
Supports:
  1. Wav2Lip Lip-Sync Reanimation (Track 2)
  2. Audio-Swap Manipulation (Track 1)
  3. Preprocessing/normalization for browser-recorded media (WebM -> MP4/WAV)
  4. Preset sample library integration for instant demo testing
"""

import os
import sys
import time
import shutil
import logging
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional

# Set up logging
logger = logging.getLogger("generator.service")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

# Resolve repository paths
REPO_ROOT = Path(__file__).resolve().parent.parent
GENERATOR_DIR = REPO_ROOT / "generator"
OUTPUTS_DIR = GENERATOR_DIR / "outputs"
TEMP_DIR = GENERATOR_DIR / "temp"
WAV2LIP_DIR = REPO_ROOT / "tools" / "Wav2Lip"
CHECKPOINT_PATH = WAV2LIP_DIR / "checkpoints" / "wav2lip_gan.pth"
if not CHECKPOINT_PATH.exists():
    fallback_ckpt = WAV2LIP_DIR / "checkpoints" / "wav2lip.pth"
    if fallback_ckpt.exists():
        CHECKPOINT_PATH = fallback_ckpt

FACE_DETECTOR_PATH = WAV2LIP_DIR / "face_detection" / "detection" / "sfd" / "s3fd.pth"

# Ensure runtime directories exist
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
TEMP_DIR.mkdir(parents=True, exist_ok=True)


def probe_media(file_path: Path) -> Dict[str, Any]:
    """Inspect media file using ffprobe and return metadata dict."""
    if not file_path.exists():
        return {"error": "File does not exist", "exists": False}

    try:
        cmd = [
            "ffprobe",
            "-v", "quiet",
            "-print_format", "json",
            "-show_streams",
            "-show_format",
            str(file_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        if result.returncode != 0:
            return {"error": result.stderr.strip(), "exists": True}

        import json
        data = json.loads(result.stdout)
        streams = data.get("streams", [])
        fmt = data.get("format", {})

        video_stream = next((s for s in streams if s.get("codec_type") == "video"), None)
        audio_stream = next((s for s in streams if s.get("codec_type") == "audio"), None)

        duration = float(fmt.get("duration", 0) or 0)
        size_bytes = int(fmt.get("size", 0) or file_path.stat().st_size)

        info = {
            "exists": True,
            "duration": round(duration, 2),
            "size_bytes": size_bytes,
            "size_kb": round(size_bytes / 1024, 1),
            "has_video": video_stream is not None,
            "has_audio": audio_stream is not None,
        }

        if video_stream:
            info["video_codec"] = video_stream.get("codec_name")
            info["width"] = int(video_stream.get("width", 0))
            info["height"] = int(video_stream.get("height", 0))
            fps_str = video_stream.get("r_frame_rate", "25/1")
            try:
                num, den = fps_str.split("/")
                info["fps"] = round(float(num) / float(den), 2)
            except Exception:
                info["fps"] = 25.0

        if audio_stream:
            info["audio_codec"] = audio_stream.get("codec_name")
            info["sample_rate"] = int(audio_stream.get("sample_rate", 0))
            info["channels"] = int(audio_stream.get("channels", 0))

        return info
    except Exception as e:
        logger.error(f"ffprobe error on {file_path}: {e}")
        return {"error": str(e), "exists": True}


def normalize_video_for_wav2lip(input_path: Path, output_path: Path) -> Path:
    """
    Convert browser-recorded WebM or non-standard videos into constant-framerate MP4
    with H.264 video encoding at 25 fps. This prevents frame dropping or OpenCV seek
    issues during face detection.
    """
    logger.info(f"Normalizing video: {input_path.name} -> {output_path.name}")
    cmd = [
        "ffmpeg", "-y",
        "-i", str(input_path),
        "-vf", "fps=25",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "22",
        "-c:a", "aac",
        "-b:a", "128k",
        "-pix_fmt", "yuv420p",
        str(output_path),
    ]
    subprocess.run(cmd, capture_output=True, check=True)
    return output_path


def extract_clean_wav(input_path: Path, output_wav_path: Path) -> Path:
    """
    Extract 16kHz mono 16-bit PCM WAV audio from any video or audio donor file.
    This guarantees 100% compatibility with Wav2Lip's Mel-spectrogram processor.
    """
    logger.info(f"Extracting clean WAV: {input_path.name} -> {output_wav_path.name}")
    cmd = [
        "ffmpeg", "-y",
        "-i", str(input_path),
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        str(output_wav_path),
    ]
    subprocess.run(cmd, capture_output=True, check=True)
    return output_wav_path


class DeepfakeGeneratorService:
    """Manages deepfake generation workflows using Wav2Lip and FFmpeg."""

    def __init__(self):
        self.wav2lip_dir = WAV2LIP_DIR
        self.checkpoint_path = CHECKPOINT_PATH
        self.outputs_dir = OUTPUTS_DIR
        self.temp_dir = TEMP_DIR

        # Validate checkpoint availability
        if not self.checkpoint_path.exists():
            logger.warning(f"Wav2Lip checkpoint not found at {self.checkpoint_path}")
        else:
            logger.info(f"Wav2Lip checkpoint loaded: {self.checkpoint_path.name}")

    def get_presets(self) -> List[Dict[str, Any]]:
        """Return demo preset clips available in the repository."""
        presets = []
        doc_dir = REPO_ROOT / "docs"
        candidate_files = [
            ("genuine happy.mp4", "Genuine Happy (Joyful Speech)", "happy"),
            ("genuine angry.mp4", "Genuine Angry (Aggressive Speech)", "angry"),
            ("genuine neutral.mp4", "Genuine Neutral (Calm Speech)", "neutral"),
            ("fake neutral.mp4", "Synthetic Audio Sample", "fake_neutral"),
        ]

        for filename, label, tag in candidate_files:
            p = doc_dir / filename
            if p.exists():
                meta = probe_media(p)
                presets.append({
                    "id": p.stem.replace(" ", "_"),
                    "filename": filename,
                    "label": label,
                    "tag": tag,
                    "path": str(p),
                    "duration": meta.get("duration", 0),
                    "has_video": meta.get("has_video", False),
                    "has_audio": meta.get("has_audio", False),
                })
        return presets

    def generate_wav2lip(
        self,
        face_path: Path,
        donor_path: Path,
        resize_factor: int = 1,
        pads: List[int] = [0, 10, 0, 0],
        nosmooth: bool = False,
    ) -> Dict[str, Any]:
        """
        Execute Wav2Lip lip-sync reanimation.
        The lips and mouth movements of the face video will be reanimated to perfectly
        match the speech/phonemes of the donor video/audio.
        """
        start_time = time.time()
        timestamp = int(start_time)
        unique_id = f"fake_wav2lip_{timestamp}"

        # Setup working files in TEMP_DIR
        norm_face = self.temp_dir / f"{unique_id}_face.mp4"
        clean_audio = self.temp_dir / f"{unique_id}_audio.wav"
        output_file = self.outputs_dir / f"{unique_id}.mp4"

        try:
            # 1. Normalize face video (handles WebM from webcam and ensures 25 fps)
            normalize_video_for_wav2lip(face_path, norm_face)

            # 2. Extract clean 16kHz mono WAV from donor material
            extract_clean_wav(donor_path, clean_audio)

            # 3. Build Wav2Lip command
            # Must run with working directory set to Wav2Lip directory for local imports
            cmd = [
                sys.executable,
                str(self.wav2lip_dir / "inference.py"),
                "--checkpoint_path", str(self.checkpoint_path),
                "--face", str(norm_face),
                "--audio", str(clean_audio),
                "--outfile", str(output_file),
                "--resize_factor", str(resize_factor),
                "--pads", str(pads[0]), str(pads[1]), str(pads[2]), str(pads[3]),
                "--wav2lip_batch_size", "128",
            ]
            if nosmooth:
                cmd.append("--nosmooth")

            logger.info(f"Starting Wav2Lip inference: {' '.join(cmd)}")
            res = subprocess.run(
                cmd,
                cwd=str(self.wav2lip_dir),
                capture_output=True,
                text=True,
                timeout=180,
            )

            if res.returncode != 0:
                logger.error(f"Wav2Lip error (code {res.returncode}):\n{res.stderr}")
                return {
                    "success": False,
                    "error": f"Wav2Lip inference failed: {res.stderr[-500:] if res.stderr else 'Unknown error'}",
                    "details": res.stderr,
                }

            if not output_file.exists() or output_file.stat().st_size < 1000:
                return {
                    "success": False,
                    "error": "Wav2Lip completed but output video file was not generated or is empty.",
                }

            elapsed = round(time.time() - start_time, 2)
            meta = probe_media(output_file)

            logger.info(f"Wav2Lip generation successful in {elapsed}s: {output_file.name}")
            return {
                "success": True,
                "mode": "wav2lip",
                "mode_title": "Wav2Lip Lip-Sync Reanimation (Track 2)",
                "filename": output_file.name,
                "output_url": f"/outputs/{output_file.name}",
                "output_path": str(output_file),
                "generation_time_sec": elapsed,
                "metadata": meta,
            }

        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": "Wav2Lip inference timed out (> 180 seconds). Consider trimming your clip or using 2x downsampling.",
            }
        except Exception as e:
            logger.exception("Unexpected error in generate_wav2lip")
            return {"success": False, "error": str(e)}
        finally:
            # Clean up intermediate normalized files
            for p in [norm_face, clean_audio]:
                try:
                    if p.exists():
                        p.unlink()
                except Exception:
                    pass

    def generate_audio_swap(
        self,
        face_path: Path,
        donor_path: Path,
    ) -> Dict[str, Any]:
        """
        Execute Audio-Swap Deepfake (Track 1).
        Preserves original facial video frames and replaces the audio stream with donor audio.
        Fast, instant (<1s) generation.
        """
        start_time = time.time()
        timestamp = int(start_time)
        unique_id = f"fake_audioswap_{timestamp}"

        clean_face = self.temp_dir / f"{unique_id}_face.mp4"
        output_file = self.outputs_dir / f"{unique_id}.mp4"

        try:
            # Normalize face video to ensure browser WebM is transcode-safe
            normalize_video_for_wav2lip(face_path, clean_face)

            # High-speed audio swap via FFmpeg
            # Replaces audio of stream 0 with audio of stream 1
            cmd = [
                "ffmpeg", "-y",
                "-i", str(clean_face),
                "-i", str(donor_path),
                "-map", "0:v:0",
                "-map", "1:a:0",
                "-c:v", "copy",
                "-c:a", "aac",
                "-b:a", "192k",
                "-shortest",
                "-movflags", "+faststart",
                str(output_file),
            ]
            logger.info(f"Running Audio Swap: {' '.join(cmd)}")
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

            if res.returncode != 0:
                logger.error(f"FFmpeg audio swap error: {res.stderr}")
                return {"success": False, "error": f"Audio swap failed: {res.stderr[-400:]}"}

            elapsed = round(time.time() - start_time, 2)
            meta = probe_media(output_file)

            logger.info(f"Audio Swap generation successful in {elapsed}s: {output_file.name}")
            return {
                "success": True,
                "mode": "audio_swap",
                "mode_title": "Audio-Swap Manipulation (Track 1)",
                "filename": output_file.name,
                "output_url": f"/outputs/{output_file.name}",
                "output_path": str(output_file),
                "generation_time_sec": elapsed,
                "metadata": meta,
            }

        except Exception as e:
            logger.exception("Unexpected error in generate_audio_swap")
            return {"success": False, "error": str(e)}
        finally:
            if clean_face.exists():
                try:
                    clean_face.unlink()
                except Exception:
                    pass

    def cleanup_old_files(self, max_age_hours: int = 24):
        """Remove temp and generated files older than max_age_hours."""
        now = time.time()
        for directory in [self.temp_dir]:
            for p in directory.glob("*"):
                if p.is_file() and (now - p.stat().st_mtime) > (max_age_hours * 3600):
                    try:
                        p.unlink()
                    except Exception:
                        pass


# Singleton instance
generator_service = DeepfakeGeneratorService()
