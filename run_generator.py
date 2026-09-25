"""
run_generator.py
================
Launcher script for the Thesis Deepfake Generator Companion Tool.
Starts the FastAPI web server on port 8002 and automatically opens the browser.

Usage:
    python run_generator.py
    python run_generator.py --port 8002 --no-browser
"""

import os
import sys
import time
import argparse
import webbrowser
import threading
from pathlib import Path

# Add project root to sys.path
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def open_browser(url: str, delay_sec: float = 1.2):
    """Open the browser after a brief delay to allow the server to start."""
    time.sleep(delay_sec)
    print(f"\n[Generator Studio] Opening interactive studio at: {url}\n")
    try:
        webbrowser.open(url)
    except Exception as e:
        print(f"Notice: Could not automatically open browser: {e}")


def main():
    parser = argparse.ArgumentParser(description="Thesis Deepfake Generator Companion Studio")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8002, help="Port (default: 8002)")
    parser.add_argument("--no-browser", action="store_true", help="Do not open browser automatically")
    args = parser.parse_args()

    # Verify checkpoints
    ckpt_path = REPO_ROOT / "tools" / "Wav2Lip" / "checkpoints" / "wav2lip_gan.pth"
    if not ckpt_path.exists():
        fallback = REPO_ROOT / "tools" / "Wav2Lip" / "checkpoints" / "wav2lip.pth"
        if not fallback.exists():
            print(f"[WARNING] Wav2Lip checkpoint not found at {ckpt_path}.")
            print("Audio-Swap mode will still work instantly; for lip-sync, ensure checkpoints are present.")
        else:
            print(f"[INFO] Using fallback checkpoint: {fallback.name}")
    else:
        print(f"[OK] Wav2Lip GAN Checkpoint Verified: {ckpt_path.name}")

    url = f"http://{args.host}:{args.port}"
    print("=" * 70)
    print("  THESIS DEEPFAKE GENERATOR COMPANION STUDIO")
    print("  Multimodal Audio-Visual Emotion Incongruence Deepfake Synthesis")
    print(f"  URL: {url}")
    print("=" * 70)

    if not args.no_browser:
        threading.Thread(target=open_browser, args=(url,), daemon=True).start()

    import uvicorn
    uvicorn.run("generator.app:app", host=args.host, port=args.port, reload=False, log_level="info")


if __name__ == "__main__":
    main()
