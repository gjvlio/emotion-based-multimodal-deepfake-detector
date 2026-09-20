import sys
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

os.environ["OPENCV_LOG_LEVEL"] = "OFF"
os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "-8"

def main():
    from webapp.model_service import ModelService

    print("Initializing ModelService...", flush=True)
    svc = ModelService()

    test_vids = [
        ("REAL", "REAL_SPEECH_A", Path("webapp/uploads/real.mp4")),
        ("REAL", "REAL_SPEECH_B", Path("webapp/uploads/trim_0_610_WIN_20260912_12_22_57_Pro.mp4")),
        ("FAKE", "FAKE_SYNTHETIC", Path("webapp/uploads/fake.mp4")),
        ("FAKE", "FAKE_CHINESE_ELON", Path("webapp/uploads/trim_0_961_AQMFIb_9Giu-DlMR3gYzkfySV_SO27ezRXU6u7timYSEXztanB0xIuGux7hO6jrYkWuLzcLlgMs8NY9CKrZHop0LhDTbNMYCynREidcDgg.mp4")),
        ("FAKE", "FAKE_ELON_WAV2LIP", Path("webapp/uploads/trim_0_565_AQMt0xSu5_Fd_JN9T63C2Aca-2elRjpAi8Jcw1-LezQU9KVx5V-R7rklkc6Vmr8jDv2aElSsdUcVu8s6sBlBtnM4UI7-rMt8B59yEUUUhg.mp4")),
    ]

    all_passed = True
    print("\n" + "=" * 75)
    print("RUNNING END-TO-END INFERENCE VERIFICATION")
    print("=" * 75, flush=True)

    for expected, label, path in test_vids:
        if not path.exists():
            print(f"Skipping {label}: file not found ({path})")
            continue

        clip_id = f"test_{label[:10]}"
        res = svc._predict_e2e(path, clip_id=clip_id)
        confidence = res.p_fake * 100 if res.verdict == "FAKE" else (1.0 - res.p_fake) * 100
        correct = (res.verdict == expected)
        if not correct:
            all_passed = False

        status_str = "PASS" if correct else "FAIL"
        print(f"\n[{status_str}] {label} ({path.name[:30]}...)")
        print(f"       Expected: {expected} | Actual: {res.verdict} (Confidence: {confidence:.1f}%)")
        print(f"       P(fake): {res.p_fake:.4f}")
        print(f"       Transcript: '{res.transcript}'")
        print(f"       Audio Emotion: {res.audio_text_emotion.label} ({res.audio_text_emotion.confidence * 100:.1f}%)")
        print(f"         Distribution: {', '.join(f'{k}: {v*100:.1f}%' for k, v in res.audio_text_emotion.distribution.items())}")
        print(f"       Visual Emotion: {res.visual_emotion.label} ({res.visual_emotion.confidence * 100:.1f}%)")
        print(f"         Distribution: {', '.join(f'{k}: {v*100:.1f}%' for k, v in res.visual_emotion.distribution.items())}")
        print(f"       Mismatch Delta: {', '.join(f'{k}: {v*100:.1f}%' for k, v in res.emotion_mismatch.items())}")
        print(f"       Sarcasm: {res.p_sarcasm * 100:.1f}%")

        # Verify floor amplification: no emotion should be < 3.0%
        for modality, emo_dist in [("Audio", res.audio_text_emotion.distribution), ("Visual", res.visual_emotion.distribution)]:
            for emo_name, val in emo_dist.items():
                if val < 0.030 and (modality == "Visual" or res.transcript):
                    print(f"       [WARNING] {modality} emotion {emo_name} is too low: {val*100:.2f}%")

    print("\n" + "=" * 75)
    if all_passed:
        print("ALL VERIFICATION TESTS PASSED SUCCESSFULLY!")
    else:
        print("SOME TESTS FAILED - REVIEW LOGS ABOVE.")
    print("=" * 75, flush=True)

if __name__ == "__main__":
    main()
