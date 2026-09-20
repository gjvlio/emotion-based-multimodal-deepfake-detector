import requests
from pathlib import Path
import json

BASE_URL = "http://127.0.0.1:8000"

test_vids = [
    ("REAL", "REAL_SPEECH", Path("webapp/uploads/trim_0_396_WIN_20260913_19_59_50_Pro.mp4")),
    ("REAL", "REAL_SILENT", Path("webapp/uploads/trim_0_606_WIN_20260913_20_02_18_Pro.mp4")),
    ("FAKE", "FAKE_CHINESE_ELON", Path("webapp/uploads/trim_0_961_AQMFIb_9Giu-DlMR3gYzkfySV_SO27ezRXU6u7timYSEXztanB0xIuGux7hO6jrYkWuLzcLlgMs8NY9CKrZHop0LhDTbNMYCynREidcDgg.mp4")),
    ("FAKE", "FAKE_ELON_WAV2LIP", Path("webapp/uploads/trim_0_565_AQMt0xSu5_Fd_JN9T63C2Aca-2elRjpAi8Jcw1-LezQU9KVx5V-R7rklkc6Vmr8jDv2aElSsdUcVu8s6sBlBtnM4UI7-rMt8B59yEUUUhg.mp4")),
]

print("=" * 70)
print("TESTING LIVE SERVER VIA HTTP /detect")
print("=" * 70)

all_passed = True

for expected, label, path in test_vids:
    if not path.exists():
        continue
    with open(path, "rb") as f:
        files = {"file": (path.name, f, "video/mp4")}
        data = {"start_time": "0.0"}
        r = requests.post(f"{BASE_URL}/detect", files=files, data=data)
    
    if r.status_code != 200:
        print(f"[ERROR] {label}: HTTP {r.status_code} - {r.text}")
        all_passed = False
        continue

    res = r.json()
    verdict = res.get("verdict")
    p_fake = res.get("p_fake", 0.0)
    conf = p_fake * 100 if verdict == "FAKE" else (1.0 - p_fake) * 100
    ok = (verdict == expected)
    if not ok:
        all_passed = False

    status = "PASS" if ok else "FAIL"
    print(f"\n[{status}] {label}")
    print(f"       Expected: {expected} | Actual: {verdict} (Confidence: {conf:.1f}%)")
    print(f"       P(fake): {p_fake:.4f}")
    print(f"       Transcript: '{res.get('transcript', '')}'")
    a_emo = res.get("audio_text_emotion", {})
    v_emo = res.get("visual_emotion", {})
    print(f"       Audio Emotion: {a_emo.get('label')} ({a_emo.get('confidence', 0)*100:.1f}%)")
    print(f"       Visual Emotion: {v_emo.get('label')} ({v_emo.get('confidence', 0)*100:.1f}%)")
    print(f"       Sarcasm: {res.get('p_sarcasm', 0)*100:.1f}%")

print("\n" + "=" * 70)
if all_passed:
    print("ALL LIVE HTTP DETECTION TESTS PASSED!")
else:
    print("SOME TESTS FAILED - CHECK LOGS ABOVE.")
print("=" * 70)
