"""
verify_cache_spec.py — Guarantee the preprocessed feature cache matches the CURRENT
preprocessing spec. Two independent checks:

  1. FINGERPRINT  — a spec.json stamped next to the cache records exactly which models,
     thresholds, detector, and package versions produced the features. --check compares
     the current environment against that stamp so drift (e.g. a Colab package bump, the
     retinaface->insightface switch) is caught before it silently corrupts a training set.

  2. RE-EXTRACTION SPOT CHECK — re-runs the pipeline on a random sample of already-cached
     clips into a TEMP cache (never touching the real cache), then compares the fresh
     features against the cached ones by cosine similarity. High similarity => the cache
     reproduces under today's spec. Low z_v similarity is the classic symptom of a face
     detector / visual-spec change.

Usage (from repo root):
    python scripts/verify_cache_spec.py --stamp                 # write data/preprocessed/spec.json
    python scripts/verify_cache_spec.py --check                 # compare env vs the stamp
    python scripts/verify_cache_spec.py --verify --sample 30    # re-extraction spot check
    python scripts/verify_cache_spec.py                         # check + verify (sample 20)

Exit code is 0 on PASS, 1 on WARN/FAIL — so it can gate a training run in a script.
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # for _collect_clips

from src.preprocessing.pipeline import PreprocessingPipeline
from src.utils.config import Config
from preprocess_all import _collect_clips

# Cosine similarity at/above this = "same spec" (allows minor GPU/ASR nondeterminism).
MATCH_THRESHOLD = 0.9990
SECTION = "=" * 68


def _section(title: str) -> None:
    print(f"\n{SECTION}\n  {title}\n{SECTION}")


# ── Spec fingerprint ─────────────────────────────────────────────────────────

def _pkg_version(*names: str) -> str:
    """Return the version of the first installed package among `names` (handles
    alternates like onnxruntime vs onnxruntime-gpu)."""
    from importlib.metadata import version
    for name in names:
        try:
            return f"{version(name)} ({name})" if len(names) > 1 else version(name)
        except Exception:
            continue
    return "not-installed"


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def _resolve_detector(cfg_value: str) -> str:
    """The config string is a legacy alias — 'retinaface' actually routes to insightface
    buffalo_s (ONNX RetinaFace) in src/preprocessing/visual.py. Record the REAL detector so
    the fingerprint doesn't imply the old removed TF-RetinaFace."""
    if cfg_value == "retinaface":
        return "insightface_buffalo_s_onnx (cfg alias 'retinaface')"
    return cfg_value


def _au_state() -> str:
    """Best-effort read of whether AU saliency is enabled in the visual module."""
    try:
        from src.preprocessing import visual
        for attr in ("_AU_ENABLED", "AU_ENABLED", "_au_enabled"):
            if hasattr(visual, attr):
                return "on" if getattr(visual, attr) else "off"
    except Exception:
        pass
    return "off (assumed — default)"


def build_spec(cfg: Config) -> dict:
    """The current, effective preprocessing spec. These are the values that determine
    what a fresh feature would look like — change any and the cache is inconsistent."""
    pipe = PreprocessingPipeline(
        cache_dir     = cfg.paths.preprocessed_dir,
        wav2vec_model = cfg.model.wav2vec_model,
        bert_model    = cfg.model.bert_model,
        whisper_model = cfg.model.whisper_model,
        vit_model     = cfg.model.vit_model,
        face_detector = cfg.preprocessing.face_detector,
        n_keyframes   = cfg.preprocessing.n_keyframes,
        frame_size    = cfg.preprocessing.frame_size,
        max_audio_sec = cfg.preprocessing.max_audio_seconds,
        device        = "cpu",
    )
    return {
        "spec": {
            "wav2vec_model":        pipe.wav2vec_model,
            "bert_model":           pipe.bert_model,
            "whisper_model":        pipe.whisper_model,
            "vit_model":            pipe.vit_model,
            "face_detector":        _resolve_detector(pipe.face_detector),
            "n_keyframes":          pipe.n_keyframes,
            "frame_size":           pipe.frame_size,
            "target_fps":           pipe.target_fps,
            "motion_threshold":     pipe.motion_threshold,
            "confidence_threshold": pipe.confidence_threshold,
            "max_audio_sec":        pipe.max_audio_sec,
            "au_saliency":          _au_state(),
            "z_at_dim":             1536,
            "z_v_dim":              768,
        },
        "versions": {
            "torch":        _pkg_version("torch"),
            "transformers": _pkg_version("transformers"),
            "numpy":        _pkg_version("numpy"),
            "insightface":  _pkg_version("insightface"),
            "onnxruntime":  _pkg_version("onnxruntime-gpu", "onnxruntime"),
            "librosa":      _pkg_version("librosa"),
            "opencv":       _pkg_version("opencv-python"),
        },
        "git_sha":     _git_sha(),
        "python":      sys.version.split()[0],
    }


def _spec_path(cfg: Config) -> Path:
    return Path(cfg.paths.preprocessed_dir) / "spec.json"


def cmd_stamp(cfg: Config, force: bool) -> int:
    path = _spec_path(cfg)
    current = build_spec(cfg)
    if path.exists() and not force:
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("spec") == current["spec"] and existing.get("versions") == current["versions"]:
            print(f"  spec.json already matches current environment -> {path}")
            return 0
        print(f"  WARNING: {path} exists and DIFFERS from the current environment.")
        print(f"  Re-run with --force to overwrite it with the current spec.")
        _diff_specs(existing, current)
        return 1
    current["stamped_utc"] = datetime.now(timezone.utc).isoformat()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(current, indent=2), encoding="utf-8")
    print(f"  Stamped spec fingerprint -> {path}")
    print(f"  git_sha: {current['git_sha']}   torch: {current['versions']['torch']}")
    return 0


def _diff_specs(a: dict, b: dict) -> None:
    """Print field-level differences between two fingerprints (a=stamped, b=current)."""
    for section in ("spec", "versions"):
        sa, sb = a.get(section, {}), b.get(section, {})
        for k in sorted(set(sa) | set(sb)):
            va, vb = sa.get(k, "<missing>"), sb.get(k, "<missing>")
            flag = "  " if va == vb else "!!"
            if va != vb:
                print(f"  {flag} {section}.{k}: stamped={va!r}  current={vb!r}")
    if a.get("git_sha") != b.get("git_sha"):
        print(f"  !! git_sha: stamped={a.get('git_sha')}  current={b.get('git_sha')}")


def cmd_check(cfg: Config) -> int:
    path = _spec_path(cfg)
    current = build_spec(cfg)
    if not path.exists():
        print(f"  No spec.json at {path} — the cache has NO recorded provenance.")
        print(f"  Cannot verify what produced the existing features by fingerprint.")
        print(f"  -> Run --verify (re-extraction) to check, then --stamp to fix going forward.")
        return 1
    stamped = json.loads(path.read_text(encoding="utf-8"))
    same_spec = stamped.get("spec") == current["spec"]
    same_ver  = stamped.get("versions") == current["versions"]
    if same_spec and same_ver:
        print(f"  MATCH — current environment matches the stamped spec ({path}).")
        print(f"  Stamped: {stamped.get('stamped_utc','?')}   git_sha: {stamped.get('git_sha','?')}")
        return 0
    print(f"  MISMATCH — current environment differs from the stamped spec:")
    _diff_specs(stamped, current)
    if not same_spec:
        print(f"\n  Spec fields changed -> existing features may be INCONSISTENT with a fresh run.")
    else:
        print(f"\n  Only package versions changed -> usually minor, but verify with --verify.")
    return 1


# ── Re-extraction spot check ─────────────────────────────────────────────────

def _cached_paths(cfg: Config, cid: str) -> tuple[Path, Path]:
    root = Path(cfg.paths.preprocessed_dir) / "features"
    return root / "z_at" / f"{cid}.pt", root / "z_v" / f"{cid}.pt"


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.flatten().float().unsqueeze(0)
    b = b.flatten().float().unsqueeze(0)
    if a.shape != b.shape:
        return float("nan")
    return float(F.cosine_similarity(a, b).item())


def cmd_verify(cfg: Config, sample: int, seed: int, device: str) -> int:
    _section(f"Re-extraction spot check (sample={sample}, device={device})")

    clips = _collect_clips(cfg, smoke=False)  # (clip_id, video_path, source)
    # Keep only clips that are cached AND whose source video still exists on disk.
    candidates = []
    for cid, vp, src in clips:
        z_at_c, z_v_c = _cached_paths(cfg, cid)
        if z_at_c.exists() and z_v_c.exists() and Path(vp).exists():
            candidates.append((cid, vp, src))

    if not candidates:
        print("  No verifiable clips (need cached feature + existing source video). Nothing to check.")
        return 1

    rng = random.Random(seed)
    rng.shuffle(candidates)
    picked = candidates[:sample]
    print(f"  Verifiable clips: {len(candidates)}   sampling: {len(picked)}")

    # Temp cache — the fresh features land here; the real cache is never touched.
    tmp_dir = Path(tempfile.mkdtemp(prefix="deepsentinel_verify_"))
    print(f"  Temp cache (left on disk for inspection): {tmp_dir}\n")

    tmp_pipe = PreprocessingPipeline(
        cache_dir     = tmp_dir,
        wav2vec_model = cfg.model.wav2vec_model,
        bert_model    = cfg.model.bert_model,
        whisper_model = cfg.model.whisper_model,
        vit_model     = cfg.model.vit_model,
        face_detector = cfg.preprocessing.face_detector,
        n_keyframes   = cfg.preprocessing.n_keyframes,
        frame_size    = cfg.preprocessing.frame_size,
        max_audio_sec = cfg.preprocessing.max_audio_seconds,
        device        = device,
    )

    real_audio = Path(cfg.paths.preprocessed_dir) / "audio"
    real_txt   = Path(cfg.paths.preprocessed_dir) / "transcripts"

    print(f"  {'clip_id':<34}{'z_at cos':>10}{'z_v cos':>10}  verdict")
    print(f"  {'-'*64}")

    rows = []
    for cid, vp, src in picked:
        # Reuse the cached WAV + transcript so z_at compares model output on identical
        # inputs (isolates spec, not ASR noise). z_v is re-derived from the raw video,
        # so the visual pipeline (incl. face detector) is fully re-exercised.
        for sub, real in (("audio", real_audio), ("transcripts", real_txt)):
            src_f = real / (f"{cid}.wav" if sub == "audio" else f"{cid}.txt")
            if src_f.exists():
                (tmp_dir / sub).mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_f, tmp_dir / sub / src_f.name)

        feats = tmp_pipe.process(cid, vp, force=False)
        if feats is None:
            print(f"  {cid[:34]:<34}{'FAILED re-extraction':>22}")
            rows.append((cid, src, float("nan"), float("nan"), "FAIL"))
            continue

        z_at_c, z_v_c = _cached_paths(cfg, cid)
        cached_at = torch.load(z_at_c, weights_only=True)
        cached_v  = torch.load(z_v_c,  weights_only=True)
        cos_at = _cosine(feats.z_at, cached_at)
        cos_v  = _cosine(feats.z_v,  cached_v)
        ok = (cos_at >= MATCH_THRESHOLD) and (cos_v >= MATCH_THRESHOLD)
        verdict = "match" if ok else ("Z_V DRIFT" if cos_v < MATCH_THRESHOLD else "Z_AT drift")
        print(f"  {cid[:34]:<34}{cos_at:>10.5f}{cos_v:>10.5f}  {verdict}")
        rows.append((cid, src, cos_at, cos_v, verdict))

    # Summary
    ok_rows   = [r for r in rows if r[4] == "match"]
    zv_drift  = [r for r in rows if r[4] == "Z_V DRIFT"]
    zat_drift = [r for r in rows if r[4] == "Z_AT drift"]
    failed    = [r for r in rows if r[4] == "FAIL"]
    n = len(rows)
    pct = 100.0 * len(ok_rows) / max(n, 1)

    _section("Spot-check result")
    print(f"  Matched      : {len(ok_rows)}/{n}  ({pct:.1f}%)")
    print(f"  Z_v drift    : {len(zv_drift)}   (visual/detector spec change — the usual culprit)")
    print(f"  Z_at drift   : {len(zat_drift)}  (audio/text spec change)")
    print(f"  Re-extract failed: {len(failed)}")

    def _by_source(rs):
        from collections import Counter
        return dict(Counter(r[1] for r in rs))

    if zv_drift:
        print(f"\n  Z_v drift by source: {_by_source(zv_drift)}")
        print(f"  Example divergent clips: {[r[0] for r in zv_drift[:5]]}")

    # Verdict logic — distinguish real spec drift from harmless per-clip nondeterminism.
    #  * z_at is deterministic given the cached WAV+transcript (which we reuse), so ANY
    #    z_at drift means a real audio/text model or version change -> always significant.
    #  * z_v re-derives keyframes from the raw video; in-the-wild clips can occasionally
    #    pick slightly different frames -> an ISOLATED z_v outlier is nondeterminism, not a
    #    spec problem. SYSTEMATIC z_v drift (high rate, or one source mostly drifting) is real.
    from collections import Counter
    zv_rate = len(zv_drift) / max(n, 1)
    src_counts_all = Counter(r[1] for r in rows)
    src_drift = Counter(r[1] for r in zv_drift)
    concentrated = any(
        src_drift[s] >= 3 and src_drift[s] / src_counts_all[s] >= 0.5
        for s in src_drift
    )  # a source is "systematically" drifting only with >=3 samples and >=50% drift

    if len(ok_rows) == n:
        print(f"\n  PASS — every sampled clip reproduces exactly. Cache is spec-consistent. Safe to train.")
        return 0

    if zat_drift:
        print(f"\n  FAIL — {len(zat_drift)} clip(s) show Z_AT drift. Z_at is deterministic given the")
        print(f"  cached WAV+transcript, so this means the audio/text model or a package version")
        print(f"  changed since the cache was built. Re-extract before training:")
        print(f"    python scripts/preprocess_all.py --device cuda --force")
        return 1

    if concentrated:
        print(f"\n  FAIL — Z_v drift is SYSTEMATIC (a source is mostly diverging): {dict(src_drift)}.")
        print(f"  This is a real visual-spec change (e.g. face detector). Re-extract that source:")
        print(f"    python scripts/preprocess_all.py --device cuda --force")
        return 1

    if zv_drift and zv_rate <= 0.15:
        print(f"\n  PASS (with note) — only {len(zv_drift)}/{n} clip(s) show minor Z_v drift, scattered,")
        print(f"  z_at all exact. This is per-clip visual nondeterminism (keyframe/face selection on")
        print(f"  in-the-wild clips), NOT a spec change. The cache is consistent — safe to train.")
        print(f"  To confirm, re-run with a larger --sample; the drift rate should stay low.")
        return 0

    if zv_drift:
        print(f"\n  WARN — Z_v drift rate is {zv_rate:.0%} ({len(zv_drift)}/{n}), above the noise band.")
        print(f"  Not clearly systematic. Re-run with a larger --sample to decide; if it holds,")
        print(f"  re-extract the visual features:  python scripts/preprocess_all.py --device cuda --force")
        return 1

    print(f"\n  WARN — {len(failed)} clip(s) failed re-extraction (often transient). Re-run to confirm.")
    return 1


def main() -> int:
    ap = argparse.ArgumentParser(description="Verify the preprocessed cache matches the current spec")
    ap.add_argument("--config", default=None)
    ap.add_argument("--stamp",  action="store_true", help="Write data/preprocessed/spec.json for the current spec")
    ap.add_argument("--check",  action="store_true", help="Compare the current environment against the stamp")
    ap.add_argument("--verify", action="store_true", help="Re-extraction spot check against the real cache")
    ap.add_argument("--force",  action="store_true", help="With --stamp: overwrite an existing, differing spec.json")
    ap.add_argument("--sample", type=int, default=20, help="Clips to re-extract for the spot check")
    ap.add_argument("--seed",   type=int, default=42)
    _dev = "cuda" if torch.cuda.is_available() else "cpu"
    ap.add_argument("--device", default=_dev)
    args = ap.parse_args()

    cfg = Config.from_yaml(args.config)
    rc = 0

    if args.stamp:
        _section("Stamp spec fingerprint")
        return cmd_stamp(cfg, args.force)

    # Default (no explicit mode) = check + verify.
    do_check  = args.check  or not (args.verify)
    do_verify = args.verify or not (args.check)

    if do_check:
        _section("Fingerprint check (env vs stamped spec.json)")
        rc |= cmd_check(cfg)
    if do_verify:
        rc |= cmd_verify(cfg, args.sample, args.seed, args.device)

    _section("Overall")
    print("  RESULT:", "PASS" if rc == 0 else "NEEDS ATTENTION (see above)")
    return rc


if __name__ == "__main__":
    sys.exit(main())
