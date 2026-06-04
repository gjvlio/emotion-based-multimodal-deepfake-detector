"""
preprocess_all.py — Run preprocessing pipeline on all clips in the dataset.

Reads metadata CSVs for all 4 tracks + real MELD + real CMU-MOSEI + MUStARD sarcasm,
then runs PreprocessingPipeline.process() on each unprocessed clip.
Progress is implicitly tracked by cached .pt files — safe to interrupt/resume.

Usage (from repo root):
    python scripts/preprocess_all.py
    python scripts/preprocess_all.py --device cuda
    python scripts/preprocess_all.py --workers 4 --device cpu
    python scripts/preprocess_all.py --smoke   # use smoke manifests (fast test run)
"""
from __future__ import annotations

import argparse
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.preprocessing.pipeline import PreprocessingPipeline
from src.utils.config import Config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

SECTION_WIDTH = 60


def _section(title: str) -> None:
    print(f"\n{'='*SECTION_WIDTH}")
    print(f"  {title}")
    print(f"{'='*SECTION_WIDTH}")


def _collect_clips(cfg: Config, smoke: bool = False) -> list[tuple[str, str, str]]:
    """Return list of (clip_id, video_path, source_name)."""
    clips: list[tuple[str, str, str]] = []

    if smoke:
        _section("SMOKE MODE — using smoke manifests")
        smoke_dir = Path("data/processed/smoke_manifests")

        detector_csv = smoke_dir / "smoke_detector.csv"
        if detector_csv.exists():
            df = pd.read_csv(detector_csv)
            for _, row in df.iterrows():
                p = str(row.get("path", ""))
                if p:
                    clips.append((Path(p).stem, p, str(row.get("source", "smoke"))))
            print(f"  smoke_detector.csv : {len(clips)} clips")
        else:
            print(f"  WARNING: {detector_csv} not found — run build_smoke_manifest.py first")

        sarcasm_csv = smoke_dir / "smoke_sarcasm.csv"
        if sarcasm_csv.exists():
            df = pd.read_csv(sarcasm_csv)
            before = len(clips)
            for _, row in df.iterrows():
                p = str(row.get("path", ""))
                if p:
                    clips.append((Path(p).stem, p, "mustard"))
            print(f"  smoke_sarcasm.csv  : {len(clips) - before} clips")
        return clips

    _section("FULL DATASET — collecting clips from all sources")

    # Fake tracks 1–4
    for csv_path, id_col, path_col, name in [
        (cfg.paths.track1_meta,  "output_stem", "output_path", "track1"),
        (cfg.paths.track2_meta,  "output_stem", "output_path", "track2"),
        (cfg.paths.track3_meta,  "output_stem", "output_path", "track3"),
        (cfg.paths.track4_meta,  "output_stem", "output_path", "track4"),
    ]:
        p = Path(csv_path)
        if not p.exists():
            print(f"  SKIP {name}: {p} not found")
            continue
        df = pd.read_csv(p)
        before = len(clips)
        for _, row in df.iterrows():
            clips.append((str(row[id_col]), str(row[path_col]), name))
        print(f"  {name}: {len(clips) - before} clips")

    # Real — MELD
    meld = Path(cfg.paths.meld_real_csv)
    if meld.exists():
        df = pd.read_csv(meld)
        before = len(clips)
        for _, row in df.iterrows():
            clips.append((str(row["clip_id"]), str(row["video_path"]), "meld_real"))
        print(f"  meld_real: {len(clips) - before} clips")
    else:
        print(f"  SKIP meld_real: {meld} not found")

    # Real — CMU-MOSEI
    mosei = Path(cfg.paths.mosei_real_csv)
    if mosei.exists():
        df = pd.read_csv(mosei)
        before = len(clips)
        for _, row in df.iterrows():
            clips.append((str(row["clip_id"]), str(row["video_path"]), "mosei_real"))
        print(f"  mosei_real: {len(clips) - before} clips")
    else:
        print(f"  SKIP mosei_real: {mosei} not found")

    # MUStARD sarcasm
    _mustard_str = getattr(cfg.paths, "mustard_csv", "")
    mustard = Path(_mustard_str) if _mustard_str else Path("__nonexistent__")
    if mustard.exists():
        df = pd.read_csv(mustard)
        before = len(clips)
        for _, row in df.iterrows():
            clips.append((str(row["clip_id"]), str(row["path"]), "mustard"))
        print(f"  mustard: {len(clips) - before} clips")
    else:
        mustard_fallback = Path("data/processed/smoke_manifests/smoke_sarcasm.csv")
        if mustard_fallback.exists():
            df = pd.read_csv(mustard_fallback)
            before = len(clips)
            for _, row in df.iterrows():
                p = str(row.get("path", ""))
                if p:
                    clips.append((Path(p).stem, p, "mustard"))
            print(f"  mustard (from smoke): {len(clips) - before} clips")

    return clips


def main():
    parser = argparse.ArgumentParser(description="Preprocess all dataset clips -> Z_at, Z_v")
    parser.add_argument("--config",  default=None,  help="Path to YAML config (default: configs/default.yaml)")
    _default_device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    parser.add_argument("--device",  default=_default_device, help="Computation device (cpu / cuda)")
    parser.add_argument("--workers", type=int, default=1, help="Parallel workers (default 1)")
    parser.add_argument("--force",     action="store_true", help="Reprocess even if cache exists")
    parser.add_argument("--smoke",     action="store_true", help="Use smoke manifests (fast ~1k clip test)")
    parser.add_argument("--max_clips", type=int, default=None,
                        help="Stop after processing this many NEW clips (resume-safe sharding)")
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config)

    _section("Loading preprocessing pipeline")
    print(f"  Wav2Vec2   : {cfg.model.wav2vec_model}")
    print(f"  BERT       : {cfg.model.bert_model}")
    print(f"  Whisper    : {cfg.model.whisper_model}")
    print(f"  ViT        : {cfg.model.vit_model}")
    print(f"  Device     : {args.device}")
    print(f"  Workers    : {args.workers}")
    print(f"  Force redo : {args.force}")

    pipeline = PreprocessingPipeline(
        cache_dir       = cfg.paths.preprocessed_dir,
        wav2vec_model   = cfg.model.wav2vec_model,
        bert_model      = cfg.model.bert_model,
        whisper_model   = cfg.model.whisper_model,
        vit_model       = cfg.model.vit_model,
        face_detector   = cfg.preprocessing.face_detector,
        n_keyframes     = cfg.preprocessing.n_keyframes,
        frame_size      = cfg.preprocessing.frame_size,
        max_audio_sec   = cfg.preprocessing.max_audio_seconds,
        device          = args.device,
    )

    clips = _collect_clips(cfg, smoke=args.smoke)

    _section("Cache check")
    already_done = sum(1 for cid, _, _ in clips if pipeline.is_cached(cid))
    to_do = [(cid, vp, src) for cid, vp, src in clips
              if not pipeline.is_cached(cid) or args.force]

    print(f"  Total clips      : {len(clips)}")
    print(f"  Already cached   : {already_done}")
    print(f"  To process       : {len(to_do)}")

    if not to_do:
        print("\n  All clips already preprocessed. Nothing to do.")
        return

    if args.max_clips and len(to_do) > args.max_clips:
        to_do = to_do[:args.max_clips]
        print(f"  Shard limit      : {args.max_clips} clips (re-run to continue next shard)")

    # Source breakdown of remaining work
    from collections import Counter
    src_counts = Counter(src for _, _, src in to_do)
    print(f"\n  Breakdown by source (this shard):")
    for src, n in sorted(src_counts.items()):
        print(f"    {src:<20} {n} clips")

    _section("Running preprocessing")
    print("  Each clip: WAV -> Wav2Vec2 mean (768) + Whisper ASR -> BERT CLS (768) -> Z_at (1536)")
    print("             MP4 -> insightface keyframes -> ViT CLS mean (768) -> Z_v (768)")
    print("  Cache: data/preprocessed/features/z_at/<id>.pt + z_v/<id>.pt\n")

    fail_file = Path(cfg.paths.preprocessed_dir) / "failed_clips.txt"
    failed: list[str] = []
    failed_by_src: dict[str, list[str]] = {}
    done_by_src:   dict[str, int] = {}
    FAIL_SAVE_EVERY = 50

    def process_one(item: tuple[str, str, str]) -> tuple[str, str, bool]:
        cid, vp, src = item
        result = pipeline.process(cid, vp, force=args.force)
        return cid, src, result is not None

    if args.workers > 1:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(process_one, item): item for item in to_do}
            pbar = tqdm(as_completed(futures), total=len(to_do),
                        desc="Preprocessing", unit="clip", dynamic_ncols=True)
            for fut in pbar:
                cid, src, ok = fut.result()
                done_by_src[src] = done_by_src.get(src, 0) + 1
                if not ok:
                    failed.append(cid)
                    failed_by_src.setdefault(src, []).append(cid)
                    if len(failed) % FAIL_SAVE_EVERY == 0:
                        fail_file.write_text("\n".join(failed), encoding="utf-8")
                pbar.set_postfix(src=src[:10], ok=len(to_do)-len(failed)-pbar.n+len(failed), fail=len(failed))
    else:
        pbar = tqdm(to_do, desc="Preprocessing", unit="clip", dynamic_ncols=True)
        for item in pbar:
            cid, src, ok = process_one(item)
            done_by_src[src] = done_by_src.get(src, 0) + 1
            if not ok:
                failed.append(cid)
                failed_by_src.setdefault(src, []).append(cid)
                if len(failed) % FAIL_SAVE_EVERY == 0:
                    fail_file.write_text("\n".join(failed), encoding="utf-8")
            pbar.set_postfix(src=src[:12], clip=cid[-30:] if len(cid)>30 else cid, fail=len(failed))

    # Final save of failed list
    if failed:
        fail_file.write_text("\n".join(failed), encoding="utf-8")

    _section("Done")
    total_done = len(to_do) - len(failed)
    print(f"  Processed : {total_done}/{len(to_do)} clips")
    print(f"  Failed    : {len(failed)}/{len(to_do)} clips")
    print(f"\n  Per-source results:")
    for src, n_done in sorted(done_by_src.items()):
        n_fail = len(failed_by_src.get(src, []))
        print(f"    {src:<20} {n_done - n_fail:>5} ok  {n_fail:>4} failed")

    if failed:
        print(f"\n  Failed clip IDs -> {fail_file}")

    total_cached = sum(1 for cid, _, _ in clips if pipeline.is_cached(cid))
    print(f"\n  Total in cache now: {total_cached}/{len(clips)} clips")
    print(f"  Preprocessing {'COMPLETE' if total_cached == len(clips) else 'PARTIAL — re-run to resume'}\n")


if __name__ == "__main__":
    main()
