"""
train.py — Entry point for two-phase deepfake detector training.

Phase 1: frozen backbones, trains emotion heads + bilinear fusion + classifier.
          Uses cached (Z_at, Z_v) feature tensors. Fast.

Phase 2: full fine-tune of all parameters including Wav2Vec2 + BERT + ViT.
          Requires raw-data DataLoader. Slower. Optional.

Usage (from repo root):
    python scripts/train.py
    python scripts/train.py --phase 1 --device cuda
    python scripts/train.py --phase 2 --device cuda --resume checkpoints/best_phase1.pt
    python scripts/train.py --config configs/default.yaml --phase 1
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models.detection_model import DeepfakeDetector
from src.training.dataset import DeepfakeDataset
from src.training.trainer import Trainer
from src.utils.config import Config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


def build_dataloaders(cfg: Config, batch_size: int):
    train_ds, val_ds, _ = DeepfakeDataset.stratified_split(
        preprocessed_dir = cfg.paths.preprocessed_dir,
        train_ratio      = cfg.training.train_ratio,
        val_ratio        = cfg.training.val_ratio,
        track1_meta      = cfg.paths.track1_meta,
        track2_meta      = cfg.paths.track2_meta,
        track3_meta      = cfg.paths.track3_meta,
        track4_meta      = cfg.paths.track4_meta,
        meld_real_csv    = cfg.paths.meld_real_csv,
        mosei_real_csv   = cfg.paths.mosei_real_csv,
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    return train_loader, val_loader


def main():
    parser = argparse.ArgumentParser(description="Train deepfake detector")
    parser.add_argument("--config",  default=None)
    parser.add_argument("--phase",   type=int, default=1, choices=[1, 2])
    parser.add_argument("--device",  default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume",  default=None, help="Checkpoint path to resume from")
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config)
    log.info(f"Device: {args.device} | Phase: {args.phase} | fp16: {cfg.training.fp16}")

    model = DeepfakeDetector(
        wav2vec_model  = cfg.model.wav2vec_model,
        bert_model     = cfg.model.bert_model,
        vit_model      = cfg.model.vit_model,
        n_emotions     = cfg.model.n_emotions,
        proj_dim       = cfg.model.proj_dim,
        dropout_heads  = cfg.model.dropout_heads,
        dropout_cls    = cfg.model.dropout_classifier,
    )

    if args.resume:
        ckpt = torch.load(args.resume, weights_only=True)
        model.load_state_dict(ckpt["model_state"], strict=False)
        log.info(f"Resumed from {args.resume}")

    train_loader, val_loader = build_dataloaders(
        cfg, batch_size=cfg.training.batch_size
    )

    trainer = Trainer(
        model          = model,
        train_loader   = train_loader,
        val_loader     = val_loader,
        checkpoint_dir = cfg.paths.checkpoints_dir,
        log_dir        = cfg.paths.logs_dir,
        fp16           = cfg.training.fp16,
        lambda_a       = cfg.training.lambda_a,
        lambda_b       = cfg.training.lambda_b,
        device         = args.device,
    )

    if args.phase == 1:
        p = cfg.training.phase1
        trainer.train_phase1(
            lr           = p.lr,
            weight_decay = p.weight_decay,
            max_epochs   = p.max_epochs,
            patience     = p.patience,
        )
    else:
        p = cfg.training.phase2
        trainer.train_phase2(
            lr           = p.lr,
            weight_decay = p.weight_decay,
            max_epochs   = p.max_epochs,
            patience     = p.patience,
        )

    log.info("Training complete.")


if __name__ == "__main__":
    main()
