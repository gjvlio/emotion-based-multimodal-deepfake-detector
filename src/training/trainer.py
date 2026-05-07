"""
trainer.py — Two-phase training loop.

Phase 1 (frozen backbones):
    - Uses forward_from_features(z_at, z_v) — no backbone forward pass
    - Higher LR (1e-3), trains only emotion heads + bilinear + classifier
    - Duration: max_epochs or until early stopping

Phase 2 (full fine-tune):
    - Calls model.load_backbones() + model.unfreeze_backbones()
    - Uses forward(...) with raw audio/text/frames — requires end-to-end batch
    - Lower LR (1e-5), all parameters updated

For the thesis scope, Phase 1 on cached features covers the primary training.
Phase 2 fine-tune is optional and requires raw-data DataLoader (not cached).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from src.models.detection_model import DeepfakeDetector
from src.training.losses import MultiTaskLoss, LossOutput
from src.utils.logging_utils import TBWriter, get_logger

log = get_logger(__name__)


class EarlyStopping:
    def __init__(self, patience: int = 5, min_delta: float = 1e-4):
        self.patience  = patience
        self.min_delta = min_delta
        self.best      = float("inf")
        self.counter   = 0
        self.triggered = False

    def step(self, val_loss: float) -> bool:
        if val_loss < self.best - self.min_delta:
            self.best    = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.triggered = True
        return self.triggered


class Trainer:
    def __init__(
        self,
        model:          DeepfakeDetector,
        train_loader:   DataLoader,
        val_loader:     DataLoader,
        checkpoint_dir: str | Path = "checkpoints",
        log_dir:        str | Path = "logs",
        fp16:           bool       = True,
        lambda_a:       float      = 0.5,
        lambda_b:       float      = 0.5,
        device:         str        = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.model   = model.to(device)
        self.device  = device
        self.fp16    = fp16 and (device == "cuda")
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.criterion    = MultiTaskLoss(lambda_a, lambda_b)
        self.scaler       = GradScaler(enabled=self.fp16)
        self.ckpt_dir     = Path(checkpoint_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.tb           = TBWriter(log_dir)

    # ── Phase 1 ───────────────────────────────────────────────────────────────

    def train_phase1(
        self,
        lr:           float = 1e-3,
        weight_decay: float = 1e-4,
        max_epochs:   int   = 10,
        patience:     int   = 5,
    ) -> None:
        """Train detection components on cached (Z_at, Z_v) features."""
        log.info("=== Phase 1: frozen backbones — training heads + fusion + classifier ===")
        optimizer = AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=lr, weight_decay=weight_decay,
        )
        scheduler   = ReduceLROnPlateau(optimizer, patience=2, factor=0.5)
        stopper     = EarlyStopping(patience=patience)
        global_step = 0

        for epoch in range(1, max_epochs + 1):
            train_loss = self._train_epoch_cached(optimizer, epoch, global_step)
            val_loss, val_acc = self._val_epoch_cached(epoch)
            scheduler.step(val_loss)
            self.tb.scalars("loss",     {"train": train_loss, "val": val_loss},     epoch)
            self.tb.scalars("accuracy", {"val": val_acc},                           epoch)
            log.info(f"Epoch {epoch:3d} | train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}")
            self._save_checkpoint("best_phase1.pt", val_loss, epoch)
            if stopper.step(val_loss):
                log.info(f"Early stopping at epoch {epoch}.")
                break

    def _train_epoch_cached(self, optimizer, epoch: int, global_step: int) -> float:
        self.model.train()
        total_loss = 0.0
        for batch in self.train_loader:
            z_at  = batch["z_at"].to(self.device)
            z_v   = batch["z_v"].to(self.device)
            fl    = batch["fake_label"].to(self.device)
            ae    = batch["audio_emotion"].to(self.device)
            ve    = batch["visual_emotion"].to(self.device)
            optimizer.zero_grad()
            with autocast(enabled=self.fp16):
                out  = self.model.forward_from_features(z_at, z_v)
                loss = self.criterion(out.logit, fl, out.emotion_a, out.emotion_b, ae, ve)
            self.scaler.scale(loss.total).backward()
            self.scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(optimizer)
            self.scaler.update()
            total_loss += loss.total.item()
            global_step += 1
        return total_loss / max(len(self.train_loader), 1)

    @torch.no_grad()
    def _val_epoch_cached(self, epoch: int):
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0
        for batch in self.val_loader:
            z_at = batch["z_at"].to(self.device)
            z_v  = batch["z_v"].to(self.device)
            fl   = batch["fake_label"].to(self.device)
            ae   = batch["audio_emotion"].to(self.device)
            ve   = batch["visual_emotion"].to(self.device)
            with autocast(enabled=self.fp16):
                out  = self.model.forward_from_features(z_at, z_v)
                loss = self.criterion(out.logit, fl, out.emotion_a, out.emotion_b, ae, ve)
            total_loss += loss.total.item()
            preds   = (torch.sigmoid(out.logit.squeeze(1)) >= 0.5).long()
            correct += (preds == fl).sum().item()
            total   += fl.size(0)
        return total_loss / max(len(self.val_loader), 1), correct / max(total, 1)

    # ── Phase 2 ───────────────────────────────────────────────────────────────

    def train_phase2(
        self,
        lr:           float = 1e-5,
        weight_decay: float = 1e-4,
        max_epochs:   int   = 20,
        patience:     int   = 5,
    ) -> None:
        """Full fine-tune — all parameters including backbones."""
        log.info("=== Phase 2: full fine-tune (all parameters) ===")
        self.model.load_backbones()
        self.model.unfreeze_backbones()
        optimizer = AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, patience=2, factor=0.5)
        stopper   = EarlyStopping(patience=patience)

        for epoch in range(1, max_epochs + 1):
            train_loss = self._train_epoch_e2e(optimizer, epoch)
            val_loss, val_acc = self._val_epoch_cached(epoch)
            scheduler.step(val_loss)
            self.tb.scalars("loss_p2", {"train": train_loss, "val": val_loss}, epoch)
            log.info(f"[P2] Epoch {epoch:3d} | train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}")
            self._save_checkpoint("best_phase2.pt", val_loss, epoch)
            if stopper.step(val_loss):
                log.info(f"Early stopping at epoch {epoch}.")
                break

    def _train_epoch_e2e(self, optimizer, epoch: int) -> float:
        """End-to-end epoch — requires DataLoader returning raw audio/text/frames."""
        self.model.train()
        total_loss = 0.0
        for batch in self.train_loader:
            audio   = batch["audio_values"].to(self.device)
            ids     = batch["input_ids"].to(self.device)
            mask    = batch["attention_mask"].to(self.device)
            pixels  = batch["keyframe_pixels"].to(self.device)
            fl      = batch["fake_label"].to(self.device)
            ae      = batch["audio_emotion"].to(self.device)
            ve      = batch["visual_emotion"].to(self.device)
            optimizer.zero_grad()
            with autocast(enabled=self.fp16):
                out  = self.model(audio, ids, mask, pixels)
                loss = self.criterion(out.logit, fl, out.emotion_a, out.emotion_b, ae, ve)
            self.scaler.scale(loss.total).backward()
            self.scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(optimizer)
            self.scaler.update()
            total_loss += loss.total.item()
        return total_loss / max(len(self.train_loader), 1)

    # ── Checkpointing ─────────────────────────────────────────────────────────

    def _save_checkpoint(self, filename: str, current_loss: float, epoch: int) -> None:
        ckpt = self.ckpt_dir / filename
        if ckpt.exists():
            saved = torch.load(ckpt, weights_only=True).get("val_loss", float("inf"))
            if current_loss >= saved:
                return
        torch.save(
            {"val_loss": current_loss, "epoch": epoch,
             "model_state": self.model.state_dict()},
            ckpt,
        )
        log.info(f"Checkpoint saved: {ckpt} (val_loss={current_loss:.4f})")

    def load_best(self, phase: int = 1) -> None:
        filename = f"best_phase{phase}.pt"
        ckpt = self.ckpt_dir / filename
        if not ckpt.exists():
            raise FileNotFoundError(f"No checkpoint at {ckpt}")
        data = torch.load(ckpt, weights_only=True)
        self.model.load_state_dict(data["model_state"])
        log.info(f"Loaded checkpoint {filename} (epoch={data['epoch']}, val_loss={data['val_loss']:.4f})")
