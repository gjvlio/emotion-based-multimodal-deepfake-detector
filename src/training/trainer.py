"""
trainer.py — Two-phase training loop.

Phase 1 (frozen backbones):
    - Uses forward_from_features(z_at, z_v) — no backbone forward pass
    - Higher LR (1e-3), trains only emotion heads + bilinear + sarcasm_head + classifier
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
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

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
        lambda_sarcasm: float      = 0.3,
        device:         str        = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.model   = model.to(device)
        self.device  = device
        self.fp16    = fp16 and (device == "cuda")
        self._amp_device = "cuda" if device == "cuda" else "cpu"
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.criterion    = MultiTaskLoss(lambda_a, lambda_b, lambda_sarcasm)
        self.scaler       = GradScaler("cuda", enabled=self.fp16)
        self.ckpt_dir     = Path(checkpoint_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.tb           = TBWriter(log_dir)

        print(f"\n{'='*60}")
        print(f"  DeepSentinel Trainer initialized")
        print(f"  Device       : {device}")
        print(f"  FP16         : {self.fp16}")
        print(f"  lambda_a     : {lambda_a}  (audio emotion loss weight)")
        print(f"  lambda_b     : {lambda_b}  (visual emotion loss weight)")
        print(f"  lambda_sarc  : {lambda_sarcasm}  (sarcasm loss weight)")
        print(f"  Train batches: {len(train_loader)}")
        print(f"  Val   batches: {len(val_loader)}")
        print(f"{'='*60}\n")

    # ── Phase 1 ───────────────────────────────────────────────────────────────

    def train_phase1(
        self,
        lr:           float = 1e-3,
        weight_decay: float = 1e-4,
        max_epochs:   int   = 10,
        patience:     int   = 5,
    ) -> None:
        """Train detection components on cached (Z_at, Z_v) features."""
        print(f"\n{'='*60}")
        print(f"  PHASE 1 — Frozen backbones")
        print(f"  Training: EmotionHeadA, EmotionHeadB, SarcasmHead,")
        print(f"            BilinearFusion, ClassifierMLP")
        print(f"  Backprop: heads + fusion + classifier (NO backbone grad)")
        print(f"  LR={lr}  weight_decay={weight_decay}  max_epochs={max_epochs}")
        print(f"{'='*60}\n")

        optimizer = AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=lr, weight_decay=weight_decay,
        )
        scheduler   = ReduceLROnPlateau(optimizer, patience=2, factor=0.5)
        stopper     = EarlyStopping(patience=patience)
        global_step = 0

        for epoch in range(1, max_epochs + 1):
            train_loss, train_components = self._train_epoch_cached(optimizer, epoch, global_step)
            val_loss, val_acc, val_components = self._val_epoch_cached(epoch)
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]["lr"]

            self.tb.scalars("loss",     {"train": train_loss, "val": val_loss},     epoch)
            self.tb.scalars("accuracy", {"val": val_acc},                           epoch)

            print(
                f"\n[P1 Epoch {epoch:3d}/{max_epochs}]  LR={current_lr:.2e}\n"
                f"  TRAIN  total={train_loss:.4f}  "
                f"bce={train_components['bce']:.4f}  "
                f"emo_a={train_components['emo_a']:.4f}  "
                f"emo_b={train_components['emo_b']:.4f}  "
                f"sarc={train_components['sarc']:.4f}\n"
                f"  VAL    total={val_loss:.4f}  "
                f"bce={val_components['bce']:.4f}  "
                f"emo_a={val_components['emo_a']:.4f}  "
                f"emo_b={val_components['emo_b']:.4f}  "
                f"sarc={val_components['sarc']:.4f}  "
                f"acc={val_acc:.4f}"
            )
            log.info(f"Epoch {epoch:3d} | train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

            self._save_checkpoint("best_phase1.pt", val_loss, epoch)
            if stopper.step(val_loss):
                print(f"\n  Early stopping triggered at epoch {epoch} (patience={patience}).")
                log.info(f"Early stopping at epoch {epoch}.")
                break

        print(f"\n  Phase 1 complete. Best val_loss={stopper.best:.4f}")
        print(f"  Checkpoint: {self.ckpt_dir}/best_phase1.pt\n")

    def _print_grad_report(self) -> float:
        """Print per-module gradient norms. Returns total grad norm."""
        modules = {
            "EmotionHeadA":   self.model.emotion_head_a,
            "EmotionHeadB":   self.model.emotion_head_b,
            "SarcasmHead":    self.model.sarcasm_head,
            "BilinearFusion": self.model.bilinear_fusion,
            "Classifier":     self.model.classifier,
        }
        print(f"\n  {'─'*52}")
        print(f"  GRADIENT FLOW REPORT")
        print(f"  {'─'*52}")
        total_norm = 0.0
        for name, module in modules.items():
            norms = [p.grad.norm().item() for p in module.parameters()
                     if p.grad is not None]
            if norms:
                mod_norm = (sum(n**2 for n in norms) ** 0.5)
                total_norm += mod_norm ** 2
                status = "OK" if mod_norm > 1e-8 else "DEAD (zero grad!)"
                print(f"  {name:<18} grad_norm={mod_norm:.6f}  [{status}]")
            else:
                print(f"  {name:<18} NO GRAD (frozen or not in graph)")
        total_norm = total_norm ** 0.5
        print(f"  {'─'*52}")
        print(f"  Total grad norm  : {total_norm:.6f}")
        print(f"  {'─'*52}\n")
        return total_norm

    def _train_epoch_cached(self, optimizer, epoch: int, global_step: int):
        self.model.train()
        total_loss = 0.0
        comp = {"bce": 0.0, "emo_a": 0.0, "emo_b": 0.0, "sarc": 0.0}
        n_batches = len(self.train_loader)
        first_batch = (epoch == 1)

        pbar = tqdm(self.train_loader, desc=f"P1 Train Ep{epoch}", unit="batch",
                    leave=False, dynamic_ncols=True)
        for batch in pbar:
            z_at  = batch["z_at"].to(self.device)
            z_v   = batch["z_v"].to(self.device)
            fl    = batch["fake_label"].to(self.device)
            ae    = batch["audio_emotion"].to(self.device)
            ve    = batch["visual_emotion"].to(self.device)
            sl    = batch["sarcasm_label"].to(self.device)

            optimizer.zero_grad()
            with autocast(self._amp_device, enabled=self.fp16):
                out  = self.model.forward_from_features(z_at, z_v)
                loss = self.criterion(
                    out.logit, fl,
                    out.emotion_a, out.emotion_b, ae, ve,
                    out.sarcasm, sl,
                )

            if first_batch:
                pbar.clear()
                print(f"\n  {'='*52}")
                print(f"  FIRST BATCH FORWARD PASS — shape check")
                print(f"  {'='*52}")
                print(f"  z_at shape      : {z_at.shape}   (expect B x 1536)")
                print(f"  z_v  shape      : {z_v.shape}    (expect B x 768)")
                print(f"  out.logit       : {out.logit.shape}    (expect B x 1)")
                print(f"  out.emotion_a   : {out.emotion_a.shape}  (expect B x 6)")
                print(f"  out.emotion_b   : {out.emotion_b.shape}  (expect B x 6)")
                print(f"  out.sarcasm     : {out.sarcasm.shape}    (expect B x 1)")
                print(f"  loss.total      : {loss.total.item():.6f}")
                print(f"  loss.bce        : {loss.bce.item():.6f}")
                print(f"  loss.emotion_a  : {loss.emotion_a.item():.6f}")
                print(f"  loss.emotion_b  : {loss.emotion_b.item():.6f}")
                print(f"  loss.sarcasm    : {loss.sarcasm.item():.6f}")
                print(f"  {'='*52}")
                print(f"  TRIGGERING BACKPROPAGATION...")

            self.scaler.scale(loss.total).backward()

            if first_batch:
                self.scaler.unscale_(optimizer)
                self._print_grad_report()
                print(f"  Backpropagation complete. Optimizer stepping...")
            else:
                self.scaler.unscale_(optimizer)

            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(optimizer)
            self.scaler.update()

            if first_batch:
                print(f"  Optimizer step complete. Training pipeline verified OK.\n")
                first_batch = False

            total_loss      += loss.total.item()
            comp["bce"]     += loss.bce.item()
            comp["emo_a"]   += loss.emotion_a.item()
            comp["emo_b"]   += loss.emotion_b.item()
            comp["sarc"]    += loss.sarcasm.item()
            global_step     += 1

            pbar.set_postfix(
                bce=f"{loss.bce.item():.3f}",
                emo_a=f"{loss.emotion_a.item():.3f}",
                emo_b=f"{loss.emotion_b.item():.3f}",
                sarc=f"{loss.sarcasm.item():.3f}",
            )

        for k in comp:
            comp[k] /= max(n_batches, 1)
        return total_loss / max(n_batches, 1), comp

    @torch.no_grad()
    def _val_epoch_cached(self, epoch: int):
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0
        comp = {"bce": 0.0, "emo_a": 0.0, "emo_b": 0.0, "sarc": 0.0}
        n_batches = len(self.val_loader)

        pbar = tqdm(self.val_loader, desc=f"P1 Val   Ep{epoch}", unit="batch",
                    leave=False, dynamic_ncols=True)
        for batch in pbar:
            z_at = batch["z_at"].to(self.device)
            z_v  = batch["z_v"].to(self.device)
            fl   = batch["fake_label"].to(self.device)
            ae   = batch["audio_emotion"].to(self.device)
            ve   = batch["visual_emotion"].to(self.device)
            sl   = batch["sarcasm_label"].to(self.device)

            with autocast(self._amp_device, enabled=self.fp16):
                out  = self.model.forward_from_features(z_at, z_v)
                loss = self.criterion(
                    out.logit, fl,
                    out.emotion_a, out.emotion_b, ae, ve,
                    out.sarcasm, sl,
                )
            total_loss     += loss.total.item()
            comp["bce"]    += loss.bce.item()
            comp["emo_a"]  += loss.emotion_a.item()
            comp["emo_b"]  += loss.emotion_b.item()
            comp["sarc"]   += loss.sarcasm.item()

            # Only count non-MUStARD clips for fake/real accuracy
            valid_mask = fl != -1
            if valid_mask.any():
                preds   = (torch.sigmoid(out.logit.squeeze(1)[valid_mask]) >= 0.5).long()
                correct += (preds == fl[valid_mask]).sum().item()
                total   += valid_mask.sum().item()

        for k in comp:
            comp[k] /= max(n_batches, 1)
        return (
            total_loss / max(n_batches, 1),
            correct / max(total, 1),
            comp,
        )

    # ── Phase 2 ───────────────────────────────────────────────────────────────

    def train_phase2(
        self,
        lr:           float = 1e-5,
        weight_decay: float = 1e-4,
        max_epochs:   int   = 20,
        patience:     int   = 5,
    ) -> None:
        """Full fine-tune — all parameters including backbones."""
        print(f"\n{'='*60}")
        print(f"  PHASE 2 — Full fine-tune (all parameters)")
        print(f"  Backprop: ALL params including Wav2Vec2 + BERT + ViT")
        print(f"  LR={lr}  weight_decay={weight_decay}  max_epochs={max_epochs}")
        print(f"{'='*60}\n")

        log.info("=== Phase 2: full fine-tune (all parameters) ===")
        self.model.load_backbones()
        self.model.unfreeze_backbones()
        optimizer = AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, patience=2, factor=0.5)
        stopper   = EarlyStopping(patience=patience)

        for epoch in range(1, max_epochs + 1):
            train_loss = self._train_epoch_e2e(optimizer, epoch)
            val_loss, val_acc, val_components = self._val_epoch_cached(epoch)
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]["lr"]

            self.tb.scalars("loss_p2", {"train": train_loss, "val": val_loss}, epoch)
            print(
                f"\n[P2 Epoch {epoch:3d}/{max_epochs}]  LR={current_lr:.2e}  "
                f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  val_acc={val_acc:.4f}"
            )
            log.info(f"[P2] Epoch {epoch:3d} | train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}")
            self._save_checkpoint("best_phase2.pt", val_loss, epoch)
            if stopper.step(val_loss):
                print(f"\n  Early stopping triggered at epoch {epoch}.")
                log.info(f"Early stopping at epoch {epoch}.")
                break

        print(f"\n  Phase 2 complete. Best val_loss={stopper.best:.4f}")

    def _train_epoch_e2e(self, optimizer, epoch: int) -> float:
        """End-to-end epoch — requires DataLoader returning raw audio/text/frames."""
        self.model.train()
        total_loss = 0.0
        pbar = tqdm(self.train_loader, desc=f"P2 Train Ep{epoch}", unit="batch",
                    leave=False, dynamic_ncols=True)
        for batch in pbar:
            audio   = batch["audio_values"].to(self.device)
            ids     = batch["input_ids"].to(self.device)
            mask    = batch["attention_mask"].to(self.device)
            pixels  = batch["keyframe_pixels"].to(self.device)
            fl      = batch["fake_label"].to(self.device)
            ae      = batch["audio_emotion"].to(self.device)
            ve      = batch["visual_emotion"].to(self.device)
            sl      = batch["sarcasm_label"].to(self.device)

            optimizer.zero_grad()
            with autocast(self._amp_device, enabled=self.fp16):
                out  = self.model(audio, ids, mask, pixels)
                loss = self.criterion(
                    out.logit, fl,
                    out.emotion_a, out.emotion_b, ae, ve,
                    out.sarcasm, sl,
                )
            self.scaler.scale(loss.total).backward()
            self.scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(optimizer)
            self.scaler.update()
            total_loss += loss.total.item()
            pbar.set_postfix(loss=f"{loss.total.item():.3f}")

        return total_loss / max(len(self.train_loader), 1)

    # ── Checkpointing ─────────────────────────────────────────────────────────

    def _save_checkpoint(self, filename: str, current_loss: float, epoch: int) -> None:
        import math
        if math.isnan(current_loss):
            return
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
        print(f"  [CKPT] Saved {filename}  (val_loss={current_loss:.4f}, epoch={epoch})")
        log.info(f"Checkpoint saved: {ckpt} (val_loss={current_loss:.4f})")

    def load_best(self, phase: int = 1) -> None:
        filename = f"best_phase{phase}.pt"
        ckpt = self.ckpt_dir / filename
        if not ckpt.exists():
            raise FileNotFoundError(f"No checkpoint at {ckpt}")
        data = torch.load(ckpt, weights_only=True)
        self.model.load_state_dict(data["model_state"])
        print(f"  [CKPT] Loaded {filename}  (epoch={data['epoch']}, val_loss={data['val_loss']:.4f})")
        log.info(f"Loaded checkpoint {filename} (epoch={data['epoch']}, val_loss={data['val_loss']:.4f})")
