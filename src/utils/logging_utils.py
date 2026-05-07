"""
logging_utils.py — Logger factory + TensorBoard writer wrapper.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s — %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


class TBWriter:
    """Thin wrapper around SummaryWriter with graceful fallback if TensorBoard is absent."""

    def __init__(self, log_dir: str | Path):
        self._writer = None
        try:
            from torch.utils.tensorboard import SummaryWriter
            self._writer = SummaryWriter(log_dir=str(log_dir))
        except ImportError:
            logging.getLogger(__name__).warning(
                "tensorboard not installed — metrics will not be written to TensorBoard."
            )

    def scalar(self, tag: str, value: float, step: int) -> None:
        if self._writer:
            self._writer.add_scalar(tag, value, step)

    def scalars(self, main_tag: str, tag_scalar_dict: dict, step: int) -> None:
        if self._writer:
            self._writer.add_scalars(main_tag, tag_scalar_dict, step)

    def close(self) -> None:
        if self._writer:
            self._writer.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
