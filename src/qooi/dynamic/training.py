"""General ML training/runtime configuration."""

from __future__ import annotations

import math

from pydantic import model_validator

from qooi.core.config import StrictConfigModel


class TrainingConfig(StrictConfigModel):
    epochs: int = 20
    batch: int = 64
    lr: float = 1e-3
    grad_clip: float = 1.0
    seed: int | None = None
    device: str = "auto"
    threads: int = 0
    log_every: int = 100
    valid_batch: int = 0
    pred_batch: int = 0
    best_metric: str = ""
    best_mode: str = "min"
    early_stop_patience: int = 0

    @model_validator(mode="after")
    def _validate(self) -> TrainingConfig:
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.batch <= 0:
            raise ValueError("batch must be positive")
        if self.lr <= 0.0 or not math.isfinite(self.lr):
            raise ValueError("lr must be a finite positive float")
        if self.grad_clip <= 0.0 or not math.isfinite(self.grad_clip):
            raise ValueError("grad_clip must be a finite positive float")
        if self.threads < 0:
            raise ValueError("threads must be non-negative")
        if self.log_every < 0:
            raise ValueError("log_every must be non-negative")
        if self.valid_batch < 0:
            raise ValueError("valid_batch must be non-negative")
        if self.pred_batch < 0:
            raise ValueError("pred_batch must be non-negative")
        if self.best_metric and self.best_mode not in {"min", "max"}:
            raise ValueError("best_mode must be 'min' or 'max' when best_metric is set")
        if self.early_stop_patience < 0:
            raise ValueError("early_stop_patience must be non-negative")
        if not self.device.strip():
            raise ValueError("device must be non-empty")
        return self

    @property
    def valid_batch_rows(self) -> int:
        return self.valid_batch if self.valid_batch > 0 else self.batch * 4

    @property
    def pred_batch_rows(self) -> int:
        return self.pred_batch if self.pred_batch > 0 else self.batch * 4

    def torch_device(self):
        import torch

        requested = self.device.strip().lower()
        if requested == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            mps = getattr(torch.backends, "mps", None)
            if mps is not None and mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("learn.train.device requests CUDA, but torch.cuda is unavailable")
        mps = getattr(torch.backends, "mps", None)
        if device.type == "mps" and (mps is None or not mps.is_available()):
            raise RuntimeError("learn.train.device requests MPS, but torch MPS is unavailable")
        return device
