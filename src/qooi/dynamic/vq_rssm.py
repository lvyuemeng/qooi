"""Fixed VQ-RSSM model, training, checkpoint, and prediction lifecycle."""

from __future__ import annotations

import csv
import logging
import math
import time
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path

import torch
from torch import nn

from qooi.dynamic.contracts import CodeSequence, SequenceDataset, SplitName, WindowDataset
from qooi.dynamic.training import TrainingConfig

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class VqRssmSpec:
    input_dim: int = 5
    hidden_dim: int = 128
    latent_dim: int = 16
    num_codes: int = 128
    ema_decay: float = 0.99
    ema_eps: float = 1e-5
    normalize_codebook: bool = False
    reset_dead_codes: bool = False
    reset_interval: int = 500
    reset_threshold: float = 0.1
    reset_fraction: float = 0.1
    reset_warmup_epochs: int = 5
    reset_candidate_similarity_max: float = 0.9
    diversity_weight: float = 0.0
    diversity_margin: float = 0.5
    eps: float = 1e-8
    objective_terms: tuple[str, ...] = ("reconstruct",)
    reconstruct_weight: float = 1.0
    future_weight: float = 0.0
    future_min_len: int = 1
    future_max_len: int = 20
    future_samples: int = 1
    future_dim: int = 32
    future_temperature: float = 0.3
    future_source: str = "features"
    future_length_policy: str = "cycle"
    future_warmup_epochs: int = 0
    future_standard_weight: float = 1.0
    future_similarity_weight: float = 0.0
    future_similarity_top_k: int = 3
    future_similarity_mse_weight: float = 1.0
    future_similarity_cosine_weight: float = 0.0
    future_similarity_max_distance: float = 0.0
    future_detrend: bool = False
    future_detrend_half_life: float = 20.0
    temporal_consistency_weight: float = 0.0
    temporal_consistency_temperature: float = 1.0

    def __post_init__(self) -> None:
        if self.input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if self.latent_dim <= 0:
            raise ValueError("latent_dim must be positive")
        if self.num_codes <= 0:
            raise ValueError("num_codes must be positive")
        if not 0.0 <= self.ema_decay < 1.0 or not math.isfinite(self.ema_decay):
            raise ValueError("ema_decay must be finite and satisfy 0 <= ema_decay < 1")
        if self.ema_eps <= 0.0 or not math.isfinite(self.ema_eps):
            raise ValueError("ema_eps must be a finite positive float")
        if self.reset_interval <= 0:
            raise ValueError("reset_interval must be positive")
        if not 0.0 < self.reset_threshold < 1.0 or not math.isfinite(self.reset_threshold):
            raise ValueError("reset_threshold must be finite and satisfy 0 < threshold < 1")
        if not 0.0 < self.reset_fraction <= 1.0 or not math.isfinite(self.reset_fraction):
            raise ValueError("reset_fraction must be finite and satisfy 0 < fraction <= 1")
        if self.reset_warmup_epochs < 0:
            raise ValueError("reset_warmup_epochs must be non-negative")
        if not -1.0 <= self.reset_candidate_similarity_max <= 1.0 or not math.isfinite(
            self.reset_candidate_similarity_max
        ):
            raise ValueError(
                "reset_candidate_similarity_max must be finite and satisfy -1 <= value <= 1"
            )
        if self.reset_dead_codes and not self.normalize_codebook:
            raise ValueError("reset_dead_codes requires normalize_codebook=True")
        if self.diversity_weight < 0.0 or not math.isfinite(self.diversity_weight):
            raise ValueError("diversity_weight must be a finite non-negative float")
        if not -1.0 <= self.diversity_margin < 1.0 or not math.isfinite(
            self.diversity_margin
        ):
            raise ValueError("diversity_margin must be finite and satisfy -1 <= margin < 1")
        if self.diversity_weight > 0.0 and not self.normalize_codebook:
            raise ValueError("diversity_weight requires normalize_codebook=True")
        if self.eps <= 0.0 or not math.isfinite(self.eps):
            raise ValueError("eps must be a finite positive float")
        if not self.objective_terms:
            raise ValueError("objective_terms must contain at least one term")
        invalid_terms = sorted(set(self.objective_terms) - {"reconstruct", "future_infonce"})
        if invalid_terms:
            raise ValueError("unsupported objective terms: " + ", ".join(invalid_terms))
        for name in ("reconstruct_weight", "future_weight"):
            value = getattr(self, name)
            if value < 0.0 or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite non-negative float")
        if self.future_min_len <= 0:
            raise ValueError("future_min_len must be positive")
        if self.future_max_len < self.future_min_len:
            raise ValueError("future_max_len must be >= future_min_len")
        if self.future_samples <= 0:
            raise ValueError("future_samples must be positive")
        if self.future_dim <= 0:
            raise ValueError("future_dim must be positive")
        if self.future_temperature <= 0.0 or not math.isfinite(self.future_temperature):
            raise ValueError("future_temperature must be a finite positive float")
        if self.future_source != "features":
            raise ValueError("future_source must be 'features'")
        if self.future_length_policy != "cycle":
            raise ValueError("future_length_policy must be 'cycle'")
        if self.future_warmup_epochs < 0:
            raise ValueError("future_warmup_epochs must be non-negative")
        for name in (
            "future_standard_weight",
            "future_similarity_weight",
            "future_similarity_mse_weight",
            "future_similarity_cosine_weight",
            "future_similarity_max_distance",
        ):
            value = getattr(self, name)
            if value < 0.0 or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite non-negative float")
        if self.future_similarity_top_k <= 0:
            raise ValueError("future_similarity_top_k must be positive")
        if self.future_detrend_half_life <= 0.0 or not math.isfinite(
            self.future_detrend_half_life
        ):
            raise ValueError("future_detrend_half_life must be a finite positive float")
        if self.future_similarity_weight > 0.0 and (
            self.future_similarity_mse_weight + self.future_similarity_cosine_weight <= 0.0
        ):
            raise ValueError(
                "future_similarity_weight requires a positive future similarity metric weight"
            )
        if (
            "future_infonce" in self.objective_terms
            and self.future_weight > 0.0
            and self.future_standard_weight + self.future_similarity_weight <= 0.0
        ):
            raise ValueError(
                "future_infonce with positive future_weight requires a positive "
                "future contrast component weight"
            )
        if not 0.0 <= self.temporal_consistency_weight <= 0.01 or not math.isfinite(
            self.temporal_consistency_weight
        ):
            raise ValueError(
                "temporal_consistency_weight must be finite and satisfy 0 <= weight <= 0.01"
            )
        if self.temporal_consistency_temperature <= 0.0 or not math.isfinite(
            self.temporal_consistency_temperature
        ):
            raise ValueError(
                "temporal_consistency_temperature must be a finite positive float"
            )


@dataclass(frozen=True)
class VqRssmCheckpoint:
    spec: VqRssmSpec
    path: Path | None
    state_dict: dict[str, torch.Tensor]


@dataclass(frozen=True)
class VqRssmScheduleStage:
    name: str = ""
    start_epoch: int = 1
    epochs: int = 1
    reconstruct_weight: float | None = None
    future_weight: float | None = None
    diversity_weight: float | None = None
    reset_fraction: float | None = None
    reset_dead_codes: bool | None = None
    freeze_encoder_blocks: tuple[str, ...] = ()
    lr: float | None = None

    def __post_init__(self) -> None:
        if self.start_epoch <= 0:
            raise ValueError("schedule stage start_epoch must be positive")
        if self.epochs <= 0:
            raise ValueError("schedule stage epochs must be positive")
        for name in (
            "reconstruct_weight",
            "future_weight",
            "diversity_weight",
            "reset_fraction",
        ):
            value = getattr(self, name)
            if value is not None and (value < 0.0 or not math.isfinite(value)):
                raise ValueError(f"schedule stage {name} must be finite and non-negative")
        if self.reset_fraction is not None and not 0.0 < self.reset_fraction <= 1.0:
            raise ValueError("schedule stage reset_fraction must satisfy 0 < fraction <= 1")
        if self.lr is not None and (self.lr <= 0.0 or not math.isfinite(self.lr)):
            raise ValueError("schedule stage lr must be a finite positive float")

    def includes(self, epoch: int) -> bool:
        return self.start_epoch <= epoch < self.start_epoch + self.epochs


@dataclass(frozen=True)
class EpochMetrics:
    epoch: int
    train_loss: float
    valid_loss: float | None
    train_rows: int
    valid_rows: int
    test_rows: int
    active_codes: int
    codebook_utilization_pct: float
    train_recon_loss: float | None = None
    train_future_loss: float | None = None
    train_future_accuracy: float | None = None
    train_future_rows: int = 0
    train_temporal_consistency_loss: float | None = None
    valid_recon_loss: float | None = None
    valid_future_loss: float | None = None
    valid_future_accuracy: float | None = None
    valid_future_rows: int = 0
    valid_temporal_consistency_loss: float | None = None
    train_z_e_norm_mean: float | None = None
    train_z_q_norm_mean: float | None = None
    train_vq_distance_mean: float | None = None
    train_vq_distance_p95: float | None = None
    train_vq_distance_max: float | None = None
    valid_z_e_norm_mean: float | None = None
    valid_z_q_norm_mean: float | None = None
    valid_vq_distance_mean: float | None = None
    valid_vq_distance_p95: float | None = None
    valid_vq_distance_max: float | None = None
    codebook_norm_mean: float | None = None
    codebook_norm_max: float | None = None
    train_diversity_loss: float | None = None
    valid_diversity_loss: float | None = None
    codebook_similarity_mean: float | None = None
    codebook_similarity_max: float | None = None

    def log_fields(self, elapsed_s: float) -> dict[str, object]:
        return {
            "epoch": self.epoch,
            "train_loss": self.train_loss,
            "valid_loss": self.valid_loss,
            "train_recon": self.train_recon_loss,
            "train_future": self.train_future_loss,
            "train_future_acc": self.train_future_accuracy,
            "train_future_rows": self.train_future_rows,
            "train_temporal_consistency": self.train_temporal_consistency_loss,
            "valid_recon": self.valid_recon_loss,
            "valid_future": self.valid_future_loss,
            "valid_future_acc": self.valid_future_accuracy,
            "valid_future_rows": self.valid_future_rows,
            "valid_temporal_consistency": self.valid_temporal_consistency_loss,
            "train_vq_distance_mean": self.train_vq_distance_mean,
            "train_vq_distance_p95": self.train_vq_distance_p95,
            "train_vq_distance_max": self.train_vq_distance_max,
            "valid_vq_distance_mean": self.valid_vq_distance_mean,
            "valid_vq_distance_p95": self.valid_vq_distance_p95,
            "valid_vq_distance_max": self.valid_vq_distance_max,
            "codebook_norm_mean": self.codebook_norm_mean,
            "codebook_norm_max": self.codebook_norm_max,
            "train_diversity": self.train_diversity_loss,
            "valid_diversity": self.valid_diversity_loss,
            "codebook_similarity_mean": self.codebook_similarity_mean,
            "codebook_similarity_max": self.codebook_similarity_max,
            "active_codes": self.active_codes,
            "utilization_pct": self.codebook_utilization_pct,
            "elapsed_s": elapsed_s,
        }

    def log_line(self, elapsed_s: float) -> str:
        return _format_log_fields(self.log_fields(elapsed_s))


@dataclass(frozen=True)
class TrainingResult:
    checkpoint: VqRssmCheckpoint
    checkpoint_path: Path | None
    metrics: tuple[EpochMetrics, ...]
    summary: EpochMetrics | None
    best_summary: EpochMetrics | None = None


@dataclass(frozen=True)
class InferenceDiagnostics:
    codes: tuple[int, ...]
    distances: tuple[float, ...]
    hidden_states: tuple[tuple[float, ...], ...]
    reconstructions: tuple[tuple[float, ...], ...]


@dataclass(frozen=True)
class SequenceRuntimeConfig:
    chunk: int = 256
    warmup: int = 64
    stride: int = 1
    carry: bool = True

    def __post_init__(self) -> None:
        for name in ("chunk", "warmup", "stride"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")


@dataclass(frozen=True)
class LossBundle:
    total: torch.Tensor
    recon: torch.Tensor
    diversity: torch.Tensor
    future: torch.Tensor | None
    future_accuracy: torch.Tensor | None
    future_rows: int
    temporal_consistency: torch.Tensor | None
    code_counts: torch.Tensor
    next_hidden: torch.Tensor
    next_input_embedding: torch.Tensor
    next_state_embedding: torch.Tensor
    z_e: torch.Tensor
    codes: torch.Tensor
    z_e_norm_mean: torch.Tensor
    z_q_norm_mean: torch.Tensor
    vq_distance_mean: torch.Tensor
    vq_distance_p95: torch.Tensor
    vq_distance_max: torch.Tensor


@dataclass(frozen=True)
class RuntimeInfo:
    device: torch.device
    device_name: str


@dataclass
class EpochAccumulator:
    loss_sum: torch.Tensor
    recon_sum: torch.Tensor
    diversity_sum: torch.Tensor
    future_sum: torch.Tensor
    future_accuracy_sum: torch.Tensor
    future_rows: int
    temporal_consistency_sum: torch.Tensor
    temporal_consistency_rows: int
    z_e_norm_sum: torch.Tensor
    z_q_norm_sum: torch.Tensor
    vq_distance_sum: torch.Tensor
    vq_distance_p95_sum: torch.Tensor
    vq_distance_max: torch.Tensor
    rows: int
    code_counts: torch.Tensor

    @classmethod
    def empty(cls, spec: VqRssmSpec, device: torch.device) -> EpochAccumulator:
        zero = torch.zeros((), dtype=torch.float32, device=device)
        return cls(
            loss_sum=zero.clone(),
            recon_sum=zero.clone(),
            diversity_sum=zero.clone(),
            future_sum=zero.clone(),
            future_accuracy_sum=zero.clone(),
            future_rows=0,
            temporal_consistency_sum=zero.clone(),
            temporal_consistency_rows=0,
            z_e_norm_sum=zero.clone(),
            z_q_norm_sum=zero.clone(),
            vq_distance_sum=zero.clone(),
            vq_distance_p95_sum=zero.clone(),
            vq_distance_max=zero.clone(),
            rows=0,
            code_counts=torch.zeros(spec.num_codes, dtype=torch.long, device=device),
        )

    def add(self, bundle: LossBundle, rows: int) -> None:
        self.loss_sum += bundle.total.detach() * rows
        self.recon_sum += bundle.recon.detach() * rows
        self.diversity_sum += bundle.diversity.detach() * rows
        if bundle.future is not None and bundle.future_accuracy is not None:
            self.future_sum += bundle.future.detach() * bundle.future_rows
            self.future_accuracy_sum += bundle.future_accuracy.detach() * bundle.future_rows
            self.future_rows += bundle.future_rows
        if bundle.temporal_consistency is not None:
            self.temporal_consistency_sum += bundle.temporal_consistency.detach() * rows
            self.temporal_consistency_rows += rows
        self.z_e_norm_sum += bundle.z_e_norm_mean.detach() * rows
        self.z_q_norm_sum += bundle.z_q_norm_mean.detach() * rows
        self.vq_distance_sum += bundle.vq_distance_mean.detach() * rows
        self.vq_distance_p95_sum += bundle.vq_distance_p95.detach() * rows
        self.vq_distance_max = torch.maximum(self.vq_distance_max, bundle.vq_distance_max.detach())
        self.rows += rows
        self.code_counts += bundle.code_counts


@dataclass(frozen=True)
class LossParts:
    total: float
    recon: float
    diversity: float
    future: float | None
    future_accuracy: float | None
    future_rows: int
    temporal_consistency: float | None
    z_e_norm_mean: float
    z_q_norm_mean: float
    vq_distance_mean: float
    vq_distance_p95: float
    vq_distance_max: float

    @classmethod
    def from_accumulator(cls, acc: EpochAccumulator) -> LossParts:
        rows = max(1, acc.rows)
        future = None
        future_accuracy = None
        if acc.future_rows > 0:
            future = float((acc.future_sum / acc.future_rows).detach().cpu())
            future_accuracy = float((acc.future_accuracy_sum / acc.future_rows).detach().cpu())
        temporal_consistency = None
        if acc.temporal_consistency_rows > 0:
            temporal_consistency = float(
                (acc.temporal_consistency_sum / acc.temporal_consistency_rows).detach().cpu()
            )
        return cls(
            total=float((acc.loss_sum / rows).detach().cpu()),
            recon=float((acc.recon_sum / rows).detach().cpu()),
            diversity=float((acc.diversity_sum / rows).detach().cpu()),
            future=future,
            future_accuracy=future_accuracy,
            future_rows=acc.future_rows,
            temporal_consistency=temporal_consistency,
            z_e_norm_mean=float((acc.z_e_norm_sum / rows).detach().cpu()),
            z_q_norm_mean=float((acc.z_q_norm_sum / rows).detach().cpu()),
            vq_distance_mean=float((acc.vq_distance_sum / rows).detach().cpu()),
            vq_distance_p95=float((acc.vq_distance_p95_sum / rows).detach().cpu()),
            vq_distance_max=float(acc.vq_distance_max.detach().cpu()),
        )


@dataclass(frozen=True)
class SequenceBatch:
    sequence_indices: tuple[int, ...]
    start: int
    batch: torch.Tensor
    hidden: torch.Tensor | None
    prev_input_embedding: torch.Tensor | None = None
    prev_state_embedding: torch.Tensor | None = None


@dataclass(frozen=True)
class FutureTargets:
    positions: tuple[tuple[int, int], ...]
    segments: torch.Tensor
    mask: torch.Tensor


@dataclass(frozen=True)
class FutureLoss:
    loss: torch.Tensor
    accuracy: torch.Tensor
    rows: int
    standard_loss: torch.Tensor | None = None
    similarity_loss: torch.Tensor | None = None


class VqRssmModel(nn.Module):
    def __init__(self, spec: VqRssmSpec) -> None:
        super().__init__()
        self.spec = spec
        self.embedding = nn.Linear(spec.input_dim, 32)
        self.initial_input_embedding = nn.Parameter(torch.zeros(32))
        self.initial_state_embedding = nn.Parameter(torch.zeros(spec.latent_dim))
        self.latent_projection = nn.Linear(spec.hidden_dim, spec.latent_dim)
        self.codebook = nn.Embedding(spec.num_codes, spec.latent_dim)
        self.codebook.weight.requires_grad_(False)
        if spec.normalize_codebook:
            with torch.no_grad():
                self.codebook.weight.copy_(
                    torch.nn.functional.normalize(self.codebook.weight, dim=-1)
                )
        self.register_buffer("ema_cluster_size", torch.zeros(spec.num_codes))
        self.register_buffer("ema_code_sum", self.codebook.weight.detach().clone())
        self.register_buffer("codebook_update_steps", torch.zeros((), dtype=torch.long))
        self.decoder = nn.Sequential(
            nn.Linear(spec.hidden_dim + spec.latent_dim, spec.hidden_dim),
            nn.ReLU(),
            nn.Linear(spec.hidden_dim, spec.input_dim),
        )
        self.transition = nn.GRUCell(32 + spec.latent_dim, spec.hidden_dim)
        self.future_encoder = nn.Sequential(
            nn.Linear(spec.input_dim, spec.hidden_dim),
            nn.ReLU(),
            nn.Linear(spec.hidden_dim, spec.future_dim),
        )
        self.future_query = nn.Linear(spec.hidden_dim + spec.latent_dim, spec.future_dim)

    def quantize(self, z_e: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        query = torch.nn.functional.normalize(z_e, dim=-1) if self.spec.normalize_codebook else z_e
        codebook = (
            torch.nn.functional.normalize(self.codebook.weight, dim=-1)
            if self.spec.normalize_codebook
            else self.codebook.weight
        )
        distances = torch.cdist(query, codebook)
        codes = distances.argmin(dim=-1)
        z_q = torch.nn.functional.embedding(codes, codebook)
        selected_distances = distances.gather(1, codes.unsqueeze(1)).squeeze(1)
        return z_q, codes, selected_distances

    def _assignment_logits(self, z_e: torch.Tensor) -> torch.Tensor:
        codebook = (
            torch.nn.functional.normalize(self.codebook.weight, dim=-1)
            if self.spec.normalize_codebook
            else self.codebook.weight
        )
        distances = torch.cdist(z_e, codebook)
        return -distances / self.spec.temporal_consistency_temperature

    def _initial_prev_inputs(
        self, batch: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prev_embed = self.initial_input_embedding.unsqueeze(0).expand(batch, -1).to(device)
        prev_state = self.initial_state_embedding.unsqueeze(0).expand(batch, -1).to(device)
        return prev_embed, prev_state

    def _step(
        self,
        x_t: torch.Tensor,
        h: torch.Tensor,
        prev_embed: torch.Tensor,
        prev_state: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        h_t = self.transition(torch.cat([prev_embed, prev_state], dim=-1), h)
        z_e = torch.nn.functional.normalize(self.latent_projection(h_t), dim=-1)
        z_q, code, distance = self.quantize(z_e)
        z_q_st = z_e + (z_q - z_e).detach()
        recon = self.decoder(torch.cat([h_t, z_q_st], dim=-1))
        embed = self.embedding(x_t)
        return {
            "h": h_t,
            "z_e": z_e,
            "z_q": z_q,
            "z_q_st": z_q_st,
            "code": code,
            "distance": distance,
            "recon": recon,
            "next_embed": embed,
            "next_state": z_q_st,
        }

    def update_codebook(
        self,
        z_e: torch.Tensor,
        codes: torch.Tensor,
        *,
        epoch: int | None = None,
    ) -> None:
        with torch.no_grad():
            flat_z = z_e.detach().reshape(-1, self.spec.latent_dim)
            if self.spec.normalize_codebook:
                flat_z = torch.nn.functional.normalize(flat_z, dim=-1)
            flat_codes = codes.detach().reshape(-1)
            counts = torch.bincount(flat_codes, minlength=self.spec.num_codes).to(flat_z.dtype)
            sums = torch.zeros_like(self.ema_code_sum)
            sums.index_add_(0, flat_codes, flat_z)
            self.ema_cluster_size.mul_(self.spec.ema_decay).add_(
                counts, alpha=1.0 - self.spec.ema_decay
            )
            self.ema_code_sum.mul_(self.spec.ema_decay).add_(sums, alpha=1.0 - self.spec.ema_decay)
            total = self.ema_cluster_size.sum()
            smoothed = (
                (self.ema_cluster_size + self.spec.ema_eps)
                / (total + self.spec.num_codes * self.spec.ema_eps)
                * total.clamp_min(self.spec.ema_eps)
            )
            updated = self.codebook.weight.detach().clone()
            live = self.ema_cluster_size > 0
            updated[live] = self.ema_code_sum[live] / smoothed[live].clamp_min(
                self.spec.ema_eps
            ).unsqueeze(1)
            if self.spec.normalize_codebook:
                updated = torch.nn.functional.normalize(updated, dim=-1)
            self.codebook.weight.copy_(updated)
            self.codebook_update_steps.add_(1)
            self._reset_dead_codes(flat_z, flat_codes, epoch)
            self._apply_codebook_diversity()

    def codebook_diversity_loss(self) -> torch.Tensor:
        if self.spec.diversity_weight <= 0.0:
            return self.codebook.weight.new_tensor(0.0)
        if self.spec.num_codes < 2:
            return self.codebook.weight.new_tensor(0.0)
        codebook = torch.nn.functional.normalize(self.codebook.weight, dim=-1)
        sim = codebook @ codebook.T
        off_diagonal = ~torch.eye(sim.shape[0], dtype=torch.bool, device=sim.device)
        violation = torch.relu(sim[off_diagonal] - self.spec.diversity_margin)
        return violation.pow(2).mean()

    def _apply_codebook_diversity(self) -> None:
        if self.spec.diversity_weight <= 0.0 or not self.spec.normalize_codebook:
            return
        codebook = torch.nn.functional.normalize(self.codebook.weight, dim=-1)
        sim = codebook @ codebook.T
        mask = ~torch.eye(sim.shape[0], dtype=torch.bool, device=sim.device)
        violation = torch.relu(sim - self.spec.diversity_margin) * mask.to(sim.dtype)
        if not violation.any().item():
            return
        neighbors = violation.count_nonzero(dim=1).clamp_min(1).unsqueeze(1)
        repulsion = violation @ codebook / neighbors
        updated = torch.nn.functional.normalize(
            codebook - self.spec.diversity_weight * repulsion, dim=-1
        )
        self.codebook.weight.copy_(updated)
        live = self.ema_cluster_size > 0
        self.ema_code_sum[live] = updated[live] * self.ema_cluster_size[live].unsqueeze(1)

    def _reset_dead_codes(
        self,
        flat_z: torch.Tensor,
        flat_codes: torch.Tensor,
        epoch: int | None,
    ) -> int:
        if not self.spec.reset_dead_codes or not self.spec.normalize_codebook:
            return 0
        if epoch is not None and epoch <= self.spec.reset_warmup_epochs:
            return 0
        if int(self.codebook_update_steps.item()) % self.spec.reset_interval != 0:
            return 0
        mean_cluster = self.ema_cluster_size.mean()
        if mean_cluster.item() <= 0.0:
            return 0
        dead_mask = self.ema_cluster_size < self.spec.reset_threshold * mean_cluster
        active_mask = (~dead_mask) & (self.ema_cluster_size > 0)
        if not dead_mask.any().item() or not active_mask.any().item():
            return 0
        dead_indices = dead_mask.nonzero(as_tuple=False).flatten()
        active_indices = active_mask.nonzero(as_tuple=False).flatten()
        max_reset = max(1, int(self.spec.num_codes * self.spec.reset_fraction))
        reset_count = min(int(dead_indices.numel()), max_reset, int(flat_z.shape[0]))
        if reset_count <= 0:
            return 0
        dead_scores = self.ema_cluster_size[dead_indices]
        selected_dead = dead_indices[dead_scores.argsort()[:reset_count]]
        active_codebook = torch.nn.functional.normalize(
            self.codebook.weight[active_indices], dim=-1
        )
        nearest_sim = (flat_z @ active_codebook.T).max(dim=1).values
        candidate_indices = self._select_reset_candidates(
            flat_z,
            1.0 - nearest_sim,
            reset_count,
        )
        reset_count = int(candidate_indices.numel())
        if reset_count <= 0:
            return 0
        selected_dead = selected_dead[:reset_count]
        new_vectors = flat_z[candidate_indices]
        self.codebook.weight[selected_dead] = new_vectors
        self.ema_code_sum[selected_dead] = new_vectors
        self.ema_cluster_size[selected_dead] = mean_cluster.clamp_min(1.0)
        self.codebook.weight.copy_(torch.nn.functional.normalize(self.codebook.weight, dim=-1))
        logger.info(
            "codebook_reset step=%s epoch=%s reset=%s dead=%s mean_cluster=%.4f",
            int(self.codebook_update_steps.item()),
            epoch,
            reset_count,
            int(dead_indices.numel()),
            float(mean_cluster.detach().cpu()),
        )
        return reset_count

    def _select_reset_candidates(
        self,
        flat_z: torch.Tensor,
        coverage_gap: torch.Tensor,
        reset_count: int,
    ) -> torch.Tensor:
        ordered = coverage_gap.argsort(descending=True)
        selected: list[torch.Tensor] = []
        for index in ordered:
            if len(selected) >= reset_count:
                break
            if selected:
                selected_indices = torch.stack(selected)
                sim = flat_z[index : index + 1] @ flat_z[selected_indices].T
                if sim.max().item() > self.spec.reset_candidate_similarity_max:
                    continue
            selected.append(index)
        if not selected:
            return ordered[:0]
        return torch.stack(selected)

    def future_loss(self, queries: torch.Tensor, targets: FutureTargets) -> FutureLoss:
        mask = targets.mask.unsqueeze(-1).to(targets.segments.dtype)
        denom = mask.sum(dim=1).clamp_min(1.0)
        pooled = (targets.segments * mask).sum(dim=1) / denom
        keys = self.future_encoder(pooled)
        q = torch.nn.functional.normalize(queries, dim=-1)
        k = torch.nn.functional.normalize(keys, dim=-1)
        logits = q @ k.T / self.spec.future_temperature
        labels = torch.arange(logits.shape[0], device=logits.device)
        standard_loss = None
        similarity_loss = None
        loss = logits.new_tensor(0.0)
        active_weight = 0.0
        if self.spec.future_standard_weight > 0.0:
            standard_loss = torch.nn.functional.cross_entropy(logits, labels)
            loss = loss + standard_loss * self.spec.future_standard_weight
            active_weight += self.spec.future_standard_weight
        if self.spec.future_similarity_weight > 0.0:
            similarity_loss = self._future_similarity_loss(logits, pooled.detach())
            loss = loss + similarity_loss * self.spec.future_similarity_weight
            active_weight += self.spec.future_similarity_weight
        if active_weight <= 0.0:
            raise ValueError("future_loss requires a positive future contrast component weight")
        loss = loss / active_weight
        accuracy = (logits.argmax(dim=1) == labels).to(torch.float32).mean()
        return FutureLoss(
            loss=loss,
            accuracy=accuracy,
            rows=logits.shape[0],
            standard_loss=standard_loss,
            similarity_loss=similarity_loss,
        )

    def _future_similarity_loss(
        self, logits: torch.Tensor, pooled_targets: torch.Tensor
    ) -> torch.Tensor:
        batch = logits.shape[0]
        device = logits.device
        diff = pooled_targets[:, None, :] - pooled_targets[None, :, :]
        mse_distance = diff.square().mean(dim=-1)
        scores = logits.new_zeros((batch, batch))
        if self.spec.future_similarity_mse_weight > 0.0:
            scores = scores - self.spec.future_similarity_mse_weight * mse_distance
        if self.spec.future_similarity_cosine_weight > 0.0:
            normalized = torch.nn.functional.normalize(pooled_targets, dim=-1)
            scores = scores + self.spec.future_similarity_cosine_weight * (
                normalized @ normalized.T
            )
        positive_mask = torch.eye(batch, dtype=torch.bool, device=device)
        candidate_scores = scores.masked_fill(positive_mask, -torch.inf)
        if self.spec.future_similarity_max_distance > 0.0:
            candidate_scores = candidate_scores.masked_fill(
                mse_distance > self.spec.future_similarity_max_distance, -torch.inf
            )
        k = min(self.spec.future_similarity_top_k, max(0, batch - 1))
        if k > 0:
            top_scores, top_indices = torch.topk(candidate_scores, k=k, dim=1)
            valid = torch.isfinite(top_scores)
            rows = torch.arange(batch, device=device).unsqueeze(1).expand_as(top_indices)
            positive_mask[rows[valid], top_indices[valid]] = True
        numerator = logits.masked_fill(~positive_mask, -torch.inf).logsumexp(dim=1)
        denominator = logits.logsumexp(dim=1)
        return (denominator - numerator).mean()

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        loss, output = self._run(x, collect=True, future_targets=None)
        output.update(
            future_loss=loss.future,
        )
        return output

    def loss_bundle(
        self,
        x: torch.Tensor,
        step_index: int = 0,
        *,
        h: torch.Tensor | None = None,
        prev_input_embedding: torch.Tensor | None = None,
        prev_state_embedding: torch.Tensor | None = None,
        future_targets: FutureTargets | None = None,
    ) -> LossBundle:
        _ = step_index
        loss, _output = self._run(
            x,
            h=h,
            prev_input_embedding=prev_input_embedding,
            prev_state_embedding=prev_state_embedding,
            collect=False,
            future_targets=future_targets,
        )
        return loss

    def training_loss(
        self, x: torch.Tensor, step_index: int = 0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        loss = self.loss_bundle(x, step_index)
        return loss.total, loss.code_counts

    def final_outputs(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        output = self.sequence_outputs(x)
        return {
            "codes": output["codes"][:, -1],
            "distances": output["distances"][:, -1],
            "hidden_states": output["hidden_states"][:, -1, :],
            "recon": output["recon"][:, -1, :],
        }

    def sequence_outputs(
        self,
        x: torch.Tensor,
        *,
        h: torch.Tensor | None = None,
        prev_input_embedding: torch.Tensor | None = None,
        prev_state_embedding: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        _loss, output = self._run(
            x,
            h=h,
            prev_input_embedding=prev_input_embedding,
            prev_state_embedding=prev_state_embedding,
            collect=True,
            future_targets=None,
        )
        return output

    def _run(
        self,
        x: torch.Tensor,
        *,
        h: torch.Tensor | None = None,
        prev_input_embedding: torch.Tensor | None = None,
        prev_state_embedding: torch.Tensor | None = None,
        collect: bool,
        future_targets: FutureTargets | None = None,
    ) -> tuple[LossBundle, dict[str, torch.Tensor]]:
        if x.ndim == 2:
            x = x.unsqueeze(0)
        batch, seq_len, _feature_width = x.shape
        if seq_len <= 0:
            raise ValueError("input sequence must contain at least one step")
        h = x.new_zeros((batch, self.spec.hidden_dim)) if h is None else h
        recon_loss = x.new_tensor(0.0)
        z_e_norm_sum = x.new_tensor(0.0)
        z_q_norm_sum = x.new_tensor(0.0)
        vq_distance_sum = x.new_tensor(0.0)
        vq_distance_max = x.new_tensor(0.0)
        code_counts = torch.zeros(self.spec.num_codes, dtype=torch.long, device=x.device)
        codes = []
        distances = []
        all_distances = []
        all_z_e = []
        all_codes = []
        all_logits = []
        future_query_inputs = []
        hidden_states = []
        recons = []
        if prev_input_embedding is None or prev_state_embedding is None:
            prev_embed, prev_state = self._initial_prev_inputs(batch, x.device)
        else:
            prev_embed = prev_input_embedding
            prev_state = prev_state_embedding
        for step in range(seq_len):
            state = self._step(x[:, step, :], h, prev_embed, prev_state)
            recon_loss = recon_loss + torch.nn.functional.mse_loss(
                state["recon"], x[:, step, :]
            )
            code_counts = code_counts + torch.bincount(
                state["code"].detach(), minlength=self.spec.num_codes
            )
            z_e_norm_sum = z_e_norm_sum + state["z_e"].norm(dim=-1).sum()
            z_q_norm_sum = z_q_norm_sum + state["z_q"].norm(dim=-1).sum()
            vq_distance_sum = vq_distance_sum + state["distance"].sum()
            vq_distance_max = torch.maximum(vq_distance_max, state["distance"].max())
            all_distances.append(state["distance"])
            all_z_e.append(state["z_e"])
            all_codes.append(state["code"])
            if self.spec.temporal_consistency_weight > 0.0:
                all_logits.append(self._assignment_logits(state["z_e"]))
            if future_targets is not None:
                future_query_inputs.append(torch.cat([state["h"], state["z_q_st"]], dim=-1))
            h = state["h"]
            prev_embed = state["next_embed"]
            prev_state = state["next_state"]
            if collect:
                codes.append(state["code"])
                distances.append(state["distance"])
                hidden_states.append(h)
                recons.append(state["recon"])
        recon = recon_loss / seq_len
        diagnostic_rows = batch * seq_len
        vq_distances = torch.cat(all_distances)
        reconstruct_component = (
            recon * self.spec.reconstruct_weight
            if "reconstruct" in self.spec.objective_terms
            else recon.new_tensor(0.0)
        )
        diversity = self.codebook_diversity_loss()
        diversity_component = self.spec.diversity_weight * diversity
        future = None
        future_accuracy = None
        future_rows = 0
        future_component = recon.new_tensor(0.0)
        if (
            future_targets is not None
            and "future_infonce" in self.spec.objective_terms
            and self.spec.future_weight > 0.0
            and len(future_targets.positions) >= 2
        ):
            query_tensor = torch.stack(future_query_inputs, dim=1)
            row_indices = torch.tensor(
                [row for row, _step in future_targets.positions], dtype=torch.long, device=x.device
            )
            step_indices = torch.tensor(
                [step for _row, step in future_targets.positions], dtype=torch.long, device=x.device
            )
            future_result = self.future_loss(
                self.future_query(query_tensor[row_indices, step_indices]), future_targets
            )
            future = future_result.loss
            future_accuracy = future_result.accuracy
            future_rows = future_result.rows
            future_component = self.spec.future_weight * future
        temporal_consistency = None
        temporal_consistency_component = recon.new_tensor(0.0)
        if self.spec.temporal_consistency_weight > 0.0 and seq_len > 1:
            logits = torch.stack(all_logits, dim=1)[:, 1:, :].reshape(-1, self.spec.num_codes)
            previous_codes = torch.stack(all_codes, dim=1)[:, :-1].detach().reshape(-1)
            temporal_consistency = torch.nn.functional.cross_entropy(logits, previous_codes)
            temporal_consistency_component = (
                self.spec.temporal_consistency_weight * temporal_consistency
            )
        z_e_sequence = torch.stack(all_z_e, dim=1)
        loss = LossBundle(
            total=reconstruct_component
            + diversity_component
            + future_component
            + temporal_consistency_component,
            recon=recon,
            diversity=diversity,
            future=future,
            future_accuracy=future_accuracy,
            future_rows=future_rows,
            temporal_consistency=temporal_consistency,
            code_counts=code_counts,
            next_hidden=h,
            next_input_embedding=prev_embed,
            next_state_embedding=prev_state,
            z_e=z_e_sequence,
            codes=torch.stack(all_codes, dim=1),
            z_e_norm_mean=z_e_norm_sum / diagnostic_rows,
            z_q_norm_mean=z_q_norm_sum / diagnostic_rows,
            vq_distance_mean=vq_distance_sum / diagnostic_rows,
            vq_distance_p95=torch.quantile(vq_distances, 0.95),
            vq_distance_max=vq_distance_max,
        )
        output = {
            "final_hidden": h,
            "final_input_embedding": prev_embed,
            "final_state_embedding": prev_state,
        }
        if collect:
            output.update(
                codes=torch.stack(codes, dim=1),
                distances=torch.stack(distances, dim=1),
                hidden_states=torch.stack(hidden_states, dim=1),
                recon=torch.stack(recons, dim=1),
            )
        return loss, output


def train(
    dataset: WindowDataset,
    spec: VqRssmSpec,
    train_cfg: TrainingConfig,
    *,
    output_dir: Path | None = None,
    schedule: tuple[VqRssmScheduleStage, ...] = (),
) -> TrainingResult:
    run_start = time.perf_counter()
    if len(dataset.feature_columns) != spec.input_dim:
        raise ValueError("dataset feature width must match VqRssmSpec.input_dim")
    train_indices = _split_indices(dataset, "train")
    valid_indices = _split_indices(dataset, "valid")
    test_indices = _split_indices(dataset, "test")
    if not train_indices:
        raise ValueError("WindowDataset contains no train split rows")
    runtime = _runtime_info(train_cfg)
    device = runtime.device
    logger.info(
        "train_runtime torch=%s device=%s cuda_device=%s threads=%s cuda_memory=%s",
        torch.__version__,
        device,
        runtime.device_name,
        torch.get_num_threads(),
        _cuda_memory_summary(device),
    )
    model, optimizer = _build_model_and_optimizer(spec, train_cfg, device)
    tensor_start = time.perf_counter()
    train_features = torch.as_tensor(
        [dataset.features[index] for index in train_indices], dtype=torch.float32, device=device
    )
    valid_features = (
        torch.as_tensor(
            [dataset.features[index] for index in valid_indices],
            dtype=torch.float32,
            device=device,
        )
        if valid_indices
        else None
    )
    logger.info(
        "train_tensors train_shape=%s valid_shape=%s elapsed_s=%.2f cuda_memory=%s",
        tuple(train_features.shape),
        tuple(valid_features.shape) if valid_features is not None else None,
        time.perf_counter() - tensor_start,
        _cuda_memory_summary(device),
    )
    metrics = []
    global_step = 0
    best_state_dict: dict[str, torch.Tensor] | None = None
    best_summary: EpochMetrics | None = None
    epochs_without_improvement = 0
    best_seen = False
    for epoch in range(1, train_cfg.epochs + 1):
        epoch_start = time.perf_counter()
        epoch_spec = _epoch_spec(spec, schedule, epoch)
        model.spec = epoch_spec
        _apply_epoch_lr(optimizer, _epoch_lr(train_cfg, schedule, epoch))
        _apply_epoch_freeze(model, schedule, epoch)
        logger.info("train_epoch_start epoch=%s/%s", epoch, train_cfg.epochs)
        model.train()
        acc = EpochAccumulator.empty(epoch_spec, device)
        batch_count = math.ceil(train_features.shape[0] / train_cfg.batch)
        for batch_index, start in enumerate(
            range(0, train_features.shape[0], train_cfg.batch), start=1
        ):
            batch = train_features[start : start + train_cfg.batch]
            optimizer.zero_grad()
            bundle = model.loss_bundle(batch, global_step)
            loss = bundle.total
            _raise_if_nonfinite_loss(loss, f"window train epoch={epoch} batch={batch_index}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
            optimizer.step()
            model.update_codebook(bundle.z_e, bundle.codes, epoch=epoch)
            global_step += 1
            batch_rows = batch.shape[0]
            with torch.no_grad():
                acc.add(bundle, batch_rows)
            if train_cfg.log_every > 0 and batch_index % train_cfg.log_every == 0:
                logger.info(
                    "train_batch epoch=%s batch=%s/%s loss=%.6f elapsed_s=%.2f",
                    epoch,
                    batch_index,
                    batch_count,
                    float(loss.detach().cpu()),
                    time.perf_counter() - epoch_start,
                )
        active_codes = int((acc.code_counts > 0).sum().item())
        train_parts = LossParts.from_accumulator(acc)
        codebook_norms = _codebook_norm_metrics(model)
        codebook_similarity = _codebook_similarity_metrics(model)
        valid_parts = None
        if valid_features is not None:
            valid_start = time.perf_counter()
            logger.info(
                "validation_start epoch=%s rows=%s batch=%s",
                epoch,
                valid_features.shape[0],
                train_cfg.valid_batch_rows,
            )
            valid_parts = _validation_loss(model, valid_features, train_cfg.valid_batch_rows)
            logger.info(
                "validation_done epoch=%s loss=%.6f elapsed_s=%.2f",
                epoch,
                valid_parts.total,
                time.perf_counter() - valid_start,
            )
        metric = _epoch_metrics(
            epoch=epoch,
            train_parts=train_parts,
            valid_parts=valid_parts,
            train_rows=len(train_indices),
            valid_rows=len(valid_indices),
            test_rows=len(test_indices),
            active_codes=active_codes,
            spec=epoch_spec,
            codebook_norms=codebook_norms,
            codebook_similarity=codebook_similarity,
        )
        metrics.append(metric)
        _write_training_metrics(output_dir, metrics)
        _log_epoch_summary("train", metric, time.perf_counter() - epoch_start)
        best_state_dict, best_summary, epochs_without_improvement, best_seen = (
            _update_best_checkpoint_state(
                model=model,
                metric=metric,
                train_cfg=train_cfg,
                best_state_dict=best_state_dict,
                best_summary=best_summary,
                epochs_without_improvement=epochs_without_improvement,
                best_seen=best_seen,
            )
        )
        if _should_stop_early(train_cfg, best_seen, epochs_without_improvement):
            logger.info(
                "early_stop epoch=%s best_metric=%s patience=%s",
                epoch,
                train_cfg.best_metric,
                train_cfg.early_stop_patience,
            )
            break
    checkpoint_path = output_dir / "behavior-state-model.pt" if output_dir is not None else None
    selected_state_dict = best_state_dict if best_state_dict is not None else _cpu_state_dict(model)
    checkpoint = VqRssmCheckpoint(
        spec=spec,
        path=checkpoint_path,
        state_dict=selected_state_dict,
    )
    if checkpoint_path is not None:
        save_checkpoint(checkpoint, checkpoint_path)
        logger.info("checkpoint_saved path=%s", checkpoint_path)
    logger.info(
        "train_done epochs=%s elapsed_s=%.2f",
        len(metrics),
        time.perf_counter() - run_start,
    )
    return TrainingResult(
        checkpoint=checkpoint,
        checkpoint_path=checkpoint_path,
        metrics=tuple(metrics),
        summary=metrics[-1] if metrics else None,
        best_summary=best_summary,
    )


def train_sequences(
    dataset: SequenceDataset,
    spec: VqRssmSpec,
    train_cfg: TrainingConfig,
    seq_cfg: SequenceRuntimeConfig,
    *,
    output_dir: Path | None = None,
    schedule: tuple[VqRssmScheduleStage, ...] = (),
) -> TrainingResult:
    run_start = time.perf_counter()
    if len(dataset.feature_columns) != spec.input_dim:
        raise ValueError("dataset feature width must match VqRssmSpec.input_dim")
    train_sequences_ = tuple(seq for seq in dataset.sequences if seq.split == "train")
    valid_sequences = tuple(seq for seq in dataset.sequences if seq.split == "valid")
    test_sequences = tuple(seq for seq in dataset.sequences if seq.split == "test")
    if not train_sequences_:
        raise ValueError("SequenceDataset contains no train split rows")
    runtime = _runtime_info(train_cfg)
    device = runtime.device
    logger.info(
        "train_sequences_runtime torch=%s device=%s cuda_device=%s threads=%s "
        "chunk=%s warmup=%s stride=%s carry=%s cuda_memory=%s",
        torch.__version__,
        device,
        runtime.device_name,
        torch.get_num_threads(),
        seq_cfg.chunk,
        seq_cfg.warmup,
        seq_cfg.stride,
        seq_cfg.carry,
        _cuda_memory_summary(device),
    )
    model, optimizer = _build_model_and_optimizer(spec, train_cfg, device)
    train_tensors = tuple(
        torch.as_tensor(sequence.features, dtype=torch.float32, device=device)
        for sequence in train_sequences_
    )
    valid_tensors = tuple(
        torch.as_tensor(sequence.features, dtype=torch.float32, device=device)
        for sequence in valid_sequences
    )
    logger.info(
        "train_sequences_tensors train_sequences=%s valid_sequences=%s train_rows=%s "
        "valid_rows=%s cuda_memory=%s",
        len(train_tensors),
        len(valid_tensors),
        sum(tensor.shape[0] for tensor in train_tensors),
        sum(tensor.shape[0] for tensor in valid_tensors),
        _cuda_memory_summary(device),
    )
    metrics = []
    global_step = 0
    train_rows = sum(len(sequence.features) for sequence in train_sequences_)
    valid_rows = sum(len(sequence.features) for sequence in valid_sequences)
    test_rows = sum(len(sequence.features) for sequence in test_sequences)
    best_state_dict: dict[str, torch.Tensor] | None = None
    best_summary: EpochMetrics | None = None
    epochs_without_improvement = 0
    best_seen = False
    for epoch in range(1, train_cfg.epochs + 1):
        epoch_start = time.perf_counter()
        epoch_spec = _epoch_spec(spec, schedule, epoch)
        model.spec = epoch_spec
        _apply_epoch_lr(optimizer, _epoch_lr(train_cfg, schedule, epoch))
        _apply_epoch_freeze(model, schedule, epoch)
        logger.info("train_sequences_epoch_start epoch=%s/%s", epoch, train_cfg.epochs)
        model.train()
        acc = EpochAccumulator.empty(epoch_spec, device)
        batch_index = 0
        hidden_by_sequence = [None] * len(train_tensors)
        prev_input_by_sequence = [None] * len(train_tensors)
        prev_state_by_sequence = [None] * len(train_tensors)
        max_rows = max(tensor.shape[0] for tensor in train_tensors)
        for start in range(0, max_rows, seq_cfg.chunk):
            for sequence_batch in _parallel_sequence_batches(
                train_tensors,
                hidden_by_sequence,
                start,
                seq_cfg,
                train_cfg.batch,
                epoch_spec.hidden_dim,
                prev_input_by_sequence,
                prev_state_by_sequence,
                latent_dim=epoch_spec.latent_dim,
            ):
                batch = sequence_batch.batch
                optimizer.zero_grad()
                future_targets = None
                if epoch > epoch_spec.future_warmup_epochs:
                    future_targets = _future_segments(
                        train_tensors, sequence_batch, epoch_spec, global_step
                    )
                bundle = model.loss_bundle(
                    batch,
                    global_step,
                    h=sequence_batch.hidden,
                    prev_input_embedding=sequence_batch.prev_input_embedding,
                    prev_state_embedding=sequence_batch.prev_state_embedding,
                    future_targets=future_targets,
                )
                loss = bundle.total
                next_h = bundle.next_hidden
                _raise_if_nonfinite_loss(
                    loss,
                    "sequence train "
                    f"epoch={epoch} batch={batch_index + 1} "
                    f"streams={batch.shape[0]} steps={batch.shape[1]}",
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
                optimizer.step()
                model.update_codebook(bundle.z_e, bundle.codes, epoch=epoch)
                global_step += 1
                batch_index += 1
                batch_rows = batch.shape[0] * batch.shape[1]
                with torch.no_grad():
                    acc.add(bundle, batch_rows)
                if seq_cfg.carry:
                    detached_h = next_h.detach()
                    detached_input = bundle.next_input_embedding.detach()
                    detached_state = bundle.next_state_embedding.detach()
                    for row, sequence_index in enumerate(sequence_batch.sequence_indices):
                        hidden_by_sequence[sequence_index] = detached_h[row : row + 1]
                        prev_input_by_sequence[sequence_index] = detached_input[row : row + 1]
                        prev_state_by_sequence[sequence_index] = detached_state[row : row + 1]
                if train_cfg.log_every > 0 and batch_index % train_cfg.log_every == 0:
                    logger.info(
                        "train_sequences_batch epoch=%s batch=%s streams=%s steps=%s "
                        "loss=%.6f elapsed_s=%.2f cuda_memory=%s",
                        epoch,
                        batch_index,
                        batch.shape[0],
                        batch.shape[1],
                        float(loss.detach().cpu()),
                        time.perf_counter() - epoch_start,
                        _cuda_memory_summary(device),
                    )
        active_codes = int((acc.code_counts > 0).sum().item())
        train_parts = LossParts.from_accumulator(acc)
        codebook_norms = _codebook_norm_metrics(model)
        codebook_similarity = _codebook_similarity_metrics(model)
        valid_parts = None
        if valid_sequences:
            valid_start = time.perf_counter()
            valid_parts = _sequence_validation_loss(
                model,
                valid_tensors,
                seq_cfg,
                train_cfg.batch,
                future_enabled=epoch > epoch_spec.future_warmup_epochs,
            )
            logger.info(
                "train_sequences_validation_done epoch=%s loss=%.6f elapsed_s=%.2f",
                epoch,
                valid_parts.total,
                time.perf_counter() - valid_start,
            )
        metric = _epoch_metrics(
            epoch=epoch,
            train_parts=train_parts,
            valid_parts=valid_parts,
            train_rows=train_rows,
            valid_rows=valid_rows,
            test_rows=test_rows,
            active_codes=active_codes,
            spec=epoch_spec,
            codebook_norms=codebook_norms,
            codebook_similarity=codebook_similarity,
        )
        metrics.append(metric)
        _write_training_metrics(output_dir, metrics)
        _log_epoch_summary("train_sequences", metric, time.perf_counter() - epoch_start)
        best_state_dict, best_summary, epochs_without_improvement, best_seen = (
            _update_best_checkpoint_state(
                model=model,
                metric=metric,
                train_cfg=train_cfg,
                best_state_dict=best_state_dict,
                best_summary=best_summary,
                epochs_without_improvement=epochs_without_improvement,
                best_seen=best_seen,
            )
        )
        if _should_stop_early(train_cfg, best_seen, epochs_without_improvement):
            logger.info(
                "train_sequences_early_stop epoch=%s best_metric=%s patience=%s",
                epoch,
                train_cfg.best_metric,
                train_cfg.early_stop_patience,
            )
            break
    checkpoint_path = output_dir / "behavior-state-model.pt" if output_dir is not None else None
    selected_state_dict = best_state_dict if best_state_dict is not None else _cpu_state_dict(model)
    checkpoint = VqRssmCheckpoint(
        spec=spec,
        path=checkpoint_path,
        state_dict=selected_state_dict,
    )
    if checkpoint_path is not None:
        save_checkpoint(checkpoint, checkpoint_path)
        logger.info("checkpoint_saved path=%s", checkpoint_path)
    logger.info(
        "train_sequences_done epochs=%s elapsed_s=%.2f",
        len(metrics),
        time.perf_counter() - run_start,
    )
    return TrainingResult(
        checkpoint=checkpoint,
        checkpoint_path=checkpoint_path,
        metrics=tuple(metrics),
        summary=metrics[-1] if metrics else None,
        best_summary=best_summary,
    )


def save_checkpoint(checkpoint: VqRssmCheckpoint, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"spec": asdict(checkpoint.spec), "state_dict": checkpoint.state_dict}, path)
    return path


def _write_training_metrics(output_dir: Path | None, metrics: list[EpochMetrics]) -> None:
    if output_dir is None:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "behavior-state-training-metrics.csv"
    fieldnames = [field.name for field in fields(EpochMetrics)]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(asdict(metric) for metric in metrics)


def _epoch_spec(
    spec: VqRssmSpec,
    schedule: tuple[VqRssmScheduleStage, ...],
    epoch: int,
) -> VqRssmSpec:
    updates: dict[str, object] = {}
    for stage in schedule:
        if not stage.includes(epoch):
            continue
        if stage.reconstruct_weight is not None:
            updates["reconstruct_weight"] = stage.reconstruct_weight
        if stage.future_weight is not None:
            updates["future_weight"] = stage.future_weight
        if stage.diversity_weight is not None:
            updates["diversity_weight"] = stage.diversity_weight
        if stage.reset_fraction is not None:
            updates["reset_fraction"] = stage.reset_fraction
        if stage.reset_dead_codes is not None:
            updates["reset_dead_codes"] = stage.reset_dead_codes
    return replace(spec, **updates) if updates else spec


def _epoch_lr(
    train_cfg: TrainingConfig,
    schedule: tuple[VqRssmScheduleStage, ...],
    epoch: int,
) -> float:
    lr = train_cfg.lr
    for stage in schedule:
        if stage.includes(epoch) and stage.lr is not None:
            lr = stage.lr
    return lr


def _apply_epoch_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def _apply_epoch_freeze(
    model: VqRssmModel,
    schedule: tuple[VqRssmScheduleStage, ...],
    epoch: int,
) -> None:
    frozen_blocks = tuple(
        block
        for stage in schedule
        if stage.includes(epoch)
        for block in stage.freeze_encoder_blocks
    )
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    model.codebook.weight.requires_grad_(False)
    if not frozen_blocks:
        return
    modules = {
        "encoder_input": (model.embedding,),
        "latent_projection": (model.latent_projection,),
        "rssm": (model.transition,),
        "decoder": (model.decoder,),
        "future_head": (model.future_encoder, model.future_query),
    }
    unknown = sorted(set(frozen_blocks) - set(modules))
    if unknown:
        raise ValueError("unknown freeze encoder blocks: " + ", ".join(unknown))
    for block in frozen_blocks:
        for module in modules[block]:
            for parameter in module.parameters():
                parameter.requires_grad_(False)


def _format_log_fields(fields_: dict[str, object]) -> str:
    parts = []
    for key, value in fields_.items():
        if isinstance(value, float):
            rendered = f"{value:.6f}"
        elif value is None:
            rendered = "none"
        else:
            rendered = str(value)
        parts.append(f"{key}={rendered}")
    return " ".join(parts)


def _runtime_info(train_cfg: TrainingConfig) -> RuntimeInfo:
    if train_cfg.threads > 0:
        torch.set_num_threads(train_cfg.threads)
    device = train_cfg.torch_device()
    if train_cfg.seed is not None:
        torch.manual_seed(train_cfg.seed)
    if device.type == "cuda":
        device_name = torch.cuda.get_device_name(device)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    else:
        device_name = ""
    return RuntimeInfo(device=device, device_name=device_name)


def _build_model_and_optimizer(
    spec: VqRssmSpec,
    train_cfg: TrainingConfig,
    device: torch.device,
) -> tuple[VqRssmModel, torch.optim.Optimizer]:
    model = VqRssmModel(spec).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg.lr)
    return model, optimizer


def _cpu_state_dict(model: VqRssmModel) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def _codebook_norm_metrics(model: VqRssmModel) -> tuple[float, float]:
    norms = model.codebook.weight.detach().norm(dim=-1)
    return float(norms.mean().cpu()), float(norms.max().cpu())


def _codebook_similarity_metrics(model: VqRssmModel) -> tuple[float, float]:
    codebook = torch.nn.functional.normalize(model.codebook.weight.detach(), dim=-1)
    sim = codebook @ codebook.T
    off_diagonal = ~torch.eye(sim.shape[0], dtype=torch.bool, device=sim.device)
    values = sim[off_diagonal]
    return float(values.mean().cpu()), float(values.max().cpu())


def _epoch_metrics(
    *,
    epoch: int,
    train_parts: LossParts,
    valid_parts: LossParts | None,
    train_rows: int,
    valid_rows: int,
    test_rows: int,
    active_codes: int,
    spec: VqRssmSpec,
    codebook_norms: tuple[float, float],
    codebook_similarity: tuple[float, float],
) -> EpochMetrics:
    return EpochMetrics(
        epoch=epoch,
        train_loss=train_parts.total,
        valid_loss=valid_parts.total if valid_parts else None,
        train_rows=train_rows,
        valid_rows=valid_rows,
        test_rows=test_rows,
        active_codes=active_codes,
        codebook_utilization_pct=active_codes / spec.num_codes * 100.0,
        train_recon_loss=train_parts.recon,
        train_diversity_loss=train_parts.diversity,
        train_future_loss=train_parts.future,
        train_future_accuracy=train_parts.future_accuracy,
        train_future_rows=train_parts.future_rows,
        train_temporal_consistency_loss=train_parts.temporal_consistency,
        valid_recon_loss=valid_parts.recon if valid_parts else None,
        valid_diversity_loss=valid_parts.diversity if valid_parts else None,
        valid_future_loss=valid_parts.future if valid_parts else None,
        valid_future_accuracy=valid_parts.future_accuracy if valid_parts else None,
        valid_future_rows=valid_parts.future_rows if valid_parts else 0,
        valid_temporal_consistency_loss=valid_parts.temporal_consistency if valid_parts else None,
        train_z_e_norm_mean=train_parts.z_e_norm_mean,
        train_z_q_norm_mean=train_parts.z_q_norm_mean,
        train_vq_distance_mean=train_parts.vq_distance_mean,
        train_vq_distance_p95=train_parts.vq_distance_p95,
        train_vq_distance_max=train_parts.vq_distance_max,
        valid_z_e_norm_mean=valid_parts.z_e_norm_mean if valid_parts else None,
        valid_z_q_norm_mean=valid_parts.z_q_norm_mean if valid_parts else None,
        valid_vq_distance_mean=valid_parts.vq_distance_mean if valid_parts else None,
        valid_vq_distance_p95=valid_parts.vq_distance_p95 if valid_parts else None,
        valid_vq_distance_max=valid_parts.vq_distance_max if valid_parts else None,
        codebook_norm_mean=codebook_norms[0],
        codebook_norm_max=codebook_norms[1],
        codebook_similarity_mean=codebook_similarity[0],
        codebook_similarity_max=codebook_similarity[1],
    )


def _log_epoch_summary(prefix: str, metric: EpochMetrics, elapsed_s: float) -> None:
    logger.info("%s_epoch_done %s", prefix, metric.log_line(elapsed_s))


def _best_metric_value(metric: EpochMetrics | None, name: str) -> float | None:
    if metric is None:
        return None
    value = getattr(metric, name, None)
    if value is None:
        return None
    value_float = float(value)
    if not math.isfinite(value_float):
        return None
    return value_float


def _is_better_metric(candidate: float, best: float, mode: str) -> bool:
    if mode == "min":
        return candidate < best
    return candidate > best


def _update_best_checkpoint_state(
    *,
    model: VqRssmModel,
    metric: EpochMetrics,
    train_cfg: TrainingConfig,
    best_state_dict: dict[str, torch.Tensor] | None,
    best_summary: EpochMetrics | None,
    epochs_without_improvement: int,
    best_seen: bool,
) -> tuple[dict[str, torch.Tensor] | None, EpochMetrics | None, int, bool]:
    if not train_cfg.best_metric:
        return best_state_dict, best_summary, epochs_without_improvement, best_seen
    value = _best_metric_value(metric, train_cfg.best_metric)
    if value is None:
        return best_state_dict, best_summary, epochs_without_improvement, best_seen
    best_value = (
        _best_metric_value(best_summary, train_cfg.best_metric)
        if best_summary is not None
        else None
    )
    if best_value is None or _is_better_metric(value, best_value, train_cfg.best_mode):
        return _cpu_state_dict(model), metric, 0, True
    return best_state_dict, best_summary, epochs_without_improvement + 1, True


def _should_stop_early(
    train_cfg: TrainingConfig,
    best_seen: bool,
    epochs_without_improvement: int,
) -> bool:
    return (
        bool(train_cfg.best_metric)
        and train_cfg.early_stop_patience > 0
        and best_seen
        and epochs_without_improvement >= train_cfg.early_stop_patience
    )


def load_checkpoint(path: Path) -> VqRssmCheckpoint:
    data = torch.load(path, map_location="cpu")
    state_dict = data["state_dict"]
    if any(key.startswith(("posterior.", "prior.")) for key in state_dict):
        raise ValueError(
            "hybrid VQ-RSSM checkpoints with posterior/prior weights are incompatible "
            "with the pure-discrete architecture"
        )
    return VqRssmCheckpoint(
        spec=VqRssmSpec(**data["spec"]),
        path=path,
        state_dict=state_dict,
    )


def load_model(
    checkpoint: VqRssmCheckpoint | Path | str,
    *,
    device: torch.device | str | None = None,
) -> VqRssmModel:
    """Instantiate a frozen eval-mode model from a VQ-RSSM checkpoint."""

    loaded = load_checkpoint(Path(checkpoint)) if isinstance(checkpoint, str | Path) else checkpoint
    target_device = torch.device(device) if device is not None else torch.device("cpu")
    model = VqRssmModel(loaded.spec).to(target_device)
    model.load_state_dict(loaded.state_dict)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def predict_codes(
    dataset: WindowDataset,
    checkpoint: VqRssmCheckpoint,
    *,
    train_cfg: TrainingConfig | None = None,
) -> CodeSequence:
    diagnostics = _predict_window_outputs(
        dataset, checkpoint, train_cfg=train_cfg, include_diagnostics=False
    )
    return CodeSequence(
        codes=diagnostics.codes,
        distances=diagnostics.distances,
        row_index=tuple(range(len(diagnostics.codes))),
        splits=dataset.splits,
    )


def predict_diagnostics(
    dataset: WindowDataset,
    checkpoint: VqRssmCheckpoint,
    *,
    train_cfg: TrainingConfig | None = None,
) -> InferenceDiagnostics:
    return _predict_window_outputs(
        dataset, checkpoint, train_cfg=train_cfg, include_diagnostics=True
    )


def _predict_window_outputs(
    dataset: WindowDataset,
    checkpoint: VqRssmCheckpoint,
    *,
    train_cfg: TrainingConfig | None,
    include_diagnostics: bool,
) -> InferenceDiagnostics:
    if len(dataset.feature_columns) != checkpoint.spec.input_dim:
        raise ValueError("dataset feature width must match checkpoint spec input_dim")
    cfg = train_cfg or TrainingConfig()
    if cfg.threads > 0:
        torch.set_num_threads(cfg.threads)
    device = cfg.torch_device()
    if not dataset.features:
        return InferenceDiagnostics(
            codes=(), distances=(), hidden_states=(), reconstructions=()
        )
    start_time = time.perf_counter()
    logger.info(
        "predict_windows_start rows=%s batch=%s device=%s diagnostics=%s",
        len(dataset.features),
        cfg.pred_batch_rows,
        device,
        include_diagnostics,
    )
    model = VqRssmModel(checkpoint.spec).to(device)
    model.load_state_dict(checkpoint.state_dict)
    model.eval()
    codes = []
    distances = []
    hidden_states = []
    reconstructions = []
    indices = tuple(range(len(dataset.features)))
    batch_count = math.ceil(len(indices) / cfg.pred_batch_rows)
    with torch.no_grad():
        for batch_index, start in enumerate(
            range(0, len(indices), cfg.pred_batch_rows), start=1
        ):
            batch_indices = indices[start : start + cfg.pred_batch_rows]
            features = torch.as_tensor(
                [dataset.features[index] for index in batch_indices],
                dtype=torch.float32,
                device=device,
            )
            output = model.final_outputs(features)
            codes.extend(output["codes"].detach().cpu().tolist())
            distances.extend(output["distances"].detach().cpu().tolist())
            if include_diagnostics:
                hidden_states.extend(output["hidden_states"].detach().cpu().tolist())
                reconstructions.extend(output["recon"].detach().cpu().tolist())
            if cfg.log_every > 0 and batch_index % cfg.log_every == 0:
                logger.info("predict_windows_batch batch=%s/%s", batch_index, batch_count)
    logger.info(
        "predict_windows_done rows=%s elapsed_s=%.2f",
        len(codes),
        time.perf_counter() - start_time,
    )
    return InferenceDiagnostics(
        codes=tuple(int(code) for code in codes),
        distances=tuple(float(distance) for distance in distances),
        hidden_states=tuple(tuple(float(value) for value in row) for row in hidden_states),
        reconstructions=tuple(
            tuple(float(value) for value in row) for row in reconstructions
        ),
    )


def predict_sequence_codes(
    dataset: SequenceDataset,
    checkpoint: VqRssmCheckpoint,
    seq_cfg: SequenceRuntimeConfig,
    *,
    train_cfg: TrainingConfig | None = None,
) -> CodeSequence:
    diagnostics = _predict_sequence_outputs(
        dataset,
        checkpoint,
        seq_cfg,
        train_cfg=train_cfg,
        include_diagnostics=False,
    )
    return CodeSequence(
        codes=diagnostics.codes,
        distances=diagnostics.distances,
        row_index=tuple(range(len(diagnostics.codes))),
        splits=_emitted_sequence_splits(dataset, seq_cfg),
    )


def predict_sequence_diagnostics(
    dataset: SequenceDataset,
    checkpoint: VqRssmCheckpoint,
    seq_cfg: SequenceRuntimeConfig,
    *,
    train_cfg: TrainingConfig | None = None,
) -> InferenceDiagnostics:
    return _predict_sequence_outputs(
        dataset,
        checkpoint,
        seq_cfg,
        train_cfg=train_cfg,
        include_diagnostics=True,
    )


def decode_codebook(
    checkpoint: VqRssmCheckpoint,
    *,
    hidden_state: tuple[float, ...] | None = None,
) -> tuple[tuple[float, ...], ...]:
    model = VqRssmModel(checkpoint.spec)
    model.load_state_dict(checkpoint.state_dict)
    model.eval()
    if hidden_state is None:
        hidden = torch.zeros((checkpoint.spec.num_codes, checkpoint.spec.hidden_dim))
    else:
        if len(hidden_state) != checkpoint.spec.hidden_dim:
            raise ValueError("hidden_state width must match checkpoint hidden_dim")
        hidden = torch.tensor([hidden_state] * checkpoint.spec.num_codes, dtype=torch.float32)
    with torch.no_grad():
        decoded = model.decoder(torch.cat([hidden, model.codebook.weight], dim=-1))
    return tuple(tuple(float(value) for value in row) for row in decoded.detach().cpu().tolist())


def _raise_if_nonfinite_loss(loss: torch.Tensor, context: str) -> None:
    if torch.isfinite(loss).all():
        return
    value = float(loss.detach().cpu())
    raise FloatingPointError(f"non-finite VQ-RSSM loss: {context} loss={value}")


def _validation_loss(
    model: VqRssmModel,
    valid_features: torch.Tensor,
    batch_size: int,
) -> LossParts:
    model.eval()
    acc = EpochAccumulator.empty(model.spec, valid_features.device)
    with torch.no_grad():
        for start in range(0, valid_features.shape[0], batch_size):
            batch = valid_features[start : start + batch_size]
            bundle = model.loss_bundle(batch)
            _raise_if_nonfinite_loss(bundle.total, "window validation")
            acc.add(bundle, batch.shape[0])
    return LossParts.from_accumulator(acc)


def _sequence_validation_loss(
    model: VqRssmModel,
    tensors,
    seq_cfg: SequenceRuntimeConfig,
    batch_size: int,
    *,
    future_enabled: bool = True,
) -> LossParts:
    model.eval()
    device = tensors[0].device
    acc = EpochAccumulator.empty(model.spec, device)
    with torch.no_grad():
        hidden_by_sequence = [None] * len(tensors)
        prev_input_by_sequence = [None] * len(tensors)
        prev_state_by_sequence = [None] * len(tensors)
        max_rows = max(tensor.shape[0] for tensor in tensors)
        for start in range(0, max_rows, seq_cfg.chunk):
            for sequence_batch in _parallel_sequence_batches(
                tensors,
                hidden_by_sequence,
                start,
                seq_cfg,
                batch_size,
                model.spec.hidden_dim,
                prev_input_by_sequence,
                prev_state_by_sequence,
                latent_dim=model.spec.latent_dim,
            ):
                batch = sequence_batch.batch
                future_targets = (
                    _future_segments(
                        tensors, sequence_batch, model.spec, 0
                    )
                    if future_enabled
                    else None
                )
                bundle = model.loss_bundle(
                    batch,
                    0,
                    h=sequence_batch.hidden,
                    prev_input_embedding=sequence_batch.prev_input_embedding,
                    prev_state_embedding=sequence_batch.prev_state_embedding,
                    future_targets=future_targets,
                )
                _raise_if_nonfinite_loss(
                    bundle.total,
                    "sequence validation "
                    f"streams={batch.shape[0]} steps={batch.shape[1]}",
                )
                batch_rows = batch.shape[0] * batch.shape[1]
                acc.add(bundle, batch_rows)
                if seq_cfg.carry:
                    detached_h = bundle.next_hidden.detach()
                    detached_input = bundle.next_input_embedding.detach()
                    detached_state = bundle.next_state_embedding.detach()
                    for row, sequence_index in enumerate(sequence_batch.sequence_indices):
                        hidden_by_sequence[sequence_index] = detached_h[row : row + 1]
                        prev_input_by_sequence[sequence_index] = detached_input[row : row + 1]
                        prev_state_by_sequence[sequence_index] = detached_state[row : row + 1]
    return LossParts.from_accumulator(acc)


def _predict_sequence_outputs(
    dataset: SequenceDataset,
    checkpoint: VqRssmCheckpoint,
    seq_cfg: SequenceRuntimeConfig,
    *,
    train_cfg: TrainingConfig | None,
    include_diagnostics: bool,
) -> InferenceDiagnostics:
    if len(dataset.feature_columns) != checkpoint.spec.input_dim:
        raise ValueError("dataset feature width must match checkpoint spec input_dim")
    cfg = train_cfg or TrainingConfig()
    if cfg.threads > 0:
        torch.set_num_threads(cfg.threads)
    device = cfg.torch_device()
    start_time = time.perf_counter()
    logger.info(
        "predict_sequences_start sequences=%s chunk=%s warmup=%s stride=%s device=%s",
        len(dataset.sequences),
        seq_cfg.chunk,
        seq_cfg.warmup,
        seq_cfg.stride,
        device,
    )
    model = VqRssmModel(checkpoint.spec).to(device)
    model.load_state_dict(checkpoint.state_dict)
    model.eval()
    codes = []
    distances = []
    hidden_states = []
    reconstructions = []
    with torch.no_grad():
        for sequence_index, sequence in enumerate(dataset.sequences, start=1):
            h = None
            prev_input = None
            prev_state = None
            row_offset = 0
            tensor = torch.as_tensor(sequence.features, dtype=torch.float32, device=device)
            for start in range(0, tensor.shape[0], seq_cfg.chunk):
                chunk = tensor[start : start + seq_cfg.chunk]
                output = model.sequence_outputs(
                    chunk,
                    h=h if seq_cfg.carry else None,
                    prev_input_embedding=prev_input if seq_cfg.carry else None,
                    prev_state_embedding=prev_state if seq_cfg.carry else None,
                )
                chunk_codes = output["codes"].squeeze(0).detach().cpu().tolist()
                chunk_distances = output["distances"].squeeze(0).detach().cpu().tolist()
                chunk_hidden = output["hidden_states"].squeeze(0).detach().cpu().tolist()
                chunk_recon = output["recon"].squeeze(0).detach().cpu().tolist()
                for local_index, code in enumerate(chunk_codes):
                    absolute_index = row_offset + local_index
                    if absolute_index < seq_cfg.warmup - 1:
                        continue
                    if (absolute_index - (seq_cfg.warmup - 1)) % seq_cfg.stride != 0:
                        continue
                    codes.append(int(code))
                    distances.append(float(chunk_distances[local_index]))
                    if include_diagnostics:
                        hidden_states.append(
                            tuple(float(value) for value in chunk_hidden[local_index])
                        )
                        reconstructions.append(
                            tuple(float(value) for value in chunk_recon[local_index])
                        )
                row_offset += chunk.shape[0]
                if seq_cfg.carry:
                    h = output["final_hidden"].detach()
                    prev_input = output["final_input_embedding"].detach()
                    prev_state = output["final_state_embedding"].detach()
                else:
                    h = None
                    prev_input = None
                    prev_state = None
            if cfg.log_every > 0 and sequence_index % cfg.log_every == 0:
                logger.info(
                    "predict_sequences_batch sequence=%s/%s",
                    sequence_index,
                    len(dataset.sequences),
                )
    logger.info(
        "predict_sequences_done rows=%s elapsed_s=%.2f",
        len(codes),
        time.perf_counter() - start_time,
    )
    return InferenceDiagnostics(
        codes=tuple(codes),
        distances=tuple(distances),
        hidden_states=tuple(hidden_states),
        reconstructions=tuple(reconstructions),
    )


def _emitted_sequence_splits(
    dataset: SequenceDataset,
    seq_cfg: SequenceRuntimeConfig,
) -> tuple[SplitName, ...]:
    splits = []
    for sequence in dataset.sequences:
        for offset in range(seq_cfg.warmup - 1, len(sequence.features), seq_cfg.stride):
            if offset >= 0:
                splits.append(sequence.split)
    return tuple(splits)


def _parallel_sequence_batches(
    tensors,
    hidden_by_sequence,
    start: int,
    seq_cfg: SequenceRuntimeConfig,
    batch_size: int,
    hidden_dim: int,
    prev_input_by_sequence=None,
    prev_state_by_sequence=None,
    *,
    input_embed_dim: int = 32,
    latent_dim: int | None = None,
):
    by_length = {}
    for sequence_index, tensor in enumerate(tensors):
        if start >= tensor.shape[0]:
            continue
        chunk = tensor[start : start + seq_cfg.chunk]
        by_length.setdefault(chunk.shape[0], []).append((sequence_index, chunk))
    for _chunk_length, items in by_length.items():
        for group_start in range(0, len(items), batch_size):
            group = items[group_start : group_start + batch_size]
            indices = tuple(index for index, _chunk in group)
            batch = torch.stack([chunk for _index, chunk in group], dim=0)
            h = None
            prev_input = None
            prev_state = None
            if seq_cfg.carry:
                hidden_rows = [hidden_by_sequence[index] for index in indices]
                if any(hidden is not None for hidden in hidden_rows):
                    h = torch.cat(
                        [
                            hidden
                            if hidden is not None
                            else batch.new_zeros((1, hidden_dim))
                            for hidden in hidden_rows
                        ],
                        dim=0,
                    )
                if prev_input_by_sequence is not None:
                    prev_input_rows = [prev_input_by_sequence[index] for index in indices]
                    if any(value is not None for value in prev_input_rows):
                        prev_input = torch.cat(
                            [
                                value
                                if value is not None
                                else batch.new_zeros((1, input_embed_dim))
                                for value in prev_input_rows
                            ],
                            dim=0,
                        )
                if prev_state_by_sequence is not None:
                    if latent_dim is None:
                        raise ValueError("latent_dim is required when carrying previous state")
                    prev_state_rows = [prev_state_by_sequence[index] for index in indices]
                    if any(value is not None for value in prev_state_rows):
                        prev_state = torch.cat(
                            [
                                value
                                if value is not None
                                else batch.new_zeros((1, latent_dim))
                                for value in prev_state_rows
                            ],
                            dim=0,
                        )
            yield SequenceBatch(
                sequence_indices=indices,
                start=start,
                batch=batch,
                hidden=h,
                prev_input_embedding=prev_input,
                prev_state_embedding=prev_state,
            )


def _future_length(global_step: int, local_step: int, spec: VqRssmSpec) -> int:
    span = spec.future_max_len - spec.future_min_len + 1
    return spec.future_min_len + ((global_step + local_step) % span)


def _future_segments(
    tensors,
    sequence_batch: SequenceBatch,
    spec: VqRssmSpec,
    global_step: int,
) -> FutureTargets | None:
    positions = []
    segments = []
    masks = []
    max_len = spec.future_max_len
    for row, sequence_index in enumerate(sequence_batch.sequence_indices):
        tensor = tensors[sequence_index]
        for local_step in range(sequence_batch.batch.shape[1]):
            absolute_step = sequence_batch.start + local_step
            length = _future_length(global_step, local_step, spec)
            future_start = absolute_step + 1
            future_end = future_start + length
            if future_end > tensor.shape[0]:
                continue
            padded = tensor.new_zeros((max_len, tensor.shape[1]))
            mask = torch.zeros((max_len,), dtype=torch.bool, device=tensor.device)
            if spec.future_detrend:
                padded[:length] = _causal_detrended_future_segment(
                    tensor, absolute_step, length, spec.future_detrend_half_life
                )
            else:
                padded[:length] = tensor[future_start:future_end]
            mask[:length] = True
            positions.append((row, local_step))
            segments.append(padded)
            masks.append(mask)
    if len(positions) < 2:
        return None
    return FutureTargets(
        positions=tuple(positions),
        segments=torch.stack(segments, dim=0),
        mask=torch.stack(masks, dim=0),
    )


def _causal_detrended_future_segment(
    tensor: torch.Tensor,
    absolute_step: int,
    length: int,
    half_life: float,
) -> torch.Tensor:
    alpha = 1.0 - math.exp(math.log(0.5) / half_life)
    ema = tensor[absolute_step]
    rows = []
    for step in range(absolute_step + 1, absolute_step + 1 + length):
        row = tensor[step]
        rows.append(row - ema)
        ema = alpha * row + (1.0 - alpha) * ema
    return torch.stack(rows, dim=0)


def _cuda_memory_summary(device: torch.device) -> str:
    if device.type != "cuda":
        return "n/a"
    index = device.index if device.index is not None else torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(index)
    return (
        f"total_gib={properties.total_memory / 1024**3:.2f} "
        f"allocated_gib={torch.cuda.memory_allocated(index) / 1024**3:.2f} "
        f"reserved_gib={torch.cuda.memory_reserved(index) / 1024**3:.2f}"
    )


def _split_indices(dataset: WindowDataset, split_name: str) -> tuple[int, ...]:
    return tuple(index for index, split in enumerate(dataset.splits) if split == split_name)

