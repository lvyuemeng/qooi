"""Fixed VQ-RSSM model, training, checkpoint, and prediction lifecycle."""

from __future__ import annotations

import csv
import logging
import math
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import torch
from torch import nn

from qooi.ai.contracts import CodeSequence, SequenceDataset, SplitName, WindowDataset
from qooi.ai.training import TrainingConfig

logger = logging.getLogger(__name__)

_LOGVAR_MIN = -20.0
_LOGVAR_MAX = 10.0
_KL_DELTA_LIMIT = 1.0e4

@dataclass(frozen=True)
class VqRssmSpec:
    input_dim: int = 5
    hidden_dim: int = 128
    latent_dim: int = 16
    num_codes: int = 128
    commitment_cost: float = 0.25
    kl_anneal_steps: int = 5000
    eps: float = 1e-8
    objective_terms: tuple[str, ...] = ("reconstruct", "vq", "kl")
    reconstruct_weight: float = 1.0
    vq_weight: float = 1.0
    kl_weight: float = 1.0
    future_weight: float = 0.0

    def __post_init__(self) -> None:
        if self.input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if self.latent_dim <= 0:
            raise ValueError("latent_dim must be positive")
        if self.num_codes <= 0:
            raise ValueError("num_codes must be positive")
        if self.commitment_cost <= 0.0 or not math.isfinite(self.commitment_cost):
            raise ValueError("commitment_cost must be a finite positive float")
        if self.kl_anneal_steps <= 0:
            raise ValueError("kl_anneal_steps must be positive")
        if self.eps <= 0.0 or not math.isfinite(self.eps):
            raise ValueError("eps must be a finite positive float")
        if not self.objective_terms:
            raise ValueError("objective_terms must contain at least one term")
        invalid_terms = sorted(
            set(self.objective_terms) - {"reconstruct", "vq", "kl", "future_infonce"}
        )
        if invalid_terms:
            raise ValueError("unsupported objective terms: " + ", ".join(invalid_terms))
        for name in ("reconstruct_weight", "vq_weight", "kl_weight", "future_weight"):
            value = getattr(self, name)
            if value < 0.0 or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite non-negative float")


@dataclass(frozen=True)
class VqRssmCheckpoint:
    spec: VqRssmSpec
    path: Path | None
    state_dict: dict[str, torch.Tensor]


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
    train_codebook_loss: float | None = None
    train_commitment_loss: float | None = None
    train_kl_loss: float | None = None
    valid_recon_loss: float | None = None
    valid_codebook_loss: float | None = None
    valid_commitment_loss: float | None = None
    valid_kl_loss: float | None = None
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


@dataclass(frozen=True)
class TrainingResult:
    checkpoint: VqRssmCheckpoint
    checkpoint_path: Path | None
    metrics: tuple[EpochMetrics, ...]
    summary: EpochMetrics | None


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
    codebook: torch.Tensor
    commitment: torch.Tensor
    kl: torch.Tensor
    kl_weight: float
    code_counts: torch.Tensor
    next_hidden: torch.Tensor
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
    codebook_sum: torch.Tensor
    commitment_sum: torch.Tensor
    kl_sum: torch.Tensor
    z_e_norm_sum: torch.Tensor
    z_q_norm_sum: torch.Tensor
    vq_distance_sum: torch.Tensor
    vq_distance_p95_sum: torch.Tensor
    vq_distance_max: torch.Tensor
    rows: int
    code_counts: torch.Tensor


@dataclass(frozen=True)
class LossParts:
    total: float
    recon: float
    codebook: float
    commitment: float
    kl: float
    z_e_norm_mean: float
    z_q_norm_mean: float
    vq_distance_mean: float
    vq_distance_p95: float
    vq_distance_max: float


class VqRssmModel(nn.Module):
    def __init__(self, spec: VqRssmSpec) -> None:
        super().__init__()
        self.spec = spec
        self.embedding = nn.Linear(spec.input_dim, 32)
        self.posterior = nn.Linear(spec.hidden_dim + 32, spec.latent_dim * 2)
        self.prior = nn.Linear(spec.hidden_dim, spec.latent_dim * 2)
        self.codebook = nn.Embedding(spec.num_codes, spec.latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(spec.hidden_dim + spec.latent_dim, spec.hidden_dim),
            nn.ReLU(),
            nn.Linear(spec.hidden_dim, spec.input_dim),
        )
        self.transition = nn.GRUCell(spec.latent_dim, spec.hidden_dim)

    def quantize(self, z_e: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        distances = torch.cdist(z_e, self.codebook.weight)
        codes = distances.argmin(dim=-1)
        z_q = self.codebook(codes)
        selected_distances = distances.gather(1, codes.unsqueeze(1)).squeeze(1)
        return z_q, codes, selected_distances

    def _step(self, x_t: torch.Tensor, h: torch.Tensor) -> dict[str, torch.Tensor]:
        embed = self.embedding(x_t)
        mu_post, logvar_post = self.posterior(torch.cat([h, embed], dim=-1)).chunk(2, dim=-1)
        mu_prior, logvar_prior = self.prior(h).chunk(2, dim=-1)
        z_e = mu_post
        z_q, code, distance = self.quantize(z_e)
        z_q_st = z_e + (z_q - z_e).detach()
        recon = self.decoder(torch.cat([h, z_q_st], dim=-1))
        next_h = self.transition(z_q_st, h)
        return {
            "mu_post": mu_post,
            "logvar_post": logvar_post,
            "mu_prior": mu_prior,
            "logvar_prior": logvar_prior,
            "z_e": z_e,
            "z_q": z_q,
            "code": code,
            "distance": distance,
            "recon": recon,
            "next_h": next_h,
        }

    def forward(self, x: torch.Tensor, kl_weight: float = 1.0) -> dict[str, torch.Tensor]:
        loss, output = self._run(x, kl_weight=kl_weight, collect=True)
        output.update(
            codebook_loss=loss.codebook,
            commitment_loss=loss.commitment,
            kl_loss=loss.kl,
        )
        return output

    def loss_bundle(
        self,
        x: torch.Tensor,
        global_step: int,
        *,
        h: torch.Tensor | None = None,
    ) -> LossBundle:
        kl_weight = min(1.0, global_step / max(1, self.spec.kl_anneal_steps))
        loss, _output = self._run(x, h=h, kl_weight=kl_weight, collect=False)
        return loss

    def training_loss(
        self, x: torch.Tensor, global_step: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        loss = self.loss_bundle(x, global_step)
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
    ) -> dict[str, torch.Tensor]:
        _loss, output = self._run(x, h=h, kl_weight=1.0, collect=True)
        return output

    def _run(
        self,
        x: torch.Tensor,
        *,
        h: torch.Tensor | None = None,
        kl_weight: float,
        collect: bool,
    ) -> tuple[LossBundle, dict[str, torch.Tensor]]:
        if x.ndim == 2:
            x = x.unsqueeze(0)
        batch, seq_len, _feature_width = x.shape
        if seq_len <= 0:
            raise ValueError("input sequence must contain at least one step")
        h = x.new_zeros((batch, self.spec.hidden_dim)) if h is None else h
        recon_loss = x.new_tensor(0.0)
        codebook_loss = x.new_tensor(0.0)
        commitment_loss = x.new_tensor(0.0)
        kl_loss = x.new_tensor(0.0)
        z_e_norm_sum = x.new_tensor(0.0)
        z_q_norm_sum = x.new_tensor(0.0)
        vq_distance_sum = x.new_tensor(0.0)
        vq_distance_max = x.new_tensor(0.0)
        code_counts = torch.zeros(self.spec.num_codes, dtype=torch.long, device=x.device)
        codes = []
        distances = []
        all_distances = []
        hidden_states = []
        recons = []
        for step in range(seq_len):
            state = self._step(x[:, step, :], h)
            recon_loss = recon_loss + torch.nn.functional.mse_loss(
                state["recon"], x[:, step, :]
            )
            codebook_loss = codebook_loss + (
                state["z_e"].detach() - state["z_q"]
            ).pow(2).mean()
            commitment_loss = commitment_loss + (
                state["z_e"] - state["z_q"].detach()
            ).pow(2).mean()
            kl_loss = kl_loss + _gaussian_kl(
                state["mu_post"],
                state["logvar_post"],
                state["mu_prior"],
                state["logvar_prior"],
            ).mean()
            code_counts = code_counts + torch.bincount(
                state["code"].detach(), minlength=self.spec.num_codes
            )
            z_e_norm_sum = z_e_norm_sum + state["z_e"].norm(dim=-1).sum()
            z_q_norm_sum = z_q_norm_sum + state["z_q"].norm(dim=-1).sum()
            vq_distance_sum = vq_distance_sum + state["distance"].sum()
            vq_distance_max = torch.maximum(vq_distance_max, state["distance"].max())
            all_distances.append(state["distance"])
            h = state["next_h"]
            if collect:
                codes.append(state["code"])
                distances.append(state["distance"])
                hidden_states.append(h)
                recons.append(state["recon"])
        recon = recon_loss / seq_len
        codebook = codebook_loss / seq_len
        commitment = commitment_loss / seq_len
        kl = kl_loss * kl_weight / seq_len
        diagnostic_rows = batch * seq_len
        vq_distances = torch.cat(all_distances)
        reconstruct_component = (
            recon * self.spec.reconstruct_weight
            if "reconstruct" in self.spec.objective_terms
            else recon.new_tensor(0.0)
        )
        vq_component = (
            self.spec.vq_weight * (codebook + self.spec.commitment_cost * commitment)
            if "vq" in self.spec.objective_terms
            else recon.new_tensor(0.0)
        )
        kl_component = (
            self.spec.kl_weight * kl
            if "kl" in self.spec.objective_terms
            else recon.new_tensor(0.0)
        )
        loss = LossBundle(
            total=reconstruct_component + vq_component + kl_component,
            recon=recon,
            codebook=codebook,
            commitment=commitment,
            kl=kl,
            kl_weight=kl_weight,
            code_counts=code_counts,
            next_hidden=h,
            z_e_norm_mean=z_e_norm_sum / diagnostic_rows,
            z_q_norm_mean=z_q_norm_sum / diagnostic_rows,
            vq_distance_mean=vq_distance_sum / diagnostic_rows,
            vq_distance_p95=torch.quantile(vq_distances, 0.95),
            vq_distance_max=vq_distance_max,
        )
        output = {"final_hidden": h}
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
    train_features = _feature_tensor(dataset, train_indices, device=device)
    valid_features = (
        _feature_tensor(dataset, valid_indices, device=device) if valid_indices else None
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
    for epoch in range(1, train_cfg.epochs + 1):
        epoch_start = time.perf_counter()
        logger.info("train_epoch_start epoch=%s/%s", epoch, train_cfg.epochs)
        model.train()
        acc = _new_epoch_accumulator(spec, device)
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
            global_step += 1
            batch_rows = batch.shape[0]
            with torch.no_grad():
                _accumulate_loss(acc, bundle, batch_rows)
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
        train_parts = _loss_parts_from_accumulator(acc)
        codebook_norms = _codebook_norm_metrics(model)
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
            spec=spec,
            codebook_norms=codebook_norms,
        )
        metrics.append(metric)
        _write_training_metrics(output_dir, metrics)
        _log_epoch_summary("train", metric, time.perf_counter() - epoch_start)
    checkpoint_path = output_dir / "behavior-state-model.pt" if output_dir is not None else None
    checkpoint = VqRssmCheckpoint(
        spec=spec,
        path=checkpoint_path,
        state_dict={key: value.detach().cpu() for key, value in model.state_dict().items()},
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
    )


def train_sequences(
    dataset: SequenceDataset,
    spec: VqRssmSpec,
    train_cfg: TrainingConfig,
    seq_cfg: SequenceRuntimeConfig,
    *,
    output_dir: Path | None = None,
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
        _sequence_feature_tensor(sequence.features, device=device)
        for sequence in train_sequences_
    )
    valid_tensors = tuple(
        _sequence_feature_tensor(sequence.features, device=device)
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
    for epoch in range(1, train_cfg.epochs + 1):
        epoch_start = time.perf_counter()
        logger.info("train_sequences_epoch_start epoch=%s/%s", epoch, train_cfg.epochs)
        model.train()
        acc = _new_epoch_accumulator(spec, device)
        batch_index = 0
        hidden_by_sequence = [None] * len(train_tensors)
        max_rows = max(tensor.shape[0] for tensor in train_tensors)
        for start in range(0, max_rows, seq_cfg.chunk):
            for sequence_indices, batch, h in _parallel_sequence_batches(
                train_tensors,
                hidden_by_sequence,
                start,
                seq_cfg,
                train_cfg.batch,
                spec.hidden_dim,
            ):
                optimizer.zero_grad()
                bundle = model.loss_bundle(batch, global_step, h=h)
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
                global_step += 1
                batch_index += 1
                batch_rows = batch.shape[0] * batch.shape[1]
                with torch.no_grad():
                    _accumulate_loss(acc, bundle, batch_rows)
                if seq_cfg.carry:
                    detached_h = next_h.detach()
                    for row, sequence_index in enumerate(sequence_indices):
                        hidden_by_sequence[sequence_index] = detached_h[row : row + 1]
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
        train_parts = _loss_parts_from_accumulator(acc)
        codebook_norms = _codebook_norm_metrics(model)
        valid_parts = None
        if valid_sequences:
            valid_start = time.perf_counter()
            valid_parts = _sequence_validation_loss(
                model,
                valid_tensors,
                seq_cfg,
                train_cfg.batch,
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
            spec=spec,
            codebook_norms=codebook_norms,
        )
        metrics.append(metric)
        _write_training_metrics(output_dir, metrics)
        _log_epoch_summary("train_sequences", metric, time.perf_counter() - epoch_start)
    checkpoint_path = output_dir / "behavior-state-model.pt" if output_dir is not None else None
    checkpoint = VqRssmCheckpoint(
        spec=spec,
        path=checkpoint_path,
        state_dict={key: value.detach().cpu() for key, value in model.state_dict().items()},
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


def _metric_float(value: float | None) -> str:
    return f"{value:.6f}" if value is not None else "none"


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


def _new_epoch_accumulator(spec: VqRssmSpec, device: torch.device) -> EpochAccumulator:
    zero = torch.zeros((), dtype=torch.float32, device=device)
    return EpochAccumulator(
        loss_sum=zero.clone(),
        recon_sum=zero.clone(),
        codebook_sum=zero.clone(),
        commitment_sum=zero.clone(),
        kl_sum=zero.clone(),
        z_e_norm_sum=zero.clone(),
        z_q_norm_sum=zero.clone(),
        vq_distance_sum=zero.clone(),
        vq_distance_p95_sum=zero.clone(),
        vq_distance_max=zero.clone(),
        rows=0,
        code_counts=torch.zeros(spec.num_codes, dtype=torch.long, device=device),
    )


def _accumulate_loss(acc: EpochAccumulator, bundle: LossBundle, rows: int) -> None:
    acc.loss_sum += bundle.total.detach() * rows
    acc.recon_sum += bundle.recon.detach() * rows
    acc.codebook_sum += bundle.codebook.detach() * rows
    acc.commitment_sum += bundle.commitment.detach() * rows
    acc.kl_sum += bundle.kl.detach() * rows
    acc.z_e_norm_sum += bundle.z_e_norm_mean.detach() * rows
    acc.z_q_norm_sum += bundle.z_q_norm_mean.detach() * rows
    acc.vq_distance_sum += bundle.vq_distance_mean.detach() * rows
    acc.vq_distance_p95_sum += bundle.vq_distance_p95.detach() * rows
    acc.vq_distance_max = torch.maximum(acc.vq_distance_max, bundle.vq_distance_max.detach())
    acc.rows += rows
    acc.code_counts += bundle.code_counts


def _loss_parts_from_accumulator(acc: EpochAccumulator) -> LossParts:
    rows = max(1, acc.rows)
    return LossParts(
        total=float((acc.loss_sum / rows).detach().cpu()),
        recon=float((acc.recon_sum / rows).detach().cpu()),
        codebook=float((acc.codebook_sum / rows).detach().cpu()),
        commitment=float((acc.commitment_sum / rows).detach().cpu()),
        kl=float((acc.kl_sum / rows).detach().cpu()),
        z_e_norm_mean=float((acc.z_e_norm_sum / rows).detach().cpu()),
        z_q_norm_mean=float((acc.z_q_norm_sum / rows).detach().cpu()),
        vq_distance_mean=float((acc.vq_distance_sum / rows).detach().cpu()),
        vq_distance_p95=float((acc.vq_distance_p95_sum / rows).detach().cpu()),
        vq_distance_max=float(acc.vq_distance_max.detach().cpu()),
    )


def _codebook_norm_metrics(model: VqRssmModel) -> tuple[float, float]:
    norms = model.codebook.weight.detach().norm(dim=-1)
    return float(norms.mean().cpu()), float(norms.max().cpu())


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
        train_codebook_loss=train_parts.codebook,
        train_commitment_loss=train_parts.commitment,
        train_kl_loss=train_parts.kl,
        valid_recon_loss=valid_parts.recon if valid_parts else None,
        valid_codebook_loss=valid_parts.codebook if valid_parts else None,
        valid_commitment_loss=valid_parts.commitment if valid_parts else None,
        valid_kl_loss=valid_parts.kl if valid_parts else None,
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
    )


def _log_epoch_summary(prefix: str, metric: EpochMetrics, elapsed_s: float) -> None:
    logger.info(
        "%s_epoch_done epoch=%s train_loss=%.6f valid_loss=%s "
        "train_recon=%.6f train_codebook=%.6f train_commitment=%.6f train_kl=%.6f "
        "valid_recon=%s valid_codebook=%s valid_commitment=%s valid_kl=%s "
        "train_z_e_norm=%.6f train_z_q_norm=%.6f "
        "train_vq_distance_mean=%.6f train_vq_distance_p95=%.6f "
        "train_vq_distance_max=%.6f valid_vq_distance_mean=%s "
        "valid_vq_distance_p95=%s valid_vq_distance_max=%s "
        "codebook_norm_mean=%.6f codebook_norm_max=%.6f active_codes=%s "
        "utilization_pct=%.2f elapsed_s=%.2f",
        prefix,
        metric.epoch,
        metric.train_loss,
        f"{metric.valid_loss:.6f}" if metric.valid_loss is not None else "none",
        metric.train_recon_loss,
        metric.train_codebook_loss,
        metric.train_commitment_loss,
        metric.train_kl_loss,
        _metric_float(metric.valid_recon_loss),
        _metric_float(metric.valid_codebook_loss),
        _metric_float(metric.valid_commitment_loss),
        _metric_float(metric.valid_kl_loss),
        metric.train_z_e_norm_mean,
        metric.train_z_q_norm_mean,
        metric.train_vq_distance_mean,
        metric.train_vq_distance_p95,
        metric.train_vq_distance_max,
        _metric_float(metric.valid_vq_distance_mean),
        _metric_float(metric.valid_vq_distance_p95),
        _metric_float(metric.valid_vq_distance_max),
        metric.codebook_norm_mean,
        metric.codebook_norm_max,
        metric.active_codes,
        metric.codebook_utilization_pct,
        elapsed_s,
    )


def load_checkpoint(path: Path) -> VqRssmCheckpoint:
    data = torch.load(path, map_location="cpu")
    return VqRssmCheckpoint(
        spec=VqRssmSpec(**data["spec"]),
        path=path,
        state_dict=data["state_dict"],
    )


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
            features = _feature_tensor(dataset, batch_indices, device=device)
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


def _gaussian_kl(
    mu_post: torch.Tensor,
    logvar_post: torch.Tensor,
    mu_prior: torch.Tensor,
    logvar_prior: torch.Tensor,
) -> torch.Tensor:
    logvar_post = logvar_post.clamp(min=_LOGVAR_MIN, max=_LOGVAR_MAX)
    logvar_prior = logvar_prior.clamp(min=_LOGVAR_MIN, max=_LOGVAR_MAX)
    delta = (mu_post - mu_prior).clamp(min=-_KL_DELTA_LIMIT, max=_KL_DELTA_LIMIT)
    return 0.5 * (
        logvar_prior
        - logvar_post
        + (logvar_post.exp() + delta.pow(2))
        / logvar_prior.exp().clamp_min(1e-8)
        - 1.0
    ).sum(dim=-1)


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
    acc = _new_epoch_accumulator(model.spec, valid_features.device)
    with torch.no_grad():
        for start in range(0, valid_features.shape[0], batch_size):
            batch = valid_features[start : start + batch_size]
            bundle = model.loss_bundle(batch, model.spec.kl_anneal_steps)
            _raise_if_nonfinite_loss(bundle.total, "window validation")
            _accumulate_loss(acc, bundle, batch.shape[0])
    return _loss_parts_from_accumulator(acc)


def _sequence_validation_loss(
    model: VqRssmModel,
    tensors,
    seq_cfg: SequenceRuntimeConfig,
    batch_size: int,
) -> LossParts:
    model.eval()
    device = tensors[0].device
    acc = _new_epoch_accumulator(model.spec, device)
    with torch.no_grad():
        hidden_by_sequence = [None] * len(tensors)
        max_rows = max(tensor.shape[0] for tensor in tensors)
        for start in range(0, max_rows, seq_cfg.chunk):
            for sequence_indices, batch, h in _parallel_sequence_batches(
                tensors,
                hidden_by_sequence,
                start,
                seq_cfg,
                batch_size,
                model.spec.hidden_dim,
            ):
                bundle = model.loss_bundle(batch, model.spec.kl_anneal_steps, h=h)
                _raise_if_nonfinite_loss(
                    bundle.total,
                    "sequence validation "
                    f"streams={batch.shape[0]} steps={batch.shape[1]}",
                )
                batch_rows = batch.shape[0] * batch.shape[1]
                _accumulate_loss(acc, bundle, batch_rows)
                if seq_cfg.carry:
                    detached_h = bundle.next_hidden.detach()
                    for row, sequence_index in enumerate(sequence_indices):
                        hidden_by_sequence[sequence_index] = detached_h[row : row + 1]
    return _loss_parts_from_accumulator(acc)


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
            row_offset = 0
            tensor = _sequence_feature_tensor(sequence.features, device=device)
            for start in range(0, tensor.shape[0], seq_cfg.chunk):
                chunk = tensor[start : start + seq_cfg.chunk]
                output = model.sequence_outputs(chunk, h=h if seq_cfg.carry else None)
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
                h = output["final_hidden"].detach() if seq_cfg.carry else None
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
            yield indices, batch, h


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


def _feature_tensor(
    dataset: WindowDataset,
    indices: tuple[int, ...],
    *,
    device: torch.device,
) -> torch.Tensor:
    return torch.as_tensor(
        [dataset.features[index] for index in indices],
        dtype=torch.float32,
        device=device,
    )


def _sequence_feature_tensor(
    features,
    *,
    device: torch.device,
) -> torch.Tensor:
    return torch.as_tensor(features, dtype=torch.float32, device=device)
