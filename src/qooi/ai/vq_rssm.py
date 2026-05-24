"""Fixed VQ-RSSM model, training, checkpoint, and prediction lifecycle."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn

from qooi.ai.contracts import CodeSequence, WindowDataset


@dataclass(frozen=True)
class VqRssmSpec:
    input_dim: int = 5
    hidden_dim: int = 128
    latent_dim: int = 16
    num_codes: int = 128
    commitment_cost: float = 0.25
    kl_anneal_steps: int = 5000
    eps: float = 1e-8

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


@dataclass(frozen=True)
class TrainSpec:
    epochs: int = 20
    batch_size: int = 64
    learning_rate: float = 1e-3
    max_grad_norm: float = 1.0
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.learning_rate <= 0.0 or not math.isfinite(self.learning_rate):
            raise ValueError("learning_rate must be a finite positive float")
        if self.max_grad_norm <= 0.0 or not math.isfinite(self.max_grad_norm):
            raise ValueError("max_grad_norm must be a finite positive float")


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


@dataclass(frozen=True)
class TrainingResult:
    checkpoint: VqRssmCheckpoint
    checkpoint_path: Path | None
    metrics: tuple[EpochMetrics, ...]
    summary: EpochMetrics | None


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

    def forward(self, x: torch.Tensor, kl_weight: float = 1.0) -> dict[str, torch.Tensor]:
        batch, seq_len, _feature_width = x.shape
        h = x.new_zeros((batch, self.spec.hidden_dim))
        recons = []
        codes = []
        distances = []
        codebook_loss = x.new_tensor(0.0)
        commitment_loss = x.new_tensor(0.0)
        kl_loss = x.new_tensor(0.0)
        for step in range(seq_len):
            x_t = x[:, step, :]
            embed = self.embedding(x_t)
            mu_post, logvar_post = self.posterior(torch.cat([h, embed], dim=-1)).chunk(2, dim=-1)
            mu_prior, logvar_prior = self.prior(h).chunk(2, dim=-1)
            z_e = mu_post
            z_q, code, distance = self.quantize(z_e)
            z_q_st = z_e + (z_q - z_e).detach()
            recons.append(self.decoder(torch.cat([h, z_q_st], dim=-1)))
            codes.append(code)
            distances.append(distance)
            codebook_loss = codebook_loss + (z_e.detach() - z_q).pow(2).mean()
            commitment_loss = commitment_loss + (z_e - z_q.detach()).pow(2).mean()
            kl_loss = kl_loss + _gaussian_kl(mu_post, logvar_post, mu_prior, logvar_prior).mean()
            h = self.transition(z_q_st, h)
        return {
            "recon": torch.stack(recons, dim=1),
            "codes": torch.stack(codes, dim=1),
            "distances": torch.stack(distances, dim=1),
            "codebook_loss": codebook_loss / seq_len,
            "commitment_loss": commitment_loss / seq_len,
            "kl_loss": kl_loss * kl_weight / seq_len,
        }


def train(
    dataset: WindowDataset,
    spec: VqRssmSpec,
    train_spec: TrainSpec,
    *,
    output_dir: Path | None = None,
) -> TrainingResult:
    if len(dataset.feature_columns) != spec.input_dim:
        raise ValueError("dataset feature width must match VqRssmSpec.input_dim")
    train_indices = _split_indices(dataset, "train")
    valid_indices = _split_indices(dataset, "valid")
    test_indices = _split_indices(dataset, "test")
    if not train_indices:
        raise ValueError("WindowDataset contains no train split rows")
    if train_spec.seed is not None:
        torch.manual_seed(train_spec.seed)
    model = VqRssmModel(spec)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_spec.learning_rate)
    train_features = _feature_tensor(dataset, train_indices)
    valid_features = _feature_tensor(dataset, valid_indices) if valid_indices else None
    metrics = []
    global_step = 0
    for epoch in range(1, train_spec.epochs + 1):
        model.train()
        epoch_losses = []
        epoch_codes = []
        for start in range(0, train_features.shape[0], train_spec.batch_size):
            batch = train_features[start : start + train_spec.batch_size]
            optimizer.zero_grad()
            loss, output = _loss(model, batch, global_step)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_spec.max_grad_norm)
            optimizer.step()
            global_step += 1
            epoch_losses.append(float(loss.detach().cpu()))
            batch_codes = output["codes"].detach().cpu().flatten().tolist()
            epoch_codes.extend(int(code) for code in batch_codes)
        active_codes = len(set(epoch_codes))
        metrics.append(
            EpochMetrics(
                epoch=epoch,
                train_loss=sum(epoch_losses) / max(1, len(epoch_losses)),
                valid_loss=(
                    _validation_loss(model, valid_features)
                    if valid_features is not None
                    else None
                ),
                train_rows=len(train_indices),
                valid_rows=len(valid_indices),
                test_rows=len(test_indices),
                active_codes=active_codes,
                codebook_utilization_pct=active_codes / spec.num_codes * 100.0,
            )
        )
    checkpoint_path = output_dir / "behavior-state-model.pt" if output_dir is not None else None
    checkpoint = VqRssmCheckpoint(
        spec=spec,
        path=checkpoint_path,
        state_dict={key: value.detach().cpu() for key, value in model.state_dict().items()},
    )
    if checkpoint_path is not None:
        save_checkpoint(checkpoint, checkpoint_path)
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


def load_checkpoint(path: Path) -> VqRssmCheckpoint:
    data = torch.load(path, map_location="cpu")
    return VqRssmCheckpoint(
        spec=VqRssmSpec(**data["spec"]),
        path=path,
        state_dict=data["state_dict"],
    )


def predict_codes(dataset: WindowDataset, checkpoint: VqRssmCheckpoint) -> CodeSequence:
    if len(dataset.feature_columns) != checkpoint.spec.input_dim:
        raise ValueError("dataset feature width must match checkpoint spec input_dim")
    model = VqRssmModel(checkpoint.spec)
    model.load_state_dict(checkpoint.state_dict)
    features = _feature_tensor(dataset, tuple(range(len(dataset.features))))
    model.eval()
    with torch.no_grad():
        output = model(features, kl_weight=1.0)
    codes = output["codes"][:, -1].detach().cpu().tolist()
    distances = output["distances"][:, -1].detach().cpu().tolist()
    return CodeSequence(
        codes=tuple(int(code) for code in codes),
        distances=tuple(float(distance) for distance in distances),
        row_index=tuple(range(len(codes))),
        splits=dataset.splits,
    )


def _gaussian_kl(
    mu_post: torch.Tensor,
    logvar_post: torch.Tensor,
    mu_prior: torch.Tensor,
    logvar_prior: torch.Tensor,
) -> torch.Tensor:
    return 0.5 * (
        logvar_prior
        - logvar_post
        + (logvar_post.exp() + (mu_post - mu_prior).pow(2))
        / logvar_prior.exp().clamp_min(1e-8)
        - 1.0
    ).sum(dim=-1)


def _loss(
    model: VqRssmModel,
    batch: torch.Tensor,
    global_step: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    kl_weight = min(1.0, global_step / max(1, model.spec.kl_anneal_steps))
    output = model(batch, kl_weight=kl_weight)
    recon_loss = torch.nn.functional.mse_loss(output["recon"], batch)
    loss = (
        recon_loss
        + output["codebook_loss"]
        + model.spec.commitment_cost * output["commitment_loss"]
        + output["kl_loss"]
    )
    return loss, output


def _validation_loss(model: VqRssmModel, valid_features: torch.Tensor) -> float:
    model.eval()
    with torch.no_grad():
        loss, _output = _loss(model, valid_features, model.spec.kl_anneal_steps)
    return float(loss.detach().cpu())


def _split_indices(dataset: WindowDataset, split_name: str) -> tuple[int, ...]:
    return tuple(index for index, split in enumerate(dataset.splits) if split == split_name)


def _feature_tensor(dataset: WindowDataset, indices: tuple[int, ...]) -> torch.Tensor:
    return torch.tensor([dataset.features[index] for index in indices], dtype=torch.float32)
