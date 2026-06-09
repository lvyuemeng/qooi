from __future__ import annotations

import importlib.util

import pytest

from qooi.dynamic.contracts import AssetFeatureSequence, SequenceDataset, WindowDataset
from qooi.dynamic.training import TrainingConfig

torch_available = importlib.util.find_spec("torch") is not None
pytestmark = pytest.mark.skipif(
    not torch_available, reason="torch optional dependency is not installed"
)


def _window_dataset() -> WindowDataset:
    return WindowDataset(
        features=(
            ((0.1, 0.2), (0.2, 0.3), (0.3, 0.4)),
            ((0.2, 0.1), (0.3, 0.2), (0.4, 0.3)),
            ((-0.1, -0.2), (-0.2, -0.3), (-0.3, -0.4)),
        ),
        feature_columns=("a", "b"),
        splits=("train", "valid", "test"),
        seq_len=3,
        stride=1,
    )


def _sequence_dataset() -> SequenceDataset:
    return SequenceDataset(
        feature_columns=("a", "b"),
        sequences=(
            AssetFeatureSequence(
                symbol="BTC-USDT-SWAP",
                split="train",
                features=((0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5)),
                row_index=(0, 1, 2, 3),
                timestamps=(10, 20, 30, 40),
            ),
            AssetFeatureSequence(
                symbol="BTC-USDT-SWAP",
                split="valid",
                features=((0.15, 0.25), (0.25, 0.35), (0.35, 0.45)),
                row_index=(4, 5, 6),
                timestamps=(50, 60, 70),
            ),
            AssetFeatureSequence(
                symbol="ETH-USDT-SWAP",
                split="test",
                features=((-0.1, -0.2), (-0.2, -0.3), (-0.3, -0.4)),
                row_index=(7, 8, 9),
                timestamps=(80, 90, 100),
            ),
        ),
    )


def _spec(**overrides):
    from qooi.dynamic import vq_rssm

    values = {
        "input_dim": 2,
        "hidden_dim": 8,
        "latent_dim": 4,
        "num_codes": 8,
        "objective_terms": ("reconstruct",),
    }
    values.update(overrides)
    return vq_rssm.VqRssmSpec(**values)


def _training_config(**overrides) -> TrainingConfig:
    values = {
        "epochs": 1,
        "batch": 2,
        "lr": 1e-3,
        "grad_clip": 1.0,
        "seed": 1,
        "threads": 1,
        "pred_batch": 2,
    }
    values.update(overrides)
    return TrainingConfig(**values)


def test_tiny_window_train_predict_and_checkpoint_roundtrip(tmp_path) -> None:
    from qooi.dynamic import vq_rssm

    dataset = _window_dataset()
    result = vq_rssm.train(
        dataset,
        _spec(),
        _training_config(),
        output_dir=tmp_path,
    )

    checkpoint_path = tmp_path / "behavior-state-model.pt"
    loaded = vq_rssm.load_checkpoint(checkpoint_path)
    codes = vq_rssm.predict_codes(dataset, loaded, train_cfg=_training_config())
    diagnostics = vq_rssm.predict_diagnostics(dataset, loaded, train_cfg=_training_config())
    decoded = vq_rssm.decode_codebook(loaded)

    assert result.summary is not None
    assert result.summary.train_rows == 1
    assert result.summary.valid_rows == 1
    assert result.summary.test_rows == 1
    assert result.checkpoint_path == checkpoint_path
    assert checkpoint_path.exists()
    assert (tmp_path / "behavior-state-training-metrics.csv").exists()
    assert len(codes.codes) == len(dataset.features)
    assert codes.row_index == (0, 1, 2)
    assert codes.splits == dataset.splits
    assert len(diagnostics.hidden_states) == len(dataset.features)
    assert len(diagnostics.hidden_states[0]) == result.checkpoint.spec.hidden_dim
    assert len(diagnostics.reconstructions[0]) == result.checkpoint.spec.input_dim
    assert len(decoded) == result.checkpoint.spec.num_codes
    assert len(decoded[0]) == result.checkpoint.spec.input_dim


def test_tiny_sequence_train_predict_and_checkpoint_roundtrip(tmp_path) -> None:
    from qooi.dynamic import vq_rssm

    dataset = _sequence_dataset()
    seq_cfg = vq_rssm.SequenceRuntimeConfig(chunk=2, warmup=1, stride=1, carry=True)
    result = vq_rssm.train_sequences(
        dataset,
        _spec(),
        _training_config(),
        seq_cfg,
        output_dir=tmp_path,
    )

    loaded = vq_rssm.load_checkpoint(tmp_path / "behavior-state-model.pt")
    codes = vq_rssm.predict_sequence_codes(
        dataset,
        loaded,
        seq_cfg,
        train_cfg=_training_config(),
    )
    diagnostics = vq_rssm.predict_sequence_diagnostics(
        dataset,
        loaded,
        seq_cfg,
        train_cfg=_training_config(),
    )

    assert result.summary is not None
    assert result.summary.train_rows == 4
    assert result.summary.valid_rows == 3
    assert result.summary.test_rows == 3
    assert len(codes.codes) == len(codes.splits)
    assert set(codes.splits) == {"train", "valid", "test"}
    assert len(diagnostics.hidden_states) == len(codes.codes)
    assert len(diagnostics.reconstructions) == len(codes.codes)


def test_model_loss_contracts_match_forward_outputs() -> None:
    import torch

    from qooi.dynamic import vq_rssm

    torch.manual_seed(1)
    model = vq_rssm.VqRssmModel(_spec())
    features = torch.tensor(
        [
            [[0.1, 0.2], [0.2, 0.3], [0.3, 0.4]],
            [[-0.1, -0.2], [-0.2, -0.3], [-0.3, -0.4]],
        ],
        dtype=torch.float32,
    )

    full = model(features)
    loss, code_counts = model.training_loss(features)
    final = model.final_outputs(features)
    manual_recon = torch.nn.functional.mse_loss(full["recon"], features)

    assert loss.item() == pytest.approx(manual_recon.item())
    assert int(code_counts.sum().item()) == features.shape[0] * features.shape[1]
    assert final["codes"].tolist() == full["codes"][:, -1].tolist()
    assert torch.allclose(final["distances"], full["distances"][:, -1])
    assert torch.allclose(final["hidden_states"], full["hidden_states"][:, -1, :])
    assert torch.allclose(final["recon"], full["recon"][:, -1, :])


def test_codebook_update_moves_selected_vectors_and_preserves_normalized_codebook() -> None:
    import torch

    from qooi.dynamic import vq_rssm

    torch.manual_seed(7)
    model = vq_rssm.VqRssmModel(_spec(ema_decay=0.0, normalize_codebook=True))
    x = torch.tensor([[0.25, -0.5], [0.5, -0.25]], dtype=torch.float32)
    h = torch.zeros((x.shape[0], model.spec.hidden_dim))
    with torch.no_grad():
        prev_embed, prev_state = model._initial_prev_inputs(x.shape[0], x.device)
        state = model._step(x, h, prev_embed, prev_state)
        z_e = state["z_e"].detach()
        codes = state["code"].detach()
        before = (z_e - model.codebook(codes)).pow(2).mean().item()
        model.update_codebook(z_e, codes)
        after = (z_e - model.codebook(codes)).pow(2).mean().item()
        norms = model.codebook.weight.norm(dim=1)

    assert after < before
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_schedule_stage_applies_weights_during_training_without_mutating_checkpoint_spec() -> None:
    from qooi.dynamic import vq_rssm

    result = vq_rssm.train(
        WindowDataset(
            features=(
                ((0.1, 0.2), (0.2, 0.3)),
                ((0.2, 0.1), (0.3, 0.2)),
            ),
            feature_columns=("a", "b"),
            splits=("train", "train"),
            seq_len=2,
            stride=1,
        ),
        _spec(),
        _training_config(epochs=2, seed=3),
        schedule=(
            vq_rssm.VqRssmScheduleStage(
                name="disable_reconstruct",
                start_epoch=2,
                epochs=1,
                reconstruct_weight=0.0,
            ),
        ),
    )

    assert result.metrics[0].train_loss > 0.0
    assert result.metrics[1].train_loss == pytest.approx(0.0)
    assert result.checkpoint.spec.reconstruct_weight == pytest.approx(1.0)


def test_spec_rejects_incompatible_optional_features() -> None:
    with pytest.raises(ValueError, match="reset_dead_codes requires normalize_codebook"):
        _spec(reset_dead_codes=True)
    with pytest.raises(ValueError, match="diversity_weight requires normalize_codebook"):
        _spec(diversity_weight=0.1)
    with pytest.raises(ValueError, match="future_detrend_half_life"):
        _spec(future_detrend_half_life=0.0)
    with pytest.raises(ValueError, match="future contrast component weight"):
        _spec(
            objective_terms=("future_infonce",),
            future_weight=1.0,
            future_standard_weight=0.0,
            future_similarity_weight=0.0,
        )


def test_train_rejects_dataset_without_train_split() -> None:
    from qooi.dynamic import vq_rssm

    dataset = WindowDataset(
        features=(((0.1, 0.2), (0.2, 0.3)),),
        feature_columns=("a", "b"),
        splits=("valid",),
        seq_len=2,
        stride=1,
    )

    with pytest.raises(ValueError, match="no train split"):
        vq_rssm.train(dataset, _spec(), _training_config())
