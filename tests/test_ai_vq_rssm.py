from __future__ import annotations

import importlib.util

import pytest

from qooi.ai.contracts import WindowDataset

torch_available = importlib.util.find_spec("torch") is not None


@pytest.mark.skipif(not torch_available, reason="torch optional dependency is not installed")
def test_tiny_vq_rssm_train_and_predict() -> None:
    from qooi.ai import vq_rssm

    dataset = WindowDataset(
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
    spec = vq_rssm.VqRssmSpec(
        input_dim=2,
        hidden_dim=8,
        latent_dim=4,
        num_codes=8,
        kl_anneal_steps=2,
    )

    result = vq_rssm.train(
        dataset,
        spec,
        vq_rssm.TrainSpec(
            epochs=1,
            batch_size=2,
            learning_rate=1e-3,
            max_grad_norm=1.0,
            seed=1,
        ),
    )
    codes = vq_rssm.predict_codes(dataset, result.checkpoint)

    assert result.metrics
    assert result.summary is not None
    assert result.summary.train_rows == 1
    assert result.summary.valid_rows == 1
    assert result.summary.test_rows == 1
    assert result.summary.valid_loss is not None
    assert len(codes.codes) == 3
    assert codes.row_index == (0, 1, 2)


@pytest.mark.skipif(not torch_available, reason="torch optional dependency is not installed")
def test_train_rejects_dataset_without_train_split() -> None:
    from qooi.ai import vq_rssm

    dataset = WindowDataset(
        features=(((0.1, 0.2),),),
        feature_columns=("a", "b"),
        splits=("valid",),
        seq_len=1,
        stride=1,
    )

    with pytest.raises(ValueError, match="no train split"):
        vq_rssm.train(dataset, vq_rssm.VqRssmSpec(input_dim=2), vq_rssm.TrainSpec())
