from __future__ import annotations

import importlib.util

import pytest

from qooi.ai.contracts import AssetFeatureSequence, SequenceDataset, WindowDataset
from qooi.ai.training import TrainingConfig

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
        TrainingConfig(
            epochs=1,
            batch=2,
            lr=1e-3,
            grad_clip=1.0,
            seed=1,
        ),
    )
    codes = vq_rssm.predict_codes(dataset, result.checkpoint)
    diagnostics = vq_rssm.predict_diagnostics(dataset, result.checkpoint)
    decoded = vq_rssm.decode_codebook(result.checkpoint)

    assert result.metrics
    assert result.summary is not None
    assert result.summary.train_rows == 1
    assert result.summary.valid_rows == 1
    assert result.summary.test_rows == 1
    assert result.summary.valid_loss is not None
    assert result.summary.train_recon_loss is not None
    assert result.summary.train_codebook_loss is not None
    assert result.summary.train_commitment_loss is not None
    assert result.summary.train_kl_loss is not None
    assert result.summary.valid_recon_loss is not None
    assert result.summary.valid_codebook_loss is not None
    assert result.summary.valid_commitment_loss is not None
    assert result.summary.valid_kl_loss is not None
    assert result.summary.train_z_e_norm_mean is not None
    assert result.summary.train_z_q_norm_mean is not None
    assert result.summary.train_vq_distance_mean is not None
    assert result.summary.train_vq_distance_p95 is not None
    assert result.summary.train_vq_distance_max is not None
    assert result.summary.valid_z_e_norm_mean is not None
    assert result.summary.valid_z_q_norm_mean is not None
    assert result.summary.valid_vq_distance_mean is not None
    assert result.summary.valid_vq_distance_p95 is not None
    assert result.summary.valid_vq_distance_max is not None
    assert result.summary.codebook_norm_mean is not None
    assert result.summary.codebook_norm_max is not None
    assert len(codes.codes) == 3
    assert codes.row_index == (0, 1, 2)
    assert len(diagnostics.hidden_states) == 3
    assert len(diagnostics.hidden_states[0]) == 8
    assert len(diagnostics.reconstructions[0]) == 2
    assert len(decoded) == 8
    assert len(decoded[0]) == 2


@pytest.mark.skipif(not torch_available, reason="torch optional dependency is not installed")
def test_explicit_vq_rssm_methods_match_full_forward() -> None:
    import torch

    from qooi.ai import vq_rssm

    torch.manual_seed(1)
    spec = vq_rssm.VqRssmSpec(
        input_dim=2,
        hidden_dim=8,
        latent_dim=4,
        num_codes=8,
        kl_anneal_steps=2,
    )
    model = vq_rssm.VqRssmModel(spec)
    features = torch.tensor(
        [
            [[0.1, 0.2], [0.2, 0.3], [0.3, 0.4]],
            [[-0.1, -0.2], [-0.2, -0.3], [-0.3, -0.4]],
        ],
        dtype=torch.float32,
    )

    full = model(features, kl_weight=1.0)
    loss, code_counts = model.training_loss(features, spec.kl_anneal_steps)
    final = model.final_outputs(features)
    manual_loss = (
        torch.nn.functional.mse_loss(full["recon"], features)
        + full["codebook_loss"]
        + spec.commitment_cost * full["commitment_loss"]
        + full["kl_loss"]
    )

    assert loss.item() == pytest.approx(manual_loss.item())
    assert int(code_counts.sum().item()) == features.shape[0] * features.shape[1]
    assert final["codes"].tolist() == full["codes"][:, -1].tolist()
    assert torch.allclose(final["distances"], full["distances"][:, -1])
    assert torch.allclose(final["hidden_states"], full["hidden_states"][:, -1, :])
    assert torch.allclose(final["recon"], full["recon"][:, -1, :])


@pytest.mark.skipif(not torch_available, reason="torch optional dependency is not installed")
def test_vq_codebook_loss_updates_selected_code_toward_encoder() -> None:
    import torch

    from qooi.ai import vq_rssm

    torch.manual_seed(7)
    model = vq_rssm.VqRssmModel(
        vq_rssm.VqRssmSpec(input_dim=2, hidden_dim=8, latent_dim=4, num_codes=8)
    )
    x = torch.tensor([[0.25, -0.5], [0.5, -0.25]], dtype=torch.float32)
    h = torch.zeros((x.shape[0], model.spec.hidden_dim))
    with torch.no_grad():
        state = model._step(x, h)
        codes = state["code"].detach()
        z_e = state["z_e"].detach()
        before = (z_e - model.codebook(codes)).pow(2).mean().item()

    optimizer = torch.optim.SGD(model.codebook.parameters(), lr=0.1)
    optimizer.zero_grad()
    selected = model.codebook(codes)
    codebook_loss = (z_e - selected).pow(2).mean()
    codebook_loss.backward()
    optimizer.step()

    with torch.no_grad():
        after = (z_e - model.codebook(codes)).pow(2).mean().item()
    assert after < before


@pytest.mark.skipif(not torch_available, reason="torch optional dependency is not installed")
def test_commitment_loss_updates_encoder_toward_selected_code() -> None:
    import torch

    from qooi.ai import vq_rssm

    torch.manual_seed(11)
    model = vq_rssm.VqRssmModel(
        vq_rssm.VqRssmSpec(input_dim=2, hidden_dim=8, latent_dim=4, num_codes=8)
    )
    x = torch.tensor([[0.25, -0.5], [0.5, -0.25]], dtype=torch.float32)
    h = torch.zeros((x.shape[0], model.spec.hidden_dim))
    with torch.no_grad():
        state = model._step(x, h)
        z_q = state["z_q"].detach()
        before = (state["z_e"] - z_q).pow(2).mean().item()

    optimizer = torch.optim.SGD(
        [*model.embedding.parameters(), *model.posterior.parameters()], lr=0.1
    )
    optimizer.zero_grad()
    state = model._step(x, h)
    commitment_loss = (state["z_e"] - z_q).pow(2).mean()
    commitment_loss.backward()
    optimizer.step()

    with torch.no_grad():
        after = (model._step(x, h)["z_e"] - z_q).pow(2).mean().item()
    assert after < before


@pytest.mark.skipif(not torch_available, reason="torch optional dependency is not installed")
def test_straight_through_reconstruction_does_not_update_codebook() -> None:
    import torch

    from qooi.ai import vq_rssm

    torch.manual_seed(13)
    model = vq_rssm.VqRssmModel(
        vq_rssm.VqRssmSpec(input_dim=2, hidden_dim=8, latent_dim=4, num_codes=8)
    )
    x = torch.tensor([[0.25, -0.5], [0.5, -0.25]], dtype=torch.float32)
    h = torch.zeros((x.shape[0], model.spec.hidden_dim))

    state = model._step(x, h)
    loss = torch.nn.functional.mse_loss(state["recon"], x)
    loss.backward()

    codebook_grad = model.codebook.weight.grad
    embedding_grad = model.embedding.weight.grad
    posterior_grad = model.posterior.weight.grad
    assert codebook_grad is None or torch.count_nonzero(codebook_grad).item() == 0
    assert embedding_grad is not None and torch.count_nonzero(embedding_grad).item() > 0
    assert posterior_grad is not None and torch.count_nonzero(posterior_grad).item() > 0


@pytest.mark.skipif(not torch_available, reason="torch optional dependency is not installed")
def test_loss_bundle_matches_forward_manual_loss_components() -> None:
    import torch

    from qooi.ai import vq_rssm

    torch.manual_seed(17)
    spec = vq_rssm.VqRssmSpec(input_dim=2, hidden_dim=8, latent_dim=4, num_codes=8)
    model = vq_rssm.VqRssmModel(spec)
    features = torch.tensor(
        [
            [[0.1, 0.2], [0.2, 0.3], [0.3, 0.4]],
            [[-0.1, -0.2], [-0.2, -0.3], [-0.3, -0.4]],
        ],
        dtype=torch.float32,
    )

    full = model(features, kl_weight=1.0)
    bundle = model.loss_bundle(features, spec.kl_anneal_steps)

    assert bundle.recon.item() == pytest.approx(
        torch.nn.functional.mse_loss(full["recon"], features).item()
    )
    assert bundle.codebook.item() == pytest.approx(full["codebook_loss"].item())
    assert bundle.commitment.item() == pytest.approx(full["commitment_loss"].item())
    assert bundle.kl.item() == pytest.approx(full["kl_loss"].item())
    assert bundle.z_e_norm_mean.item() >= 0.0
    assert bundle.z_q_norm_mean.item() >= 0.0
    assert bundle.vq_distance_mean.item() >= 0.0
    assert bundle.vq_distance_p95.item() >= 0.0
    assert bundle.vq_distance_max.item() >= bundle.vq_distance_p95.item()
    assert bundle.total.item() == pytest.approx(
        (
            bundle.recon
            + bundle.codebook
            + spec.commitment_cost * bundle.commitment
            + bundle.kl
        ).item()
    )


@pytest.mark.skipif(not torch_available, reason="torch optional dependency is not installed")
def test_objective_terms_and_weights_compose_total_loss() -> None:
    import torch

    from qooi.ai import vq_rssm

    torch.manual_seed(18)
    features = torch.tensor(
        [[[0.1, 0.2], [0.2, 0.3], [0.3, 0.4]]], dtype=torch.float32
    )
    base = vq_rssm.VqRssmSpec(input_dim=2, hidden_dim=8, latent_dim=4, num_codes=8)
    weighted = vq_rssm.VqRssmSpec(
        input_dim=2,
        hidden_dim=8,
        latent_dim=4,
        num_codes=8,
        objective_terms=("reconstruct", "vq"),
        reconstruct_weight=0.5,
        vq_weight=2.0,
        kl_weight=0.0,
    )
    model = vq_rssm.VqRssmModel(weighted)
    model.load_state_dict(vq_rssm.VqRssmModel(base).state_dict())

    bundle = model.loss_bundle(features, weighted.kl_anneal_steps)

    assert bundle.total.item() == pytest.approx(
        (
            bundle.recon * 0.5
            + (bundle.codebook + weighted.commitment_cost * bundle.commitment) * 2.0
        ).item()
    )


@pytest.mark.skipif(not torch_available, reason="torch optional dependency is not installed")
def test_window_and_sequence_loss_match_when_hidden_zero() -> None:
    import torch

    from qooi.ai import vq_rssm

    torch.manual_seed(19)
    spec = vq_rssm.VqRssmSpec(input_dim=2, hidden_dim=8, latent_dim=4, num_codes=8)
    model = vq_rssm.VqRssmModel(spec)
    features = torch.tensor(
        [[[0.1, 0.2], [0.2, 0.3], [0.3, 0.4]]], dtype=torch.float32
    )

    window_bundle = model.loss_bundle(features, spec.kl_anneal_steps)
    zero_h = torch.zeros((1, spec.hidden_dim))
    sequence_bundle = model.loss_bundle(features.squeeze(0), spec.kl_anneal_steps, h=zero_h)
    sequence_output = model.sequence_outputs(features.squeeze(0), h=zero_h)

    assert sequence_bundle.total.item() == pytest.approx(window_bundle.total.item())
    assert sequence_bundle.recon.item() == pytest.approx(window_bundle.recon.item())
    assert sequence_bundle.codebook.item() == pytest.approx(window_bundle.codebook.item())
    assert sequence_bundle.commitment.item() == pytest.approx(window_bundle.commitment.item())
    assert sequence_bundle.kl.item() == pytest.approx(window_bundle.kl.item())
    assert torch.allclose(sequence_output["final_hidden"], window_bundle.next_hidden)


@pytest.mark.skipif(not torch_available, reason="torch optional dependency is not installed")
def test_kl_weight_affects_only_kl_component() -> None:
    import torch

    from qooi.ai import vq_rssm

    torch.manual_seed(23)
    spec = vq_rssm.VqRssmSpec(input_dim=2, hidden_dim=8, latent_dim=4, num_codes=8)
    model = vq_rssm.VqRssmModel(spec)
    features = torch.tensor(
        [[[0.1, 0.2], [0.2, 0.3], [0.3, 0.4]]], dtype=torch.float32
    )

    no_kl = model.loss_bundle(features, 0)
    full_kl = model.loss_bundle(features, spec.kl_anneal_steps)

    assert full_kl.recon.item() == pytest.approx(no_kl.recon.item())
    assert full_kl.codebook.item() == pytest.approx(no_kl.codebook.item())
    assert full_kl.commitment.item() == pytest.approx(no_kl.commitment.item())
    assert no_kl.kl.item() == pytest.approx(0.0)
    assert full_kl.kl.item() > no_kl.kl.item()
    assert (full_kl.total - no_kl.total).item() == pytest.approx(
        (full_kl.kl - no_kl.kl).item()
    )


@pytest.mark.skipif(not torch_available, reason="torch optional dependency is not installed")
def test_parallel_sequence_batches_preserve_sequence_hidden_slots() -> None:
    import torch

    from qooi.ai import vq_rssm

    tensors = (
        torch.ones((5, 2)),
        torch.full((4, 2), 2.0),
        torch.full((2, 2), 3.0),
    )
    hidden_by_sequence = [
        torch.full((1, 3), 10.0),
        None,
        torch.full((1, 3), 30.0),
    ]
    cfg = vq_rssm.SequenceRuntimeConfig(chunk=2, warmup=1, stride=1, carry=True)

    batches = list(
        vq_rssm._parallel_sequence_batches(
            tensors, hidden_by_sequence, 0, cfg, batch_size=3, hidden_dim=3
        )
    )

    assert len(batches) == 1
    sequence_indices, batch, h = batches[0]
    assert sequence_indices == (0, 1, 2)
    assert batch.shape == (3, 2, 2)
    assert h is not None
    assert h.tolist() == [[10.0, 10.0, 10.0], [0.0, 0.0, 0.0], [30.0, 30.0, 30.0]]


@pytest.mark.skipif(not torch_available, reason="torch optional dependency is not installed")
def test_batched_sequence_loss_equals_unbatched_loss_without_optimizer_step() -> None:
    import torch

    from qooi.ai import vq_rssm

    torch.manual_seed(29)
    spec = vq_rssm.VqRssmSpec(input_dim=2, hidden_dim=8, latent_dim=4, num_codes=8)
    model = vq_rssm.VqRssmModel(spec)
    batch = torch.tensor(
        [
            [[0.1, 0.2], [0.2, 0.3], [0.3, 0.4]],
            [[-0.1, -0.2], [-0.2, -0.3], [-0.3, -0.4]],
        ],
        dtype=torch.float32,
    )
    h = torch.randn((2, spec.hidden_dim))

    batched = model.loss_bundle(batch, spec.kl_anneal_steps, h=h)
    first = model.loss_bundle(batch[0], spec.kl_anneal_steps, h=h[0:1])
    second = model.loss_bundle(batch[1], spec.kl_anneal_steps, h=h[1:2])

    assert batched.total.item() == pytest.approx(
        ((first.total + second.total) / 2.0).item()
    )
    assert batched.recon.item() == pytest.approx(((first.recon + second.recon) / 2.0).item())
    assert batched.codebook.item() == pytest.approx(
        ((first.codebook + second.codebook) / 2.0).item()
    )
    assert batched.commitment.item() == pytest.approx(
        ((first.commitment + second.commitment) / 2.0).item()
    )
    assert batched.kl.item() == pytest.approx(((first.kl + second.kl) / 2.0).item())


@pytest.mark.skipif(not torch_available, reason="torch optional dependency is not installed")
def test_carry_false_resets_every_chunk() -> None:
    import torch

    from qooi.ai import vq_rssm

    tensors = (torch.ones((5, 2)), torch.full((5, 2), 2.0))
    hidden_by_sequence = [torch.full((1, 3), 10.0), torch.full((1, 3), 20.0)]
    cfg = vq_rssm.SequenceRuntimeConfig(chunk=2, warmup=1, stride=1, carry=False)

    batches = list(
        vq_rssm._parallel_sequence_batches(
            tensors, hidden_by_sequence, 2, cfg, batch_size=2, hidden_dim=3
        )
    )

    assert len(batches) == 1
    sequence_indices, batch, h = batches[0]
    assert sequence_indices == (0, 1)
    assert batch.shape == (2, 2, 2)
    assert h is None


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
        vq_rssm.train(dataset, vq_rssm.VqRssmSpec(input_dim=2), TrainingConfig())


@pytest.mark.skipif(not torch_available, reason="torch optional dependency is not installed")
def test_tiny_vq_rssm_train_and_predict_sequences(tmp_path) -> None:
    from qooi.ai import vq_rssm

    dataset = SequenceDataset(
        sequences=(
            AssetFeatureSequence(
                symbol="BTC",
                split="train",
                features=((0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5)),
                row_index=(0, 1, 2, 3),
                timestamps=(0, 1, 2, 3),
            ),
            AssetFeatureSequence(
                symbol="BTC",
                split="valid",
                features=((-0.1, -0.2), (-0.2, -0.3), (-0.3, -0.4)),
                row_index=(4, 5, 6),
                timestamps=(4, 5, 6),
            ),
            AssetFeatureSequence(
                symbol="BTC",
                split="test",
                features=((0.5, 0.4), (0.6, 0.5), (0.7, 0.6)),
                row_index=(7, 8, 9),
                timestamps=(7, 8, 9),
            ),
        ),
        feature_columns=("a", "b"),
    )
    spec = vq_rssm.VqRssmSpec(
        input_dim=2,
        hidden_dim=8,
        latent_dim=4,
        num_codes=8,
        kl_anneal_steps=2,
    )
    seq_cfg = vq_rssm.SequenceRuntimeConfig(chunk=2, warmup=2, stride=1)

    result = vq_rssm.train_sequences(
        dataset,
        spec,
        TrainingConfig(epochs=1, batch=2, lr=1e-3, grad_clip=1.0, seed=1),
        seq_cfg,
        output_dir=tmp_path,
    )
    codes = vq_rssm.predict_sequence_codes(dataset, result.checkpoint, seq_cfg)
    diagnostics = vq_rssm.predict_sequence_diagnostics(dataset, result.checkpoint, seq_cfg)

    assert result.summary is not None
    assert result.summary.train_rows == 4
    assert result.summary.valid_rows == 3
    assert result.summary.test_rows == 3
    assert result.summary.valid_loss is not None
    assert result.summary.train_recon_loss is not None
    assert result.summary.valid_recon_loss is not None
    assert result.summary.train_vq_distance_mean is not None
    assert result.summary.valid_vq_distance_mean is not None
    assert result.summary.codebook_norm_mean is not None
    metrics_text = (tmp_path / "behavior-state-training-metrics.csv").read_text()
    assert "train_vq_distance_mean" in metrics_text
    assert "codebook_norm_mean" in metrics_text
    assert len(codes.codes) == 7
    assert codes.row_index == tuple(range(7))
    assert codes.splits == ("train", "train", "train", "valid", "valid", "test", "test")
    assert len(diagnostics.hidden_states) == 7
    assert len(diagnostics.reconstructions[0]) == 2


@pytest.mark.skipif(not torch_available, reason="torch optional dependency is not installed")
def test_sequence_inference_resets_between_asset_split_sequences() -> None:
    from qooi.ai import vq_rssm

    spec = vq_rssm.VqRssmSpec(input_dim=2, hidden_dim=8, latent_dim=4, num_codes=8)
    model = vq_rssm.VqRssmModel(spec)
    checkpoint = vq_rssm.VqRssmCheckpoint(spec=spec, path=None, state_dict=model.state_dict())
    valid = AssetFeatureSequence(
        symbol="BTC",
        split="valid",
        features=((0.1, 0.2), (0.2, 0.3), (0.3, 0.4)),
        row_index=(10, 11, 12),
        timestamps=(10, 11, 12),
    )
    base_train = AssetFeatureSequence(
        symbol="BTC",
        split="train",
        features=((9.0, 9.0), (8.0, 8.0), (7.0, 7.0)),
        row_index=(0, 1, 2),
        timestamps=(0, 1, 2),
    )
    changed_train = AssetFeatureSequence(
        symbol="BTC",
        split="train",
        features=((-9.0, -9.0), (-8.0, -8.0), (-7.0, -7.0)),
        row_index=(0, 1, 2),
        timestamps=(0, 1, 2),
    )
    seq_cfg = vq_rssm.SequenceRuntimeConfig(chunk=2, warmup=1, stride=1)

    base_codes = vq_rssm.predict_sequence_codes(
        SequenceDataset((base_train, valid), ("a", "b")), checkpoint, seq_cfg
    )
    changed_codes = vq_rssm.predict_sequence_codes(
        SequenceDataset((changed_train, valid), ("a", "b")), checkpoint, seq_cfg
    )

    assert base_codes.codes[3:] == changed_codes.codes[3:]
    assert base_codes.distances[3:] == changed_codes.distances[3:]


@pytest.mark.skipif(not torch_available, reason="torch optional dependency is not installed")
def test_sequence_inference_is_chunk_size_invariant_when_carrying_hidden() -> None:
    from qooi.ai import vq_rssm

    spec = vq_rssm.VqRssmSpec(input_dim=2, hidden_dim=8, latent_dim=4, num_codes=8)
    model = vq_rssm.VqRssmModel(spec)
    checkpoint = vq_rssm.VqRssmCheckpoint(spec=spec, path=None, state_dict=model.state_dict())
    dataset = SequenceDataset(
        (
            AssetFeatureSequence(
                symbol="BTC",
                split="train",
                features=((0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5), (0.5, 0.6)),
                row_index=(0, 1, 2, 3, 4),
                timestamps=(0, 1, 2, 3, 4),
            ),
        ),
        ("a", "b"),
    )

    chunk_two = vq_rssm.predict_sequence_codes(
        dataset, checkpoint, vq_rssm.SequenceRuntimeConfig(chunk=2, warmup=2, stride=1)
    )
    chunk_five = vq_rssm.predict_sequence_codes(
        dataset, checkpoint, vq_rssm.SequenceRuntimeConfig(chunk=5, warmup=2, stride=1)
    )

    assert chunk_two.codes == chunk_five.codes
    assert chunk_two.distances == pytest.approx(chunk_five.distances)


@pytest.mark.skipif(not torch_available, reason="torch optional dependency is not installed")
def test_sequence_training_remains_finite_with_large_scaled_inputs() -> None:
    from qooi.ai import vq_rssm

    sequences = tuple(
        AssetFeatureSequence(
            symbol=f"S{index}",
            split="train" if index < 3 else "valid",
            features=tuple(
                (float(step * 25), float(-step * 30)) for step in range(1, 9)
            ),
            row_index=tuple(range(index * 10, index * 10 + 8)),
            timestamps=tuple(range(index * 10, index * 10 + 8)),
        )
        for index in range(4)
    )
    dataset = SequenceDataset(sequences, ("a", "b"))
    spec = vq_rssm.VqRssmSpec(
        input_dim=2,
        hidden_dim=8,
        latent_dim=4,
        num_codes=8,
        kl_anneal_steps=2,
    )

    result = vq_rssm.train_sequences(
        dataset,
        spec,
        TrainingConfig(epochs=1, batch=4, lr=1e-3, grad_clip=1.0, seed=1),
        vq_rssm.SequenceRuntimeConfig(chunk=4, warmup=2, stride=1),
    )

    assert result.summary is not None
    assert result.summary.train_loss == pytest.approx(result.summary.train_loss)
    assert result.summary.valid_loss == pytest.approx(result.summary.valid_loss)


def test_sequence_preparation_warmup_and_provenance_alignment() -> None:
    import polars as pl

    from qooi.research.states import LearnedStateConfig, SequenceConfig, WindowConfig

    frame = pl.DataFrame(
        {
            "timestamp": [100, 200, 300, 400, 500, 600],
            "symbol": ["BTC"] * 6,
            "open": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
            "high": [11.0, 12.0, 13.0, 14.0, 15.0, 16.0],
            "low": [9.0, 10.0, 11.0, 12.0, 13.0, 14.0],
            "close": [10.5, 11.5, 12.5, 13.5, 14.5, 15.5],
            "vol": [100.0, 110.0, 120.0, 130.0, 140.0, 150.0],
        }
    )
    config = LearnedStateConfig(
        input="sequence",
        win=WindowConfig(len=3, stride=1, train=0.5, valid=0.25),
        seq=SequenceConfig(chunk=2, warmup=2, stride=1),
    )

    prepared = config.prepare(frame)

    assert len(prepared.sequence_dataset.sequences) == 3
    assert prepared.sequence_provenance.symbols == ("BTC", "BTC")
    assert prepared.sequence_provenance.timestamps == (300, 600)
    assert prepared.sequence_provenance.splits == ("train", "test")
