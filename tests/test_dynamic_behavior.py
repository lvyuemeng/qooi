from __future__ import annotations

import polars as pl
import pytest

from qooi.ai.contracts import CodeSequence
from qooi.research import states


def _frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "timestamp": [1, 2, 3, 4, 5, 6],
            "symbol": ["BTC-USDT-SWAP"] * 6,
            "open": [100.0, 110.0, 120.0, 130.0, 140.0, 150.0],
            "high": [110.0, 121.0, 132.0, 143.0, 154.0, 165.0],
            "low": [90.0, 99.0, 108.0, 117.0, 126.0, 135.0],
            "close": [100.0, 110.0, 120.0, 130.0, 140.0, 150.0],
            "vol": [10.0, 20.0, 40.0, 80.0, 160.0, 320.0],
            "liquidity_event_type": ["none"] * 6,
        }
    )


def _frame_for_symbol(symbol: str, multiplier: float = 1.0) -> pl.DataFrame:
    base = _frame()
    return base.with_columns(
        pl.lit(symbol).alias("symbol"),
        (pl.col("open") * multiplier).alias("open"),
        (pl.col("high") * multiplier).alias("high"),
        (pl.col("low") * multiplier).alias("low"),
        (pl.col("close") * multiplier).alias("close"),
    )


def test_config_features_use_previous_close_and_volume() -> None:
    config = states.LearnedStateConfig()
    features = config.window.features(_frame(), config.columns)

    assert features.height == 5
    first = features.row(0, named=True)
    assert first["timestamp"] == 2
    assert first["open_rel"] == pytest.approx(0.10)
    assert first["high_rel"] == pytest.approx(0.21)
    assert first["low_rel"] == pytest.approx(-0.01)
    assert first["close_rel"] == pytest.approx(0.10)
    assert first["volume_log_rel"] == pytest.approx(0.69314718056)


def test_learn_run_phases_are_ordered_and_validated() -> None:
    config = states.LearnedStateConfig(run={"phases": ["predict", "evaluate", "predict"]})

    assert config.run.phases == ("predict", "evaluate")
    assert config.run_checkpoint_path() == config.checkpoint_path()
    assert config.run_states_path() == config.out / "behavior-state-sequence.csv"
    with pytest.raises(ValueError, match="learn.run.phases"):
        states.LearnedStateConfig(run={"phases": []})
    with pytest.raises(ValueError):
        states.LearnedStateConfig(run={"phases": ["train", "unknown"]})


def test_legacy_learn_predict_false_maps_to_train_only() -> None:
    config = states.LearnedStateConfig(predict=False)

    assert config.run.phases == ("train",)


def test_learned_objective_terms_are_named_and_weighted() -> None:
    config = states.LearnedStateConfig(
        objective={
            "terms": ["reconstruct", "vq", "kl", "vq"],
            "reconstruct": 0.5,
            "vq": 2.0,
            "kl": 0.25,
            "future": 0.0,
        }
    )

    assert config.objective.terms == ("reconstruct", "vq", "kl")
    assert config.objective.reconstruct == pytest.approx(0.5)
    assert config.objective.vq == pytest.approx(2.0)
    assert config.objective.kl == pytest.approx(0.25)
    with pytest.raises(ValueError):
        states.LearnedStateConfig(objective={"terms": ["unknown"]})
    with pytest.raises(ValueError, match="objective vq weight"):
        states.LearnedStateConfig(objective={"vq": -1.0})


def test_custom_columns_and_required_columns() -> None:
    config = states.LearnedStateConfig(
        columns={
            "open": "o",
            "high": "h",
            "low": "l",
            "close": "c",
            "volume": "v",
            "timestamp": "ts",
            "symbol": "sym",
        },
        win={"len": 2, "stride": 1},
    )
    frame = _frame().rename(
        {
            "open": "o",
            "high": "h",
            "low": "l",
            "close": "c",
            "vol": "v",
            "timestamp": "ts",
            "symbol": "sym",
        }
    )

    features = config.window.features(frame, config.columns)

    assert config.required_columns() == ("o", "h", "l", "c", "v")
    assert features.row(0, named=True)["open_rel"] == pytest.approx(0.10)


def test_causal_volatility_scaling_is_per_symbol() -> None:
    config = states.LearnedStateConfig(
        scale={"on": True, "floor": 0.001, "cap": 1.0},
    )
    frame = pl.concat(
        [
            _frame_for_symbol("BTC"),
            pl.DataFrame(
                {
                    "timestamp": [1, 2, 3, 4, 5, 6],
                    "symbol": ["ETH"] * 6,
                    "open": [100.0, 105.0, 95.0, 115.0, 90.0, 130.0],
                    "high": [101.0, 107.0, 98.0, 118.0, 92.0, 133.0],
                    "low": [99.0, 103.0, 93.0, 112.0, 88.0, 126.0],
                    "close": [100.0, 105.0, 95.0, 115.0, 90.0, 130.0],
                    "vol": [10.0, 20.0, 10.0, 40.0, 10.0, 80.0],
                    "liquidity_event_type": ["none"] * 6,
                }
            ),
        ],
        how="diagonal_relaxed",
    )

    features = config.window.features(frame, config.columns, config.volatility_scaling)
    first_by_symbol = features.group_by("symbol", maintain_order=True).first()

    btc_first_scale = first_by_symbol.filter(pl.col("symbol") == "BTC").select(
        "volatility_scale"
    ).item()
    eth_first_scale = first_by_symbol.filter(pl.col("symbol") == "ETH").select(
        "volatility_scale"
    ).item()
    btc_max_scale = features.filter(pl.col("symbol") == "BTC").select(
        "volatility_scale"
    ).max().item()
    eth_max_scale = features.filter(pl.col("symbol") == "ETH").select(
        "volatility_scale"
    ).max().item()

    assert btc_first_scale == pytest.approx(0.001)
    assert eth_first_scale == pytest.approx(0.001)
    assert eth_max_scale > btc_max_scale


def test_sequence_warmup_resets_per_split_not_per_asset_only() -> None:
    config = states.LearnedStateConfig()
    frame = pl.DataFrame(
        {
            "timestamp": list(range(9)),
            "symbol": ["BTC"] * 9,
            "open_rel": [0.1] * 9,
            "high_rel": [0.2] * 9,
            "low_rel": [-0.1] * 9,
            "close_rel": [0.05] * 9,
            "volume_log_rel": [0.0] * 9,
            "split": ["train"] * 3 + ["valid"] * 3 + ["test"] * 3,
            "volatility_scale": [0.01] * 9,
        }
    )

    sequences, provenance = config.window.sequences(
        frame,
        config.columns,
        states.SequenceConfig(chunk=2, warmup=2, stride=1),
        config.feature_columns,
    )

    assert tuple(sequence.split for sequence in sequences.sequences) == ("train", "valid", "test")
    assert tuple(len(sequence.features) for sequence in sequences.sequences) == (3, 3, 3)
    assert provenance.splits == ("train", "train", "valid", "valid", "test", "test")
    assert provenance.row_index == (1, 2, 4, 5, 7, 8)
    assert provenance.volatility_scale == pytest.approx((0.01,) * 6)


def test_sequence_features_are_finite_after_scaling() -> None:
    config = states.LearnedStateConfig(
        scale={"on": True, "floor": 1e-6, "cap": 1.0, "min_periods": 1},
        win={"len": 2, "stride": 1},
        seq={"chunk": 2, "warmup": 2},
    )
    frame = pl.DataFrame(
        {
            "timestamp": [1, 2, 3, 4, 5],
            "symbol": ["BTC"] * 5,
            "open": [100.0, 100.0001, 100.0002, 100.0001, 100.0003],
            "high": [100.1, 100.1001, 100.1002, 100.1001, 100.1003],
            "low": [99.9, 99.9001, 99.9002, 99.9001, 99.9003],
            "close": [100.0, 100.0001, 100.0002, 100.0001, 100.0003],
            "vol": [1.0, 10.0, 1.0, 100.0, 1.0],
        }
    )

    prepared = config.prepare(frame)
    feature_values = [
        value
        for sequence in prepared.sequence_dataset.sequences
        for row in sequence.features
        for value in row
    ]

    assert feature_values
    assert all(value == pytest.approx(value) for value in feature_values)
    assert all(
        len(sequence.features) == len(sequence.row_index)
        for sequence in prepared.sequence_dataset.sequences
    )
    assert len(prepared.sequence_provenance.row_index) == len(prepared.sequence_provenance.splits)


def test_volume_log_rel_not_scaled_is_explicit() -> None:
    unscaled = states.LearnedStateConfig(scale={"on": False})
    scaled = states.LearnedStateConfig(
        scale={"on": True, "floor": 0.001, "cap": 1.0, "min_periods": 1}
    )

    base_features = unscaled.window.features(_frame(), unscaled.columns)
    scaled_features = scaled.window.features(_frame(), scaled.columns, scaled.volatility_scaling)
    base_first = base_features.row(0, named=True)
    scaled_first = scaled_features.row(0, named=True)

    assert scaled_first["volume_log_rel"] == pytest.approx(base_first["volume_log_rel"])
    assert scaled_first["open_rel"] == pytest.approx(
        base_first["open_rel"] / scaled_first["volatility_scale"]
    )


def test_method_flow_preserves_window_alignment_and_strips_provenance() -> None:
    config = states.LearnedStateConfig(
        win={"len": 3, "stride": 1, "train": 0.6, "valid": 0.2}
    )
    features = config.window.features(_frame(), config.columns)
    split = config.window.split(features.height)
    split_frame = config.window.assign_split(features, split, config.columns)
    windows, provenance = config.window.windows(split_frame, config.columns, config.feature_columns)

    assert split == states.Split(train_end=3, valid_end=4)
    assert provenance.row_index == (3, 4, 5)
    assert provenance.timestamps == (4, 5, 6)
    assert windows.splits == ("train", "valid", "test")
    generic = windows.to_dataset()
    assert generic.feature_columns == states.LEARNED_STATE_FEATURE_COLUMNS
    assert not hasattr(generic, "row_index")


def test_prepare_many_splits_per_symbol_and_merges_provenance() -> None:
    config = states.LearnedStateConfig(
        win={"len": 2, "stride": 1, "train": 0.4, "valid": 0.2}
    )

    prepared = config.prepare_many(
        (_frame_for_symbol("BTC"), _frame_for_symbol("ETH", multiplier=10.0))
    )

    assert len(prepared.windows.features) == 8
    assert prepared.provenance.symbols == ("BTC", "BTC", "BTC", "BTC", "ETH", "ETH", "ETH", "ETH")
    assert prepared.windows.splits == ("train", "valid", "test", "test") * 2
    assert prepared.provenance.row_index == (2, 3, 4, 5, 2, 3, 4, 5)


def test_codes_to_states_and_research_frame() -> None:
    config = states.LearnedStateConfig(
        win={"len": 3, "stride": 1, "train": 0.6, "valid": 0.2}
    )
    prepared = config.prepare(_frame())
    codes = CodeSequence(
        codes=(7, 8, 9),
        distances=(0.1, 0.2, 0.3),
        row_index=(0, 1, 2),
        splits=prepared.windows.splits,
    )

    sequence = prepared.provenance.states_from_codes(codes)
    attached = sequence.attach_to(_frame())
    research = sequence.research_frame(attached, symbol="BTC", timeframe="1H")

    assert sequence.frame.select("behavior_state_id").to_series().to_list() == [7, 8, 9]
    assert attached.get_column("behavior_state_id").to_list() == [None, None, None, 7, 8, 9]
    assert set(research.get_column("state_source")) == {"vq_rssm"}
    assert set(research.get_column("state_column")) == {"behavior_state_id"}


def test_state_sequence_attach_is_symbol_safe() -> None:
    sequence = states.StateSequence(
        pl.DataFrame(
            {
                "row_index": [1, 1],
                "timestamp": [2, 2],
                "symbol": ["BTC", "ETH"],
                "split": ["train", "train"],
                "behavior_state_id": [7, 8],
                "code_distance": [0.1, 0.2],
            }
        )
    )
    market = pl.concat([_frame_for_symbol("BTC"), _frame_for_symbol("ETH", 10.0)])

    attached = sequence.attach_to(market)

    rows = attached.filter(pl.col("row_index") == 1).select("symbol", "behavior_state_id").rows()
    assert sorted(rows) == [("BTC", 7), ("ETH", 8)]


def test_state_sequence_event_frame_is_dense_and_symbol_safe() -> None:
    sequence = states.StateSequence(
        pl.DataFrame(
            {
                "row_index": [1, 3, 1],
                "timestamp": [2, 4, 2],
                "symbol": ["BTC", "BTC", "ETH"],
                "split": ["train", "test", "test"],
                "behavior_state_id": [7, 8, 9],
                "code_distance": [0.1, 0.2, 0.3],
            }
        )
    )
    market = pl.concat([_frame_for_symbol("BTC"), _frame_for_symbol("ETH", 10.0)])

    events = sequence.event_frame(market)

    assert events.height == 3
    assert events.get_column("behavior_state_id").null_count() == 0
    eth = events.filter(pl.col("symbol") == "ETH").row(0, named=True)
    assert eth["close"] == 1100.0
    assert eth["behavior_state_id"] == 9


def test_no_public_build_function_api() -> None:
    assert not hasattr(states, "build_features")
    assert not hasattr(states, "build_windows")
    assert not hasattr(states, "sequence_from_codes")


def test_sequence_runtime_config_maps_research_config_once() -> None:
    config = states.LearnedStateConfig(
        input="sequence",
        win={"len": 2, "stride": 1, "train": 0.5, "valid": 0.25},
        seq={"chunk": 3, "warmup": 2, "stride": 2, "carry": False},
    )

    prepared = config.prepare(_frame())
    runtime = prepared.sequence_runtime_config()

    assert runtime.chunk == 3
    assert runtime.warmup == 2
    assert runtime.stride == 2
    assert runtime.carry is False
