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
        window={"seq_len": 2, "stride": 1},
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
        volatility_scaling={"enabled": True, "floor": 0.001, "cap": 1.0},
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


def test_method_flow_preserves_window_alignment_and_strips_provenance() -> None:
    config = states.LearnedStateConfig(
        window={"seq_len": 3, "stride": 1, "train_pct": 0.6, "valid_pct": 0.2}
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
        window={"seq_len": 2, "stride": 1, "train_pct": 0.4, "valid_pct": 0.2}
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
        window={"seq_len": 3, "stride": 1, "train_pct": 0.6, "valid_pct": 0.2}
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


def test_no_public_build_function_api() -> None:
    assert not hasattr(states, "build_features")
    assert not hasattr(states, "build_windows")
    assert not hasattr(states, "sequence_from_codes")
