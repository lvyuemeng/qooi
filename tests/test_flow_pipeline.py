"""Signal pipeline unit tests — OFI flow, regime features, gates."""

import polars as pl
import pytest

from qooi.strategies.indicators import (
    add_indicators,
    add_ofi_flow_columns,
    add_regime_features,
    apply_regime_gate,
    compute_flow_pipeline_frame,
)

# ── fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def sample_df() -> pl.DataFrame:
    """100 bars of synthetic OHLCV with ATR."""
    import math

    n = 100
    ts = list(range(1_700_000_000_000, 1_700_000_000_000 + n * 14_400_000, 14_400_000))
    rng = 42
    closes = [100.0 + sum(math.sin(i / 10 + rng) * 5 for _ in range(1)) for i in range(n)]
    closes = [
        closes[i - 1] + math.sin(i / 8) * 3 + (1 if math.sin(i / 20) > 0 else -1) * 1.5
        for i in range(n)
    ]
    closes[0] = 100.0
    for i in range(1, n):
        closes[i] = closes[i - 1] + math.sin(i / 8) * 3 + (1 if math.sin(i / 20) > 0 else -1) * 1.5

    return pl.DataFrame(
        {
            "timestamp": ts[:n],
            "open": [c - 2 for c in closes],
            "high": [c + 4 for c in closes],
            "low": [c - 4 for c in closes],
            "close": closes,
            "vol": [500 + abs(math.sin(i / 5)) * 2000 for i in range(n)],
            "atr_14": [abs(math.sin(i / 7)) * 2 + 0.5 for i in range(n)],
        }
    )


# ── OFI flow ────────────────────────────────────────────────────────────────


class TestOfiFlow:
    def test_basic_computation(self, sample_df):
        df = add_ofi_flow_columns(sample_df)
        assert "ofi_flow_score" in df.columns
        assert "ofi_signed_vol" in df.columns
        assert "ofi_net_flow" in df.columns
        # Score should be in [-1, 1]
        assert df["ofi_flow_score"].min() >= -1.0
        assert df["ofi_flow_score"].max() <= 1.0

    def test_scale_invariance(self):
        """Scaling all vols by 10x should not change flow_score."""
        import math

        n = 50
        ts = list(range(1_700_000_000_000, 1_700_000_000_000 + n * 14_400_000, 14_400_000))
        closes = [100.0 + math.sin(i / 10) * 5 for i in range(n)]
        base = pl.DataFrame(
            {
                "timestamp": ts,
                "open": [c - 1 for c in closes],
                "high": [c + 2 for c in closes],
                "low": [c - 2 for c in closes],
                "close": closes,
                "vol": [500.0 for _ in range(n)],
                "atr_14": [2.0 for _ in range(n)],
            }
        )
        scaled = base.with_columns(pl.col("vol") * 10)

        df1 = add_ofi_flow_columns(base)
        df2 = add_ofi_flow_columns(scaled)

        # Flow scores should be identical (volume-fraction normalization)
        tail1 = df1["ofi_flow_score"][-20:].to_list()
        tail2 = df2["ofi_flow_score"][-20:].to_list()
        for a, b in zip(tail1, tail2):
            assert abs(a - b) < 0.001, f"{a} != {b}"

    def test_volume_gate_dead_bars(self):
        """Near-zero volume bars should produce zero OFI score."""
        n = 30
        ts = list(range(1_700_000_000_000, 1_700_000_000_000 + n * 14_400_000, 14_400_000))
        df = pl.DataFrame(
            {
                "timestamp": ts,
                "open": [100.0] * n,
                "high": [101.0] * n,
                "low": [99.0] * n,
                "close": [101.0 if i % 3 == 0 else 99.0 for i in range(n)],
                "vol": [1e-10] * n,  # near-zero volume
                "atr_14": [2.0] * n,
            }
        )
        df = add_ofi_flow_columns(df)
        scores = df["ofi_flow_score"].drop_nulls().to_list()
        # All valid scores should be near 0 (gated)
        for s in scores:
            assert abs(s) < 0.01, f"Expected ~0, got {s}"


# ── regime features ─────────────────────────────────────────────────────────


class TestRegimeFeatures:
    def test_basic(self, sample_df):
        df = add_regime_features(sample_df)
        for col in (
            "regime_score",
            "regime_strength",
            "regime_mom_fast",
            "regime_mom_mid",
            "regime_mom_slow",
            "regime_vol_conf",
        ):
            assert col in df.columns
            assert df[col].drop_nulls().len() > 0

    def test_regime_clipped(self, sample_df):
        df = add_regime_features(sample_df)
        assert df["regime_score"].min() >= -1.0
        assert df["regime_score"].max() <= 1.0
        assert df["regime_mom_fast"].min() >= -1.0
        assert df["regime_mom_fast"].max() <= 1.0

    def test_regime_gate(self, sample_df):
        """Regime gate zeros signal when regime is strong."""
        df = add_regime_features(sample_df)
        # Add synthetic signal
        df = df.with_columns(pl.lit(0.8).alias("signal"))
        df = apply_regime_gate(df, max_regime=0.5)
        # Where regime_score > 0.5, signal should be 0
        gated = df.filter(pl.col("regime_score").abs() > 0.5)
        assert gated["signal"].sum() == 0.0, "Signal not gated in strong regime"

    def test_empty_df(self):
        df = pl.DataFrame()
        df = add_regime_features(df)
        df = add_ofi_flow_columns(df)
        assert df.is_empty()


def test_flow_pipeline_matches_manual_composition_on_synthetic_data(sample_df):
    manual = add_indicators(sample_df.clone())
    manual = add_regime_features(manual)
    manual = add_ofi_flow_columns(manual)
    manual = apply_regime_gate(manual, signal_col="ofi_flow_score")
    ofi = pl.col("ofi_flow_score")
    manual = manual.with_columns(
        pl.when(ofi.abs() >= 0.25).then(ofi).otherwise(0.0).alias("signal")
    )

    computed = compute_flow_pipeline_frame(sample_df.clone(), 0.25)

    assert computed["signal"].to_list() == manual["signal"].to_list()
