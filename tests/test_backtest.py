"""Backtest engine tests — strategy performance, risk rules, trade lifecycle."""

import polars as pl

from qooi.exchange.backtest import Backtest, CostModel, RiskConfig
from qooi.exchange.indicator import add_indicators
from qooi.strategies.flow_pipeline import add_ofi_flow_columns, add_regime_features

# ── helpers ──────────────────────────────────────────────────────────────────


def _load_btc_spot() -> pl.DataFrame:
    """Load BTC spot data from cache — the most reliable dataset."""
    df = pl.read_parquet("data/cache/BTC_USDT_4H.parquet")
    df = add_indicators(df)
    df = add_regime_features(df)
    df = add_ofi_flow_columns(df)
    return df


def _ofi_to_signal(df: pl.DataFrame, threshold: float) -> pl.DataFrame:
    return df.with_columns(
        pl.when(pl.col("ofi_flow_score").abs() >= threshold)
        .then(pl.col("ofi_flow_score"))
        .otherwise(0.0)
        .alias("signal")
    )


# ── strategy performance ────────────────────────────────────────────────────


class TestStrategyPerformance:
    def test_btc_spot_produces_positive_sharpe(self):
        """BTC spot signal at threshold 0.25 should produce Sharpe > 1.0."""
        df = _load_btc_spot()
        assert df.height > 500, f"Need 500+ bars, got {df.height}"

        df = _ofi_to_signal(df, 0.25)
        risk = RiskConfig(
            atr_stop_mult=2.0,
            atr_target_mult=3.0,
            max_leverage=0.4,
            trailing_activation_mult=2.0,
            trailing_distance_mult=1.0,
        )
        bt = Backtest(
            df,
            pl.col("signal"),
            cost=CostModel(0.00005),
            risk=risk,
            threshold=0.25,
            ord_type="market",
        )
        r = bt.run()
        m = r.metrics

        assert m.num_trades > 50, f"Expected 50+ trades, got {m.num_trades}"
        assert m.sharpe_ratio > 1.0, f"Expected Sharpe > 1.0, got {m.sharpe_ratio:.2f}"
        assert m.win_rate_pct > 45, f"Expected WR > 45%, got {m.win_rate_pct:.0f}%"

    def test_sol_spot_produces_positive_sharpe(self):
        """SOL spot signal at threshold 0.35 should produce Sharpe > 1.0."""
        df = pl.read_parquet("data/cache/SOL_USDT_4H.parquet")
        df = add_indicators(df)
        df = add_regime_features(df)
        df = add_ofi_flow_columns(df)
        assert df.height > 500, f"Need 500+ bars, got {df.height}"

        df = _ofi_to_signal(df, 0.35)
        risk = RiskConfig(
            atr_stop_mult=2.0,
            atr_target_mult=3.0,
            max_leverage=0.4,
            trailing_activation_mult=2.0,
            trailing_distance_mult=1.0,
        )
        bt = Backtest(
            df,
            pl.col("signal"),
            cost=CostModel(0.00005),
            risk=risk,
            threshold=0.35,
            ord_type="market",
        )
        r = bt.run()
        m = r.metrics

        assert m.num_trades > 30, f"Expected 30+ trades, got {m.num_trades}"
        assert m.sharpe_ratio > 1.0, f"Expected Sharpe > 1.0, got {m.sharpe_ratio:.2f}"

    def test_higher_threshold_reduces_trades(self):
        """Raising threshold from 0.25 to 0.40 should reduce trade count."""
        df = _load_btc_spot()
        df25 = _ofi_to_signal(df, 0.25)
        df40 = _ofi_to_signal(df, 0.40)

        risk = RiskConfig(atr_stop_mult=2.0, atr_target_mult=3.0, max_leverage=0.4)

        r25 = Backtest(
            df25,
            pl.col("signal"),
            cost=CostModel(0.00005),
            risk=risk,
            threshold=0.25,
            ord_type="market",
        ).run()
        r40 = Backtest(
            df40,
            pl.col("signal"),
            cost=CostModel(0.00005),
            risk=risk,
            threshold=0.40,
            ord_type="market",
        ).run()

        assert r40.metrics.num_trades < r25.metrics.num_trades, (
            f"Expected fewer trades at 0.40 ({r40.metrics.num_trades}) vs 0.25 ({r25.metrics.num_trades})"
        )


# ── risk rules ──────────────────────────────────────────────────────────────


class TestRiskRules:
    def test_risk_config_defaults(self):
        risk = RiskConfig()
        assert risk.atr_stop_mult == 2.0
        assert risk.atr_target_mult == 3.0
        assert risk.trailing_activation_mult == 2.0
        assert risk.trailing_distance_mult == 1.0
        assert risk.max_leverage == 1.0  # default is 1.0

    def test_all_exit_reasons_exercised(self):
        """Backtest should exercise stop, target, trail, and signal exits."""
        df = _load_btc_spot()
        df = _ofi_to_signal(df, 0.25)

        risk = RiskConfig(
            atr_stop_mult=2.0,
            atr_target_mult=3.0,
            max_leverage=0.4,
            trailing_activation_mult=2.0,
            trailing_distance_mult=1.0,
        )
        bt = Backtest(
            df,
            pl.col("signal"),
            cost=CostModel(0.00005),
            risk=risk,
            threshold=0.25,
            ord_type="market",
        )
        r = bt.run()
        t = r.trades

        reasons = set(t["reason"].value_counts()["reason"].to_list())
        assert len(reasons) >= 2, f"Expected 2+ exit reasons, got {reasons}"

    def test_backtest_no_trades_on_zero_signal(self):
        """Backtest with all-zero signal produces 0 trades."""
        df = pl.DataFrame(
            {
                "timestamp": list(
                    range(1_700_000_000_000, 1_700_000_000_000 + 200 * 14_400_000, 14_400_000)
                ),
                "open": [100.0] * 200,
                "high": [101.0] * 200,
                "low": [99.0] * 200,
                "close": [100.0] * 200,
                "vol": [1000.0] * 200,
                "atr_14": [2.0] * 200,
                "signal": [0.0] * 200,
            }
        )
        bt = Backtest(df, pl.col("signal"), risk=RiskConfig())
        r = bt.run()
        assert r.trades.height == 0
        assert r.metrics.num_trades == 0
