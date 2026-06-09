"""BacktestExecutor boundary integration tests."""

import polars as pl
import pytest

from qooi.core.basket import Basket, BasketState, ExitConfig
from qooi.core.config import CORE_UNIVERSE, AssetConfig, PairConfig
from qooi.core.executor import BacktestExecutor
from qooi.core.recovery import GridRecovery, HedgeRecovery, MartingaleRecovery


def _load(sig_symbol: str) -> pl.DataFrame:
    df = pl.read_parquet(f"data/cache/{sig_symbol.replace('-', '_')}_1H.parquet")
    if "vol" not in df.columns and "volume" in df.columns:
        df = df.rename({"volume": "vol"})
    return df


def _pair(
    *,
    capital: float = 500.0,
    max_risk_pct: float = 0.01,
    leverage: float = 10.0,
    max_notional_pct_per_basket: float | None = 1.0,
    ct_val: float = 0.1,
    min_contracts: float = 0.01,
    lot_size: float = 0.01,
) -> PairConfig:
    asset_kwargs = {
        "symbol": "TEST-USDT-SWAP",
        "sig_symbol": "TEST-USDT",
        "capital": capital,
        "max_risk_pct": max_risk_pct,
        "leverage": leverage,
        "ct_val": ct_val,
        "min_contracts": min_contracts,
        "lot_size": lot_size,
    }
    if max_notional_pct_per_basket is not None:
        asset_kwargs["max_notional_pct_per_basket"] = max_notional_pct_per_basket
    return PairConfig(asset=AssetConfig(**asset_kwargs))


def _tiny_unsizeable_pair() -> PairConfig:
    return _pair(
        capital=1.0,
        max_risk_pct=0.001,
        leverage=1.0,
        max_notional_pct_per_basket=None,
        ct_val=1.0,
        min_contracts=1.0,
        lot_size=1.0,
    )


def _seed_basket() -> Basket:
    return Basket(
        basket_id="seed",
        symbol="TEST-USDT-SWAP",
        strategy="momentum_burst",
        side="buy",
        state=BasketState.ACTIVE,
        entry_px=100.0,
        current_sz=1.0,
        stop_px=98.0,
        target_px=103.0,
    )


def _signal_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "timestamp": list(range(8)),
            "open": [100.0] * 8,
            "high": [101.0] * 8,
            "low": [99.0] * 8,
            "close": [100.0] * 8,
            "vol": [1000.0] * 8,
            "atr_14": [1.0] * 8,
            "signal": [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
            "position_signal": [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
            "entry_signal": [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
            "exit_signal": [False] * 8,
            "signal_strength": [1.0] * 8,
            "signal_id": ["none", "entry", "entry", "entry", "entry", "entry", "none", "none"],
        }
    )


def _mechanics_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "timestamp": list(range(5)),
            "open": [100.0, 100.0, 98.5, 98.5, 98.5],
            "high": [100.0, 100.0, 98.5, 98.5, 98.5],
            "low": [100.0, 100.0, 98.5, 98.5, 98.5],
            "close": [100.0, 100.0, 98.5, 98.5, 98.5],
            "vol": [1000.0] * 5,
            "atr_14": [1.0] * 5,
            "signal": [0.0, 1.0, 1.0, 1.0, 0.0],
            "position_signal": [0.0, 1.0, 1.0, 1.0, 0.0],
            "entry_signal": [0.0, 1.0, 0.0, 0.0, 0.0],
            "exit_signal": [False] * 5,
            "signal_strength": [1.0] * 5,
            "signal_id": ["none", "entry", "hold", "hold", "none"],
        }
    )


def _cooldown_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "timestamp": list(range(6)),
            "open": [100.0] * 6,
            "high": [100.0] * 6,
            "low": [100.0, 100.0, 98.0, 100.0, 100.0, 100.0],
            "close": [100.0, 100.0, 98.0, 100.0, 100.0, 100.0],
            "vol": [1000.0] * 6,
            "atr_14": [1.0] * 6,
            "signal": [0.0, 1.0, 1.0, 1.0, 0.0, 0.0],
            "position_signal": [0.0, 1.0, 1.0, 1.0, 0.0, 0.0],
            "entry_signal": [0.0, 1.0, 0.0, 1.0, 0.0, 0.0],
            "exit_signal": [False] * 6,
            "signal_strength": [1.0] * 6,
            "signal_id": ["none", "entry", "hold", "entry", "none", "none"],
        }
    )


def _ambiguous_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "timestamp": list(range(4)),
            "open": [100.0] * 4,
            "high": [100.0, 100.0, 103.0, 100.0],
            "low": [100.0, 100.0, 98.0, 100.0],
            "close": [100.0] * 4,
            "vol": [1000.0] * 4,
            "atr_14": [1.0] * 4,
            "signal": [0.0, 1.0, 1.0, 0.0],
            "position_signal": [0.0, 1.0, 1.0, 0.0],
            "entry_signal": [0.0, 1.0, 0.0, 0.0],
            "exit_signal": [False] * 4,
            "signal_strength": [1.0] * 4,
            "signal_id": ["none", "entry", "hold", "none"],
        }
    )


def test_cached_data_run_report_preserves_trade_schema_and_pnl_direction():
    pair = CORE_UNIVERSE[0]
    report = BacktestExecutor(initial_capital=pair.asset.capital, cost_pct=0.0).run_report(
        _load(pair.asset.sig_symbol),
        pair,
    )

    assert report.metrics.num_trades > 0
    assert report.equity.height > 10
    assert float(report.equity["portfolio_value"][-1]) > 0
    assert float(report.equity["portfolio_value"][-1]) != pair.asset.capital
    expected_columns = {
        "pnl_usd",
        "bars_held",
        "entry_ts",
        "exit_ts",
        "entry_notional_usd",
        "notional_pct_capital",
        "pre_entry_total_notional_pct",
        "post_entry_total_notional_pct",
        "exit_family",
        "recovery_active_at_exit",
        "sizing_binding",
    }
    assert expected_columns <= set(report.trades.columns)
    assert any(abs(float(t["pnl_usd"])) > 0 for t in report.trades.iter_rows(named=True))
    for trade in report.trades.iter_rows(named=True):
        entry = float(trade["entry_px"])
        exit_px = float(trade["exit_px"])
        pnl = float(trade["pnl"])
        if trade["side"] == "buy":
            assert (exit_px > entry and pnl > 0) or (exit_px <= entry and pnl <= 0)
        else:
            assert (exit_px < entry and pnl > 0) or (exit_px >= entry and pnl <= 0)


def test_executor_populates_lifecycle_risk_and_drawdown_diagnostics():
    pair = CORE_UNIVERSE[0]
    report = BacktestExecutor(initial_capital=pair.asset.capital, cost_pct=0.0).run_report(
        _signal_frame(),
        pair,
        exit_cfg=ExitConfig(max_bars=2),
        precomputed_signal=True,
    )

    assert report.diagnostics is not None
    lifecycle = report.diagnostics.lifecycle
    risk = report.diagnostics.risk
    assert lifecycle.entry_actions >= 1
    assert lifecycle.exit_actions >= 1
    assert lifecycle.entry_signals >= lifecycle.entry_actions
    assert lifecycle.entry_acceptance_rate_pct > 0
    assert lifecycle.max_simultaneous_baskets >= 1
    assert lifecycle.blocked_entry_signals >= 0
    assert lifecycle.duplicate_entry_suppressed >= 0
    assert isinstance(lifecycle.blocked_entry_reasons, dict)
    assert risk.max_notional_exposure_pct >= 0
    assert {
        "entry_equity",
        "entry_drawdown_pct",
        "pre_entry_total_notional_pct",
        "post_entry_total_notional_pct",
        "exit_drawdown_pct_before",
    } <= set(report.trades.columns)
    table = report.diagnostics_table()
    assert "Basket Lifecycle Diagnostics" in table
    assert "EntryAccept" in table
    assert "Risk Control Diagnostics" in table
    assert "Drawdown Path Diagnostics" in table


@pytest.mark.parametrize(
    "recovery_cfg, action_field",
    [
        (GridRecovery(zone_atr=1.0, multiplier=0.5, max_levels=1), "grid_actions"),
        (HedgeRecovery(zone_atr=1.0, multiplier=1.0, max_levels=1), "hedge_actions"),
    ],
)
def test_recovery_actions_are_sized_allowed_and_diagnosed(recovery_cfg, action_field):
    pair = _pair()
    report = BacktestExecutor(initial_capital=pair.asset.capital, cost_pct=0.0).run_report(
        _mechanics_frame(),
        pair,
        exit_cfg=ExitConfig(max_bars=10),
        recovery_cfg=recovery_cfg,
        precomputed_signal=True,
    )

    assert report.diagnostics is not None
    risk = report.diagnostics.risk
    lifecycle = report.diagnostics.lifecycle
    assert risk.recovery_unsized_actions == 0
    assert risk.recovery_allowed_actions >= 1
    assert getattr(lifecycle, action_field) >= 1
    assert lifecycle.action_event_count >= 1
    table = report.diagnostics_table()
    assert "Recovery Mechanics Diagnostics" in table
    assert "UnsizedActions=0" in table


def test_grid_recovery_blocked_by_sizing_reports_concrete_reason():
    pair = _tiny_unsizeable_pair()
    executor = BacktestExecutor(
        initial_capital=pair.asset.capital,
        cost_pct=0.0,
        close_open_positions=False,
    )
    executor.run(
        _mechanics_frame(),
        pair,
        exit_cfg=ExitConfig(max_bars=10),
        recovery_cfg=GridRecovery(zone_atr=1.0, multiplier=1.0, max_levels=1),
        initial_baskets=[_seed_basket()],
        precomputed_signal=True,
    )

    risk = executor._last_diagnostics.risk
    assert risk.recovery_unsized_actions == 0
    assert risk.recovery_allowed_actions == 0
    assert risk.recovery_blocked_reasons["below_min_contracts_1"] >= 1


def test_martingale_recovery_group_is_blocked_atomically_when_reversal_sizing_fails():
    pair = _tiny_unsizeable_pair()
    executor = BacktestExecutor(
        initial_capital=pair.asset.capital,
        cost_pct=0.0,
        close_open_positions=False,
    )
    executor.run(
        _mechanics_frame(),
        pair,
        exit_cfg=ExitConfig(max_bars=10),
        recovery_cfg=MartingaleRecovery(zone_atr=1.0, max_levels=1),
        initial_baskets=[_seed_basket()],
        precomputed_signal=True,
    )

    risk = executor._last_diagnostics.risk
    lifecycle = executor._last_diagnostics.lifecycle
    assert lifecycle.recovery_actions == 0
    assert lifecycle.exit_actions == 0
    assert risk.recovery_unsized_actions == 0
    assert risk.recovery_blocked_reasons["below_min_contracts_1"] >= 1
    assert risk.recovery_blocked_reasons["paired_below_min_contracts_1"] >= 1
    assert risk.recovery_allowed_actions == 0


def test_same_bar_terminal_entry_block_reports_sizing_not_duplicate():
    pair = _pair(
        capital=10.0,
        leverage=1.0,
        max_notional_pct_per_basket=None,
        ct_val=1.0,
        min_contracts=1.0,
    )
    frame = pl.DataFrame(
        {
            "timestamp": [0, 1],
            "open": [100.0, 100.0],
            "high": [100.0, 100.0],
            "low": [100.0, 100.0],
            "close": [100.0, 100.0],
            "vol": [1000.0, 1000.0],
            "atr_14": [1.0, 1.0],
            "signal": [1.0, 1.0],
            "position_signal": [1.0, 1.0],
            "entry_signal": [0.0, 1.0],
            "exit_signal": [False, True],
            "signal_strength": [1.0, 1.0],
            "signal_id": ["seed", "same_bar_entry"],
        }
    )
    executor = BacktestExecutor(initial_capital=pair.asset.capital, cost_pct=0.0)

    executor.run(frame, pair, initial_baskets=[_seed_basket()], precomputed_signal=True)

    diagnostics = executor._last_diagnostics
    assert diagnostics.lifecycle.sizing_blocked_entries == 1
    assert diagnostics.lifecycle.duplicate_entry_suppressed == 0
    assert diagnostics.lifecycle.blocked_entry_reasons["below_min_contracts_1"] == 1


def test_same_bar_stop_target_ambiguity_is_diagnosed():
    pair = CORE_UNIVERSE[0]
    report = BacktestExecutor(initial_capital=pair.asset.capital, cost_pct=0.0).run_report(
        _ambiguous_frame(),
        pair,
        exit_cfg=ExitConfig(max_bars=10),
        precomputed_signal=True,
    )

    assert report.diagnostics is not None
    assert report.diagnostics.risk.ambiguous_stop_target_count >= 1
    assert report.diagnostics.risk.target_first_counterfactual_net_pnl_usd > 0
    assert "ambiguous_stop_target_bar" in report.trades.columns
    assert bool(report.trades["ambiguous_stop_target_bar"].any()) is True
    assert "Intrabar Ambiguity Diagnostics" in report.diagnostics_table()
    assert "TargetFirstNet" in report.diagnostics_table()


def test_loss_cooldown_blocks_same_side_entries_after_loss():
    pair = CORE_UNIVERSE[0]
    report = BacktestExecutor(
        initial_capital=pair.asset.capital,
        cost_pct=0.0,
        loss_cooldown_bars=3,
    ).run_report(
        _cooldown_frame(),
        pair,
        exit_cfg=ExitConfig(max_bars=10),
        precomputed_signal=True,
    )

    assert report.diagnostics is not None
    lifecycle = report.diagnostics.lifecycle
    assert lifecycle.entry_signals == 2
    assert lifecycle.entry_actions == 1
    assert lifecycle.blocked_entry_reasons["loss_cooldown_buy"] == 1
