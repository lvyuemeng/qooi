"""Evaluation layer boundary tests."""

from qooi.core.config import CORE_UNIVERSE
from qooi.core.evaluate import (
    Report,
    compare,
    format_pair_attribution,
    format_strategy_recommendations,
    format_symbol_rankings,
)
from qooi.core.metrics import compute_metrics


def _trade(**overrides):
    trade = {
        "side": "buy",
        "entry_px": 100.0,
        "exit_px": 101.0,
        "pnl": 0.01,
        "pnl_usd": 1.0,
        "net_pnl_usd": 1.0,
        "reason": "strategy_exit",
    }
    trade.update(overrides)
    return trade


def _report(trades=None, equity=None, *, label=None, metadata=()):
    return Report.from_raw(
        trades or [],
        equity or [100.0, 100.0],
        CORE_UNIVERSE[0],
        label=label,
        metadata=metadata,
    )


def test_report_metric_contract_uses_trade_returns_and_core_metrics():
    trades = [
        _trade(reason="x"),
        _trade(exit_px=99.0, pnl=-0.005, pnl_usd=-0.5, net_pnl_usd=-0.5, reason="y"),
    ]
    report = _report(trades, [100.0, 101.0, 100.495])
    metrics = compute_metrics(report.equity, trades=report.trades)

    assert abs(report.metrics.avg_win_pct - 1.0) < 1e-6
    assert abs(report.metrics.avg_loss_pct - 0.5) < 1e-6
    assert report.trade_expectancy_pct > 0
    assert report.trade_expectancy_usd > 0
    assert report.trade_sharpe != 0
    assert report.metrics.total_return_pct == metrics.total_return_pct


def test_report_marks_unstable_annualization_and_active_exposure():
    sparse = _report([_trade(reason="x")], [100.0] + [100.0] * 50)
    active = Report.from_raw(
        [],
        [100.0, 100.5, 100.5, 101.0, 101.0],
        CORE_UNIVERSE[0],
        active_exposure=[0.0, 1.0, 0.0, 2.0, 0.0],
    )

    assert sparse.unstable_annualization is True
    assert active.active_bar_pct > 0
    assert active.active_bar_sharpe != 0 or active.active_bar_pct > 0


def test_report_and_compare_tables_expose_primary_metric_labels():
    report = _report(
        [
            _trade(exit_px=102.0, pnl=0.02, pnl_usd=2.0, net_pnl_usd=2.0, reason="x"),
            _trade(exit_px=99.0, pnl=-0.01, pnl_usd=-1.0, net_pnl_usd=-1.0, reason="y"),
        ],
        [100.0, 102.0, 100.98],
    )
    comparison = compare(_report(label="A"), _report(label="B"))

    assert "P/L=" in report.table()
    assert "PF=" in report.table()
    assert "Label" in comparison
    assert "PF" in comparison
    assert "PL" not in comparison
    assert "A" in comparison and "B" in comparison


def test_report_derives_entry_context_buckets_and_sections_from_trade_rows():
    report = _report(
        [
            _trade(
                entry_adx_14=42.0,
                entry_volatility_ratio=1.8,
                entry_trend_return=0.04,
                entry_close_z_score=-3.8,
                entry_liquidity_event_type="breakout_acceptance_high",
                entry_event_quality_score=2.7,
                entry_market_stage_reason="trend_without_range_break",
                entry_stage_unknown_reason="trend_without_range_break",
                entry_atr_percentile_100=92.0,
                entry_key_level_proximity_bucket="near_prior_high_no_breach",
                entry_z_pressure_side="short_pressure",
                entry_mtf_state_key="uptrend|markup|accumulation",
                entry_mtf_structure_key="uptrend|uptrend|range",
                entry_mtf_stage_key="markup|range|accumulation",
                entry_mtf_event_state_key="uptrend|markup|accumulation|failed_breakout_low",
            ),
            _trade(
                pnl=-0.01,
                pnl_usd=-1.0,
                net_pnl_usd=-1.0,
                reason="stop",
                entry_adx_14=15.0,
                entry_volatility_ratio=0.7,
                entry_trend_return=0.0,
                entry_close_z_score=2.1,
                entry_liquidity_event_type="none",
                entry_event_quality_score=1.0,
                entry_market_stage_reason="compressed_mid_range",
                entry_stage_unknown_reason="none",
                entry_atr_percentile_100=50.0,
                entry_key_level_proximity_bucket="mid_range_far_from_key_level",
                entry_z_pressure_side="long_pressure",
            ),
        ],
        [100.0, 101.0, 100.0],
    )
    trades = report.trades
    sections = report.metric_sections()

    assert trades["entry_adx_bucket"].to_list() == ["high", "low"]
    assert trades["entry_volatility_bucket"].to_list() == ["expanded", "compressed"]
    assert trades["entry_trend_bucket"].to_list() == ["uptrend", "flat"]
    assert trades["entry_zscore_bucket"].to_list() == ["tail", "moderate"]
    assert trades["entry_liquidity_event_type_bucket"].to_list() == [
        "breakout_acceptance_high",
        "none",
    ]
    assert trades["entry_event_quality_bucket"].to_list() == ["high", "low"]
    assert trades["entry_market_stage_reason_bucket"].to_list()[0] == "trend_without_range_break"
    assert trades["entry_stage_unknown_reason_bucket"].to_list()[0] == "trend_without_range_break"
    assert trades["entry_atr_percentile_bucket"].to_list() == ["extreme", "normal"]
    assert trades["entry_z_pressure_side_bucket"].to_list() == [
        "short_pressure",
        "long_pressure",
    ]
    assert trades["entry_mtf_state_bucket"].to_list()[0] == "uptrend|markup|accumulation"
    assert trades["entry_mtf_structure_bucket"].to_list()[0] == "uptrend|uptrend|range"
    assert trades["entry_mtf_stage_bucket"].to_list()[0] == "markup|range|accumulation"
    assert trades["entry_mtf_event_state_bucket"].to_list()[0] == (
        "uptrend|markup|accumulation|failed_breakout_low"
    )
    assert "By entry liquidity event type" in sections
    assert "By entry event quality" in sections
    assert "By entry market stage reason" in sections
    assert "By entry unknown stage reason" in sections
    assert "By entry ATR percentile" in report.group_attribution()
    assert "By entry key-level proximity" in report.group_attribution()
    assert "By entry Z pressure side" in report.group_attribution()


def test_report_derives_exit_family_and_cross_attribution_sections():
    report = _report(
        [
            _trade(pnl=-0.01, pnl_usd=-1.0, net_pnl_usd=-1.0, reason="stop", side="sell"),
            _trade(
                reason="strategy_exit",
                entry_market_stage="range",
                entry_liquidity_event_type="bullish_reclaim",
            ),
            _trade(pnl=-0.03, pnl_usd=-3.0, net_pnl_usd=-3.0, reason="martingale_reverse"),
        ],
        [100.0, 99.0, 101.0, 98.0],
    )
    sections = report.metric_sections()

    assert report.trades["exit_family"].to_list() == ["risk_stop", "strategy", "recovery"]
    assert "risk_stop" in sections
    assert "strategy" in sections
    assert "recovery" in sections
    assert "By exit family" in sections
    assert "By stage x event" in sections
    assert format_pair_attribution(report.trades, "missing", "also_missing") == ""


def test_yield_attribution_reconciles_price_edge_and_dollar_loss():
    report = _report(
        [
            _trade(
                reason="target",
                gross_pnl_usd=5.0,
                fee_usd=1.0,
                net_pnl_usd=4.0,
                pnl_usd=4.0,
                signal_id="long_rule",
                entry_notional_usd=100.0,
                notional_pct_capital=100.0,
                contracts=1.0,
            ),
            _trade(
                side="sell",
                exit_px=99.5,
                pnl=0.005,
                gross_pnl_usd=-10.0,
                fee_usd=1.0,
                net_pnl_usd=-11.0,
                pnl_usd=-11.0,
                reason="stop",
                signal_id="short_rule",
                entry_notional_usd=400.0,
                notional_pct_capital=400.0,
                contracts=4.0,
            ),
            _trade(
                side="sell",
                exit_px=99.5,
                pnl=0.005,
                gross_pnl_usd=-8.0,
                fee_usd=1.0,
                net_pnl_usd=-9.0,
                pnl_usd=-9.0,
                reason="stop",
                signal_id="short_rule",
                entry_notional_usd=400.0,
                notional_pct_capital=400.0,
                contracts=4.0,
            ),
        ],
        [100.0, 90.0],
    )
    y = report.yield_attribution
    sections = report.metric_sections()

    assert report.trade_expectancy_pct > 0
    assert report.trade_expectancy_usd < 0
    assert y.net_pnl_usd < 0
    assert y.fee_drag_pct > 0
    assert y.worst_exit_reason == "stop"
    assert y.worst_side == "sell"
    assert y.worst_signal_id == "short_rule"
    assert y.loss_to_win_notional_ratio == 4.0
    assert y.avg_loss_contracts == 4.0
    assert "Yield Attribution" in sections
    assert "SizeWExp" in sections
    assert "Group Attribution" in sections
    assert "By side" in sections


def test_stop_effectiveness_identifies_worst_side_signal_and_sections():
    report = _report(
        [
            _trade(
                exit_px=98.0,
                pnl=-0.02,
                gross_pnl_usd=-20.0,
                net_pnl_usd=-21.0,
                pnl_usd=-21.0,
                reason="stop",
                signal_id="long_rule",
                notional_pct_capital=100.0,
            ),
            _trade(
                side="sell",
                exit_px=101.0,
                pnl=-0.01,
                gross_pnl_usd=-5.0,
                net_pnl_usd=-6.0,
                pnl_usd=-6.0,
                reason="stop",
                signal_id="short_rule",
                notional_pct_capital=50.0,
            ),
            _trade(gross_pnl_usd=10.0, net_pnl_usd=9.0, pnl_usd=9.0, signal_id="long_rule"),
        ],
        [100.0, 79.0, 73.0, 82.0],
    )
    stop = report.stop_effectiveness
    sections = report.metric_sections()

    assert stop.stop_trades == 2
    assert stop.stop_loss_share_pct == 100.0
    assert stop.worst_stop_side == "buy"
    assert stop.worst_stop_signal_id == "long_rule"
    assert "Stop Effectiveness" in sections
    assert "By exit family" in sections
    assert "risk_stop" in sections


def test_drawdown_path_diagnostics_from_trade_rows():
    report = _report(
        [
            _trade(entry_drawdown_pct=0.0, entry_total_notional_pct=0.0, entry_signal_id="clean"),
            _trade(
                side="sell",
                exit_px=101.0,
                pnl=-0.01,
                pnl_usd=-1.0,
                net_pnl_usd=-1.0,
                reason="stop",
                entry_drawdown_pct=12.5,
                entry_total_notional_pct=150.0,
                post_entry_total_notional_pct=175.0,
                entry_signal_id="dd_signal",
            ),
        ],
        [100.0, 101.0, 99.0],
    )
    dd = report.drawdown_path

    assert dd.entries_while_drawdown_count == 1
    assert dd.max_entry_drawdown_pct == 12.5
    assert dd.max_notional_during_drawdown_pct == 175.0
    assert dd.worst_drawdown_entry_signal_id == "dd_signal"


def test_strategy_recommendations_and_symbol_rankings_keep_decision_contracts():
    rejected = _report(
        [_trade(exit_px=99.0, pnl=-0.01, pnl_usd=-1.0, net_pnl_usd=-1.0, reason="x")],
        [100.0, 99.0],
    )
    traded = _report([_trade()], [100.0, 101.0], label="TRADED")
    incomplete = _report(
        [],
        [100.0, 100.0],
        label="XAU placeholder",
        metadata=("data_quality=data_incomplete",),
    )

    recommendation = format_strategy_recommendations("momentum_burst", [rejected])
    ranking = format_symbol_rankings([traded, incomplete])

    assert "Reject current momentum_burst baseline" in recommendation
    assert "do not loosen filters" in recommendation
    assert "calendar Sharpe/Sortino as tertiary" in recommendation
    assert "Return best=TRADED" in ranking
    assert "RankingSkipped=1" in ranking
    assert "XAU placeholder inf" not in ranking
