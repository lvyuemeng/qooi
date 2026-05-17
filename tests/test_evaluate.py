"""Evaluation layer tests."""

from qooi.core.config import PAIRS
from qooi.core.evaluate import (
    Report,
    _trades_frame,
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
        "reason": "strategy_exit",
    }
    trade.update(overrides)
    return trade


def _report(trades=None, equity=None, *, label=None, metadata=()):
    return Report.from_raw(
        trades or [],
        equity or [100.0, 100.0],
        PAIRS[0],
        label=label,
        metadata=metadata,
    )


def test_report_uses_return_ratios_not_usd():
    trades = [
        _trade(reason="x"),
        _trade(exit_px=99.0, pnl=-0.01, pnl_usd=-1.0, reason="y"),
    ]
    report = _report(trades, [100.0, 101.0, 99.99])
    assert abs(report.metrics.avg_win_pct - 1.0) < 1e-6
    assert abs(report.metrics.avg_loss_pct - 1.0) < 1e-6
    assert report.trade_expectancy_usd == 0.0


def test_compare_formats_multiple_reports():
    r1 = _report(label="A")
    r2 = _report(label="B")
    table = compare(r1, r2)
    assert "Label" in table
    assert "PF" in table
    assert "PL" not in table
    assert "A" in table and "B" in table


def test_report_marks_unstable_annualization_for_sparse_equity():
    report = _report([_trade(reason="x")], [100.0] + [100.0] * 50)
    assert report.unstable_annualization is True


def test_trade_expectancy_fields_populated():
    trades = [
        _trade(reason="x"),
        _trade(exit_px=99.0, pnl=-0.005, pnl_usd=-0.5, reason="y"),
    ]
    report = _report(trades, [100.0, 101.0, 100.495])
    assert report.trade_expectancy_pct > 0
    assert report.trade_expectancy_usd > 0
    assert report.trade_sharpe != 0


def test_regime_buckets_include_mtf_state_keys():
    trades = _trades_frame(
        [
            _trade(
                entry_mtf_state_key="uptrend|markup|accumulation",
                entry_mtf_structure_key="uptrend|uptrend|range",
                entry_mtf_stage_key="markup|range|accumulation",
                entry_mtf_event_state_key="uptrend|markup|accumulation|failed_breakout_low",
            )
        ]
    )

    assert trades["entry_mtf_state_bucket"].to_list() == ["uptrend|markup|accumulation"]
    assert trades["entry_mtf_structure_bucket"].to_list() == ["uptrend|uptrend|range"]
    assert trades["entry_mtf_stage_bucket"].to_list() == ["markup|range|accumulation"]
    assert trades["entry_mtf_event_state_bucket"].to_list() == [
        "uptrend|markup|accumulation|failed_breakout_low"
    ]


def test_active_bar_sharpe_uses_active_exposure():
    equity = [100.0, 100.5, 100.5, 101.0, 101.0]
    active_exposure = [0.0, 1.0, 0.0, 2.0, 0.0]
    report = Report.from_raw([], equity, PAIRS[0], active_exposure=active_exposure)
    assert report.active_bar_pct > 0
    assert report.active_bar_sharpe != 0 or report.active_bar_pct > 0


def test_report_table_distinguishes_pl_ratio_and_profit_factor():
    trades = [
        _trade(exit_px=102.0, pnl=0.02, reason="x"),
        _trade(exit_px=99.0, pnl=-0.01, pnl_usd=-1.0, reason="y"),
    ]
    report = _report(trades, [100.0, 102.0, 100.98])
    table = report.table()
    assert "P/L=" in table
    assert "PF=" in table


def test_metrics_live_in_core_metrics():
    report = _report(equity=[100.0, 101.0])
    metrics = compute_metrics(report.equity, trades=report.trades)
    assert metrics.total_return_pct == report.metrics.total_return_pct


def test_strategy_recommendation_prioritizes_primary_metrics():
    report = _report([_trade(exit_px=99.0, pnl=-0.01, pnl_usd=-1.0, reason="x")], [100.0, 99.0])

    text = format_strategy_recommendations("momentum_burst", [report])

    assert "Reject current momentum_burst baseline" in text
    assert "do not loosen filters" in text
    assert "calendar Sharpe/Sortino as tertiary" in text


def test_yield_attribution_reconciles_price_edge_and_dollar_loss():
    pair = PAIRS[0]
    trades = [
        {
            "side": "buy",
            "entry_px": 100.0,
            "exit_px": 101.0,
            "pnl": 0.01,
            "gross_pnl_usd": 5.0,
            "fee_usd": 1.0,
            "net_pnl_usd": 4.0,
            "pnl_usd": 4.0,
            "reason": "target",
            "signal_id": "long_rule",
            "entry_notional_usd": 100.0,
            "notional_pct_capital": 100.0,
            "contracts": 1.0,
        },
        {
            "side": "sell",
            "entry_px": 100.0,
            "exit_px": 99.5,
            "pnl": 0.005,
            "gross_pnl_usd": -10.0,
            "fee_usd": 1.0,
            "net_pnl_usd": -11.0,
            "pnl_usd": -11.0,
            "reason": "stop",
            "signal_id": "short_rule",
            "entry_notional_usd": 400.0,
            "notional_pct_capital": 400.0,
            "contracts": 4.0,
        },
        {
            "side": "sell",
            "entry_px": 100.0,
            "exit_px": 99.5,
            "pnl": 0.005,
            "gross_pnl_usd": -8.0,
            "fee_usd": 1.0,
            "net_pnl_usd": -9.0,
            "pnl_usd": -9.0,
            "reason": "stop",
            "signal_id": "short_rule",
            "entry_notional_usd": 400.0,
            "notional_pct_capital": 400.0,
            "contracts": 4.0,
        },
    ]
    report = Report.from_raw(trades, [100.0, 90.0], pair)

    assert report.trade_expectancy_pct > 0
    assert report.trade_expectancy_usd < 0
    assert report.yield_attribution.net_pnl_usd < 0
    assert report.yield_attribution.fee_drag_pct > 0
    assert report.yield_attribution.worst_exit_reason == "stop"
    assert report.yield_attribution.worst_side == "sell"
    assert report.yield_attribution.worst_signal_id == "short_rule"
    assert report.yield_attribution.loss_to_win_notional_ratio == 4.0
    assert report.yield_attribution.avg_loss_contracts == 4.0
    sections = report.metric_sections()
    assert "Yield Attribution" in sections
    assert "SizeWExp" in sections
    assert "Group Attribution" in sections
    assert "By side" in sections


def test_stop_effectiveness_identifies_worst_side_and_signal():
    pair = PAIRS[0]
    trades = [
        {
            "side": "buy",
            "entry_px": 100.0,
            "exit_px": 98.0,
            "pnl": -0.02,
            "gross_pnl_usd": -20.0,
            "net_pnl_usd": -21.0,
            "pnl_usd": -21.0,
            "reason": "stop",
            "signal_id": "long_rule",
            "notional_pct_capital": 100.0,
        },
        {
            "side": "sell",
            "entry_px": 100.0,
            "exit_px": 101.0,
            "pnl": -0.01,
            "gross_pnl_usd": -5.0,
            "net_pnl_usd": -6.0,
            "pnl_usd": -6.0,
            "reason": "stop",
            "signal_id": "short_rule",
            "notional_pct_capital": 50.0,
        },
        {
            "side": "buy",
            "entry_px": 100.0,
            "exit_px": 101.0,
            "pnl": 0.01,
            "gross_pnl_usd": 10.0,
            "net_pnl_usd": 9.0,
            "pnl_usd": 9.0,
            "reason": "strategy_exit",
            "signal_id": "long_rule",
            "notional_pct_capital": 100.0,
        },
    ]
    report = Report.from_raw(trades, [100.0, 79.0, 73.0, 82.0], pair)

    stop = report.stop_effectiveness

    assert stop.stop_trades == 2
    assert stop.stop_loss_share_pct == 100.0
    assert stop.worst_stop_side == "buy"
    assert stop.worst_stop_signal_id == "long_rule"
    assert "Stop Effectiveness" in report.metric_sections()
    assert "By exit family" in report.metric_sections()
    assert "risk_stop" in report.metric_sections()


def test_report_buckets_liquidity_event_type_and_quality():
    pair = PAIRS[0]
    trades = [
        {
            "side": "buy",
            "entry_px": 100.0,
            "exit_px": 101.0,
            "pnl": 0.01,
            "pnl_usd": 1.0,
            "net_pnl_usd": 1.0,
            "reason": "strategy_exit",
            "entry_liquidity_event_type": "breakout_acceptance_high",
            "entry_event_quality_score": 2.7,
        }
    ]

    report = Report.from_raw(trades, [100.0, 101.0], pair)

    assert report.trades["entry_liquidity_event_type_bucket"][0] == "breakout_acceptance_high"
    assert report.trades["entry_event_quality_bucket"][0] == "high"
    sections = report.metric_sections()
    assert "By entry liquidity event type" in sections
    assert "By entry event quality" in sections


def test_report_buckets_stage_reasons_and_cross_attribution():
    pair = PAIRS[0]
    trades = [
        {
            "side": "sell",
            "entry_px": 100.0,
            "exit_px": 101.0,
            "pnl": -0.01,
            "pnl_usd": -1.0,
            "net_pnl_usd": -1.0,
            "reason": "stop",
            "entry_structure_trend_state": "uptrend",
            "entry_market_stage": "unknown",
            "entry_market_stage_reason": "trend_without_range_break",
            "entry_stage_unknown_reason": "trend_without_range_break",
            "entry_liquidity_event_type": "none",
        },
        {
            "side": "buy",
            "entry_px": 100.0,
            "exit_px": 101.0,
            "pnl": 0.01,
            "pnl_usd": 1.0,
            "net_pnl_usd": 1.0,
            "reason": "strategy_exit",
            "entry_structure_trend_state": "downtrend",
            "entry_market_stage": "range",
            "entry_market_stage_reason": "compressed_mid_range",
            "entry_stage_unknown_reason": "none",
            "entry_liquidity_event_type": "bullish_reclaim",
        },
    ]

    report = Report.from_raw(trades, [100.0, 99.0, 100.0], pair)
    sections = report.metric_sections()

    assert report.trades["entry_market_stage_reason_bucket"][0] == "trend_without_range_break"
    assert report.trades["entry_stage_unknown_reason_bucket"][0] == "trend_without_range_break"
    assert "By entry market stage reason" in sections
    assert "By entry unknown stage reason" in sections
    assert "By side x structure" in sections
    assert "By stage x event" in sections


def test_pair_attribution_safely_omits_missing_columns():
    pair = PAIRS[0]
    report = Report.from_raw([{"pnl": 0.01, "pnl_usd": 1.0}], [100.0, 101.0], pair)

    assert format_pair_attribution(report.trades, "missing", "also_missing") == ""


def test_drawdown_path_diagnostics_from_trade_rows():
    pair = PAIRS[0]
    trades = [
        {
            "side": "buy",
            "entry_px": 100.0,
            "exit_px": 101.0,
            "pnl": 0.01,
            "pnl_usd": 1.0,
            "reason": "strategy_exit",
            "entry_drawdown_pct": 0.0,
            "entry_total_notional_pct": 0.0,
            "entry_signal_id": "clean",
        },
        {
            "side": "sell",
            "entry_px": 100.0,
            "exit_px": 101.0,
            "pnl": -0.01,
            "pnl_usd": -1.0,
            "reason": "stop",
            "entry_drawdown_pct": 12.5,
            "entry_total_notional_pct": 150.0,
            "post_entry_total_notional_pct": 175.0,
            "entry_signal_id": "dd_signal",
        },
    ]
    report = Report.from_raw(trades, [100.0, 101.0, 99.0], pair)

    dd = report.drawdown_path

    assert dd.entries_while_drawdown_count == 1
    assert dd.max_entry_drawdown_pct == 12.5
    assert dd.max_notional_during_drawdown_pct == 175.0
    assert dd.worst_drawdown_entry_signal_id == "dd_signal"


def test_report_derives_exit_family_when_missing_from_trade_rows():
    pair = PAIRS[0]
    trades = [
        {"pnl": -0.01, "pnl_usd": -1.0, "reason": "stop"},
        {"pnl": 0.02, "pnl_usd": 2.0, "reason": "strategy_exit"},
        {"pnl": -0.03, "pnl_usd": -3.0, "reason": "martingale_reverse"},
    ]

    report = Report.from_raw(trades, [100.0, 99.0, 101.0, 98.0], pair)

    assert report.trades["exit_family"].to_list() == ["risk_stop", "strategy", "recovery"]
    sections = report.metric_sections()
    assert "risk_stop" in sections
    assert "strategy" in sections
    assert "recovery" in sections


def test_report_derives_entry_regime_buckets_from_trade_rows():
    pair = PAIRS[0]
    trades = [
        {
            "pnl": -0.01,
            "pnl_usd": -1.0,
            "reason": "stop",
            "entry_adx_14": 42.0,
            "entry_volatility_ratio": 1.8,
            "entry_trend_return": 0.04,
            "entry_close_z_score": -3.8,
        },
        {
            "pnl": 0.02,
            "pnl_usd": 2.0,
            "reason": "strategy_exit",
            "entry_adx_14": 15.0,
            "entry_volatility_ratio": 0.7,
            "entry_trend_return": 0.0,
            "entry_close_z_score": 2.1,
        },
    ]

    report = Report.from_raw(trades, [100.0, 99.0, 101.0], pair)

    assert report.trades["entry_adx_bucket"].to_list() == ["high", "low"]
    assert report.trades["entry_volatility_bucket"].to_list() == ["expanded", "compressed"]
    assert report.trades["entry_trend_bucket"].to_list() == ["uptrend", "flat"]
    assert report.trades["entry_zscore_bucket"].to_list() == ["tail", "moderate"]
    sections = report.metric_sections()
    assert "By entry trend regime" in sections
    assert "By entry volatility regime" in sections


def test_report_derives_none_context_buckets_from_trade_rows():
    pair = PAIRS[0]
    trades = [
        {
            "pnl": -0.01,
            "pnl_usd": -1.0,
            "reason": "stop",
            "entry_atr_percentile_100": 92.0,
            "entry_key_level_proximity_bucket": "near_prior_high_no_breach",
            "entry_z_pressure_side": "short_pressure",
        },
        {
            "pnl": 0.02,
            "pnl_usd": 2.0,
            "reason": "strategy_exit",
            "entry_atr_percentile_100": 50.0,
            "entry_key_level_proximity_bucket": "mid_range_far_from_key_level",
            "entry_z_pressure_side": "long_pressure",
        },
    ]

    report = Report.from_raw(trades, [100.0, 99.0, 101.0], pair)

    assert report.trades["entry_atr_percentile_bucket"].to_list() == ["extreme", "normal"]
    assert report.trades["entry_z_pressure_side_bucket"].to_list() == [
        "short_pressure",
        "long_pressure",
    ]
    sections = report.group_attribution()
    assert "By entry ATR percentile" in sections
    assert "By entry key-level proximity" in sections
    assert "By entry Z pressure side" in sections


def test_symbol_rankings_skip_data_incomplete_placeholders():
    pair = PAIRS[0]
    traded = Report.from_raw(
        [{"pnl": 0.01, "pnl_usd": 1.0}],
        [100.0, 101.0],
        pair,
        label="TRADED",
    )
    incomplete = Report.from_raw(
        [],
        [100.0, 100.0],
        pair,
        label="XAU placeholder",
        metadata=("data_quality=data_incomplete",),
    )

    text = format_symbol_rankings([traded, incomplete])

    assert "Return best=TRADED" in text
    assert "RankingSkipped=1" in text
    assert "XAU placeholder inf" not in text
