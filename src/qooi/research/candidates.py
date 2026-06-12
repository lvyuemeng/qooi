"""Candidate pattern trade diagnostics for behavior-state research."""

from __future__ import annotations

import math
import random
from bisect import bisect_right

import polars as pl

from qooi.research.artifacts import empty_frame, ensure_columns
from qooi.research.patterns import SCORED_PATTERN_SCHEMA

CANDIDATE_TRADE_SCHEMA: dict[str, pl.DataType] = {
    "pattern_id": pl.Utf8,
    "pattern_family": pl.Utf8,
    "pattern_source": pl.Utf8,
    "symbol": pl.Utf8,
    "timeframe": pl.Utf8,
    "horizon": pl.Int64,
    "side": pl.Utf8,
    "pattern_value": pl.Utf8,
    "signal_timestamp": pl.Int64,
    "entry_timestamp": pl.Int64,
    "exit_timestamp": pl.Int64,
    "entry_close": pl.Float64,
    "exit_close": pl.Float64,
    "market_return_pct": pl.Float64,
    "side_return_pct": pl.Float64,
    "transaction_cost_bps": pl.Float64,
    "nonoverlap": pl.Boolean,
    "split": pl.Utf8,
}

CANDIDATE_BOOTSTRAP_SCHEMA: dict[str, pl.DataType] = {
    "pattern_id": pl.Utf8,
    "symbol": pl.Utf8,
    "horizon": pl.Int64,
    "trades": pl.Int64,
    "mean_return_pct": pl.Float64,
    "mean_ci_low_95": pl.Float64,
    "mean_ci_high_95": pl.Float64,
    "win_rate_pct": pl.Float64,
    "win_rate_ci_low_95": pl.Float64,
    "win_rate_ci_high_95": pl.Float64,
    "median_return_pct": pl.Float64,
    "median_ci_low_95": pl.Float64,
    "median_ci_high_95": pl.Float64,
    "cumulative_return_pct": pl.Float64,
    "cumulative_ci_low_95": pl.Float64,
    "cumulative_ci_high_95": pl.Float64,
}

CANDIDATE_DIRECTION_SCHEMA: dict[str, pl.DataType] = {
    "pattern_id": pl.Utf8,
    "symbol": pl.Utf8,
    "horizon": pl.Int64,
    "trades": pl.Int64,
    "long_signal_pct": pl.Float64,
    "short_signal_pct": pl.Float64,
    "long_mean_return_pct": pl.Float64,
    "short_or_contrarian_mean_return_pct": pl.Float64,
    "matched_buy_hold_mean_pct": pl.Float64,
    "excess_mean_pct": pl.Float64,
    "rule_sharpe": pl.Float64,
    "matched_buy_hold_sharpe": pl.Float64,
    "rule_minus_bh_sharpe": pl.Float64,
}

CANDIDATE_ALPHA_BETA_SCHEMA: dict[str, pl.DataType] = {
    "pattern_id": pl.Utf8,
    "symbol": pl.Utf8,
    "horizon": pl.Int64,
    "trades": pl.Int64,
    "alpha_pct": pl.Float64,
    "alpha_ci_low_95": pl.Float64,
    "alpha_ci_high_95": pl.Float64,
    "beta": pl.Float64,
    "beta_ci_low_95": pl.Float64,
    "beta_ci_high_95": pl.Float64,
    "r_squared": pl.Float64,
    "residual_std_pct": pl.Float64,
    "alpha_information_ratio": pl.Float64,
    "mean_excess_return_pct": pl.Float64,
    "excess_ci_low_95": pl.Float64,
    "excess_ci_high_95": pl.Float64,
    "beta_proxy_warning": pl.Utf8,
}

CANDIDATE_REGIME_SCHEMA: dict[str, pl.DataType] = {
    "pattern_id": pl.Utf8,
    "symbol": pl.Utf8,
    "horizon": pl.Int64,
    "regime": pl.Utf8,
    "eligible_bars": pl.Int64,
    "trigger_count": pl.Int64,
    "trigger_rate_per_1000_bars": pl.Float64,
    "nonoverlap_trades": pl.Int64,
    "mean_return_pct": pl.Float64,
    "median_return_pct": pl.Float64,
    "win_rate_pct": pl.Float64,
    "trade_sharpe": pl.Float64,
    "max_drawdown_pct": pl.Float64,
    "matched_buy_hold_mean_pct": pl.Float64,
    "mean_excess_return_pct": pl.Float64,
}


def build_candidate_nonoverlap_trades(
    patterns: pl.DataFrame,
    market_frame: pl.DataFrame,
    scored: pl.DataFrame,
    *,
    returns_split: str = "test",
    transaction_cost_bps: float = 0.0,
) -> pl.DataFrame:
    """Build candidate trades with signal-at-close and next-bar entry semantics."""
    candidates = _candidate_rows(scored)
    if patterns.is_empty() or market_frame.is_empty() or candidates.is_empty():
        return _empty_frame(CANDIDATE_TRADE_SCHEMA)
    market_lookup = _market_lookup(market_frame)
    pattern_rows = patterns.sort(["symbol", "timeframe", "timestamp"]).to_dicts()
    pattern_index: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in pattern_rows:
        key = (str(row.get("pattern_id")), str(row.get("symbol")))
        pattern_index.setdefault(key, []).append(row)
    cost_pct = transaction_cost_bps / 100.0
    accepted_by_group: dict[tuple[str, str, str, int, str | None], int] = {}
    rows = []
    for candidate in candidates.sort(["symbol", "pattern_id", "horizon"]).to_dicts():
        horizon = int(candidate.get("horizon") or 0)
        if horizon <= 0:
            continue
        symbol = str(candidate.get("symbol"))
        side = candidate.get("side")
        pattern_id = str(candidate.get("pattern_id"))
        occurrences = pattern_index.get((pattern_id, symbol), ())
        for occurrence in occurrences:
            split = occurrence.get("split")
            if returns_split != "all" and split is not None and str(split) != returns_split:
                continue
            timeframe = str(occurrence.get("timeframe"))
            market_key = (symbol, timeframe)
            market = market_lookup.get(market_key) or market_lookup.get((symbol, "unknown"))
            if market is None:
                continue
            signal_timestamp = occurrence.get("timestamp")
            if signal_timestamp is None:
                continue
            entry_index = bisect_right(market["timestamps"], int(signal_timestamp))
            exit_index = entry_index + horizon
            if exit_index >= len(market["timestamps"]):
                continue
            entry_timestamp = market["timestamps"][entry_index]
            exit_timestamp = market["timestamps"][exit_index]
            group_key = (
                pattern_id,
                symbol,
                timeframe,
                horizon,
                str(side) if side is not None else None,
            )
            if entry_timestamp <= accepted_by_group.get(group_key, -1):
                continue
            entry_close = market["closes"][entry_index]
            exit_close = market["closes"][exit_index]
            if entry_close is None or exit_close is None or entry_close == 0:
                continue
            market_return = (float(exit_close) - float(entry_close)) / float(entry_close) * 100.0
            side_return = -market_return if side == "short" else market_return
            accepted_by_group[group_key] = exit_timestamp
            rows.append(
                {
                    "pattern_id": pattern_id,
                    "pattern_family": candidate.get("pattern_family"),
                    "pattern_source": candidate.get("pattern_source"),
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "horizon": horizon,
                    "side": side,
                    "pattern_value": occurrence.get("pattern_value"),
                    "signal_timestamp": int(signal_timestamp),
                    "entry_timestamp": entry_timestamp,
                    "exit_timestamp": exit_timestamp,
                    "entry_close": float(entry_close),
                    "exit_close": float(exit_close),
                    "market_return_pct": market_return,
                    "side_return_pct": side_return - cost_pct,
                    "transaction_cost_bps": transaction_cost_bps,
                    "nonoverlap": True,
                    "split": str(split) if split is not None else None,
                }
            )
    return _ensure_columns(pl.DataFrame(rows), CANDIDATE_TRADE_SCHEMA)


def bootstrap_candidate_trades(
    trades: pl.DataFrame, *, samples: int = 1000, seed: int = 7
) -> pl.DataFrame:
    if trades.is_empty():
        return _empty_frame(CANDIDATE_BOOTSTRAP_SCHEMA)
    rows = []
    for key, group in trades.partition_by(
        ["pattern_id", "symbol", "horizon"], as_dict=True
    ).items():
        pattern_id, symbol, horizon = key if isinstance(key, tuple) else (key, None, None)
        returns = [float(value) for value in group.get_column("side_return_pct").drop_nulls()]
        if not returns:
            continue
        boot = _bootstrap_return_stats(returns, samples=samples, seed=seed)
        rows.append(
            {
                "pattern_id": pattern_id,
                "symbol": symbol,
                "horizon": horizon,
                "trades": len(returns),
                "mean_return_pct": _mean(returns),
                "mean_ci_low_95": _percentile(boot["mean"], 0.025),
                "mean_ci_high_95": _percentile(boot["mean"], 0.975),
                "win_rate_pct": _win_rate(returns),
                "win_rate_ci_low_95": _percentile(boot["win_rate"], 0.025),
                "win_rate_ci_high_95": _percentile(boot["win_rate"], 0.975),
                "median_return_pct": _median(returns),
                "median_ci_low_95": _percentile(boot["median"], 0.025),
                "median_ci_high_95": _percentile(boot["median"], 0.975),
                "cumulative_return_pct": _sum(returns),
                "cumulative_ci_low_95": _percentile(boot["cumulative"], 0.025),
                "cumulative_ci_high_95": _percentile(boot["cumulative"], 0.975),
            }
        )
    return _ensure_columns(pl.DataFrame(rows), CANDIDATE_BOOTSTRAP_SCHEMA)


def summarize_candidate_direction_asymmetry(trades: pl.DataFrame) -> pl.DataFrame:
    if trades.is_empty():
        return _empty_frame(CANDIDATE_DIRECTION_SCHEMA)
    rows = []
    for key, group in trades.partition_by(
        ["pattern_id", "symbol", "horizon"], as_dict=True
    ).items():
        pattern_id, symbol, horizon = key if isinstance(key, tuple) else (key, None, None)
        rule_returns = [float(value) for value in group.get_column("side_return_pct").drop_nulls()]
        market_returns = [
            float(value) for value in group.get_column("market_return_pct").drop_nulls()
        ]
        sides = group.get_column("side").to_list() if "side" in group.columns else []
        trades_count = len(rule_returns)
        if trades_count == 0:
            continue
        long_count = sum(1 for side in sides if side != "short")
        short_count = sum(1 for side in sides if side == "short")
        short_returns = [-value for value in market_returns]
        rows.append(
            {
                "pattern_id": pattern_id,
                "symbol": symbol,
                "horizon": horizon,
                "trades": trades_count,
                "long_signal_pct": long_count / trades_count * 100.0,
                "short_signal_pct": short_count / trades_count * 100.0,
                "long_mean_return_pct": _mean(rule_returns),
                "short_or_contrarian_mean_return_pct": _mean(short_returns),
                "matched_buy_hold_mean_pct": _mean(market_returns),
                "excess_mean_pct": _mean(
                    [r - m for r, m in zip(rule_returns, market_returns, strict=False)]
                ),
                "rule_sharpe": _sharpe_like(rule_returns),
                "matched_buy_hold_sharpe": _sharpe_like(market_returns),
                "rule_minus_bh_sharpe": _sharpe_like(rule_returns) - _sharpe_like(market_returns),
            }
        )
    return _ensure_columns(pl.DataFrame(rows), CANDIDATE_DIRECTION_SCHEMA)


def summarize_candidate_alpha_beta(
    trades: pl.DataFrame, *, samples: int = 1000, seed: int = 7
) -> pl.DataFrame:
    if trades.is_empty():
        return _empty_frame(CANDIDATE_ALPHA_BETA_SCHEMA)
    rows = []
    rng = random.Random(seed)
    btc = (
        trades.filter(pl.col("symbol").str.starts_with("BTC"))
        if "symbol" in trades.columns
        else trades
    )
    for key, group in btc.partition_by(["pattern_id", "symbol", "horizon"], as_dict=True).items():
        pattern_id, symbol, horizon = key if isinstance(key, tuple) else (key, None, None)
        pairs = [
            (float(row["side_return_pct"]), float(row["market_return_pct"]))
            for row in group.select("side_return_pct", "market_return_pct").drop_nulls().to_dicts()
        ]
        if not pairs:
            continue
        alpha, beta, r_squared, residual_std = _ols_alpha_beta(pairs)
        excess = [rule - market for rule, market in pairs]
        boot_alpha = []
        boot_beta = []
        boot_excess = []
        for _ in range(samples):
            sample = [pairs[rng.randrange(len(pairs))] for _item in pairs]
            sample_alpha, sample_beta, _sample_r2, _sample_resid = _ols_alpha_beta(sample)
            boot_alpha.append(sample_alpha)
            boot_beta.append(sample_beta)
            boot_excess.append(_mean([rule - market for rule, market in sample]))
        alpha_std = _std(boot_alpha)
        warning = "none"
        if beta >= 0.8 and r_squared >= 0.8 and _percentile(boot_alpha, 0.025) <= 0:
            warning = "likely_beta_proxy"
        elif _percentile(boot_alpha, 0.025) <= 0:
            warning = "alpha_not_significant"
        rows.append(
            {
                "pattern_id": pattern_id,
                "symbol": symbol,
                "horizon": horizon,
                "trades": len(pairs),
                "alpha_pct": alpha,
                "alpha_ci_low_95": _percentile(boot_alpha, 0.025),
                "alpha_ci_high_95": _percentile(boot_alpha, 0.975),
                "beta": beta,
                "beta_ci_low_95": _percentile(boot_beta, 0.025),
                "beta_ci_high_95": _percentile(boot_beta, 0.975),
                "r_squared": r_squared,
                "residual_std_pct": residual_std,
                "alpha_information_ratio": alpha / alpha_std if alpha_std > 0 else None,
                "mean_excess_return_pct": _mean(excess),
                "excess_ci_low_95": _percentile(boot_excess, 0.025),
                "excess_ci_high_95": _percentile(boot_excess, 0.975),
                "beta_proxy_warning": warning,
            }
        )
    return _ensure_columns(pl.DataFrame(rows), CANDIDATE_ALPHA_BETA_SCHEMA)


def summarize_candidate_regime_segments(
    trades: pl.DataFrame, market_frame: pl.DataFrame
) -> pl.DataFrame:
    if trades.is_empty() or market_frame.is_empty():
        return _empty_frame(CANDIDATE_REGIME_SCHEMA)
    regimes = _btc_regime_frame(market_frame)
    if regimes.is_empty():
        return _empty_frame(CANDIDATE_REGIME_SCHEMA)
    eligible = regimes.group_by("regime").agg(pl.len().alias("eligible_bars"))
    with_regime = trades.join(
        regimes.select("timestamp", "regime"),
        left_on="signal_timestamp",
        right_on="timestamp",
        how="left",
    ).with_columns(pl.col("regime").fill_null("unknown"))
    rows = []
    eligible_map = {row["regime"]: int(row["eligible_bars"]) for row in eligible.to_dicts()}
    for key, group in with_regime.partition_by(
        ["pattern_id", "symbol", "horizon", "regime"], as_dict=True
    ).items():
        pattern_id, symbol, horizon, regime = (
            key if isinstance(key, tuple) else (key, None, None, None)
        )
        returns = [float(value) for value in group.get_column("side_return_pct").drop_nulls()]
        market_returns = [
            float(value) for value in group.get_column("market_return_pct").drop_nulls()
        ]
        bars = eligible_map.get(str(regime), 0)
        triggers = group.height
        rows.append(
            {
                "pattern_id": pattern_id,
                "symbol": symbol,
                "horizon": horizon,
                "regime": regime,
                "eligible_bars": bars,
                "trigger_count": triggers,
                "trigger_rate_per_1000_bars": triggers / bars * 1000.0 if bars else None,
                "nonoverlap_trades": len(returns),
                "mean_return_pct": _mean(returns),
                "median_return_pct": _median(returns),
                "win_rate_pct": _win_rate(returns),
                "trade_sharpe": _sharpe_like(returns),
                "max_drawdown_pct": _max_drawdown(returns),
                "matched_buy_hold_mean_pct": _mean(market_returns),
                "mean_excess_return_pct": _mean(
                    [r - m for r, m in zip(returns, market_returns, strict=False)]
                ),
            }
        )
    return _ensure_columns(pl.DataFrame(rows), CANDIDATE_REGIME_SCHEMA)


def _candidate_rows(scored: pl.DataFrame) -> pl.DataFrame:
    if scored.is_empty() or "passes_candidate_gate" not in scored.columns:
        return _empty_frame(SCORED_PATTERN_SCHEMA)
    return scored.filter(pl.col("passes_candidate_gate").fill_null(False))


def _market_with_group_columns(market: pl.DataFrame) -> pl.DataFrame:
    work = market
    if "symbol" not in work.columns:
        work = work.with_columns(pl.lit("unknown").alias("symbol"))
    if "timeframe" not in work.columns:
        work = work.with_columns(pl.lit("unknown").alias("timeframe"))
    return work


def _market_lookup(market_frame: pl.DataFrame) -> dict[tuple[str, str], dict[str, list[object]]]:
    if market_frame.is_empty() or not {"timestamp", "close"} <= set(market_frame.columns):
        return {}
    work = _market_with_group_columns(market_frame)
    out: dict[tuple[str, str], dict[str, list[object]]] = {}
    for key, group in (
        work.sort(["symbol", "timeframe", "timestamp"])
        .partition_by(["symbol", "timeframe"], as_dict=True)
        .items()
    ):
        symbol, timeframe = key if isinstance(key, tuple) else (key, "unknown")
        out[(str(symbol), str(timeframe))] = {
            "timestamps": [int(value) for value in group.get_column("timestamp").to_list()],
            "closes": group.get_column("close").to_list(),
        }
    return out


def _iter_market_groups(market: pl.DataFrame):
    if market.is_empty() or "timestamp" not in market.columns:
        return
    work = market
    if "symbol" not in work.columns:
        work = work.with_columns(pl.lit("unknown").alias("symbol"))
    if "timeframe" not in work.columns:
        work = work.with_columns(pl.lit("unknown").alias("timeframe"))
    for key, group in (
        work.sort(["symbol", "timeframe", "timestamp"])
        .partition_by(["symbol", "timeframe"], as_dict=True)
        .items()
    ):
        symbol, timeframe = key if isinstance(key, tuple) else (key, "unknown")
        yield (str(symbol), str(timeframe)), group.to_dicts()


def _market_row_index(
    market: pl.DataFrame,
) -> dict[tuple[str, str, int], tuple[list[dict[str, object]], int]]:
    out = {}
    for (_symbol, _timeframe), rows in _iter_market_groups(market):
        for index, row in enumerate(rows):
            if row.get("timestamp") is not None:
                out[(str(row.get("symbol")), str(row.get("timeframe")), int(row["timestamp"]))] = (
                    rows,
                    index,
                )
    return out


def _bootstrap_return_stats(
    values: list[float], *, samples: int, seed: int
) -> dict[str, list[float]]:
    rng = random.Random(seed)
    means = []
    medians = []
    win_rates = []
    cumulative = []
    if not values:
        return {"mean": means, "median": medians, "win_rate": win_rates, "cumulative": cumulative}
    for _ in range(samples):
        sample = [values[rng.randrange(len(values))] for _item in values]
        means.append(_mean(sample))
        medians.append(_median(sample))
        win_rates.append(_win_rate(sample))
        cumulative.append(_sum(sample))
    return {"mean": means, "median": medians, "win_rate": win_rates, "cumulative": cumulative}


def _ols_alpha_beta(pairs: list[tuple[float, float]]) -> tuple[float, float, float, float]:
    if not pairs:
        return 0.0, 0.0, 0.0, 0.0
    y = [rule for rule, _market in pairs]
    x = [market for _rule, market in pairs]
    x_mean = _mean(x)
    y_mean = _mean(y)
    var_x = _mean([(value - x_mean) ** 2 for value in x])
    beta = (
        _mean([(xv - x_mean) * (yv - y_mean) for xv, yv in zip(x, y, strict=False)]) / var_x
        if var_x > 0
        else 0.0
    )
    alpha = y_mean - beta * x_mean
    fitted = [alpha + beta * value for value in x]
    residuals = [actual - fit for actual, fit in zip(y, fitted, strict=False)]
    ss_res = _sum([value**2 for value in residuals])
    ss_tot = _sum([(value - y_mean) ** 2 for value in y])
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return alpha, beta, r_squared, _std(residuals)


def _btc_regime_frame(market_frame: pl.DataFrame) -> pl.DataFrame:
    if market_frame.is_empty() or not {"symbol", "timestamp", "close"} <= set(market_frame.columns):
        return pl.DataFrame()
    btc = market_frame.filter(pl.col("symbol").str.starts_with("BTC")).sort("timestamp")
    if btc.is_empty():
        return pl.DataFrame()
    ma = pl.col("close").rolling_mean(window_size=4800, min_periods=1)
    return (
        btc.with_columns(ma.alias("ma_200d"))
        .with_columns((pl.col("ma_200d") - pl.col("ma_200d").shift(168)).alias("ma_slope"))
        .with_columns(
            pl.when((pl.col("close") > pl.col("ma_200d")) & (pl.col("ma_slope") > 0))
            .then(pl.lit("bull"))
            .when((pl.col("close") < pl.col("ma_200d")) & (pl.col("ma_slope") < 0))
            .then(pl.lit("bear"))
            .otherwise(pl.lit("range"))
            .alias("regime")
        )
        .select("timestamp", "regime")
    )


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _sum(values: list[float]) -> float:
    return sum(values) if values else 0.0


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(max(int(round((len(ordered) - 1) * q)), 0), len(ordered) - 1)
    return ordered[index]


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = _mean(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))


def _win_rate(values: list[float]) -> float:
    return sum(1 for value in values if value > 0) / len(values) * 100.0 if values else 0.0


def _sharpe_like(values: list[float]) -> float:
    std = _std(values)
    return _mean(values) / std if std > 0 else 0.0


def _max_drawdown(values: list[float]) -> float:
    equity = 0.0
    peak = 0.0
    max_drawdown = 0.0
    for value in values:
        equity += value
        peak = max(peak, equity)
        max_drawdown = min(max_drawdown, equity - peak)
    return max_drawdown


def _ensure_columns(frame: pl.DataFrame, schema: dict[str, pl.DataType]) -> pl.DataFrame:
    return ensure_columns(frame, schema)


def _empty_frame(schema: dict[str, pl.DataType]) -> pl.DataFrame:
    return empty_frame(schema)
