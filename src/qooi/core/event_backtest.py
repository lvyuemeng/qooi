"""Historical event extraction for scored research events."""

from __future__ import annotations

import statistics

import polars as pl

BACKTEST_EVENT_SCHEMA: dict[str, pl.DataType] = {
    "timestamp": pl.Int64,
    "symbol": pl.String,
    "alert_level": pl.String,
    "score_total": pl.Int64,
    "entry_close": pl.Float64,
    "return_3h": pl.Float64,
    "return_7h": pl.Float64,
    "return_24h": pl.Float64,
    "return_3d": pl.Float64,
    "return_7d": pl.Float64,
    "max_drawdown_24h": pl.Float64,
    "max_drawdown_7d": pl.Float64,
    "hit_take_profit_5pct_7d": pl.Boolean,
    "hit_stop_loss_5pct_7d": pl.Boolean,
}


def empty_backtest_event_frame() -> pl.DataFrame:
    return pl.DataFrame(schema=BACKTEST_EVENT_SCHEMA)

HORIZON_HOURS: dict[str, int] = {"3h": 3, "7h": 7, "24h": 24, "3d": 72, "7d": 168}


def _forward_return(prices: pl.DataFrame, timestamp: int, entry: float, hours: int) -> float | None:
    target_ts = timestamp + hours * 3_600_000
    future = prices.filter(pl.col("timestamp") >= target_ts).head(1)
    if future.is_empty() or entry <= 0:
        return None
    return float(future["close"][0]) / entry - 1.0


def _max_drawdown(prices: pl.DataFrame, timestamp: int, entry: float, hours: int) -> float | None:
    window = prices.filter(
        (pl.col("timestamp") > timestamp) & (pl.col("timestamp") <= timestamp + hours * 3_600_000)
    )
    if window.is_empty() or entry <= 0:
        return None
    low_col = "low" if "low" in window.columns else "close"
    return float(window[low_col].min()) / entry - 1.0


def build_backtest_events(
    scores: pl.DataFrame,
    price_frame: pl.DataFrame,
    *,
    alert_levels: tuple[str, ...] = ("yellow", "orange", "red"),
    take_profit_pct: float = 0.05,
    stop_loss_pct: float = 0.05,
) -> pl.DataFrame:
    if scores.is_empty() or price_frame.is_empty():
        return empty_backtest_event_frame()
    prices = price_frame.sort("timestamp")
    events = scores.filter(pl.col("alert_level").is_in(alert_levels)).sort("timestamp")
    rows = []
    for row in events.to_dicts():
        ts = int(row["timestamp"])
        entry_row = prices.filter(pl.col("timestamp") <= ts).tail(1)
        if entry_row.is_empty():
            continue
        entry = float(entry_row["close"][0])
        window_7d = prices.filter(
            (pl.col("timestamp") > ts) & (pl.col("timestamp") <= ts + 168 * 3_600_000)
        )
        high_col = "high" if "high" in window_7d.columns else "close"
        low_col = "low" if "low" in window_7d.columns else "close"
        rows.append(
            {
                "timestamp": ts,
                "symbol": str(row["symbol"]),
                "alert_level": str(row["alert_level"]),
                "score_total": int(row["score_total"]),
                "entry_close": entry,
                "return_3h": _forward_return(prices, ts, entry, 3),
                "return_7h": _forward_return(prices, ts, entry, 7),
                "return_24h": _forward_return(prices, ts, entry, 24),
                "return_3d": _forward_return(prices, ts, entry, 72),
                "return_7d": _forward_return(prices, ts, entry, 168),
                "max_drawdown_24h": _max_drawdown(prices, ts, entry, 24),
                "max_drawdown_7d": _max_drawdown(prices, ts, entry, 168),
                "hit_take_profit_5pct_7d": bool(
                    not window_7d.is_empty()
                    and float(window_7d[high_col].max()) >= entry * (1.0 + take_profit_pct)
                ),
                "hit_stop_loss_5pct_7d": bool(
                    not window_7d.is_empty()
                    and float(window_7d[low_col].min()) <= entry * (1.0 - stop_loss_pct)
                ),
            }
        )
    return (
        pl.DataFrame(rows, schema=BACKTEST_EVENT_SCHEMA) if rows else empty_backtest_event_frame()
    )


def summarize_backtest_events(events: pl.DataFrame) -> pl.DataFrame:
    if events.is_empty():
        return pl.DataFrame(
            schema={
                "alert_level": pl.String,
                "symbol": pl.String,
                "signal_count": pl.Int64,
                "win_rate_24h": pl.Float64,
                "win_rate_3d": pl.Float64,
                "win_rate_7d": pl.Float64,
                "mean_return_24h": pl.Float64,
                "median_return_24h": pl.Float64,
                "payoff_ratio_24h": pl.Float64,
                "mean_max_drawdown_7d": pl.Float64,
            }
        )
    rows = []
    for group in events.partition_by(["alert_level", "symbol"], as_dict=False):
        returns = [x for x in group["return_24h"].to_list() if x is not None]
        wins = [x for x in returns if x > 0]
        losses = [x for x in returns if x < 0]
        payoff = None
        if wins and losses:
            payoff = statistics.mean(wins) / abs(statistics.mean(losses))
        rows.append(
            {
                "alert_level": str(group["alert_level"][0]),
                "symbol": str(group["symbol"][0]),
                "signal_count": group.height,
                "win_rate_24h": sum(1 for x in returns if x > 0) / len(returns)
                if returns
                else None,
                "win_rate_3d": sum(1 for x in group["return_3d"].drop_nulls().to_list() if x > 0)
                / group["return_3d"].drop_nulls().len()
                if group["return_3d"].drop_nulls().len()
                else None,
                "win_rate_7d": sum(1 for x in group["return_7d"].drop_nulls().to_list() if x > 0)
                / group["return_7d"].drop_nulls().len()
                if group["return_7d"].drop_nulls().len()
                else None,
                "mean_return_24h": statistics.mean(returns) if returns else None,
                "median_return_24h": statistics.median(returns) if returns else None,
                "payoff_ratio_24h": payoff,
                "mean_max_drawdown_7d": statistics.mean(
                    group["max_drawdown_7d"].drop_nulls().to_list()
                )
                if group["max_drawdown_7d"].drop_nulls().len()
                else None,
            }
        )
    return pl.DataFrame(rows).sort(["alert_level", "symbol"])
