"""Trend-pullback — stateful strategy requiring a full DataFrame.

Uses Polars expressions for all rolling computations (EMA slope, ADX,
trend count) and delegates the position-hold state machine to a small
``_PositionManager`` helper so the main loop has no nested conditionals.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl


@dataclass
class _PositionManager:
    direction: int = 0
    entry: float = 0.0
    trail_high: float = 0.0
    trail_low: float = 0.0
    bars_since_entry: int = 0
    use_chandelier: bool = False
    max_hold_bars: int = 200

    def step(self, close: float, atr: float, opposite_entry: bool, regime_ok: bool) -> int:
        if not regime_ok:
            self.direction = 0
            return 0

        if self.direction == 0:
            return 0

        self.bars_since_entry += 1

        if self.direction == 1:
            self.trail_high = max(self.trail_high, close)
            if opposite_entry:
                self.direction = 0
            elif self.use_chandelier and close < self.trail_high - 3.0 * atr:
                self.direction = 0
            elif self.use_chandelier and self.bars_since_entry > self.max_hold_bars:
                self.direction = 0
        else:
            self.trail_low = min(self.trail_low, close)
            if opposite_entry:
                self.direction = 0
            elif self.use_chandelier and close > self.trail_low + 3.0 * atr:
                self.direction = 0
            elif self.use_chandelier and self.bars_since_entry > self.max_hold_bars:
                self.direction = 0

        return self.direction

    def enter(self, direction: int, price: float, atr: float, atr_stop_mult: float) -> None:
        self.direction = direction
        self.entry = price
        self.trail_high = price if direction == 1 else 0.0
        self.trail_low = price if direction == -1 else 0.0
        self.bars_since_entry = 0


def trend_pullback_signal(
    df: pl.DataFrame,
    ema_length: int = 20,
    trend_bars_min: int = 17,
    atr_period: int = 14,
    adx_threshold: float = 22.0,
    bar_hours: float | None = None,
) -> pl.DataFrame:
    if bar_hours:
        trend_bars_min = max(17, int(17 * 24 / bar_hours))
        trend_bars_min = min(trend_bars_min, 40)

    ema = f"ema_{ema_length}"
    atr = f"atr_{atr_period}"

    # All filters as Polars expressions — zero Python loops
    atr_ratio = pl.col(atr) / pl.col(atr).rolling_mean(100).fill_null(pl.col(atr))
    regime_ok = atr_ratio > 0.7

    ema_slope = (pl.col(ema) - pl.col(ema).shift(20)) / pl.col(ema).shift(20).fill_null(1)

    far_enough = (pl.col("close") - pl.col(ema)).abs() > 0.35 * pl.col(atr)

    trend_dir = (
        pl.when((pl.col("close") > pl.col(ema)) & far_enough & (ema_slope > 0))
        .then(1)
        .when((pl.col("close") < pl.col(ema)) & far_enough & (ema_slope < 0))
        .then(-1)
        .otherwise(0)
    )

    chg = trend_dir.cast(pl.Int32) != trend_dir.cast(pl.Int32).shift(1).fill_null(0)
    group_id = chg.fill_null(True).cum_sum()
    td = trend_dir.fill_null(0).cast(pl.Int32)
    trend_count = ((1 + pl.int_range(0, pl.len()).over(group_id)) * (td != 0).cast(pl.Int32)).cast(
        pl.Int32
    )

    ema_dist = (pl.col("close") - pl.col(ema)).abs() / pl.col(atr)

    entry_mask = (
        (trend_count >= trend_bars_min)
        & pl.col("adx_14").is_not_null()
        & (pl.col("adx_14") > adx_threshold)
        & (ema_dist < 1.0)
        & regime_ok
    )

    entry_long = (entry_mask & (trend_dir == 1)).cast(pl.Int32)
    entry_short = (entry_mask & (trend_dir == -1)).cast(pl.Int32)

    temp = df.select(
        pl.col("timestamp"),
        entry_long.alias("entry_long"),
        entry_short.alias("entry_short"),
        regime_ok.cast(pl.Int32).alias("regime_ok"),
    )

    close_arr = df["close"].to_list()
    atr_arr = df[atr].fill_nan(0).fill_null(0).to_list()
    long_arr = temp["entry_long"].to_list()
    short_arr = temp["entry_short"].to_list()
    regime_arr = temp["regime_ok"].to_list()

    mgr = _PositionManager(
        use_chandelier=bar_hours is not None and bar_hours <= 4,
        max_hold_bars=trend_bars_min * 6,
    )

    signal = [0.0] * len(df)
    for i in range(len(df)):
        c = close_arr[i]
        a = atr_arr[i] if atr_arr[i] > 0 else 1.0
        r = regime_arr[i] == 1

        if mgr.direction == 0:
            if r and long_arr[i]:
                mgr.enter(1, c, a, atr_stop_mult=0)
            elif r and short_arr[i]:
                mgr.enter(-1, c, a, atr_stop_mult=0)

        d = mgr.step(c, a, short_arr[i] if mgr.direction == 1 else long_arr[i], r)
        signal[i] = float(d)

    return df.with_columns(pl.Series(signal).alias("signal"))
