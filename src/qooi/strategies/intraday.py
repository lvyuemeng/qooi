"""Multi-factor intraday ensemble — production signal used by backtest + live.

Single function: ``multi_factor_intraday_signal(df)`` which fuses trend,
momentum, CVD-proxy, order-book, funding rate, pair z-score, and regime
bias into one fractional signal column.

All intermediate computations use Polars Series/Expr where possible.
Only the position-hold state machine remains a lightweight Python loop.
"""

from __future__ import annotations

import math

import polars as pl


def _ema(arr: list[float], period: int) -> list[float]:
    result = [0.0] * len(arr)
    alpha = 2.0 / (period + 1)
    result[0] = arr[0] if arr else 0.0
    for i in range(1, len(arr)):
        result[i] = alpha * arr[i] + (1 - alpha) * result[i - 1]
    return result


def _clip(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _safe(arr: list[float], idx: int, fallback: float = 1.0) -> float:
    return arr[idx] if idx < len(arr) and arr[idx] is not None and arr[idx] > 0 else fallback


def multi_factor_intraday_signal(
    df: pl.DataFrame,
    ob_bid_vol: str = "ob_bid_vol",
    ob_ask_vol: str = "ob_ask_vol",
    pair_close_col: str | None = None,
    funding_rate_col: str | None = None,
    score_entry_threshold: float = 0.40,
    score_exit_threshold: float = 0.15,
    atr_period: int = 14,
    atr_stop_mult: float = 2.0,
    risk_reward: float = 1.8,
) -> pl.DataFrame:
    """Multi-factor intraday ensemble.

    Fuses: trend (EMA slow), slope (EMA fast), momentum (fast/slow),
    CVD-divergence, optional order-book, optional funding rate, optional
    pair z-score, and regime bias from ``regime_score`` column.

    Returns ``df`` with a ``signal`` column in [-1, 1] where magnitude
    encodes position size and sign encodes direction.
    """
    if df.is_empty():
        return df.with_columns(pl.lit(0.0).alias("signal"))

    close = df["close"].to_list()
    vol = df["vol"].fill_nan(0).fill_null(0).to_list()

    # ---- precompute shared arrays (Polars rolling where possible) ----------
    atr = (
        df[f"atr_{atr_period}"].fill_nan(0).fill_null(0).to_list()
        if f"atr_{atr_period}" in df.columns
        else [1.0] * len(df)
    )
    atr_ma = list(pl.Series(atr).rolling_mean(100)) if len(df) >= 100 else atr[:]

    # EMA
    bar_hours = 1.0  # assume 1H by default; adjusted if column exists
    if df.height >= 2:
        bar_hours = max(0.25, (df["timestamp"][1] - df["timestamp"][0]) / 3_600_000.0)
    ema_fast_period = max(8, int(round(12 / max(bar_hours, 0.25))))
    ema_slow_period = max(ema_fast_period * 4, 32)
    ema_fast = _ema(close, ema_fast_period)
    ema_slow = _ema(close, ema_slow_period)

    fast_mom = max(3, int(round(8 / max(bar_hours, 0.25))))
    slow_mom = max(12, int(round(24 / max(bar_hours, 0.25))))

    # CVD proxy
    signed = [0.0] * len(df)
    for i in range(1, len(df)):
        d = 1.0 if close[i] > close[i - 1] else (-1.0 if close[i] < close[i - 1] else 0.0)
        signed[i] = d * vol[i]
    cvd = [0.0] * len(df)
    for i in range(1, len(df)):
        cvd[i] = cvd[i - 1] + signed[i]

    # optional columns
    has_ob = ob_bid_vol in df.columns and ob_ask_vol in df.columns
    bid_v = df[ob_bid_vol].fill_nan(0).fill_null(0).to_list() if has_ob else [0.0] * len(df)
    ask_v = df[ob_ask_vol].fill_nan(0).fill_null(0).to_list() if has_ob else [0.0] * len(df)

    _has_regime = "regime_score" in df.columns

    fr_col = funding_rate_col or ""
    has_fr = bool(fr_col) and fr_col in df.columns
    funding = df[fr_col].fill_nan(0).fill_null(0).to_list() if has_fr else [0.0] * len(df)

    pair_col = pair_close_col or ""
    has_pair = bool(pair_col) and pair_col in df.columns
    pair_close = df[pair_col].fill_nan(0).fill_null(0).to_list() if has_pair else [0.0] * len(df)

    # ---- stateful loop ----------------------------------------------------
    signal = [0.0] * len(df)
    # compact state machine — no dataclass needed for the small loop
    direction = 0
    entry = 0.0
    sl = 0.0
    tp = 0.0
    trail_high = 0.0
    trail_low = 0.0
    trail_active = False
    cooldown = 0
    size = 0.0
    loss_streak = 0
    prev_trend_dir = 0
    trend_cooldown = 0
    prev_net_score = 0.0
    trade_entry_price = 0.0

    for i in range(max(200, ema_slow_period, slow_mom, 120), len(df)):
        if cooldown:
            cooldown -= 1

        atr_v = _safe(atr, i, 1.0)
        atr_ref = _safe(atr_ma, i, atr_v)
        atr_ratio = atr_v / atr_ref if atr_ref > 0 else 1.0
        if not (0.6 <= atr_ratio <= 2.5):
            if direction:
                cooldown = 3
                direction = 0
            continue

        # ---- scores -------------------------------------------------------
        trend_score = _clip((close[i] - ema_slow[i]) / max(atr_v * 2.0, 1e-9), -1.0, 1.0)
        trend_dir = 1 if trend_score > 0 else (-1 if trend_score < 0 else 0)
        if prev_trend_dir and trend_dir and trend_dir != prev_trend_dir:
            trend_cooldown = max(trend_cooldown, 4)
        prev_trend_dir = trend_dir or prev_trend_dir
        if trend_cooldown:
            trend_cooldown -= 1

        slope_score = _clip(
            (ema_fast[i] - ema_fast[i - fast_mom]) / max(atr_v * 2.0, 1e-9), -1.0, 1.0
        )
        mom_fast_score = _clip((close[i] - close[i - fast_mom]) / max(atr_v * 2.0, 1e-9), -1.0, 1.0)
        mom_slow_score = _clip((close[i] - close[i - slow_mom]) / max(atr_v * 3.0, 1e-9), -1.0, 1.0)

        # CVD
        cvd_score = 0.0
        if i >= 24:
            ph = max(close[i - 24 : i])
            pl_ = min(close[i - 24 : i])
            ch = max(cvd[i - 24 : i])
            cl = min(cvd[i - 24 : i])
            if close[i] >= ph and cvd[i] < ch:
                cvd_score = -0.6
            elif close[i] <= pl_ and cvd[i] > cl:
                cvd_score = 0.6

        # funding rate
        fr_score = 0.0
        if has_fr:
            r = funding[i]
            if r >= 0.002:
                fr_score = -_clip(r / 0.00375, 0.0, 1.0)
            elif r <= -0.002:
                fr_score = _clip(abs(r) / 0.00375, 0.0, 1.0)

        # pair z
        pair_score = 0.0
        if has_pair and pair_close[i] > 0 and i >= 120:
            sw = [
                math.log(close[j]) - math.log(pair_close[j])
                for j in range(i - 120, i)
                if pair_close[j] > 0 and close[j] > 0
            ]
            if sw:
                mu = sum(sw) / len(sw)
                va = sum((x - mu) ** 2 for x in sw) / max(1, len(sw) - 1)
                sd = math.sqrt(va) if va > 0 else 0.0
                if sd > 0 and close[i] > 0:
                    z = (math.log(close[i]) - math.log(pair_close[i]) - mu) / sd
                    pair_score = _clip(-z / 2.5, -1.0, 1.0)

        ob_score = 0.0
        if has_ob:
            total = bid_v[i] + ask_v[i]
            if total > 1e-8:
                ob_score = _clip(((bid_v[i] - ask_v[i]) / total) / 0.35, -1.0, 1.0)

        # ---- fusion -------------------------------------------------------
        net_score = (
            0.28 * trend_score
            + 0.18 * slope_score
            + 0.16 * mom_fast_score
            + 0.16 * mom_slow_score
            + 0.10 * cvd_score
            + 0.06 * fr_score
            + 0.06 * pair_score
            + 0.06 * ob_score
        )
        persisted = 0.5 * (net_score + prev_net_score)
        prev_net_score = net_score

        regime_bias = 0.0
        if _has_regime:
            rs = df["regime_score"][i] if i < len(df) else 0.0
            rs2 = (
                df["regime_strength"][i] if i < len(df) and "regime_strength" in df.columns else 0.0
            )
            if rs2 > 0.3:
                regime_bias = _clip(rs * 0.15, -0.15, 0.15)
        persisted += regime_bias

        # ---- sizing -------------------------------------------------------
        vol_factor = _clip(atr_ref / max(atr_v, 1e-9), 0.25, 1.0)
        lf = 1.0
        if loss_streak >= 3:
            lf = 0.25
        elif loss_streak == 2:
            lf = 0.5
        elif loss_streak == 1:
            lf = 0.75
        ru = atr_v / max(close[i], 1e-9) * atr_stop_mult
        rb = 0.02 / max(ru, 1e-9)
        size_target = _clip(abs(persisted) * vol_factor * lf, 0.0, rb)

        # ---- exit / hold --------------------------------------------------
        if direction:
            tp_hit = (direction == 1 and close[i] >= tp) or (direction == -1 and close[i] <= tp)
            sl_hit = (direction == 1 and close[i] <= sl) or (direction == -1 and close[i] >= sl)
            if tp_hit or sl_hit:
                direction = 0
                cooldown = 4
            elif direction == 1:
                trail_high = max(trail_high, close[i])
                if trail_active and close[i] < trail_high - atr_v * 4.0:
                    direction = 0
                    cooldown = 4
                elif not trail_active and close[i] >= entry + atr_v * 2.0:
                    trail_active = True
            elif direction == -1:
                trail_low = min(trail_low, close[i])
                if trail_active and close[i] > trail_low + atr_v * 4.0:
                    direction = 0
                    cooldown = 4
                elif not trail_active and close[i] <= entry - atr_v * 2.0:
                    trail_active = True

            if direction:
                signal[i] = float(direction) * size
            else:
                pnl = (
                    (close[i] / trade_entry_price - 1.0) * (1 if direction == 1 else 1)
                    if trade_entry_price > 0
                    else 0.0
                )
                loss_streak = loss_streak + 1 if pnl < 0 else 0
                size = 0.0
            continue

        # ---- entry --------------------------------------------------------
        if cooldown or trend_cooldown or abs(persisted) < score_entry_threshold:
            continue

        direction = 1 if persisted > 0 else -1
        stop_dist = atr_stop_mult * atr_v
        rr = risk_reward + 0.3 * min(1.0, abs(persisted))
        entry = close[i]
        if direction == 1:
            sl = entry - stop_dist
            tp = entry + stop_dist * rr
            trail_high = entry
        else:
            sl = entry + stop_dist
            tp = entry - stop_dist * rr
            trail_low = entry
        trail_active = False
        trade_entry_price = entry
        size = size_target
        signal[i] = float(direction) * size

    return df.with_columns(pl.Series(signal).alias("signal"))
