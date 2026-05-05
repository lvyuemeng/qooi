"""Intraday limit-order strategies — book imbalance + momentum + risk management."""

from __future__ import annotations

import math
from dataclasses import dataclass

import polars as pl


@dataclass
class _TradeState:
    direction: int = 0
    entry: float = 0.0
    sl: float = 0.0
    tp: float = 0.0
    trail_high: float = 0.0
    trail_low: float = 0.0
    trail_active: bool = False
    cooldown: int = 0

    def exit_if(self, close: float, atr: float) -> bool:
        if self.direction == 1:
            if close >= self.tp or close <= self.sl:
                return True
            self.trail_high = max(self.trail_high, close)
            if self.trail_active and close < self.trail_high - atr * 4.0:
                return True
            if not self.trail_active and close >= self.entry + atr * 2.0:
                self.trail_active = True
        elif self.direction == -1:
            if close <= self.tp or close >= self.sl:
                return True
            self.trail_low = min(self.trail_low, close)
            if self.trail_active and close > self.trail_low + atr * 4.0:
                return True
            if not self.trail_active and close <= self.entry - atr * 2.0:
                self.trail_active = True
        return False

    def enter(self, direction: int, price: float, stop: float, target: float) -> None:
        self.direction = direction
        self.entry = price
        self.sl = stop
        self.tp = target
        self.trail_high = price if direction == 1 else 0.0
        self.trail_low = price if direction == -1 else 0.0
        self.trail_active = False

    def reset(self, cooldown_bars: int = 5) -> None:
        self.direction = 0
        self.cooldown = cooldown_bars


def _safe(arr: list[float], idx: int, fallback: float = 1.0) -> float:
    return arr[idx] if idx < len(arr) and arr[idx] is not None and arr[idx] > 0 else fallback


def _ema(arr: list[float], period: int) -> list[float]:
    """Simple EMA for use in the loop (not a Polars Expr)."""
    result = [0.0] * len(arr)
    alpha = 2.0 / (period + 1)
    result[0] = arr[0] if arr else 0.0
    for i in range(1, len(arr)):
        result[i] = alpha * arr[i] + (1 - alpha) * result[i - 1]
    return result


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _bar_hours(df: pl.DataFrame) -> float:
    if df.height < 2:
        return 1.0
    delta_ms = float(df["timestamp"][1] - df["timestamp"][0])
    return max(1.0 / 60.0, delta_ms / 3_600_000.0)


def order_book_imbalance_signal(
    df: pl.DataFrame,
    ob_bid_vol: str = "ob_bid_vol",
    ob_ask_vol: str = "ob_ask_vol",
    ob_bid_price: str = "ob_bid_price",
    ob_ask_price: str = "ob_ask_price",
    imbalance_threshold: float = 0.15,
    atr_period: int = 14,
    atr_stop_mult: float = 2.5,
    risk_reward: float = 1.5,
    momentum_bars: int = 12,
    higher_tf_ratio: int = 4,
    cooldown_bars: int = 8,
    vol_ma_period: int = 20,
    entry_slippage_pct: float = 0.0002,
) -> pl.DataFrame:
    """Order-book imbalance with higher-timeframe trend filter.

    Multi-timeframe confirmation:
      - **Higher timeframe EMA20 trend**: ``close >/ < EMA(price, 20)``
        computed on bars aggregated by ``higher_tf_ratio`` (e.g. 1H →
        effective 4H EMA20).  This prevents trading against the macro
        direction (catches the 61% WR with 60% DD problem).

      - **Position sizing via Kelly / volatility scaling**: signal
        magnitude = imbalance_strength × volatility_factor, where
        volatility_factor = max(0.3, min(1.0, atr_ma / atr)) so that
        entries in high-volatility periods take smaller risk.

      - **Adaptive momentum**: ``momentum_bars`` replaced by
        ``max(3, avg_bars_per_hour * higher_tf_ratio)`` so 1H uses
        ~12-bar momentum (12 hours), 4H uses ~3-bar (12 hours).
    """
    close = df["close"].to_list()
    vol = df["vol"].fill_nan(0).fill_null(0).to_list()

    bid_v = (
        df[ob_bid_vol].fill_nan(0).fill_null(0).to_list()
        if ob_bid_vol in df.columns
        else [0.0] * len(df)
    )
    ask_v = (
        df[ob_ask_vol].fill_nan(0).fill_null(0).to_list()
        if ob_ask_vol in df.columns
        else [0.0] * len(df)
    )
    bid_p = df[ob_bid_price].to_list() if ob_bid_price in df.columns else close
    ask_p = df[ob_ask_price].to_list() if ob_ask_price in df.columns else close

    # ATR
    atr_raw = [1.0] * len(df)
    atr_ma = [1.0] * len(df)
    if f"atr_{atr_period}" in df.columns:
        s = df[f"atr_{atr_period}"].fill_nan(0).fill_null(0).to_list()
        atr_raw = s
        atr_ma = list(pl.Series(s).rolling_mean(20)) if len(s) >= 20 else s[:]

    # Volume SMA
    vol_sma = (
        list(pl.Series(vol).rolling_mean(vol_ma_period)) if len(vol) >= vol_ma_period else vol[:]
    )

    # Higher-timeframe trend: EMA20 on price sampled at higher_tf_ratio bars
    # e.g. higher_tf_ratio=4 on 1H data ≈ 4H EMA20
    sampled = [close[i] for i in range(0, len(close), higher_tf_ratio)]
    ht_ema20 = _ema(sampled, 20)

    # Map back to original bar count
    ht_ema_full = [0.0] * len(close)
    for i in range(len(close)):
        ht_ema_full[i] = ht_ema20[min(i // higher_tf_ratio, len(ht_ema20) - 1)]

    # Keep the momentum horizon stable in wall-clock time instead of raw bar count.
    bar_hours = _bar_hours(df)
    effective_momentum = max(3, int(round(momentum_bars / max(bar_hours, 1.0 / 60.0))))

    signal = [0.0] * len(df)
    t = _TradeState()
    active_strength = 0.0

    for i in range(effective_momentum, len(df)):
        if t.cooldown > 0:
            t.cooldown -= 1
            if t.direction == 0:
                continue

        a_v = _safe(atr_raw, i, 1.0)
        a_ma = _safe(atr_ma, i, a_v)
        v_ma = _safe(vol_sma, i, 1.0)
        bv = _safe(bid_v, i, 0.0)
        av = _safe(ask_v, i, 0.0)

        # --- exit ---
        if t.direction != 0:
            if t.exit_if(close[i], a_v):
                t.reset(cooldown_bars)
                active_strength = 0.0
                signal[i] = 0.0
                continue
            signal[i] = float(t.direction) * active_strength
            continue

        # --- entry gates ---
        if not (a_v >= a_ma * 0.5 and (v_ma <= 0 or vol[i] >= v_ma * 0.7) and (bv + av) > 1e-8):
            continue

        imbalance = (bv - av) / (bv + av)
        eff_threshold = (
            imbalance_threshold * max(0.5, a_v / a_ma) if a_ma > 0 else imbalance_threshold
        )
        if not (imbalance > eff_threshold or imbalance < -eff_threshold):
            continue

        ob_dir = 1 if imbalance > eff_threshold else -1

        # --- higher-timeframe trend agreement ---
        trend_dir = 1 if close[i] > ht_ema_full[i] else (-1 if close[i] < ht_ema_full[i] else 0)
        if ob_dir != trend_dir:
            continue

        # --- momentum (original bar scale) ---
        price_change = (close[i] - close[i - effective_momentum]) / close[i - effective_momentum]
        momentum_dir = 1 if price_change > 0 else (-1 if price_change < 0 else 0)
        if ob_dir != momentum_dir:
            continue

        # --- enter ---
        if ob_dir == 1:
            entry_price = ask_p[i] * (1.0 + entry_slippage_pct)
            stop = entry_price - atr_stop_mult * a_v
            target = entry_price + atr_stop_mult * a_v * risk_reward
        else:
            entry_price = bid_p[i] * (1.0 - entry_slippage_pct)
            stop = entry_price + atr_stop_mult * a_v
            target = entry_price - atr_stop_mult * a_v * risk_reward

        t.enter(ob_dir, entry_price, stop, target)
        # Volatility-scaled position size
        vol_factor = max(0.3, min(1.0, a_ma / a_v)) if a_v > 0 else 1.0
        active_strength = max(0.3, min(1.0, abs(imbalance) / eff_threshold)) * vol_factor
        signal[i] = float(ob_dir) * float(active_strength)

    return df.with_columns(pl.Series(signal).alias("signal"))


def ensemble_intraday_signal(
    df: pl.DataFrame,
    ob_bid_vol: str = "ob_bid_vol",
    ob_ask_vol: str = "ob_ask_vol",
    prehistory_bars: int = 200,
    atr_period: int = 14,
    score_entry_threshold: float = 0.45,
    score_exit_threshold: float = 0.15,
    min_hold_bars: int = 2,
    atr_stop_mult: float = 2.2,
    base_risk_reward: float = 1.8,
) -> pl.DataFrame:
    """Self-adaptive intraday ensemble using OHLCV and optional order-book.

    Design goals:
    - adapt across assets by normalizing all signals with ATR / rolling stats
    - adapt across timeframes by scaling momentum windows from bar duration
    - use prehistory to estimate regime before first trade
    - unify trend, momentum, volume and optional order-book into one score
    - emit fractional target position for dynamic sizing
    """
    if df.is_empty():
        return df.with_columns(pl.lit(0.0).alias("signal"))

    bar_hours = _bar_hours(df)
    close = df["close"].to_list()
    vol = df["vol"].fill_nan(0).fill_null(0).to_list()

    atr_values = [1.0] * len(df)
    if f"atr_{atr_period}" in df.columns:
        atr_values = df[f"atr_{atr_period}"].fill_nan(0).fill_null(0).to_list()
    atr_ma = list(pl.Series(atr_values).rolling_mean(100)) if len(df) >= 100 else atr_values[:]

    ema_fast_period = max(12, int(round(24 / bar_hours)))
    ema_slow_period = max(ema_fast_period * 3, 36)
    ema_fast = _ema(close, ema_fast_period)
    ema_slow = _ema(close, ema_slow_period)

    fast_mom_bars = max(3, int(round(12 / bar_hours)))
    slow_mom_bars = max(fast_mom_bars * 4, 12)

    vol_ma = list(pl.Series(vol).rolling_mean(20)) if len(df) >= 20 else vol[:]
    vol_std = list(pl.Series(vol).rolling_std(20)) if len(df) >= 20 else [0.0] * len(df)

    has_ob = ob_bid_vol in df.columns and ob_ask_vol in df.columns
    bid_v = df[ob_bid_vol].fill_nan(0).fill_null(0).to_list() if has_ob else [0.0] * len(df)
    ask_v = df[ob_ask_vol].fill_nan(0).fill_null(0).to_list() if has_ob else [0.0] * len(df)

    signal = [0.0] * len(df)
    trade = _TradeState()
    active_size = 0.0
    active_entry_idx = -1

    for i in range(max(prehistory_bars, slow_mom_bars, ema_slow_period), len(df)):
        if trade.cooldown > 0:
            trade.cooldown -= 1

        atr_v = _safe(atr_values, i, 1.0)
        atr_ref = _safe(atr_ma, i, atr_v)
        vol_ref = _safe(vol_ma, i, 1.0)
        vol_sigma = _safe(vol_std, i, 0.0)

        # 1) regime: avoid dead and explosive zones
        atr_ratio = atr_v / atr_ref if atr_ref > 0 else 1.0
        regime_ok = 0.6 <= atr_ratio <= 2.5
        if not regime_ok:
            trade.reset(cooldown_bars=3)
            active_size = 0.0
            continue

        # 2) multi-timescale trend identification
        trend_raw = (close[i] - ema_slow[i]) / max(atr_v, 1e-9)
        trend_score = _clip(trend_raw / 2.0, -1.0, 1.0)
        slope_raw = (ema_fast[i] - ema_fast[i - fast_mom_bars]) / max(atr_v, 1e-9)
        slope_score = _clip(slope_raw / 2.0, -1.0, 1.0)

        # 3) momentum across two horizons
        mom_fast = (close[i] - close[i - fast_mom_bars]) / max(atr_v, 1e-9)
        mom_slow = (close[i] - close[i - slow_mom_bars]) / max(atr_v, 1e-9)
        mom_fast_score = _clip(mom_fast / 2.0, -1.0, 1.0)
        mom_slow_score = _clip(mom_slow / 3.0, -1.0, 1.0)

        # 4) volume as confidence, not primary direction
        vol_z = (vol[i] - vol_ref) / vol_sigma if vol_sigma > 0 else 0.0
        volume_conf = _clip(0.5 + max(0.0, vol_z) / 4.0, 0.25, 1.0)

        # 5) optional order-book as additive directional edge
        ob_score = 0.0
        if has_ob:
            total = bid_v[i] + ask_v[i]
            if total > 1e-8:
                ob_score = _clip(((bid_v[i] - ask_v[i]) / total) / 0.35, -1.0, 1.0)

        # 6) confidence-weighted ensemble score
        core_score = (
            0.35 * trend_score
            + 0.20 * slope_score
            + 0.20 * mom_fast_score
            + 0.20 * mom_slow_score
            + 0.05 * ob_score
        )
        net_score = core_score * volume_conf

        # 7) dynamic position sizing
        volatility_factor = _clip(atr_ref / max(atr_v, 1e-9), 0.35, 1.0)
        confidence_size = _clip(abs(net_score), 0.0, 1.0)
        target_size = _clip(confidence_size * volatility_factor, 0.0, 1.0)

        # exit / hold
        if trade.direction != 0:
            should_exit = trade.exit_if(close[i], atr_v)
            score_flip = (trade.direction == 1 and net_score < -score_exit_threshold) or (
                trade.direction == -1 and net_score > score_exit_threshold
            )
            stale = (
                active_entry_idx >= 0
                and (i - active_entry_idx) >= min_hold_bars
                and abs(net_score) < score_exit_threshold
            )
            if should_exit or score_flip or stale:
                trade.reset(cooldown_bars=5)
                active_size = 0.0
            signal[i] = float(trade.direction) * active_size
            continue

        if trade.cooldown > 0:
            continue

        if abs(net_score) < score_entry_threshold:
            continue

        direction = 1 if net_score > 0 else -1
        entry = close[i]
        stop_dist = atr_stop_mult * atr_v
        rr = base_risk_reward + 0.5 * min(1.0, abs(net_score))
        if direction == 1:
            trade.enter(direction, entry, entry - stop_dist, entry + stop_dist * rr)
        else:
            trade.enter(direction, entry, entry + stop_dist, entry - stop_dist * rr)
        active_size = target_size
        active_entry_idx = i
        signal[i] = float(direction) * active_size

    return df.with_columns(pl.Series(signal).alias("signal"))


def cvd_proxy_signal(
    df: pl.DataFrame,
    atr_period: int = 14,
    lookback: int = 24,
    divergence_threshold: float = 0.8,
) -> pl.DataFrame:
    """CVD proxy using signed volume from OHLCV only.

    Since historical aggressive trade classification is unavailable in the
    current stack, approximate CVD as signed volume:

      signed_volume = sign(close - close[-1]) * volume

    Then compare price breakout vs rolling CVD breakout for divergence.
    """
    if df.is_empty():
        return df.with_columns(pl.lit(0.0).alias("signal"))

    close = df["close"].to_list()
    vol = df["vol"].fill_nan(0).fill_null(0).to_list()

    signed = [0.0] * len(df)
    for i in range(1, len(df)):
        direction = 1.0 if close[i] > close[i - 1] else (-1.0 if close[i] < close[i - 1] else 0.0)
        signed[i] = direction * vol[i]

    cvd = [0.0] * len(df)
    for i in range(1, len(df)):
        cvd[i] = cvd[i - 1] + signed[i]

    signal = [0.0] * len(df)
    for i in range(lookback, len(df)):
        price_high = max(close[i - lookback : i])
        price_low = min(close[i - lookback : i])
        cvd_high = max(cvd[i - lookback : i])
        cvd_low = min(cvd[i - lookback : i])

        price_new_high = close[i] >= price_high
        price_new_low = close[i] <= price_low
        cvd_confirm_high = cvd[i] >= cvd_high * divergence_threshold if cvd_high != 0 else False
        cvd_confirm_low = cvd[i] <= cvd_low * divergence_threshold if cvd_low != 0 else False

        if price_new_high and not cvd_confirm_high:
            signal[i] = -1.0
        elif price_new_low and not cvd_confirm_low:
            signal[i] = 1.0

    return df.with_columns(
        [
            pl.Series(cvd).alias("cvd_proxy"),
            pl.Series(signal).alias("signal"),
        ]
    )


def pair_zscore_signal(
    df: pl.DataFrame,
    pair_close_col: str = "pair_close",
    lookback: int = 120,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
) -> pl.DataFrame:
    """Single-leg expression of a pair-trading signal.

    ``pair_close_col`` should contain the comparison asset close (same
    timestamps). We approximate the spread with the log-price ratio and
    trade the primary asset direction only:

      spread = log(close) - log(pair_close)

    If spread is too low, the primary asset is relatively cheap → long.
    If too high, primary asset is relatively expensive → short.
    """
    if df.is_empty() or pair_close_col not in df.columns:
        return df.with_columns(pl.lit(0.0).alias("signal"))

    close = df["close"].to_list()
    pair_close = df[pair_close_col].fill_nan(0).fill_null(0).to_list()

    spread = [0.0] * len(df)
    for i in range(len(df)):
        if close[i] > 0 and pair_close[i] > 0:
            spread[i] = math.log(close[i]) - math.log(pair_close[i])

    signal = [0.0] * len(df)
    active = 0
    for i in range(lookback, len(df)):
        window = spread[i - lookback : i]
        mean = sum(window) / len(window)
        var = sum((x - mean) ** 2 for x in window) / max(1, len(window) - 1)
        std = math.sqrt(var) if var > 0 else 0.0
        z = (spread[i] - mean) / std if std > 0 else 0.0

        if active == 0:
            if z <= -entry_z:
                active = 1
            elif z >= entry_z:
                active = -1
        elif abs(z) <= exit_z:
            active = 0

        signal[i] = float(active)

    return df.with_columns(
        [
            pl.Series(spread).alias("pair_spread"),
            pl.Series(signal).alias("signal"),
        ]
    )


def multi_factor_intraday_signal(
    df: pl.DataFrame,
    pair_close_col: str | None = None,
    funding_rate_col: str | None = None,
    score_entry_threshold: float = 0.40,
    score_exit_threshold: float = 0.15,
    atr_period: int = 14,
    atr_stop_mult: float = 2.0,
    risk_reward: float = 1.8,
) -> pl.DataFrame:
    """Multi-factor intraday ensemble using currently available data.

    Modules:
    - trend + momentum + volatility regime
    - optional order-book score
    - optional funding-rate extreme score
    - CVD-proxy divergence score
    - optional pair z-score score

    This is the practical version of the suggested architecture that can
    run today on the current codebase.
    """
    if df.is_empty():
        return df.with_columns(pl.lit(0.0).alias("signal"))

    close = df["close"].to_list()
    vol = df["vol"].fill_nan(0).fill_null(0).to_list()
    bar_hours = _bar_hours(df)

    atr = (
        df[f"atr_{atr_period}"].fill_nan(0).fill_null(0).to_list()
        if f"atr_{atr_period}" in df.columns
        else [1.0] * len(df)
    )
    atr_ma = list(pl.Series(atr).rolling_mean(100)) if len(df) >= 100 else atr[:]

    ema_fast_period = max(8, int(round(12 / bar_hours)))
    ema_slow_period = max(ema_fast_period * 4, 32)
    ema_fast = _ema(close, ema_fast_period)
    ema_slow = _ema(close, ema_slow_period)

    fast_mom = max(3, int(round(8 / bar_hours)))
    slow_mom = max(12, int(round(24 / bar_hours)))

    # CVD proxy precompute
    signed = [0.0] * len(df)
    for i in range(1, len(df)):
        d = 1.0 if close[i] > close[i - 1] else (-1.0 if close[i] < close[i - 1] else 0.0)
        signed[i] = d * vol[i]
    cvd = [0.0] * len(df)
    for i in range(1, len(df)):
        cvd[i] = cvd[i - 1] + signed[i]

    # Optional order-book arrays
    has_ob = "ob_bid_vol" in df.columns and "ob_ask_vol" in df.columns
    bid_v = df["ob_bid_vol"].fill_nan(0).fill_null(0).to_list() if has_ob else [0.0] * len(df)
    ask_v = df["ob_ask_vol"].fill_nan(0).fill_null(0).to_list() if has_ob else [0.0] * len(df)

    # Optional regime features
    _has_regime = "regime_score" in df.columns

    # Optional funding rate
    fr_col = funding_rate_col if funding_rate_col is not None else ""
    has_fr = bool(fr_col) and fr_col in df.columns
    funding = df[fr_col].fill_nan(0).fill_null(0).to_list() if has_fr else [0.0] * len(df)

    # Optional pair pricing
    pair_col = pair_close_col if pair_close_col is not None else ""
    has_pair = bool(pair_col) and pair_col in df.columns
    pair_close = df[pair_col].fill_nan(0).fill_null(0).to_list() if has_pair else [0.0] * len(df)

    signal = [0.0] * len(df)
    trade = _TradeState()
    size = 0.0
    loss_streak = 0
    prev_trend_dir = 0
    trend_cooldown = 0
    prev_net_score = 0.0
    trade_entry_price = 0.0

    for i in range(max(200, ema_slow_period, slow_mom, 120), len(df)):
        if trade.cooldown > 0:
            trade.cooldown -= 1

        atr_v = _safe(atr, i, 1.0)
        atr_ref = _safe(atr_ma, i, atr_v)
        atr_ratio = atr_v / atr_ref if atr_ref > 0 else 1.0
        if not (0.6 <= atr_ratio <= 2.5):
            if trade.direction != 0:
                trade.reset(cooldown_bars=3)
            continue

        # Trend / momentum core
        trend_score = _clip((close[i] - ema_slow[i]) / max(atr_v * 2.0, 1e-9), -1.0, 1.0)
        trend_dir = 1 if trend_score > 0 else (-1 if trend_score < 0 else 0)
        if prev_trend_dir != 0 and trend_dir != 0 and trend_dir != prev_trend_dir:
            trend_cooldown = max(trend_cooldown, 4)
        prev_trend_dir = trend_dir or prev_trend_dir
        if trend_cooldown > 0:
            trend_cooldown -= 1
        slope_score = _clip(
            (ema_fast[i] - ema_fast[i - fast_mom]) / max(atr_v * 2.0, 1e-9), -1.0, 1.0
        )
        mom_fast_score = _clip((close[i] - close[i - fast_mom]) / max(atr_v * 2.0, 1e-9), -1.0, 1.0)
        mom_slow_score = _clip((close[i] - close[i - slow_mom]) / max(atr_v * 3.0, 1e-9), -1.0, 1.0)

        # CVD divergence score
        cvd_score = 0.0
        if i >= 24:
            price_high = max(close[i - 24 : i])
            price_low = min(close[i - 24 : i])
            cvd_high = max(cvd[i - 24 : i])
            cvd_low = min(cvd[i - 24 : i])
            if close[i] >= price_high and cvd[i] < cvd_high:
                cvd_score = -0.6
            elif close[i] <= price_low and cvd[i] > cvd_low:
                cvd_score = 0.6

        # Funding-rate extreme score
        funding_score = 0.0
        if has_fr:
            r = funding[i]
            if r >= 0.002:
                funding_score = -_clip(r / 0.00375, 0.0, 1.0)
            elif r <= -0.002:
                funding_score = _clip(abs(r) / 0.00375, 0.0, 1.0)

        # Pair z-score
        pair_score = 0.0
        if has_pair and pair_close[i] > 0 and i >= 120:
            spread_window = [
                math.log(close[j]) - math.log(pair_close[j])
                for j in range(i - 120, i)
                if pair_close[j] > 0 and close[j] > 0
            ]
            if spread_window:
                mean = sum(spread_window) / len(spread_window)
                var = sum((x - mean) ** 2 for x in spread_window) / max(1, len(spread_window) - 1)
                std = math.sqrt(var) if var > 0 else 0.0
                if std > 0 and close[i] > 0:
                    z = (math.log(close[i]) - math.log(pair_close[i]) - mean) / std
                    pair_score = _clip(-z / 2.5, -1.0, 1.0)

        # Optional OBI score
        ob_score = 0.0
        if has_ob:
            total = bid_v[i] + ask_v[i]
            if total > 1e-8:
                ob_score = _clip(((bid_v[i] - ask_v[i]) / total) / 0.35, -1.0, 1.0)

        # Confidence-weighted fusion
        net_score = (
            0.28 * trend_score
            + 0.18 * slope_score
            + 0.16 * mom_fast_score
            + 0.16 * mom_slow_score
            + 0.10 * cvd_score
            + 0.06 * funding_score
            + 0.06 * pair_score
            + 0.06 * ob_score
        )
        persisted_score = 0.5 * (net_score + prev_net_score)
        prev_net_score = net_score

        # Regime-aware bias: if regime_score column exists, damp entries
        # against the macro direction and amplify aligned entries
        regime_bias = 0.0
        if _has_regime:
            rs = df["regime_score"][i] if i < len(df) else 0.0
            rstr = (
                df["regime_strength"][i] if i < len(df) and "regime_strength" in df.columns else 0.0
            )
            # Only apply bias when regime conviction is strong (>0.3)
            if rstr > 0.3:
                regime_bias = _clip(rs * 0.15, -0.15, 0.15)
        persisted_score = persisted_score + regime_bias

        # Dynamic sizing with risk budget
        vol_factor = _clip(atr_ref / max(atr_v, 1e-9), 0.25, 1.0)
        loss_factor = 1.0
        if loss_streak >= 3:
            loss_factor = 0.25
        elif loss_streak == 2:
            loss_factor = 0.5
        elif loss_streak == 1:
            loss_factor = 0.75
        # Risk budget: cap position size so that a 2×ATR stop costs ≤ 2% equity
        risk_per_unit = atr_v / max(close[i], 1e-9) * atr_stop_mult
        risk_budget = 0.02 / max(risk_per_unit, 1e-9)
        size_target = _clip(abs(persisted_score) * vol_factor * loss_factor, 0.0, risk_budget)

        if trade.direction != 0:
            flip = (trade.direction == 1 and persisted_score < -score_exit_threshold) or (
                trade.direction == -1 and persisted_score > score_exit_threshold
            )
            weak = abs(persisted_score) < score_exit_threshold
            should_exit = trade.exit_if(close[i], atr_v) or flip or weak
            if should_exit:
                pnl = (
                    (close[i] / trade_entry_price - 1.0) * trade.direction
                    if trade_entry_price > 0
                    else 0.0
                )
                loss_streak = loss_streak + 1 if pnl < 0 else 0
                trade.reset(cooldown_bars=4)
                size = 0.0
            signal[i] = float(trade.direction) * size
            continue

        if trade.cooldown > 0 or trend_cooldown > 0 or abs(persisted_score) < score_entry_threshold:
            continue

        direction = 1 if persisted_score > 0 else -1
        # Long-short symmetry: apply equal threshold in both directions
        if abs(persisted_score) < score_entry_threshold:
            continue
        stop_dist = atr_stop_mult * atr_v
        rr = risk_reward + 0.3 * min(1.0, abs(persisted_score))
        entry = close[i]
        if direction == 1:
            trade.enter(direction, entry, entry - stop_dist, entry + stop_dist * rr)
        else:
            trade.enter(direction, entry, entry + stop_dist, entry - stop_dist * rr)
        trade_entry_price = entry
        size = size_target
        signal[i] = float(direction) * size

    return df.with_columns(pl.Series(signal).alias("signal"))
