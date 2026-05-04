"""Intraday limit-order strategies — book imbalance + momentum + risk management."""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl


@dataclass
class _TradeState:
    """Purity: pure state machine — no column access, no DataFrame coupling."""

    direction: int = 0  # 1 long, -1 short, 0 flat
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
            if self.trail_active and close < self.trail_high - atr * 2.0:
                return True
            if not self.trail_active and close >= self.entry + atr * 1.0:
                self.trail_active = True
        elif self.direction == -1:
            if close <= self.tp or close >= self.sl:
                return True
            self.trail_low = min(self.trail_low, close)
            if self.trail_active and close > self.trail_low + atr * 2.0:
                return True
            if not self.trail_active and close <= self.entry - atr * 1.0:
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


def order_book_imbalance_signal(
    df: pl.DataFrame,
    ob_bid_vol: str = "ob_bid_vol",
    ob_ask_vol: str = "ob_ask_vol",
    ob_bid_price: str = "ob_bid_price",
    ob_ask_price: str = "ob_ask_price",
    imbalance_threshold: float = 0.25,
    atr_period: int = 14,
    atr_stop_mult: float = 1.5,
    risk_reward: float = 2.0,
    momentum_bars: int = 3,
    cooldown_bars: int = 5,
    vol_ma_period: int = 20,
    entry_slippage_pct: float = 0.0002,
) -> pl.DataFrame:
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

    atr_raw = [1.0] * len(df)
    atr_ma = [1.0] * len(df)
    if f"atr_{atr_period}" in df.columns:
        s = df[f"atr_{atr_period}"].fill_nan(0).fill_null(0).to_list()
        atr_raw = s
        atr_ma = list(pl.Series(s).rolling_mean(20)) if len(s) >= 20 else s[:]

    vol_sma = (
        list(pl.Series(vol).rolling_mean(vol_ma_period)) if len(vol) >= vol_ma_period else vol[:]
    )

    signal = [0.0] * len(df)
    t = _TradeState()

    for i in range(momentum_bars, len(df)):
        # --- cool-down ---
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
            signal[i] = float(t.direction)
            continue

        # --- entry gates (short-circuit chain) ---
        atr_ok = a_v >= a_ma * 0.5
        vol_ok = v_ma <= 0 or vol[i] >= v_ma * 0.7
        total = bv + av
        if not (atr_ok and vol_ok and total > 1e-8):
            continue

        imbalance = (bv - av) / total
        eff_threshold = (
            imbalance_threshold * max(0.5, a_v / a_ma) if a_ma > 0 else imbalance_threshold
        )
        if not (imbalance > eff_threshold or imbalance < -eff_threshold):
            continue

        price_change = (close[i] - close[i - momentum_bars]) / close[i - momentum_bars]
        momentum_dir = 1 if price_change > 0 else (-1 if price_change < 0 else 0)
        ob_dir = 1 if imbalance > eff_threshold else -1
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
        strength = max(0.5, min(1.0, abs(imbalance) / eff_threshold))
        signal[i] = float(ob_dir) * strength

    return df.with_columns(pl.Series(signal).alias("signal"))
