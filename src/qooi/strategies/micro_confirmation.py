"""Microstructure confirmation layer — OFI proxy + OBI snapshot + funding alignment.

Used as a post-filter on any directional signal to improve alpha quality
by checking whether real order-flow agrees with the signal direction.
"""

from __future__ import annotations

import polars as pl


def _clip(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def add_ofi_flow_columns(
    df: pl.DataFrame,
    *,
    flow_window: int = 12,
    atr_col: str = "atr_14",
) -> pl.DataFrame:
    """Compute order-flow proxy from OHLCV signed volume.

    OFI (Order Flow Imbalance) proxy:
        signed_volume = sign(close - open) * volume
        net_flow      = sum of signed_volume over flow_window
        flow_score    = normalized net_flow in [-1, 1]

    These columns can be used to check whether a directional signal
    (e.g. a long entry) is supported by aggressive flow in the same
    direction.
    """
    if df.is_empty():
        return df

    close = df["close"].to_list()
    open_prices = df["open"].to_list()
    vol = df["vol"].fill_nan(0).fill_null(0).to_list()
    atr = (
        df[atr_col].fill_nan(0).fill_null(0).to_list() if atr_col in df.columns else [1.0] * len(df)
    )

    signed_vol = [0.0] * len(df)
    for i in range(len(df)):
        direction = (
            1.0 if close[i] > open_prices[i] else (-1.0 if close[i] < open_prices[i] else 0.0)
        )
        signed_vol[i] = direction * vol[i]

    net_flow = [0.0] * len(df)
    flow_score = [0.0] * len(df)

    for i in range(flow_window, len(df)):
        nf = sum(signed_vol[i - flow_window + 1 : i + 1])
        net_flow[i] = nf
        a = atr[i] if i < len(atr) and atr[i] > 0 else 1.0
        avg_price = close[i] if close[i] > 0 else 1.0
        scaled = nf / (a * avg_price) if a > 0 else 0.0
        flow_score[i] = _clip(scaled, -1.0, 1.0)

    return df.with_columns(
        [
            pl.Series(signed_vol).alias("ofi_signed_vol"),
            pl.Series(net_flow).alias("ofi_net_flow"),
            pl.Series(flow_score).alias("ofi_flow_score"),
        ]
    )


def check_obi_alignment(
    ob_imbalance: float,
    signal_direction: float,
    min_imbalance: float = 0.15,
) -> float:
    """Validate a signal direction against a real OBI snapshot.

    Args:
        ob_imbalance: Current OBI value in [-1, 1] (from ObSnapshot)
        signal_direction: +1 for long, -1 for short
        min_imbalance: Minimum absolute imbalance to trust

    Returns:
        A multiplier in [0.0, 1.0] — 1.0 means OBI fully confirms,
        0.3 means OBI is uncertain, 0.0 means OBI contradicts.
    """
    if abs(ob_imbalance) < min_imbalance:
        return 0.3  # uncertain — reduce size
    if ob_imbalance > 0 and signal_direction > 0:
        return 1.0  # bid dominant + long → confirm
    if ob_imbalance < 0 and signal_direction < 0:
        return 1.0  # ask dominant + short → confirm
    return 0.0  # contradiction — reject entry


def apply_micro_confirmation(
    df: pl.DataFrame,
    signal_col: str = "signal",
    *,
    flow_col: str = "ofi_flow_score",
    ob_enabled: bool = False,
) -> pl.DataFrame:
    """Post-filter a signal column using OFI flow and optional OBI.

    For each bar where signal != 0:
      - If flow_score disagrees with signal direction → 0.4× multiplier
      - If flow_score agrees → 1.0× multiplier
      - If flow_score is flat (near zero) → 0.6× multiplier

    Returns the input DataFrame with a filtered ``signal`` column.
    """
    if signal_col not in df.columns:
        return df

    sig = df[signal_col].to_list()
    flow = df[flow_col].to_list() if flow_col in df.columns else [0.0] * len(df)

    new_sig = [0.0] * len(df)
    for i in range(len(df)):
        s = sig[i]
        if abs(s) < 1e-9:
            continue
        direction = 1 if s > 0 else -1
        fs = flow[i] if i < len(flow) else 0.0

        if abs(fs) < 0.05:
            multiplier = 0.6  # flat flow — reduced confidence
        elif (fs > 0 and direction > 0) or (fs < 0 and direction < 0):
            multiplier = 1.0  # flow confirms
        else:
            multiplier = 0.4  # flow contradicts

        new_sig[i] = s * multiplier

    return df.with_columns(pl.Series(new_sig).alias(signal_col))
