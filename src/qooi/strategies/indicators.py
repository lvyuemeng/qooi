"""Strategy-owned indicator and signal feature precompute."""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl


@dataclass(frozen=True)
class IndicatorSources:
    ohlcv: pl.DataFrame
    order_book: pl.DataFrame | None = None
    funding: pl.DataFrame | None = None


ORDER_BOOK_FEATURE_SCHEMA = {
    "ob_bid_price": pl.Float64,
    "ob_ask_price": pl.Float64,
    "ob_bid_vol_5": pl.Float64,
    "ob_ask_vol_5": pl.Float64,
    "ob_bid_vol_25": pl.Float64,
    "ob_ask_vol_25": pl.Float64,
    "ob_bid_vol": pl.Float64,
    "ob_ask_vol": pl.Float64,
    "ob_imbalance_5": pl.Float64,
    "ob_imbalance_25": pl.Float64,
    "ob_samples": pl.Int64,
}


def sma(df: pl.DataFrame, period: int = 20, col: str = "close") -> pl.Series:
    """Simple Moving Average."""
    return df[col].rolling_mean(period)


def ema(df: pl.DataFrame, period: int = 20, col: str = "close") -> pl.Series:
    """Exponential Moving Average (span = period)."""
    return df[col].ewm_mean(span=period, min_samples=period)


def rsi(df: pl.DataFrame, period: int = 14, col: str = "close") -> pl.Series:
    """Relative Strength Index."""
    delta = df[col].diff()
    gain = delta.to_list()
    loss = delta.to_list()

    gain_series = pl.Series([0.0 if v is None or v < 0 else float(v) for v in gain])
    loss_series = pl.Series([0.0 if v is None or v > 0 else abs(float(v)) for v in loss])

    avg_gain = gain_series.rolling_mean(period)
    avg_loss = loss_series.rolling_mean(period)

    rs = avg_gain / avg_loss.replace(0, 1e-10)
    return 100.0 - (100.0 / (1.0 + rs))


def atr(df: pl.DataFrame, period: int = 14) -> pl.Series:
    """Average True Range."""
    high = df["high"]
    low = df["low"]
    close = df["close"]

    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()

    true_range = pl.Series(
        [
            max(a or 0.0, b or 0.0, c or 0.0)
            for a, b, c in zip(tr1.to_list(), tr2.to_list(), tr3.to_list())
        ]
    )
    return true_range.rolling_mean(period)


def adx(df: pl.DataFrame, period: int = 14) -> pl.Series:
    """Average Directional Index — trend strength (0-100)."""
    high = df["high"].to_list()
    low = df["low"].to_list()
    close = df["close"].to_list()

    plus_dm = [0.0] * len(df)
    minus_dm = [0.0] * len(df)
    tr = [0.0] * len(df)

    for i in range(1, len(df)):
        up_move = high[i] - high[i - 1]
        down_move = low[i - 1] - low[i]
        plus_dm[i] = up_move if up_move > down_move and up_move > 0 else 0.0
        minus_dm[i] = down_move if down_move > up_move and down_move > 0 else 0.0
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1]),
        )

    plus_dm_smooth = pl.Series(plus_dm).rolling_mean(period)
    minus_dm_smooth = pl.Series(minus_dm).rolling_mean(period)
    tr_smooth = pl.Series(tr).rolling_mean(period)

    plus_di = 100.0 * plus_dm_smooth / tr_smooth.replace(0, 1e-10)
    minus_di = 100.0 * minus_dm_smooth / tr_smooth.replace(0, 1e-10)

    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, 1e-10)
    adx_vals = dx.rolling_mean(period)
    return adx_vals


def bollinger_bands(
    df: pl.DataFrame, period: int = 20, std_dev: float = 2.0, col: str = "close"
) -> tuple[pl.Series, pl.Series, pl.Series]:
    """Bollinger Bands — returns (middle, upper, lower)."""
    middle = sma(df, period, col)
    std = df[col].rolling_std(period)
    upper = middle + std_dev * std
    lower = middle - std_dev * std
    return middle, upper, lower


def volatility(df: pl.DataFrame, period: int = 20, col: str = "close") -> pl.Series:
    """Historical volatility — standard deviation of log returns."""
    log_ret = df[col].log().diff()
    return log_ret.rolling_std(period)


def vumanchu_swing(
    df: pl.DataFrame,
    period: int = 20,
    multiplier: float = 3.5,
    channel_deviation_mult: float = 0.0,
) -> tuple[pl.Series, pl.Series]:
    """VuManChu Swing Free — range filter with long/short conditions."""
    close = df["close"]
    range_raw = (close - close.shift(1)).abs().ewm_mean(span=period, min_samples=period)
    range_size = range_raw * multiplier
    rf = close.ewm_mean(span=period, min_samples=period)
    rf_upper = rf + range_size
    rf_lower = rf - range_size

    long_entry = close > rf_upper
    short_entry = close < rf_lower

    if channel_deviation_mult > 0:
        exit_threshold = range_size * channel_deviation_mult
        too_far = (close > rf + exit_threshold) | (close < rf - exit_threshold)
    else:
        too_far = None

    state = pl.Series([0.0] * len(df))
    prev = 0.0
    for i in range(len(df)):
        if too_far is not None and too_far[i] and prev != 0:
            prev = 0.0
        elif long_entry[i]:
            prev = 1.0
        elif short_entry[i]:
            prev = -1.0
        state[i] = prev

    long_signal = (state == 1.0).cast(pl.Int32)
    short_signal = (state == -1.0).cast(pl.Int32)
    return long_signal, short_signal


def add_indicators(df: pl.DataFrame, vm_channel_deviation_mult: float = 1.5) -> pl.DataFrame:
    """Add generic reusable technical feature columns."""
    if "vol" not in df.columns and "volume" in df.columns:
        df = df.rename({"volume": "vol"})
    long_sig, short_sig = vumanchu_swing(df, channel_deviation_mult=vm_channel_deviation_mult)
    adx_vals = adx(df, 14)
    return df.with_columns(
        [
            sma(df, 20).alias("sma_20"),
            sma(df, 50).alias("sma_50"),
            ema(df, 12).alias("ema_12"),
            ema(df, 26).alias("ema_26"),
            ema(df, 20).alias("ema_20"),
            ema(df, 50).alias("ema_50"),
            ema(df, 200).alias("ema_200"),
            rsi(df, 14).alias("rsi_14"),
            atr(df, 14).alias("atr_14"),
            volatility(df, 20).alias("volatility_20"),
            sma(df, 20).alias("bb_middle"),
            long_sig.alias("vm_long"),
            short_sig.alias("vm_short"),
            adx_vals.alias("adx_14"),
        ]
    ).with_columns(
        [
            (pl.col("bb_middle") + 2.0 * pl.col("close").rolling_std(20)).alias("bb_upper"),
            (pl.col("bb_middle") - 2.0 * pl.col("close").rolling_std(20)).alias("bb_lower"),
        ]
    )


def add_ofi_flow_columns(
    df: pl.DataFrame,
    *,
    flow_window: int = 12,
    atr_col: str = "atr_14",
) -> pl.DataFrame:
    if df.is_empty():
        return df

    close = pl.col("close")
    open_p = pl.col("open")
    vol = pl.col("vol").fill_nan(0).fill_null(0)
    signed = pl.when(close > open_p).then(vol).when(close < open_p).then(-vol).otherwise(0.0)

    net_flow = signed.rolling_sum(flow_window)
    vol_total = signed.abs().rolling_sum(flow_window).clip(1e-9)
    flow_score = (net_flow / vol_total).fill_null(0).clip(-1.0, 1.0)
    flow_score = pl.when(vol_total < 1e-6).then(0.0).otherwise(flow_score)

    return df.with_columns(
        signed.alias("ofi_signed_vol"),
        net_flow.alias("ofi_net_flow"),
        flow_score.fill_null(0).alias("ofi_flow_score"),
    )


def apply_micro_confirmation(
    df: pl.DataFrame,
    signal_col: str = "signal",
    flow_col: str = "ofi_flow_score",
) -> pl.DataFrame:
    if signal_col not in df.columns or flow_col not in df.columns:
        return df

    s = pl.col(signal_col)
    f = pl.col(flow_col)
    direction = s.sign()
    multiplier = pl.when(f.abs() < 0.05).then(0.6).when(direction * f > 0).then(1.0).otherwise(0.4)
    return df.with_columns((s * multiplier).alias(signal_col))


def add_regime_features(
    df: pl.DataFrame,
    *,
    atr_col: str = "atr_14",
    ema_slow_period: int = 200,
    momentum_bars: tuple[int, int, int] = (6, 24, 96),
) -> pl.DataFrame:
    if df.is_empty():
        return df

    close = pl.col("close")
    atr_col_expr = (
        pl.col(atr_col).fill_nan(0).fill_null(0) if atr_col in df.columns else pl.lit(1.0)
    )
    ema_slow = close.ewm_mean(span=ema_slow_period, min_samples=ema_slow_period)
    vol_ma = pl.col("vol").fill_nan(0).fill_null(0).rolling_mean(20)

    regime_score = ((close - ema_slow) / (atr_col_expr * 3.0)).clip(-1.0, 1.0)
    regime_strength = regime_score.abs()

    mf, mm, ms = momentum_bars
    mom_fast = ((close - close.shift(mf)) / (atr_col_expr * 2.0)).clip(-1.0, 1.0)
    mom_mid = ((close - close.shift(mm)) / (atr_col_expr * 3.0)).clip(-1.0, 1.0)
    mom_slow = ((close - close.shift(ms)) / (atr_col_expr * 4.0)).clip(-1.0, 1.0)

    vol_conf = (0.5 + (pl.col("vol").fill_nan(0) / vol_ma - 1.0) * 0.3).clip(0.25, 1.0)

    return df.with_columns(
        regime_score.fill_null(0).alias("regime_score"),
        regime_strength.fill_null(0).alias("regime_strength"),
        mom_fast.fill_null(0).alias("regime_mom_fast"),
        mom_mid.fill_null(0).alias("regime_mom_mid"),
        mom_slow.fill_null(0).alias("regime_mom_slow"),
        vol_conf.fill_null(0.25).alias("regime_vol_conf"),
    )


def apply_regime_gate(
    df: pl.DataFrame,
    signal_col: str = "signal",
    regime_col: str = "regime_score",
    max_regime: float = 0.7,
) -> pl.DataFrame:
    if signal_col not in df.columns or regime_col not in df.columns:
        return df
    return df.with_columns(
        pl.when(pl.col(regime_col).abs() > max_regime)
        .then(0.0)
        .otherwise(pl.col(signal_col))
        .alias(signal_col)
    )


def normalize_order_book_snapshots(df: pl.DataFrame) -> pl.DataFrame:
    """Normalize raw cached order-book snapshots to required feature columns."""
    if df.is_empty():
        schema = {
            "timestamp": pl.Int64,
            **{k: v for k, v in ORDER_BOOK_FEATURE_SCHEMA.items() if k != "ob_samples"},
        }
        return pl.DataFrame(schema=schema)

    required = ["timestamp", *(col for col in ORDER_BOOK_FEATURE_SCHEMA if col != "ob_samples")]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Order-book frame missing columns: {missing}")
    return df.select(required).unique(subset=["timestamp"]).sort("timestamp")


def attach_order_book_features(df: pl.DataFrame, snapshot_df: pl.DataFrame) -> pl.DataFrame:
    """Aggregate recorded order-book snapshots into the OHLCV bar grid."""
    if df.is_empty():
        return _with_empty_order_book_features(df)

    snapshots = normalize_order_book_snapshots(snapshot_df)
    if snapshots.is_empty():
        return _with_empty_order_book_features(df)

    bar_ms = _bar_interval_ms(df)
    bars = df.select(pl.col("timestamp").alias("bar_timestamp")).sort("bar_timestamp")
    mapped = (
        snapshots.rename({"timestamp": "snapshot_timestamp"})
        .sort("snapshot_timestamp")
        .join_asof(
            bars,
            left_on="snapshot_timestamp",
            right_on="bar_timestamp",
            strategy="backward",
        )
        .drop_nulls(["bar_timestamp"])
        .filter((pl.col("snapshot_timestamp") - pl.col("bar_timestamp")) < bar_ms)
    )
    if mapped.is_empty():
        return _with_empty_order_book_features(df)

    per_bar = (
        mapped.group_by("bar_timestamp")
        .agg(
            [
                pl.col("ob_bid_price").last().alias("ob_bid_price"),
                pl.col("ob_ask_price").last().alias("ob_ask_price"),
                pl.col("ob_bid_vol_5").mean().alias("ob_bid_vol_5"),
                pl.col("ob_ask_vol_5").mean().alias("ob_ask_vol_5"),
                pl.col("ob_bid_vol_25").mean().alias("ob_bid_vol_25"),
                pl.col("ob_ask_vol_25").mean().alias("ob_ask_vol_25"),
                pl.col("ob_bid_vol").mean().alias("ob_bid_vol"),
                pl.col("ob_ask_vol").mean().alias("ob_ask_vol"),
                pl.col("ob_imbalance_5").mean().alias("ob_imbalance_5"),
                pl.col("ob_imbalance_25").mean().alias("ob_imbalance_25"),
                pl.len().alias("ob_samples"),
            ]
        )
        .rename({"bar_timestamp": "timestamp"})
        .sort("timestamp")
    )
    return df.sort("timestamp").join(per_bar, on="timestamp", how="left")


def compute_indicator_frame(
    ohlcv: pl.DataFrame,
    *,
    order_book: pl.DataFrame | None = None,
    threshold: float = 0.25,
) -> pl.DataFrame:
    """Compute a unified indicator/signal frame from available market sources."""
    df = ohlcv
    if "volume" in df.columns and "vol" not in df.columns:
        df = df.rename({"volume": "vol"})
    df = add_indicators(df)
    if order_book is not None:
        df = attach_order_book_features(df, order_book)
    df = add_regime_features(df)
    df = add_ofi_flow_columns(df)
    df = apply_regime_gate(df, signal_col="ofi_flow_score")
    ofi = pl.col("ofi_flow_score")
    return df.with_columns(pl.when(ofi.abs() >= threshold).then(ofi).otherwise(0.0).alias("signal"))


def compute_flow_pipeline_frame(df: pl.DataFrame, threshold: float = 0.25) -> pl.DataFrame:
    """Full OFI/regime signal pipeline on OHLCV data."""
    return compute_indicator_frame(df, threshold=threshold)


def _with_empty_order_book_features(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        [
            pl.lit(0, dtype=dtype).alias(name)
            if name == "ob_samples"
            else pl.lit(None, dtype=dtype).alias(name)
            for name, dtype in ORDER_BOOK_FEATURE_SCHEMA.items()
        ]
    )


def _bar_interval_ms(df: pl.DataFrame) -> int:
    if df.height < 2:
        return 3_600_000
    timestamps = df["timestamp"].to_list()
    deltas = [int(timestamps[i] - timestamps[i - 1]) for i in range(1, len(timestamps))]
    positive = [delta for delta in deltas if delta > 0]
    return min(positive) if positive else 3_600_000
