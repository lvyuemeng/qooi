"""Small-cap reversal strategy — local backtest using BigQuant BigTrader.

Data sources:
  - Daily bars: TickFlow free tier (no auth needed)
  - Fundamentals (ROE, debt ratio): BigQuant DAI (requires API key)
  - Factor data (market cap, listing days): BigQuant DAI

No database needed — all data fetched on-the-fly and cached in memory.
"""

from __future__ import annotations

import warnings
from datetime import datetime

import numpy as np
import polars as pl
from bigquant import bigtrader, dai

from qooi.data import DataSource

warnings.filterwarnings("ignore")

# ============================================================
# Configuration
# ============================================================
HOLD_NUM = 5
TARGET_VOL = 0.25
VOL_LOOKBACK = 20
STOP_LOSS_PCT = 0.20
GRACE_DAYS = 10
START_DATE = "2020-01-01"
END_DATE = "2024-08-14"
INITIAL_CAPITAL = 500_000


# ============================================================
# 1. Data preparation
# ============================================================
def prepare_data() -> pl.DataFrame:
    """Fetch and merge all data into a single Polars DataFrame.

    Returns columns: date, instrument, float_market_cap, roe, debt_to_asset,
                     close, ret_20d, avg_turn_20, vol_20d, volume_ratio
    """
    tf = DataSource()

    # --- Daily bars from TickFlow (free, no auth) ---
    # Get full A-share universe
    all_a_shares = tf.instruments(
        ["600000.SH", "000001.SZ", "600519.SH"]
    )  # placeholder
    # In production, fetch from BigQuant DAI universe:
    #   SELECT instrument FROM cn_stock_prefactors_community WHERE date = '...'
    # For now use DAI to get the full list

    bars = dai.query(
        """
        SELECT date, instrument, close, turn, amount
        FROM cn_stock_bar1d
        WHERE date >= '2018-11-01'
        """,
        filters={"date": ["2018-11-01", END_DATE]},
    ).pl()

    # --- Fundamentals from BigQuant DAI ---
    base = dai.query(
        """
        SELECT date, instrument, float_market_cap, list_days, st_status, suspended
        FROM cn_stock_prefactors_community
        WHERE list_days > 365 AND st_status = 0 AND suspended = 0 AND float_market_cap > 0
        """,
        filters={"date": [START_DATE, END_DATE]},
    ).pl()

    fin = dai.query(
        """
        SELECT date, instrument, roe_avg_deduct_ttm, debt_to_asset
        FROM cn_stock_prefactors
        """,
        filters={"date": [START_DATE, END_DATE]},
    ).pl()

    # --- Compute derived factors ---
    bars = bars.sort(["instrument", "date"])
    bars = bars.with_columns(
        [
            pl.col("close").pct_change().over("instrument").alias("ret_1d"),
            (pl.col("close") / pl.col("close").shift(20).over("instrument") - 1).alias(
                "ret_20d"
            ),
            pl.col("turn").rolling_mean(20).over("instrument").alias("avg_turn_20"),
            pl.col("amount").rolling_mean(20).over("instrument").alias("avg_amount_20"),
            pl.col("amount").rolling_mean(5).over("instrument").alias("avg_amount_5"),
            pl.col("ret_1d").rolling_std(20).over("instrument").alias("vol_20d"),
        ]
    )
    bars = bars.with_columns(
        (pl.col("avg_amount_5") / pl.col("avg_amount_20")).alias("volume_ratio")
    )
    bars = bars.drop_nulls(["ret_20d", "avg_turn_20", "vol_20d", "volume_ratio"])

    # --- Merge ---
    merged = base.join(fin, on=["date", "instrument"], how="inner")
    merged = merged.join(
        bars.select(
            ["date", "instrument", "ret_20d", "avg_turn_20", "vol_20d", "volume_ratio"]
        ),
        on=["date", "instrument"],
        how="inner",
    )

    # --- Negative screens ---
    merged = merged.filter(
        (pl.col("avg_turn_20") <= 0.10)
        & (pl.col("vol_20d") <= 0.04)
        & (pl.col("volume_ratio") < 3.0)
    )

    return merged


# ============================================================
# 2. Stock selection
# ============================================================
def select_stocks(data: pl.DataFrame, hold_num: int = HOLD_NUM) -> pl.DataFrame:
    """Select stocks: 60% small-cap factor + 40% reversal factor."""
    data = data.with_columns(
        [
            pl.col("float_market_cap").rank("ordinal").over("date").alias("size_rank"),
            pl.col("ret_20d").rank("ordinal").over("date").alias("reversal_rank"),
        ]
    )
    n = data.group_by("date").agg(pl.len().alias("n"))
    data = data.join(n, on="date")
    data = data.with_columns(
        [
            (pl.col("size_rank") / pl.col("n")).alias("size_score"),
            (pl.col("reversal_rank") / pl.col("n")).alias("reversal_score"),
        ]
    )
    data = data.with_columns(
        (0.6 * pl.col("size_score") + 0.4 * pl.col("reversal_score")).alias(
            "final_score"
        )
    )
    selected = (
        data.sort(["date", "final_score"], descending=[False, True])
        .group_by("date")
        .head(hold_num)
        .with_columns(pl.lit(1.0 / hold_num).alias("position"))
    )
    return selected.select(["date", "instrument", "position", "final_score"])


# ============================================================
# 3. Strategy callbacks
# ============================================================
def initialize(context):
    context.set_commission(
        bigtrader.PerOrder(buy_cost=0.0003, sell_cost=0.0013, min_cost=5)
    )
    context.target_vol = TARGET_VOL
    context.vol_lookback = VOL_LOOKBACK
    context.stop_loss_pct = STOP_LOSS_PCT
    context.grace_days = GRACE_DAYS
    context.daily_returns: list[float] = []
    context.prev_value: float | None = None


def handle_data(context, data):
    if not context.rebalance_period.is_signal_date(data.current_dt.date()):
        _check_stop_loss(context, data)
        return

    today_str = data.current_dt.strftime("%Y-%m-%d")
    today_df = context.signal_data[context.signal_data["date"] == today_str]
    if today_df.is_empty():
        return

    # --- Volatility scaling ---
    scale = 1.0
    if len(context.daily_returns) >= context.vol_lookback:
        recent = np.array(context.daily_returns[-context.vol_lookback :])
        realized = float(np.std(recent) * np.sqrt(252))
        if realized > 0:
            scale = min(1.0, context.target_vol / realized)

    # --- Determine tradable instruments ---
    holdings = context.get_account_positions()
    targets = set()
    for row in today_df.iter_rows(named=True):
        inst = row["instrument"]
        if data.can_trade(inst):
            targets.add(inst)
        elif inst in holdings:
            context.order_target_percent(inst, 0)

    # Sell non-targets
    for inst in list(holdings.keys()):
        if inst not in targets:
            context.order_target_percent(inst, 0)

    # Buy / adjust targets
    for row in today_df.iter_rows(named=True):
        inst = row["instrument"]
        if inst not in targets:
            continue
        pos = float(row["position"]) * scale
        if pos > 0:
            context.order_target_percent(inst, pos)


def _check_stop_loss(context, data):
    positions = context.get_account_positions()
    for inst, pos in positions.items():
        if pos.amount == 0:
            continue
        open_dt = pos.open_date
        if isinstance(open_dt, str):
            open_dt = datetime.strptime(open_dt, "%Y-%m-%d")
        if not isinstance(open_dt, datetime):
            continue
        holding = (data.current_dt.date() - open_dt.date()).days
        if holding < context.grace_days:
            continue
        try:
            hist = data.history(inst, "close", 300, "1d")
            if hist.is_empty():
                continue
            mask = hist["date"] >= open_dt.strftime("%Y-%m-%d")
            if mask.sum() < 2:
                continue
            relevant = hist.filter(mask)
            high = relevant["close"].max()
            current = relevant["close"][-1]
            dd = (high - current) / high
            if dd >= context.stop_loss_pct:
                context.order_target_percent(inst, 0)
        except Exception:
            pass


def after_trading(context, _data):
    if context.prev_value is None:
        context.prev_value = float(context.portfolio.portfolio_value)
        return
    cur = float(context.portfolio.portfolio_value)
    ret = (cur - context.prev_value) / context.prev_value
    context.daily_returns.append(ret)
    if len(context.daily_returns) > 252:
        context.daily_returns.pop(0)
    context.prev_value = cur


# ============================================================
# 4. Run
# ============================================================
def main() -> None:
    print("==> Preparing data (fetching from TickFlow + BigQuant DAI)...")
    data = prepare_data()

    print("==> Selecting stocks...")
    # Danger months (Jan, Apr): additional ROE > 3% & debt < 80%
    data = data.with_columns(
        pl.col("date").str.to_date("%Y-%m-%d").dt.month().is_in([1, 4]).alias("danger")
    )
    danger_pool = data.filter(
        pl.col("danger")
        & (pl.col("roe_avg_deduct_ttm") > 3.0)
        & (pl.col("debt_to_asset") < 0.8)
    )
    safe_pool = data.filter(~pl.col("danger"))

    selected = pl.concat(
        [select_stocks(safe_pool), select_stocks(danger_pool)],
        how="vertical",
    ).sort(["date", "instrument"])

    # Write to BigQuant temporary DataSource
    pdf = selected.to_pandas()
    stock_data = dai.DataSource.write_bdb(pdf)
    print(f"    Written {len(pdf)} rows to DataSource {stock_data.id}")

    print(f"==> Running backtest ({START_DATE} → {END_DATE})...")
    perf = bigtrader.run(
        data=stock_data,
        start_date=START_DATE,
        end_date=END_DATE,
        initialize=initialize,
        handle_data=handle_data,
        after_trading=after_trading,
        capital_base=INITIAL_CAPITAL,
        frequency="daily",
        product_type="股票",
        rebalance_period_type="交易日",
        rebalance_period_days="20",
        rebalance_period_roll_forward=True,
        benchmark="000300.SH",
        volume_limit=1,
        order_price_field_buy="open",
        order_price_field_sell="open",
    )
    print(perf.summary)


if __name__ == "__main__":
    main()
