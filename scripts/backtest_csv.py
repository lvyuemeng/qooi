"""Small-cap backtest — signal CSV loaded from file, bars from TickFlow free tier.

Workflow:
  1. In BigQuant AI Studio (web), run the data-prep SQL to get selected stocks.
  2. Export the resulting DataFrame as CSV, save to ``data/signals/``.
  3. Run:  uv run python scripts/backtest_csv.py --csv data/signals/signals.csv
"""

from __future__ import annotations

import argparse
from datetime import datetime

import numpy as np
import polars as pl
from bigquant import bigtrader

from qooi.data import DataSource

# ============================================================
# Configuration
# ============================================================
HOLD_NUM = 5
TARGET_VOL = 0.25
VOL_LOOKBACK = 20
STOP_LOSS_PCT = 0.20
GRACE_DAYS = 10
INITIAL_CAPITAL = 500_000


# ============================================================
# Strategy callbacks
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

    scale = 1.0
    if len(context.daily_returns) >= context.vol_lookback:
        recent = np.array(context.daily_returns[-context.vol_lookback :])
        realized = float(np.std(recent) * np.sqrt(252))
        if realized > 0:
            scale = min(1.0, context.target_vol / realized)

    holdings = context.get_account_positions()
    targets = set()
    for row in today_df.iter_rows(named=True):
        inst = row["instrument"]
        if data.can_trade(inst):
            targets.add(inst)
        elif inst in holdings:
            context.order_target_percent(inst, 0)

    for inst in list(holdings.keys()):
        if inst not in targets:
            context.order_target_percent(inst, 0)

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
# Main
# ============================================================
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run small-cap backtest from signal CSV"
    )
    parser.add_argument(
        "--csv",
        required=True,
        help="Path to signal CSV (date,instrument,position,score)",
    )
    parser.add_argument(
        "--start", default=None, help="Override start date (YYYY-MM-DD)"
    )
    parser.add_argument("--end", default=None, help="Override end date (YYYY-MM-DD)")
    args = parser.parse_args()

    signals = pl.read_csv(args.csv, try_parse_dates=True)
    print(f"Loaded {len(signals)} signal rows from {args.csv}")

    start = args.start or signals["date"].min()
    end = args.end or signals["date"].max()

    # Write to BigQuant DataSource
    pdf = signals.to_pandas()
    stock_data = bigquant_ds = __import__("bigquant").dai.DataSource.write_bdb(pdf)
    print(f"Written to DataSource {stock_data.id}")

    print(f"Running backtest {start} → {end}...")
    perf = bigtrader.run(
        data=stock_data,
        start_date=start,
        end_date=end,
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
