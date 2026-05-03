"""
Run this script inside BigQuant AI Studio (web) to execute the complete
value-inversion research pipeline and export signal CSVs for all 7 strategies.

Steps:
  1. Open https://bigquant.com/aistudio
  2. Create a new strategy / notebook
  3. Paste this entire script into a code cell
  4. Run it
  5. Download the CSVs from the AI Studio file explorer (in /tmp/ or work dir)
  6. Save to data/signals/ locally
  7. Run:  uv run python scripts/backtest_csv.py --csv data/signals/<strategy>.csv
"""

from bigmodule import M, I
import dai
import pandas as pd
import numpy as np
import os
from datetime import timedelta

# ============================================================
# Config
# ============================================================
start_date_str = "2020-01-01"
end_date_str = "2026-04-01"
DATA_START = "2019-01-01"

LIST_SECTOR_MAINBOARD = 1
CANDIDATE_SIZE_RANK = 600
PANEL_REBALANCE_DAYS = 20
LOOKBACK_DAYS = 220
FORWARD_DAYS = 60
MIN_LIST_DAYS = 365
MIN_TRADING_DAYS = 240
DEFAULT_HOLD_NUM = 5
DEFAULT_BUFFER_NUM = 15
DEFAULT_REBALANCE_DAYS = 20
MIN_LIQUIDITY_PCT = 0.20
MAX_INDUSTRY_COUNT = 2
BENCHMARK = "上证指数"
USE_PANEL_CACHE = True
PANEL_CACHE_PATH = "/home/aiuser/work/mainboard_smallcap_panel.pkl"


# ============================================================
# 1. Panel building (same as notebook cells 1-4)
# ============================================================
def load_trade_dates():
    sql = f"""
    SELECT DISTINCT date
    FROM cn_stock_factors_base
    WHERE date >= '{DATA_START}' AND date <= '{end_date_str}'
    ORDER BY date
    """
    trade_dates = dai.query(sql).df()
    trade_dates["date"] = pd.to_datetime(trade_dates["date"]).dt.strftime("%Y-%m-%d")
    trade_dates = (
        trade_dates.drop_duplicates().sort_values("date").reset_index(drop=True)
    )
    return trade_dates


def make_signal_dates(trade_dates, rebalance_days=PANEL_REBALANCE_DAYS):
    x = trade_dates[trade_dates["date"] >= start_date_str].copy().reset_index(drop=True)
    x["idx"] = np.arange(len(x))
    signal_dates = list(x.loc[x["idx"] % rebalance_days == 0, "date"])
    return signal_dates


def get_window_dates(
    signal_date, trade_dates, lookback_days=LOOKBACK_DAYS, forward_days=FORWARD_DAYS
):
    dates = list(trade_dates["date"])
    idx = dates.index(signal_date)
    start_idx = max(0, idx - lookback_days)
    end_idx = min(len(dates) - 1, idx + forward_days)
    return dates[start_idx], dates[end_idx]


def load_one_signal_panel(signal_date, trade_dates):
    candidate_sql = f"""
    SELECT instrument
    FROM cn_stock_factors_base
    WHERE date = '{signal_date}'
      AND list_sector = {LIST_SECTOR_MAINBOARD}
      AND st_status = 0 AND suspended = 0
      AND list_days > {MIN_LIST_DAYS}
      AND trading_days > {MIN_TRADING_DAYS}
      AND float_market_cap > 0
    ORDER BY float_market_cap ASC
    LIMIT {CANDIDATE_SIZE_RANK}
    """
    candidate_df = dai.query(candidate_sql).df()
    candidates = candidate_df["instrument"].dropna().astype(str).unique().tolist()
    if len(candidates) == 0:
        return pd.DataFrame()

    instruments_sql = ",".join([f"'{x}'" for x in candidates])
    window_start, window_end = get_window_dates(signal_date, trade_dates)

    raw_sql = f"""
    SELECT
        b.date, b.instrument,
        b.close, b.amount, b.volume, b.open, b.high, b.low,
        b.price_limit_status, b.st_status, b.suspended,
        b.list_days, b.trading_days,
        b.list_sector, b.total_market_cap, b.float_market_cap,
        b.sw2021_level1,
        v.pb, v.pe_ttm, v.pcf_op_ttm, v.pcf_net_ttm,
        f.operating_net_income_ttm, f.net_profit_deducted_lf,
        f.fcff_ttm, f.fcfe_ttm, f.ebit_ttm, f.ebitda_ttm,
        f.nopat_ttm, f.invested_capital_lf
    FROM cn_stock_factors_base AS b
    LEFT JOIN cn_stock_valuation AS v ON b.date = v.date AND b.instrument = v.instrument
    LEFT JOIN cn_stock_factors_financial_indicators AS f
        ON b.date = f.date AND b.instrument = f.instrument
    WHERE b.date >= '{window_start}' AND b.date <= '{window_end}'
      AND b.instrument IN ({instruments_sql})
      AND b.list_sector = {LIST_SECTOR_MAINBOARD}
    """

    df = dai.query(raw_sql).df()
    if df.empty:
        return pd.DataFrame()

    df = df.dropna(subset=["date", "instrument", "close"]).copy()
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df["instrument"] = df["instrument"].astype(str)
    df = df.sort_values(["instrument", "date"]).reset_index(drop=True)

    float_cols = [
        "close",
        "amount",
        "volume",
        "open",
        "high",
        "low",
        "total_market_cap",
        "float_market_cap",
        "pb",
        "pe_ttm",
        "pcf_op_ttm",
        "pcf_net_ttm",
        "operating_net_income_ttm",
        "net_profit_deducted_lf",
        "fcff_ttm",
        "fcfe_ttm",
        "ebit_ttm",
        "ebitda_ttm",
        "nopat_ttm",
        "invested_capital_lf",
    ]
    for col in float_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")

    int_cols = [
        "price_limit_status",
        "st_status",
        "suspended",
        "list_days",
        "trading_days",
        "list_sector",
    ]
    for col in int_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    g = df.groupby("instrument")
    df["ret_120d"] = g["close"].transform(lambda x: x / x.shift(120) - 1)
    df["ret_60d"] = g["close"].transform(lambda x: x / x.shift(60) - 1)
    df["ret_20d"] = g["close"].transform(lambda x: x / x.shift(20) - 1)
    df["ret_5d"] = g["close"].transform(lambda x: x / x.shift(5) - 1)
    df["daily_ret"] = g["close"].transform(lambda x: x / x.shift(1) - 1)
    df["amount_20d"] = g["amount"].transform(
        lambda x: x.rolling(20, min_periods=10).mean()
    )
    df["amount_to_float_mv"] = df["amount_20d"] / df["float_market_cap"]
    df["vol_20d"] = g["daily_ret"].transform(
        lambda x: x.rolling(20, min_periods=15).std()
    )
    df["vol_60d"] = g["daily_ret"].transform(
        lambda x: x.rolling(60, min_periods=40).std()
    )
    df["ma_20"] = g["close"].transform(lambda x: x.rolling(20, min_periods=15).mean())
    df["ma_60"] = g["close"].transform(lambda x: x.rolling(60, min_periods=40).mean())
    df["operating_yield"] = df["operating_net_income_ttm"] / df["total_market_cap"]
    df["fwd_20d"] = g["close"].transform(lambda x: x.shift(-20) / x - 1)
    df["fwd_30d"] = g["close"].transform(lambda x: x.shift(-30) / x - 1)
    df["fwd_40d"] = g["close"].transform(lambda x: x.shift(-40) / x - 1)

    panel = df[df["date"] == signal_date].copy()
    panel = panel[panel["instrument"].isin(candidates)].copy()
    panel["signal_date"] = signal_date
    return panel


def build_panel(trade_dates, use_cache=USE_PANEL_CACHE):
    if use_cache and os.path.exists(PANEL_CACHE_PATH):
        panel_df = pd.read_pickle(PANEL_CACHE_PATH)
        print("loaded panel cache:", PANEL_CACHE_PATH)
        print("panel_df shape:", panel_df.shape)
        return panel_df

    signal_dates = make_signal_dates(trade_dates)
    panels = []
    for i, d in enumerate(signal_dates):
        try:
            one = load_one_signal_panel(d, trade_dates)
            if len(one) > 0:
                panels.append(one)
            print(f"[{i + 1}/{len(signal_dates)}] {d}, shape={one.shape}")
        except Exception as e:
            print(f"[ERROR] {d}: {e}")
            continue

    if len(panels) == 0:
        raise ValueError("No panel data generated.")
    panel_df = pd.concat(panels, axis=0, ignore_index=True)
    panel_df = panel_df.sort_values(["date", "instrument"]).reset_index(drop=True)
    panel_df.to_pickle(PANEL_CACHE_PATH)
    print("saved panel cache:", PANEL_CACHE_PATH)
    print("panel_df shape:", panel_df.shape)
    return panel_df


# ============================================================
# 2. Factor ranking & filtering (notebook cell 6)
# ============================================================
def apply_base_filter(
    df,
    require_pcf_positive=False,
    require_profit=False,
    use_trend_filter=False,
    use_vol_filter=False,
    use_reversal_band=False,
    min_liquidity_pct=MIN_LIQUIDITY_PCT,
):
    x = df.copy()
    x = x[
        (x["list_sector"] == LIST_SECTOR_MAINBOARD)
        & (x["list_days"] > MIN_LIST_DAYS)
        & (x["trading_days"] > MIN_TRADING_DAYS)
        & (x["st_status"] == 0)
        & (x["suspended"] == 0)
        & (x["price_limit_status"] == 2)
        & (x["float_market_cap"] > 0)
        & (x["total_market_cap"] > 0)
        & (x["amount_20d"] > 0)
        & (x["ret_60d"].notna())
        & (x["ret_20d"].notna())
        & (x["ret_5d"].notna())
        & (x["vol_20d"].notna())
        & (x["ma_20"].notna())
        & (x["ma_60"].notna())
        & (x["sw2021_level1"].notna())
    ].copy()
    x = x[(x["pb"].notna()) & (x["pe_ttm"].notna())].copy()

    if require_pcf_positive:
        x = x[(x["pcf_op_ttm"] > 0)].copy()
    if require_profit:
        x = x[
            (
                (x["operating_net_income_ttm"].fillna(0) > 0)
                | (x["net_profit_deducted_lf"].fillna(0) > 0)
                | (x["fcff_ttm"].fillna(0) > 0)
            )
        ].copy()

    x["liquidity_rank"] = x.groupby("date")["amount_20d"].rank(pct=True, ascending=True)
    x = x[x["liquidity_rank"] >= min_liquidity_pct].copy()

    if use_reversal_band:
        x["rev_pct"] = x.groupby("date")["ret_60d"].rank(pct=True, ascending=True)
        x = x[(x["rev_pct"] >= 0.10) & (x["rev_pct"] <= 0.80)].copy()
    if use_trend_filter:
        x["trend_ok"] = (x["close"] > x["ma_20"] * 0.92) | (
            x["close"] > x["ma_60"] * 0.92
        )
        x = x[x["trend_ok"]].copy()
    if use_vol_filter:
        x["vol_pct"] = x.groupby("date")["vol_20d"].rank(pct=True, ascending=True)
        x = x[x["vol_pct"] <= 0.90].copy()

    return x


def add_factor_ranks(df):
    x = df.copy()
    x["rank_size"] = x.groupby("date")["float_market_cap"].rank(
        pct=True, ascending=True
    )
    x["rank_lowvol"] = x.groupby("date")["vol_20d"].rank(pct=True, ascending=True)
    x["rank_pb"] = x.groupby("date")["pb"].rank(pct=True, ascending=True)
    x["rank_pcf"] = x.groupby("date")["pcf_op_ttm"].rank(pct=True, ascending=True)
    x["rank_rev60"] = x.groupby("date")["ret_60d"].rank(pct=True, ascending=True)
    x["rank_mom20"] = x.groupby("date")["ret_20d"].rank(pct=True, ascending=False)
    x["rank_quality"] = x.groupby("date")["operating_yield"].rank(
        pct=True, ascending=False
    )
    if "amount_to_float_mv" in x.columns:
        x["rank_low_turnover_proxy"] = x.groupby("date")["amount_to_float_mv"].rank(
            pct=True, ascending=True
        )

    x["rev60_pct"] = x.groupby("date")["ret_60d"].rank(pct=True, ascending=True)
    x["mid_rev60_score_raw"] = (x["rev60_pct"] - 0.40).abs()
    x["rank_mid_rev60"] = x.groupby("date")["mid_rev60_score_raw"].rank(
        pct=True, ascending=True
    )

    return x


# ============================================================
# 3. Strategy definitions (notebook cell 12)
# ============================================================
STRATEGIES = {
    "pure_size": {
        "weights": {"rank_size": 1.00},
        "require_pcf_positive": False,
        "require_profit": False,
        "use_trend_filter": False,
        "use_vol_filter": False,
        "use_reversal_band": False,
        "hold_num": 5,
        "buffer_num": 15,
        "rebalance_days": 20,
        "exposure": 1.00,
    },
    "size_lowvol": {
        "weights": {"rank_size": 0.70, "rank_lowvol": 0.30},
        "require_pcf_positive": False,
        "require_profit": False,
        "use_trend_filter": False,
        "use_vol_filter": False,
        "use_reversal_band": False,
        "hold_num": 5,
        "buffer_num": 15,
        "rebalance_days": 20,
        "exposure": 1.00,
    },
    "size_mom20": {
        "weights": {"rank_size": 0.70, "rank_mom20": 0.30},
        "require_pcf_positive": False,
        "require_profit": False,
        "use_trend_filter": False,
        "use_vol_filter": False,
        "use_reversal_band": False,
        "hold_num": 5,
        "buffer_num": 15,
        "rebalance_days": 20,
        "exposure": 1.00,
    },
    "size_reversal": {
        "weights": {"rank_size": 0.70, "rank_rev60": 0.30},
        "require_pcf_positive": False,
        "require_profit": False,
        "use_trend_filter": False,
        "use_vol_filter": False,
        "use_reversal_band": True,
        "hold_num": 5,
        "buffer_num": 15,
        "rebalance_days": 20,
        "exposure": 1.00,
    },
    "size_value": {
        "weights": {"rank_size": 0.60, "rank_pb": 0.20, "rank_pcf": 0.20},
        "require_pcf_positive": True,
        "require_profit": False,
        "use_trend_filter": False,
        "use_vol_filter": False,
        "use_reversal_band": False,
        "hold_num": 5,
        "buffer_num": 15,
        "rebalance_days": 20,
        "exposure": 1.00,
    },
    "size_quality": {
        "weights": {"rank_size": 0.70, "rank_quality": 0.30},
        "require_pcf_positive": False,
        "require_profit": True,
        "use_trend_filter": False,
        "use_vol_filter": False,
        "use_reversal_band": False,
        "hold_num": 5,
        "buffer_num": 15,
        "rebalance_days": 20,
        "exposure": 1.00,
    },
    "balanced": {
        "weights": {
            "rank_size": 0.50,
            "rank_lowvol": 0.15,
            "rank_mom20": 0.15,
            "rank_pb": 0.10,
            "rank_quality": 0.10,
        },
        "require_pcf_positive": False,
        "require_profit": False,
        "use_trend_filter": True,
        "use_vol_filter": True,
        "use_reversal_band": False,
        "hold_num": 5,
        "buffer_num": 15,
        "rebalance_days": 20,
        "exposure": 1.00,
    },
}


# ============================================================
# 4. Stock selection with industry limit
# ============================================================
def get_trade_dates(df):
    trade_dates = (
        df[["date"]].drop_duplicates().sort_values("date").reset_index(drop=True)
    )
    trade_dates["next_date"] = trade_dates["date"].shift(-1)
    return trade_dates


def make_rebalance_dates(trade_dates, rebalance_days):
    trade_date_df = (
        trade_dates[trade_dates["date"] >= start_date_str].copy().reset_index(drop=True)
    )
    trade_date_df["rebalance_index"] = np.arange(len(trade_date_df))
    rebalance_dates = set(
        trade_date_df.loc[
            trade_date_df["rebalance_index"] % rebalance_days == 0, "date"
        ]
    )
    return rebalance_dates


def select_with_industry_limit(df_day, last_holdings, hold_num, buffer_num):
    df_day = df_day.sort_values("score").copy()
    df_day["score_rank"] = np.arange(1, len(df_day) + 1)
    selected_rows = []
    selected_set = set()
    industry_count = {}

    keep_candidates = df_day[
        (df_day["instrument"].isin(last_holdings))
        & (df_day["score_rank"] <= buffer_num)
    ].copy()
    for _, row in keep_candidates.iterrows():
        inst = row["instrument"]
        ind = row["sw2021_level1"]
        if industry_count.get(ind, 0) >= MAX_INDUSTRY_COUNT:
            continue
        selected_rows.append(row)
        selected_set.add(inst)
        industry_count[ind] = industry_count.get(ind, 0) + 1
        if len(selected_rows) >= hold_num:
            break

    for _, row in df_day.iterrows():
        if len(selected_rows) >= hold_num:
            break
        inst = row["instrument"]
        ind = row["sw2021_level1"]
        if inst in selected_set:
            continue
        if industry_count.get(ind, 0) >= MAX_INDUSTRY_COUNT:
            continue
        selected_rows.append(row)
        selected_set.add(inst)
        industry_count[ind] = industry_count.get(ind, 0) + 1

    if len(selected_rows) == 0:
        return pd.DataFrame(columns=df_day.columns)
    return pd.DataFrame(selected_rows)


def build_selected(feature_df, strategy_name, strategy):
    print("=" * 80)
    print("build strategy:", strategy_name)
    print(strategy)

    df = apply_base_filter(
        feature_df,
        require_pcf_positive=strategy.get("require_pcf_positive", False),
        require_profit=strategy.get("require_profit", False),
        use_trend_filter=strategy.get("use_trend_filter", False),
        use_vol_filter=strategy.get("use_vol_filter", False),
        use_reversal_band=strategy.get("use_reversal_band", False),
        min_liquidity_pct=strategy.get("min_liquidity_pct", MIN_LIQUIDITY_PCT),
    )
    df = add_factor_ranks(df)

    df["score"] = 0.0
    for rank_col, weight in strategy["weights"].items():
        if rank_col not in df.columns:
            raise ValueError(f"rank column not found: {rank_col}")
        df["score"] += weight * df[rank_col]
    df = df.dropna(subset=["score"]).copy()
    print("after filter and score:", df.shape)

    trade_dates = get_trade_dates(feature_df)
    date_map = dict(zip(trade_dates["date"], trade_dates["next_date"]))
    df["trade_date"] = df["date"].map(date_map)
    df = df.dropna(subset=["trade_date"]).copy()
    df = df[df["trade_date"] >= start_date_str].copy()

    rebalance_dates = make_rebalance_dates(
        trade_dates, strategy.get("rebalance_days", DEFAULT_REBALANCE_DAYS)
    )
    df = df[df["trade_date"].isin(rebalance_dates)].copy()
    df = df.sort_values(["trade_date", "score"]).copy()
    print("after rebalance filter:", df.shape)

    hold_num = strategy.get("hold_num", DEFAULT_HOLD_NUM)
    buffer_num = strategy.get("buffer_num", DEFAULT_BUFFER_NUM)
    exposure = strategy.get("exposure", 1.0)

    selected_rows = []
    last_holdings = set()
    for trade_date, df_day in df.groupby("trade_date", sort=True):
        final = select_with_industry_limit(
            df_day=df_day,
            last_holdings=last_holdings,
            hold_num=hold_num,
            buffer_num=buffer_num,
        )
        if len(final) == 0:
            continue
        final = final.head(hold_num).copy()
        final["date"] = trade_date
        final["position"] = exposure / len(final)
        final["strategy"] = strategy_name
        selected_rows.append(
            final[["date", "instrument", "position", "score", "strategy"]]
        )
        last_holdings = set(final["instrument"])

    if len(selected_rows) == 0:
        raise ValueError(f"No selected stocks for strategy: {strategy_name}")

    selected = pd.concat(selected_rows, axis=0)
    selected["date"] = pd.to_datetime(selected["date"])
    selected["instrument"] = selected["instrument"].astype(str)
    selected = selected.sort_values(["date", "score"]).reset_index(drop=True)

    print("selected shape:", selected.shape)
    print("date range:", selected["date"].min(), selected["date"].max())
    print(selected.head(20))
    return selected


# ============================================================
# 5. Factor quantile analysis
# ============================================================
def factor_quantile_report(
    df,
    factor_col,
    forward_col="fwd_20d",
    n_quantiles=5,
    ascending=True,
    min_count_per_date=50,
):
    cols = ["date", "instrument", factor_col, forward_col]
    x = df[cols].dropna().copy()
    date_counts = x.groupby("date")["instrument"].count()
    valid_dates = date_counts[date_counts >= min_count_per_date].index
    x = x[x["date"].isin(valid_dates)].copy()

    def assign_quantile(s):
        rank = s.rank(method="first", ascending=ascending)
        try:
            return pd.qcut(rank, n_quantiles, labels=False) + 1
        except ValueError:
            return pd.Series(np.nan, index=s.index)

    x["q"] = x.groupby("date")[factor_col].transform(assign_quantile)
    x = x.dropna(subset=["q"]).copy()
    x["q"] = x["q"].astype(int)

    report = (
        x.groupby("q")[forward_col]
        .agg(["mean", "median", "std", "count"])
        .reset_index()
    )
    if 1 in set(report["q"]) and n_quantiles in set(report["q"]):
        q1 = report.loc[report["q"] == 1, "mean"].iloc[0]
        qn = report.loc[report["q"] == n_quantiles, "mean"].iloc[0]
        print(f"{factor_col}, {forward_col}, Q1-Q{n_quantiles}: {q1 - qn}")
    return report


# ============================================================
# 6. Backtest via M.bigtrader (runs in AI Studio)
# ============================================================
def m5_initialize_bigquant_run(context):
    from bigtrader.finance.commission import PerOrder

    context.set_commission(PerOrder(buy_cost=0.0003, sell_cost=0.0013, min_cost=5))
    context.stop_loss_pct = 0.20
    context.stop_loss_grace_days = 10
    context.target_volatility = 0.25
    context.vol_lookback = 20
    context.daily_returns = []
    context.prev_value = None


def m5_before_trading_start_bigquant_run(context, data):
    pass


def m5_handle_tick_bigquant_run(context, tick):
    pass


def m5_handle_data_bigquant_run(context, data):
    if not context.rebalance_period.is_signal_date(data.current_dt.date()):
        check_stop_loss(context, data)
        return
    today_str = data.current_dt.strftime("%Y-%m-%d")
    today_df = context.data[context.data["date"] == today_str]
    if today_df.empty:
        return

    position_scale = 1.0
    if len(context.daily_returns) >= context.vol_lookback:
        recent_rets = context.daily_returns[-context.vol_lookback :]
        realized_vol = np.std(recent_rets) * np.sqrt(252)
        if realized_vol > 0:
            position_scale = min(1.0, context.target_volatility / realized_vol)

    holdings = context.get_account_positions()
    target_instruments = set()
    for _, row in today_df.iterrows():
        inst = row["instrument"]
        if data.can_trade(inst):
            target_instruments.add(inst)
        elif inst in holdings:
            context.order_target_percent(inst, 0)

    for inst in list(holdings.keys()):
        if inst not in target_instruments:
            context.order_target_percent(inst, 0)

    for _, row in today_df.iterrows():
        inst = row["instrument"]
        if inst not in target_instruments:
            continue
        base_pos = 0.0 if pd.isnull(row["position"]) else float(row["position"])
        final_pos = base_pos * position_scale
        if final_pos > 0:
            context.order_target_percent(inst, final_pos)


def check_stop_loss(context, data):
    positions = context.get_account_positions()
    for inst, pos in positions.items():
        if pos.amount == 0:
            continue
        open_dt = pos.open_date
        if isinstance(open_dt, str):
            open_dt = pd.Timestamp(open_dt).to_pydatetime()
        if not hasattr(open_dt, "date"):
            continue
        holding_days = (data.current_dt.date() - open_dt.date()).days
        if holding_days < context.stop_loss_grace_days:
            continue
        try:
            hist = data.history(inst, "close", 300, "1d")
            if hist.empty:
                continue
            mask = hist.index >= open_dt.strftime("%Y-%m-%d")
            if mask.sum() < 2:
                continue
            relevant = hist[mask]
            highest = relevant.max()
            current = relevant.iloc[-1]
            drawdown = (highest - current) / highest
            if drawdown >= context.stop_loss_pct:
                context.order_target_percent(inst, 0)
        except:
            pass


def m5_handle_trade_bigquant_run(context, trade):
    pass


def m5_handle_order_bigquant_run(context, order):
    pass


def m5_after_trading_bigquant_run(context, data):
    if context.prev_value is None:
        context.prev_value = context.portfolio.portfolio_value
        return
    current_value = context.portfolio.portfolio_value
    ret = (current_value - context.prev_value) / context.prev_value
    context.daily_returns.append(ret)
    if len(context.daily_returns) > 252:
        context.daily_returns.pop(0)
    context.prev_value = current_value


def run_backtest(selected, strategy_name, plot_charts=False):
    selected_to_write = selected[["date", "instrument", "position", "score"]].copy()
    selected_to_write["date"] = pd.to_datetime(selected_to_write["date"])
    selected_to_write["instrument"] = selected_to_write["instrument"].astype(str)
    selected_to_write["position"] = selected_to_write["position"].astype(float)
    selected_to_write["score"] = selected_to_write["score"].astype(float)

    stock_data = dai.DataSource.write_bdb(selected_to_write)

    m5 = M.bigtrader.v30(
        data=stock_data,
        start_date=start_date_str,
        end_date=end_date_str,
        initialize=m5_initialize_bigquant_run,
        before_trading_start=m5_before_trading_start_bigquant_run,
        handle_tick=m5_handle_tick_bigquant_run,
        handle_data=m5_handle_data_bigquant_run,
        handle_trade=m5_handle_trade_bigquant_run,
        handle_order=m5_handle_order_bigquant_run,
        after_trading=m5_after_trading_bigquant_run,
        capital_base=500000,
        frequency="daily",
        product_type="股票",
        rebalance_period_type="交易日",
        rebalance_period_days="1",
        rebalance_period_roll_forward=True,
        backtest_engine_mode="标准模式",
        before_start_days=0,
        volume_limit=1,
        order_price_field_buy="open",
        order_price_field_sell="open",
        benchmark=BENCHMARK,
        plot_charts=plot_charts,
        debug=False,
        backtest_only=False,
        m_name=f"主板_因子测试_{strategy_name}",
    )
    return m5


# ============================================================
# 7. Main execution
# ============================================================
if __name__ == "__main__" or True:
    print("=" * 80)
    print("Step 1: Loading trade dates...")
    trade_dates = load_trade_dates()
    print(f"trade_dates: {trade_dates.shape}")

    print("\nStep 2: Building panel...")
    panel_df = build_panel(trade_dates)

    print("\nStep 3: Factor analysis...")
    research_base = apply_base_filter(panel_df, min_liquidity_pct=MIN_LIQUIDITY_PCT)
    research_ranked = add_factor_ranks(research_base)
    print(f"research_ranked shape: {research_ranked.shape}")

    print("\n--- Factor quantile reports ---")
    for factor in [
        "rank_size",
        "rank_lowvol",
        "rank_pb",
        "rank_rev60",
        "rank_mom20",
        "rank_quality",
    ]:
        try:
            rpt = factor_quantile_report(
                research_ranked, factor_col=factor, forward_col="fwd_20d"
            )
            print(
                factor,
                "- Q1 mean:",
                rpt.loc[rpt["q"] == 1, "mean"].iloc[0]
                if 1 in rpt["q"].values
                else "N/A",
            )
        except Exception as e:
            print(f"{factor}: {e}")

    # Export CSV for each strategy
    os.makedirs("/tmp/signals", exist_ok=True)
    print("\nStep 4: Building and exporting strategies...")
    for strategy_name in STRATEGIES:
        print(f"\n--- Strategy: {strategy_name} ---")
        try:
            selected = build_selected(
                panel_df, strategy_name, STRATEGIES[strategy_name]
            )
            csv_path = f"/tmp/signals/{strategy_name}.csv"
            selected.to_csv(csv_path, index=False)
            print(f"Exported: {csv_path} ({len(selected)} rows)")

            # Optional: run backtest
            # print(f"\nRunning backtest for {strategy_name}...")
            # m = run_backtest(selected, strategy_name, plot_charts=False)
        except Exception as e:
            print(f"[ERROR] {strategy_name}: {e}")

    print("\nDone. Download CSVs from /tmp/signals/")
