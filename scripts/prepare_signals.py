"""Run this on BigQuant AI Studio (cloud) via:  bq aistudio run --file scripts/prepare_signals.py

This script has DAI access (which your local SDK doesn't).
It selects stocks and writes the result to a CSV file.
"""

import pandas as pd
from bigquant import dai

# ============================================================
# Config
# ============================================================
HOLD_NUM = 5
START_DATE = "2019-01-01"
END_DATE = "2024-08-14"

# ============================================================
# 1. Fetch data
# ============================================================
print("Fetching base factors...")
base = (
    dai.query(
        """
    SELECT date, instrument, float_market_cap,
           list_days, st_status, suspended, is_bz50
    FROM cn_stock_prefactors_community
    WHERE list_days > 365 AND st_status = 0
      AND suspended = 0 AND float_market_cap > 0
      AND is_bz50 = 0
    """,
        filters={"date": [START_DATE, END_DATE]},
    )
    .df()
    .dropna()
)

print(f"  base: {len(base)} rows")

print("Fetching financial factors...")
fin = (
    dai.query(
        """
    SELECT date, instrument, roe_avg_deduct_ttm, debt_to_asset
    FROM cn_stock_prefactors
    """,
        filters={"date": [START_DATE, END_DATE]},
    )
    .df()
    .dropna()
)
fin = fin[fin["debt_to_asset"].notna()]
print(f"  fin: {len(fin)} rows")

print("Fetching daily bars...")
bars = (
    dai.query(
        """
    SELECT date, instrument, close, turn, amount
    FROM cn_stock_bar1d
    WHERE date >= '2018-11-01'
    """,
        filters={"date": ["2018-11-01", END_DATE]},
    )
    .df()
    .dropna()
)
print(f"  bars: {len(bars)} rows")

# ============================================================
# 2. Compute factors
# ============================================================
print("Computing factors...")
bars = bars.sort_values(["instrument", "date"])
bars["ret_1d"] = bars.groupby("instrument")["close"].pct_change()
bars["ret_20d"] = bars.groupby("instrument")["close"].transform(
    lambda x: x / x.shift(20) - 1
)
bars["avg_turn_20"] = bars.groupby("instrument")["turn"].transform(
    lambda x: x.rolling(20, min_periods=20).mean()
)
bars["avg_amount_20"] = bars.groupby("instrument")["amount"].transform(
    lambda x: x.rolling(20, min_periods=20).mean()
)
bars["avg_amount_5"] = bars.groupby("instrument")["amount"].transform(
    lambda x: x.rolling(5, min_periods=5).mean()
)
bars["vol_20d"] = bars.groupby("instrument")["ret_1d"].transform(
    lambda x: x.rolling(20, min_periods=20).std()
)
bars["volume_ratio"] = bars["avg_amount_5"] / bars["avg_amount_20"]
bars = bars[bars["date"] >= START_DATE].dropna(
    subset=["ret_20d", "avg_turn_20", "vol_20d", "volume_ratio"]
)

# ============================================================
# 3. Merge & filter
# ============================================================
print("Merging...")
merged = pd.merge(base, fin, on=["date", "instrument"], how="inner")
merged = pd.merge(merged, bars, on=["date", "instrument"], how="inner")

merged = merged[
    (merged["avg_turn_20"] <= 0.10)
    & (merged["vol_20d"] <= 0.04)
    & (merged["volume_ratio"] < 3.0)
]

# ============================================================
# 4. Select stocks
# ============================================================
print("Selecting stocks...")


def select_stocks(data, hold_num):
    data = data.copy()
    data["size_score"] = data.groupby("date")["float_market_cap"].rank(
        pct=True, ascending=True
    )
    data["reversal_score"] = data.groupby("date")["ret_20d"].rank(
        pct=True, ascending=True
    )
    data["final_score"] = 0.6 * data["size_score"] + 0.4 * data["reversal_score"]
    selected = (
        data.sort_values(["date", "final_score"], ascending=[True, False])
        .groupby("date")
        .head(hold_num)
        .copy()
    )
    return selected


merged["danger"] = merged["date"].apply(lambda d: pd.Timestamp(d).month in [1, 4])

danger = merged[
    merged["danger"]
    & (merged["roe_avg_deduct_ttm"] > 3)
    & (merged["debt_to_asset"] < 0.8)
]
safe = merged[~merged["danger"]]

selected = pd.concat(
    [select_stocks(safe, HOLD_NUM), select_stocks(danger, HOLD_NUM)],
    ignore_index=True,
).sort_values(["date", "instrument"])

selected["position"] = 1.0 / HOLD_NUM
selected = selected[["date", "instrument", "position", "final_score"]]
selected.columns = ["date", "instrument", "position", "score"]

# ============================================================
# 5. Save & upload as DataSource
# ============================================================
print(f"Saving {len(selected)} signal rows...")

# Save to CSV (downloadable from AI Studio)
csv_path = "/tmp/signals.csv"
selected.to_csv(csv_path, index=False)
print(f"CSV saved to {csv_path}")

# Also register as a BigQuant DataSource for direct use
ds = dai.DataSource.write_bdb(selected)
print(f"DataSource created: {ds.id}")
print("Done.")
