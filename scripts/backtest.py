"""Pipeline backtest — ensemble comparison and per-strategy detail.

Uses BacktestExecutor from qooi.core.executor and Report from qooi.core.evaluate.
Strategy-independent: works identically for any PairConfig.
"""

import polars as pl

from qooi.core.config import PAIRS
from qooi.core.evaluate import Report, compare
from qooi.core.executor import BacktestExecutor


def _cache_path(sig_symbol: str) -> str:
    return f"data/cache/{sig_symbol.replace('-', '')}_1H.parquet"


if __name__ == "__main__":
    reports = []
    for pair in PAIRS:
        path = _cache_path(pair.asset.sig_symbol)
        df = pl.read_parquet(path)
        bt = BacktestExecutor(initial_capital=pair.asset.capital, cost_pct=0.00005)
        trades, equity = bt.run(df, pair)
        reports.append(Report.from_raw(trades, equity, pair))

    print(compare(*reports))
    for r in reports:
        print(f"\n{r.label}")
        print(r.table())
