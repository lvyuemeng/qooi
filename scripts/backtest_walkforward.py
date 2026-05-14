"""Walk-forward backtest — OOS stability analysis.

Uses walk_forward from qooi.core.styles.  Strategy-independent.
"""

import polars as pl

from qooi.core.config import PAIRS
from qooi.core.executor import BacktestExecutor
from qooi.core.styles import walk_forward


def _cache_path(sig_symbol: str) -> str:
    return f"data/cache/{sig_symbol.replace('-', '_')}_1H.parquet"


if __name__ == "__main__":
    for pair in PAIRS:
        path = _cache_path(pair.asset.sig_symbol)
        df = pl.read_parquet(path)

        def trades_fn(seg):
            bt = BacktestExecutor(initial_capital=pair.asset.capital)
            return bt.run(seg, pair, strategy="momentum_burst")

        result = walk_forward(trades_fn, df, train_bars=500, test_bars=100)
        print(result.summary())
