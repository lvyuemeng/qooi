"""Strategy functions — all signals used by the backtester.

Only ``flow_pipeline`` is active in production.  Other modules are kept for
backtest compatibility (``backtest.py`` imports ``portfolio``).
"""

from __future__ import annotations

from typing import Any

from qooi.strategies.flow_pipeline import (
    add_ofi_flow_columns,
    add_regime_features,
    apply_micro_confirmation,
    apply_regime_gate,
)
from qooi.strategies.intraday import multi_factor_intraday_signal
from qooi.strategies.portfolio import (
    AssetSignalState,
    PortfolioLimits,
    allocate_portfolio_weights,
    qualify_asset,
)

__all__ = [
    "add_ofi_flow_columns",
    "add_regime_features",
    "apply_micro_confirmation",
    "apply_regime_gate",
    "multi_factor_intraday_signal",
    "AssetSignalState",
    "PortfolioLimits",
    "allocate_portfolio_weights",
    "qualify_asset",
    "trace_signal_pipeline",
]


# =============================================================================
# Diagnostic utility — trace signal pipeline step by step
# =============================================================================


def trace_signal_pipeline(
    symbol: str = "ETH-USDT",
    timeframe: str = "4h",
    *,
    verbose: bool = True,
) -> dict[str, Any]:
    """Run the full signal pipeline and return intermediate values.

    Returns a dict with keys: bars, last_close, atr_14, regime_score,
    regime_strength, mom_fast, mom_mid, mom_slow, raw_signal,
    ofi_flow_score, ofi_net_flow, signal_after_mc, thresh_long,
    thresh_short, final_signal.

    When ``verbose=True``, also prints each step to stdout.
    """
    from qooi.exchange.indicator import add_indicators
    from qooi.exchange.market import MarketData

    result: dict[str, Any] = {}

    with MarketData("okx") as md:
        df = md.candles(symbol, timeframe=timeframe, limit=500)
    if df.is_empty():
        raise RuntimeError(f"No market data for {symbol} {timeframe}")

    result["bars"] = df.height
    result["last_close"] = float(df["close"][-1])
    if verbose:
        print(f"=== Signal trace: {symbol} {timeframe} ===\n")
        print(
            f"Bars: {df.height}  |  Close range: {df['close'].min():.2f} - {df['close'].max():.2f}"
        )
        print(f"Last close: {df['close'][-1]:.2f}  at ts={df['timestamp'][-1]}")

    # Step 1: indicators
    df = add_indicators(df)
    result["atr_14"] = float(df["atr_14"][-1])
    if verbose:
        print(f"\n[1] add_indicators: atr_14={df['atr_14'][-1]:.4f}")

    # Step 2: regime features
    df = add_regime_features(df)
    result["regime_score"] = float(df["regime_score"][-1])
    result["regime_strength"] = float(df["regime_strength"][-1])
    result["mom_fast"] = float(df["regime_mom_fast"][-1])
    result["mom_mid"] = float(df["regime_mom_mid"][-1])
    result["mom_slow"] = float(df["regime_mom_slow"][-1])
    if verbose:
        print(
            f"[2] regime:   score={df['regime_score'][-1]:.4f}  strength={df['regime_strength'][-1]:.4f}"
        )
        print(
            f"    mom_fast={df['regime_mom_fast'][-1]:.4f}  mom_mid={df['regime_mom_mid'][-1]:.4f}  mom_slow={df['regime_mom_slow'][-1]:.4f}"
        )

    # Step 3: multi-factor intraday
    df = multi_factor_intraday_signal(df)
    raw_signal = float(df["signal"][-1])
    result["raw_signal"] = raw_signal
    tail_sig = df["signal"][-50:].to_list()
    nonzero = [s for s in tail_sig if abs(s) > 0.001]
    result["nonzero_bars"] = len(nonzero)
    result["avg_nonzero_signal"] = sum(nonzero) / max(len(nonzero), 1)
    if verbose:
        print(f"[3] multi_factor_intraday_signal: raw_signal={raw_signal:.4f}")
        print(
            f"    Non-zero signals in last 50 bars: {len(nonzero)} (avg={result['avg_nonzero_signal']:.4f})"
        )

    # Step 4: OFI flow
    df = add_ofi_flow_columns(df)
    result["ofi_flow_score"] = float(df["ofi_flow_score"][-1])
    result["ofi_net_flow"] = float(df["ofi_net_flow"][-1])
    if verbose:
        print(
            f"[4] OFI flow:  score={df['ofi_flow_score'][-1]:.4f}  net_flow={df['ofi_net_flow'][-1]:.4f}"
        )

    # Step 5: micro confirmation
    before_mc = float(df["signal"][-1])
    df = apply_micro_confirmation(df)
    final = float(df["signal"][-1])
    result["final_signal"] = final
    result["signal_after_mc"] = final
    if verbose:
        print(f"[5] micro_conf: before={before_mc:.4f}  after={final:.4f}")
        print(f"\n=== FINAL SIGNAL: {final:.4f} ===")

    return result
