"""Research backtest CLI."""

from __future__ import annotations

import argparse

from qooi.research.config import (
    DATA_SOURCE_CHOICES,
    PROFILE_CHOICES,
    STYLE_CHOICES,
    UNIVERSE_CHOICES,
)
from qooi.research.run import run_command
from qooi.research.strategies import (
    BENCHMARK_GROUP_CHOICES,
    DEFAULT_STRATEGY,
    STRATEGY_CHOICES,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run QOOI strategy backtests")
    parser.add_argument("--profile", choices=PROFILE_CHOICES, default="research")
    parser.add_argument(
        "--mode", choices=("base", "grid", "martingale", "reverse", "hedge"), default="base"
    )
    parser.add_argument("--strategy", choices=STRATEGY_CHOICES, default=DEFAULT_STRATEGY)
    parser.add_argument("--strategies", default="")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument(
        "--benchmark-group", choices=BENCHMARK_GROUP_CHOICES, default="zscore-family"
    )
    parser.add_argument("--cache-audit", action="store_true")
    parser.add_argument("--symbol", default="")
    parser.add_argument("--exclude-symbol", default="")
    parser.add_argument("--universe", choices=UNIVERSE_CHOICES, default="core")
    parser.add_argument("--data-source", choices=DATA_SOURCE_CHOICES, default="swap")
    parser.add_argument(
        "--allow-swap-signal-fallback",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--days", type=int, default=None)
    parser.add_argument("--min-bars", type=int, default=None)
    parser.add_argument("--min-coverage-pct", type=float, default=None)

    parser.add_argument("--style", choices=STYLE_CHOICES, default="single")
    parser.add_argument("--train-bars", type=int, default=500)
    parser.add_argument("--test-bars", type=int, default=100)
    parser.add_argument("--step-bars", type=int, default=100)
    parser.add_argument("--folds", type=int, default=5)

    parser.add_argument("--entry-z", type=float, default=2.0)
    parser.add_argument("--exit-z", type=float, default=0.25)
    parser.add_argument("--z-period", type=int, default=20)
    parser.add_argument("--ewma-span", type=int, default=48)
    parser.add_argument("--robust-period", type=int, default=96)
    parser.add_argument("--volatility-ratio-max", type=float, default=2.5)
    parser.add_argument("--adx-max", type=float, default=25.0)
    parser.add_argument("--adx-threshold", type=float, default=15.0)
    parser.add_argument("--volume-mult", type=float, default=1.1)
    parser.add_argument("--trend-maturity", type=int, default=12)
    parser.add_argument("--mom-threshold", type=float, default=0.003)

    parser.add_argument("--normalize-sizing", action="store_true")
    parser.add_argument("--risk-pct", type=float, default=None)
    parser.add_argument("--max-notional-pct", type=float, default=None)
    parser.add_argument("--leverage", type=float, default=None)
    parser.add_argument("--capital", type=float, default=None)
    parser.add_argument("--min-contracts", type=int, default=None)

    parser.add_argument("--max-dd-pct", type=float, default=None)
    parser.add_argument("--max-notional-exposure-pct", type=float, default=None)
    parser.add_argument("--min-trades", type=int, default=0)
    parser.add_argument("--min-pf", type=float, default=0.0)
    parser.add_argument("--min-expectancy-pct", type=float, default=None)
    parser.add_argument("--fail-on-risk", action="store_true")

    parser.add_argument("--drawdown-stop-pct", type=float, default=None)
    parser.add_argument("--no-drawdown-stop", action="store_true")
    parser.add_argument("--max-bars", type=int, default=10)
    parser.add_argument("--stop-mult", type=float, default=1.5)
    parser.add_argument("--target-mult", type=float, default=1.3)
    parser.add_argument("--trail-mult", type=float, default=2.0)

    parser.add_argument("--detail", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--diagnostics", action="store_true")
    parser.add_argument("--explain-layers", action="store_true")
    parser.add_argument("--show-status", action="store_true")
    args = parser.parse_args()
    if args.no_drawdown_stop:
        args.drawdown_stop_pct = None
    return args


def main() -> None:
    print(run_command(_parse_args()))


if __name__ == "__main__":
    main()
