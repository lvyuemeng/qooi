"""Data loading and source-policy alignment for research backtests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import polars as pl

from qooi.core.config import PairConfig
from qooi.exchange.store import CacheStore, HistoryCoverage, HistoryRequest, validate_history
from qooi.research.config import ResolvedBacktestConfig
from qooi.strategies import StrategyBehavior, compute_signal_frame
from qooi.strategies.indicators import add_indicators


@dataclass(frozen=True)
class PreparedBacktestFrame:
    pair: PairConfig
    frame: pl.DataFrame
    precomputed_signal: bool
    signal_inst_id: str
    execution_inst_id: str
    signal_coverage: HistoryCoverage
    execution_coverage: HistoryCoverage
    metadata: tuple[str, ...]


def source_inst_ids(pair: PairConfig, data_source: str) -> tuple[str, str]:
    if data_source == "spot_signal_swap_exec":
        return pair.asset.sig_symbol, pair.asset.symbol
    if data_source == "spot":
        return pair.asset.sig_symbol, pair.asset.sig_symbol
    return pair.asset.symbol, pair.asset.symbol


def load_cache(
    store: CacheStore,
    inst_id: str,
    timeframe: str,
    config: ResolvedBacktestConfig,
    *,
    refresh: bool,
    trim_to_target: bool = True,
) -> tuple[pl.DataFrame, HistoryCoverage]:
    df, coverage = store.load_history(
        HistoryRequest(
            inst_id=inst_id,
            bar=timeframe,
            days=config.days,
            min_bars=config.min_bars,
            refresh=refresh,
        )
    )
    if trim_to_target and df.height > coverage.target.target_bars:
        df = df.tail(coverage.target.target_bars)
        coverage = validate_history(df, coverage.target, refreshed=coverage.refreshed)
    if config.risk_gates.min_coverage_pct > 0 and (
        coverage.coverage_pct < config.risk_gates.min_coverage_pct
    ):
        raise RuntimeError(
            f"cache coverage below threshold for {inst_id}: "
            f"{coverage.coverage_pct:.1f}% < {config.risk_gates.min_coverage_pct:.1f}%"
        )
    return df, coverage


def prepare_backtest_frame(
    store: CacheStore,
    pair: PairConfig,
    strategy: StrategyBehavior,
    args: Any,
    config: ResolvedBacktestConfig,
) -> PreparedBacktestFrame:
    signal_inst_id, execution_inst_id = source_inst_ids(pair, config.data_source)
    timeframe = pair.asset.timeframe
    refresh = bool(getattr(args, "refresh_cache", False))
    try:
        signal_df, signal_summary = load_cache(
            store, signal_inst_id, timeframe, config, refresh=refresh
        )
    except FileNotFoundError:
        if not bool(getattr(args, "allow_swap_signal_fallback", False)) or (
            signal_inst_id == pair.asset.symbol
        ):
            raise
        signal_inst_id = pair.asset.symbol
        signal_df, signal_summary = load_cache(
            store, signal_inst_id, timeframe, config, refresh=refresh
        )

    if execution_inst_id == signal_inst_id:
        execution_df = signal_df
        execution_summary = signal_summary
        frame = execution_df
        precomputed_signal = False
    else:
        execution_df, execution_summary = load_cache(
            store, execution_inst_id, timeframe, config, refresh=refresh
        )
        signal_frame = compute_signal_frame(signal_df, strategy).select(
            "timestamp",
            "raw_entry_signal",
            "entry_signal",
            "position_signal",
            "exit_signal",
            "signal_strength",
            "signal_id",
            "signal",
        )
        frame = add_indicators(execution_df).join(signal_frame, on="timestamp", how="inner")
        precomputed_signal = True

    metadata = (
        *config.metadata(),
        f"signal_inst={signal_inst_id}",
        f"execution_inst={execution_inst_id}",
        signal_summary.note(),
        execution_summary.note(),
    )
    if signal_summary.notes:
        print(f"cache warning {signal_inst_id}: {', '.join(signal_summary.notes)}")
    if execution_summary.notes and execution_inst_id != signal_inst_id:
        print(f"cache warning {execution_inst_id}: {', '.join(execution_summary.notes)}")

    return PreparedBacktestFrame(
        pair=pair,
        frame=frame,
        precomputed_signal=precomputed_signal,
        signal_inst_id=signal_inst_id,
        execution_inst_id=execution_inst_id,
        signal_coverage=signal_summary,
        execution_coverage=execution_summary,
        metadata=metadata,
    )


def cache_audit_rows(
    pairs: tuple[PairConfig, ...],
    args: Any,
    config: ResolvedBacktestConfig,
) -> list[list[str]]:
    store = CacheStore()
    rows: list[list[str]] = []
    for pair in pairs:
        for inst_id in dict.fromkeys(source_inst_ids(pair, config.data_source)):
            try:
                _df, coverage = load_cache(
                    store,
                    inst_id,
                    pair.asset.timeframe,
                    config,
                    refresh=bool(getattr(args, "refresh_cache", False)),
                    trim_to_target=False,
                )
                status = (
                    "PASS"
                    if coverage.coverage_pct >= config.risk_gates.min_coverage_pct
                    else "LOW"
                )
                rows.append(
                    [
                        status,
                        inst_id,
                        pair.asset.timeframe,
                        str(coverage.actual_bars),
                        str(coverage.target.target_bars),
                        f"{coverage.coverage_pct:.1f}",
                        str(coverage.actual_start_ms or "n/a"),
                        str(coverage.actual_end_ms or "n/a"),
                        ",".join(coverage.notes),
                    ]
                )
            except Exception as exc:
                rows.append(
                    [
                        "ERROR",
                        inst_id,
                        pair.asset.timeframe,
                        "0",
                        "0",
                        "0.0",
                        "n/a",
                        "n/a",
                        str(exc),
                    ]
                )
    return rows
