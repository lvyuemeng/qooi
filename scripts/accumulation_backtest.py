from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

from qooi.accumulation.config import load_accumulation_config
from qooi.accumulation.csv_io import read_artifact, write_csv_artifacts
from qooi.core.event_backtest import build_backtest_events, summarize_backtest_events
from qooi.sources.coverage import manifest_frame, source_manifest_row

SUPPORTED_HORIZONS = "3h,7h,24h,3d,7d"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backtest accumulation alert artifacts")
    parser.add_argument("--config", default="configs/research/accumulation-mvp.toml")
    parser.add_argument("--horizons", default="3h,7h,24h,3d,7d")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.horizons != SUPPORTED_HORIZONS:
        raise ValueError(
            f"custom horizons are not implemented yet; supported value is {SUPPORTED_HORIZONS}"
        )
    config = load_accumulation_config(Path(args.config))
    scores = read_artifact(config.output_dir, "scores")
    discovery = read_artifact(config.output_dir, "candidate_discovery")
    bars = read_artifact(config.output_dir, "source_bars")
    event_frames = []
    coverage_rows = []
    for symbol in _symbols_from_scores_or_discovery(scores, discovery):
        symbol_scores = _symbol_frame(scores, symbol)
        prices = _symbol_frame(bars, symbol)
        if prices.is_empty():
            coverage_rows.append(
                source_manifest_row(
                    symbol=symbol,
                    source="bars",
                    phase="backtest",
                    status="missing",
                    warning="price_missing;backtest_skipped",
                    stop_reason="price_missing",
                )
            )
            continue
        coverage_rows.append(
            source_manifest_row(
                symbol=symbol,
                source="bars",
                phase="backtest",
                status="ok",
                rows=prices.height,
            )
        )
        event_frames.append(
            build_backtest_events(
                symbol_scores,
                prices,
                take_profit_pct=config.scoring.stop_loss_pct,
                stop_loss_pct=config.scoring.stop_loss_pct,
            )
        )
    events = (
        pl.concat(event_frames, how="vertical")
        if event_frames
        else build_backtest_events(pl.DataFrame(), pl.DataFrame())
    )
    summary = summarize_backtest_events(events)
    coverage = manifest_frame(coverage_rows)
    write_csv_artifacts(
        config.output_dir,
        backtest_events=events,
        backtest_summary=summary,
        data_coverage=coverage,
    )
    print(
        f"wrote backtest_events={events.height} summary_rows={summary.height} "
        f"out={config.output_dir}"
    )


def _symbols_from_scores_or_discovery(
    scores: pl.DataFrame, discovery: pl.DataFrame
) -> tuple[str, ...]:
    if not scores.is_empty() and "symbol" in scores.columns:
        return tuple(dict.fromkeys(str(symbol) for symbol in scores["symbol"].to_list()))
    if discovery.is_empty() or "symbol" not in discovery.columns:
        return ()
    frame = discovery
    if "eligible" in discovery.columns:
        eligible = discovery.filter(pl.col("eligible"))
        if not eligible.is_empty():
            frame = eligible
    return tuple(dict.fromkeys(str(symbol) for symbol in frame["symbol"].to_list()))


def _symbol_frame(frame: pl.DataFrame, symbol: str) -> pl.DataFrame:
    if frame.is_empty() or "symbol" not in frame.columns:
        return pl.DataFrame()
    return frame.filter(pl.col("symbol") == symbol)


if __name__ == "__main__":
    main()
