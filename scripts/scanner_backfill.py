from __future__ import annotations

import argparse
import asyncio
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from qooi.pipeline import HOUR_MS, now_ms
from qooi.pipeline.coverage import CoverageRunPolicy, coverage_summary, median_span_days
from qooi.pipeline.discovery import rank_discovery, select_symbols
from qooi.pipeline.load import DEFAULT_CACHE_ROOT, MarketLoadPolicy, load_market
from qooi.scanner.workflow import load_config, scanner_market_request
from qooi.transport.okx import OkxClient


def stamp() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(message: str) -> None:
    print(f"{stamp()} {message}", flush=True)


def source_frame(name: str) -> pl.DataFrame:
    path = DEFAULT_CACHE_ROOT.parent / "sources" / f"{name}.parquet"
    return pl.read_parquet(path) if path.exists() else pl.DataFrame()


def bar_frame(symbols: tuple[str, ...], timeframe: str) -> pl.DataFrame:
    frames = []
    for symbol in symbols:
        path = DEFAULT_CACHE_ROOT / symbol / f"bars_{timeframe}.parquet"
        if path.exists():
            frame = pl.read_parquet(path)
            if "symbol" not in frame.columns:
                frame = frame.with_columns(pl.lit(symbol).alias("symbol"))
            frames.append(frame)
    return pl.concat(frames, how="diagonal_relaxed") if frames else pl.DataFrame()


def span_summary(
    frame: pl.DataFrame, symbols: tuple[str, ...], target_rows: int, target_days: int
) -> str:
    if frame.is_empty() or "symbol" not in frame.columns or "timestamp" not in frame.columns:
        return f"symbols=0/{len(symbols)} rows=0 complete=0/{len(symbols)} median_days=0.0"
    frame = frame.filter(pl.col("symbol").is_in(symbols))
    if frame.is_empty():
        return f"symbols=0/{len(symbols)} rows=0 complete=0/{len(symbols)} median_days=0.0"
    grouped = frame.group_by("symbol").agg(
        pl.len().alias("rows"),
        pl.col("timestamp").min().alias("earliest"),
        pl.col("timestamp").max().alias("latest"),
    )
    target_start = now_ms() - target_days * 24 * HOUR_MS
    complete = grouped.filter(
        (pl.col("rows") >= target_rows) & (pl.col("earliest") <= target_start)
    ).height
    spans = (
        (grouped.get_column("latest") - grouped.get_column("earliest")) / (24 * HOUR_MS)
    ).to_list()
    spans = sorted(float(value) for value in spans if value is not None)
    median = spans[len(spans) // 2] if spans else 0.0
    return (
        f"symbols={grouped.height}/{len(symbols)} rows={frame.height} "
        f"complete={complete}/{len(symbols)} median_days={median:.1f}"
    )


def log_completeness(label: str, symbols: tuple[str, ...], target_days: int) -> None:
    bars = span_summary(bar_frame(symbols, "1H"), symbols, target_days * 24, target_days)
    log(f"{label} bars {bars}")
    for name, target_rows in {
        "funding": target_days * 3,
        "open_interest": target_days * 24,
        "taker_volume": target_days * 24,
        "long_short_ratios": target_days * 24,
    }.items():
        log(f"{label} {name} {span_summary(source_frame(name), symbols, target_rows, target_days)}")


def log_coverage(label: str, plans) -> None:
    for name, plan in plans.items():
        fetched_jobs = len(plan.jobs)
        job_pages = sum(job.max_pages for job in plan.jobs)
        log(
            f"{label} {name} statuses={coverage_summary(plan)} "
            f"median_days={median_span_days(plan):.1f} "
            f"jobs={fetched_jobs} job_pages={job_pages} "
            f"estimated_remaining_pages={plan.estimated_pages}"
        )


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/potential-daily-deep-profile-live.toml")
    parser.add_argument("--max-requests", type=int, default=1000)
    parser.add_argument("--max-seconds", type=int, default=900)
    parser.add_argument("--max-requests-per-symbol-product", type=int, default=24)
    args = parser.parse_args()

    config = load_config(Path(args.config))
    async with OkxClient() as okx:
        log("discover start")
        instruments = await okx.instruments()
        tickers = await okx.tickers()
        discovery = rank_discovery(instruments.frame, tickers.frame)
        symbols = select_symbols(discovery, top_n=config.max_symbols)
        request = scanner_market_request(config, symbols)
        policy = MarketLoadPolicy(
            coverage=CoverageRunPolicy(
                max_requests=args.max_requests,
                max_seconds=args.max_seconds,
                max_requests_per_symbol_product=args.max_requests_per_symbol_product,
                concurrency=max(1, config.fetch_concurrency),
                allow_partial=True,
            )
        )
        log(
            f"symbols={len(symbols)} target_days={request.bars.target_days} "
            f"max_requests={args.max_requests} max_seconds={args.max_seconds} "
            f"max_requests_per_symbol_product={args.max_requests_per_symbol_product}"
        )
        log_completeness("before", symbols, request.bars.target_days)
        loaded = await load_market(okx, request, policy, instrument_frame=discovery)
        log_coverage("coverage_before", loaded.coverage_before)
        log(
            "load done "
            f"bar_pages={loaded.stats.bar_pages} "
            f"source_pages={loaded.stats.source_pages} "
            f"provider_bounded={loaded.stats.provider_bounded}"
        )
        log_coverage("coverage_after", loaded.coverage_after)
        log_completeness("after", symbols, request.bars.target_days)


if __name__ == "__main__":
    asyncio.run(main())
