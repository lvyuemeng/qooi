from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

import httpx
import polars as pl

from qooi.accumulation.config import AccumulationConfig, load_accumulation_config
from qooi.accumulation.csv_io import (
    artifact_path,
    read_artifact,
    read_source_bundle,
    write_csv_artifacts,
    write_source_bundle,
    write_text_artifact,
)
from qooi.accumulation.database import maybe_store_frame
from qooi.accumulation.discovery import discover_candidates, select_candidate_symbols
from qooi.accumulation.features import join_hourly_accumulation_features_batch
from qooi.accumulation.scoring import score_accumulation_features
from qooi.accumulation.summary import (
    CandidateReadoutSettings,
    build_candidate_detail,
    build_candidate_summary,
    build_next_fetch_actions,
    render_candidate_rationale,
)
from qooi.exchange.store import (
    AsyncCacheStore,
    BooksRequest,
    CacheStore,
    HistoryRefreshRequest,
    _normalize_bars,
    _read_frame,
)
from qooi.sources.coverage import (
    compute_source_coverage_score,
    manifest_frame,
    manifest_row_from_history_coverage,
    source_manifest_row,
)
from qooi.sources.messages import (
    LocalMessageSettings,
    classify_message_rows,
    normalize_local_messages,
)
from qooi.sources.okx import (
    OKX_BASE_URL,
    fetch_okx_book_snapshot_async,
    fetch_okx_funding_history_async,
    fetch_okx_open_interest_async,
    fetch_okx_recent_trades_async,
)
from qooi.sources.polymarket import (
    POLYMARKET_GAMMA_BASE_URL,
    fetch_polymarket_search_async,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run accumulation scanner")
    parser.add_argument("--config", default="configs/research/accumulation-mvp.toml")
    parser.add_argument(
        "--phase",
        choices=(
            "discover",
            "collect-market",
            "collect-onchain",
            "collect-context",
            "score",
            "summarize",
            "all",
        ),
        default="score",
    )
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--symbols", default="")
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--min-volume-usd", type=float, default=None)
    parser.add_argument("--min-coverage-pct", type=float, default=None)
    parser.add_argument("--fetch-concurrency", type=int, default=None)
    parser.add_argument("--book-mode", choices=("snapshot", "sample", "off"), default=None)
    parser.add_argument("--refresh-discovery", action="store_true")
    parser.add_argument("--refresh-bars", action="store_true")
    parser.add_argument("--refresh-trades", action="store_true")
    parser.add_argument("--refresh-context", action="store_true")
    parser.add_argument(
        "--summary-latest-only", action=argparse.BooleanOptionalAction, default=True
    )
    return parser.parse_args()


def _parse_symbols(raw: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in raw.split(",") if part.strip())


def run_discover(
    config: AccumulationConfig,
    *,
    top_n: int,
    min_volume_usd: float | None,
    manual_symbols: tuple[str, ...] = (),
) -> tuple[tuple[str, ...], pl.DataFrame, pl.DataFrame]:
    result = discover_candidates(
        config,
        top_n=top_n,
        min_volume_usd=min_volume_usd,
        symbols=manual_symbols,
    )
    write_csv_artifacts(
        config.output_dir,
        discovery=result.discovery,
        source_manifest=result.manifest,
        data_coverage=result.manifest,
    )
    _maybe_store(config, "candidate_discovery", result.discovery)
    _maybe_store(config, "source_manifest", result.manifest)
    return result.symbols, result.discovery, result.manifest


def resolve_symbols_or_discover(
    config: AccumulationConfig,
    args: argparse.Namespace,
    ) -> tuple[tuple[str, ...], pl.DataFrame, pl.DataFrame]:
    manual_symbols = _parse_symbols(args.symbols)
    discovery_path = config.output_dir / "candidate-discovery.csv"
    if args.phase == "discover":
        return run_discover(
            config,
            top_n=args.top_n,
            min_volume_usd=args.min_volume_usd,
            manual_symbols=manual_symbols,
        )
    if args.phase == "all" and not manual_symbols:
        return run_discover(
            config,
            top_n=args.top_n,
            min_volume_usd=args.min_volume_usd,
        )
    if manual_symbols:
        discovery = read_artifact(config.output_dir, "candidate_discovery")
        return manual_symbols, discovery, read_artifact(config.output_dir, "source_manifest")
    if args.refresh_discovery and args.phase in {"collect-market", "all"}:
        return run_discover(config, top_n=args.top_n, min_volume_usd=args.min_volume_usd)
    if not discovery_path.exists():
        raise SystemExit(
            "candidate-discovery.csv is missing; run --phase discover first or pass --symbols"
        )
    discovery = read_artifact(config.output_dir, "candidate_discovery")
    symbols = select_candidate_symbols(discovery, top_n=args.top_n)
    return symbols, discovery, read_artifact(config.output_dir, "source_manifest")


async def _refresh_bars(
    config: AccumulationConfig, symbols: tuple[str, ...], *, concurrency: int, refresh: bool
) -> tuple[pl.DataFrame, pl.DataFrame]:
    requests = tuple(
        HistoryRefreshRequest(
            inst_id=symbol,
            bar=config.market.bar,
            days=config.market.days,
            min_bars=config.market.days * 24,
            refresh=refresh or config.market.refresh,
        )
        for symbol in symbols
    )
    async with AsyncCacheStore() as store:
        results = await store.many(requests, concurrency=concurrency, fail_fast=False)
    bars = []
    for result in results:
        if result.path.exists():
            frame = _read_frame(result.path, _normalize_bars)
            if not frame.is_empty():
                bars.append(frame.with_columns(pl.lit(result.request.inst_id).alias("symbol")))
    return (
        manifest_frame(
            [
                manifest_row_from_history_coverage(
                    result.coverage,
                    phase="collect-market",
                    error=result.error,
                )
                for result in results
            ]
        ),
        pl.concat(bars, how="vertical") if bars else pl.DataFrame(),
    )


async def _collect_public_sources(
    config: AccumulationConfig,
    symbols: tuple[str, ...],
    discovery: pl.DataFrame,
    *,
    concurrency: int,
    book_mode: str,
    refresh_trades: bool,
    refresh_context: bool,
    collect_books: bool = True,
) -> tuple[pl.DataFrame, dict[str, pl.DataFrame]]:
    semaphore = asyncio.Semaphore(max(1, concurrency))
    manifests = []
    frames_by_source: dict[str, list[pl.DataFrame]] = {
        "books": [],
        "trades": [],
        "funding": [],
        "open_interest": [],
    }
    async with httpx.AsyncClient(base_url=OKX_BASE_URL, timeout=20.0) as client:

        async def run_symbol(
            symbol: str,
        ) -> tuple[list[pl.DataFrame], dict[str, pl.DataFrame]]:
            async with semaphore:
                return await _collect_symbol_sources(
                    client,
                    config,
                    symbol,
                    _discovery_row(discovery, symbol),
                    book_mode=book_mode,
                    refresh_trades=refresh_trades,
                    refresh_context=refresh_context,
                    collect_books=collect_books,
                )

        for symbol_manifests, symbol_frames in await asyncio.gather(
            *(run_symbol(symbol) for symbol in symbols)
        ):
            manifests.extend(symbol_manifests)
            for source, frame in symbol_frames.items():
                if not frame.is_empty():
                    frames_by_source[source].append(frame)
    frames = {
        source: pl.concat(source_frames, how="vertical") if source_frames else pl.DataFrame()
        for source, source_frames in frames_by_source.items()
    }
    return pl.concat(manifests, how="vertical") if manifests else manifest_frame([]), frames


async def _collect_symbol_sources(
    client: httpx.AsyncClient,
    config: AccumulationConfig,
    symbol: str,
    discovery_row: dict[str, object] | None,
    *,
    book_mode: str,
    refresh_trades: bool,
    refresh_context: bool,
    collect_books: bool = True,
) -> tuple[list[pl.DataFrame], dict[str, pl.DataFrame]]:
    manifests = []
    frames: dict[str, pl.DataFrame] = {}
    if not collect_books:
        pass
    elif book_mode == "off":
        manifests.append(
            manifest_frame(
                [
                    source_manifest_row(
                        symbol=symbol, source="books", phase="collect-market", status="skipped"
                    )
                ]
            )
        )
    elif book_mode == "snapshot":
        result = await fetch_okx_book_snapshot_async(client, symbol, limit=config.market.book_depth)
        if not result.frame.is_empty():
            frames["books"] = _with_symbol(result.frame, symbol)
        manifests.append(result.manifest)
    contract_value = _float_or_none(discovery_row, "ct_val")
    contract_ccy = _str_or_none(discovery_row, "ct_val_ccy")
    contract_base = _str_or_none(discovery_row, "base_ccy")
    trades_path = artifact_path(config.output_dir, "source_trades")
    if refresh_trades or not trades_path.exists():
        trades = await fetch_okx_recent_trades_async(
            client,
            symbol,
            limit=config.sources.trade_limit,
            contract_value=contract_value,
            contract_value_currency=contract_ccy,
            contract_base_currency=contract_base,
        )
        if not trades.frame.is_empty():
            frames["trades"] = _with_symbol(trades.frame, symbol)
        manifests.append(trades.manifest)
    else:
        manifests.append(_local_file_manifest(symbol, "trades", trades_path))
    funding_path = artifact_path(config.output_dir, "source_funding")
    if refresh_context or not funding_path.exists():
        funding = await fetch_okx_funding_history_async(
            client, symbol, limit=config.sources.funding_limit
        )
        if not funding.frame.is_empty():
            frames["funding"] = _with_symbol(funding.frame, symbol)
        manifests.append(funding.manifest)
    else:
        manifests.append(_local_file_manifest(symbol, "funding", funding_path))
    oi_path = artifact_path(config.output_dir, "source_open_interest")
    if refresh_context or not oi_path.exists():
        oi = await fetch_okx_open_interest_async(client, symbol)
        if not oi.frame.is_empty():
            frames["open_interest"] = _with_symbol(oi.frame, symbol)
        manifests.append(oi.manifest)
    else:
        manifests.append(_local_file_manifest(symbol, "open_interest", oi_path))
    return manifests, frames


def run_collect_market(
    config: AccumulationConfig,
    symbols: tuple[str, ...],
    discovery: pl.DataFrame,
    *,
    concurrency: int,
    book_mode: str,
    refresh_bars: bool,
    refresh_trades: bool,
    refresh_context: bool,
) -> pl.DataFrame:
    bars_result = asyncio.run(
        _refresh_bars(config, symbols, concurrency=concurrency, refresh=refresh_bars)
    )
    if isinstance(bars_result, tuple):
        bars_manifest, bars_frame = bars_result
    else:
        bars_manifest, bars_frame = bars_result, pl.DataFrame()
    if book_mode == "sample":
        sample_symbols = symbols[: config.market.book_sample_symbols_max]
        with CacheStore() as store:
            sampled_books = []
            for symbol in sample_symbols:
                books = store.books(
                    BooksRequest(
                        inst_id=symbol,
                        samples=config.market.book_samples,
                        limit=config.market.book_depth,
                        refresh=True,
                        append=True,
                        transport="rest",
                        every_seconds=config.market.book_every_seconds,
                    )
                )
                if not books.is_empty():
                    sampled_books.append(_with_symbol(books, symbol))
        sampled = set(sample_symbols)
        book_manifest = manifest_frame(
            [
                source_manifest_row(
                    symbol=s,
                    source="books",
                    phase="collect-market",
                    status="ok" if s in sampled else "skipped",
                    warning="" if s in sampled else "book_sample_symbol_limit",
                )
                for s in symbols
            ]
        )
        other_result = asyncio.run(
            _collect_public_sources(
                config,
                symbols,
                discovery,
                concurrency=concurrency,
                book_mode="off",
                refresh_trades=refresh_trades,
                refresh_context=refresh_context,
                collect_books=False,
            )
        )
        if isinstance(other_result, tuple):
            other_manifest, other_frames = other_result
        else:
            other_manifest, other_frames = other_result, {}
        public_manifest = pl.concat([book_manifest, other_manifest], how="vertical")
        public_frames = {
            **other_frames,
            "books": pl.concat(sampled_books, how="vertical") if sampled_books else pl.DataFrame(),
        }
    else:
        public_result = asyncio.run(
            _collect_public_sources(
                config,
                symbols,
                discovery,
                concurrency=concurrency,
                book_mode=book_mode,
                refresh_trades=refresh_trades,
                refresh_context=refresh_context,
            )
        )
        if isinstance(public_result, tuple):
            public_manifest, public_frames = public_result
        else:
            public_manifest, public_frames = public_result, {}
    existing = read_artifact(config.output_dir, "source_manifest")
    frames = [frame for frame in (existing, bars_manifest, public_manifest) if not frame.is_empty()]
    manifest = pl.concat(frames, how="vertical") if frames else manifest_frame([])
    write_csv_artifacts(config.output_dir, source_manifest=manifest, data_coverage=manifest)
    write_source_bundle(
        config.output_dir,
        bars=bars_frame,
        books=public_frames.get("books"),
        trades=public_frames.get("trades"),
        funding=public_frames.get("funding"),
        open_interest=public_frames.get("open_interest"),
    )
    _maybe_store(config, "source_manifest", manifest)
    return manifest


def run_collect_context(
    config: AccumulationConfig,
    symbols: tuple[str, ...],
    *,
    concurrency: int,
) -> pl.DataFrame:
    manifests = []
    market_frames = []
    messages = pl.DataFrame()
    message_classifications = pl.DataFrame()
    polymarket = config.sources.polymarket
    if polymarket.enabled:
        context_manifest, market_frames = asyncio.run(
            _collect_polymarket_context(config, symbols, concurrency=concurrency)
        )
        manifests.append(context_manifest)
    else:
        manifests.append(
            _context_manifest_rows(
                symbols,
                source="polymarket_markets",
                status="skipped",
                warning="polymarket_disabled",
            )
        )
    message_manifest, messages, message_classifications = _collect_local_message_context(
        config, symbols
    )
    manifests.append(message_manifest)
    manifest = _concat_frames(read_artifact(config.output_dir, "source_manifest"), *manifests)
    write_csv_artifacts(config.output_dir, source_manifest=manifest, data_coverage=manifest)
    write_source_bundle(
        config.output_dir,
        messages=messages,
        message_classifications=message_classifications,
        polymarket_markets=pl.concat(market_frames, how="vertical")
        if market_frames
        else pl.DataFrame(),
    )
    _maybe_store(config, "source_manifest", manifest)
    return manifest


def _collect_local_message_context(
    config: AccumulationConfig, symbols: tuple[str, ...]
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    messages_config = config.sources.messages
    if not messages_config.enabled:
        return (
            _context_manifest_rows(
                symbols,
                source="messages",
                status="skipped",
                warning="messages_disabled",
            ),
            pl.DataFrame(),
            pl.DataFrame(),
        )
    if not messages_config.path.strip():
        return (
            _context_manifest_rows(
                symbols,
                source="messages",
                status="missing",
                warning="local_messages_path_missing",
            ),
            pl.DataFrame(),
            pl.DataFrame(),
        )
    path = Path(messages_config.path)
    if not path.exists():
        return (
            _context_manifest_rows(
                symbols,
                source="messages",
                status="missing",
                warning="local_messages_file_missing",
            ),
            pl.DataFrame(),
            pl.DataFrame(),
        )
    settings = LocalMessageSettings(default_source=messages_config.default_source)
    normalized = normalize_local_messages(pl.read_csv(path), settings=settings)
    if symbols and "symbol" in normalized.columns:
        normalized = normalized.filter(pl.col("symbol").is_in(symbols))
    classifications = classify_message_rows(normalized, settings=settings)
    rows = []
    for symbol in symbols:
        symbol_messages = normalized.filter(pl.col("symbol") == symbol)
        rows.append(
            source_manifest_row(
                symbol=symbol,
                source="messages",
                phase="collect-context",
                status="ok" if not symbol_messages.is_empty() else "missing",
                backend="local_csv",
                endpoint=str(path),
                rows=symbol_messages.height,
                warning="" if not symbol_messages.is_empty() else "local_messages_missing",
            )
        )
    return manifest_frame(rows), normalized, classifications


async def _collect_polymarket_context(
    config: AccumulationConfig,
    symbols: tuple[str, ...],
    *,
    concurrency: int,
) -> tuple[pl.DataFrame, list[pl.DataFrame]]:
    semaphore = asyncio.Semaphore(max(1, concurrency))
    manifests = []
    frames = []
    polymarket = config.sources.polymarket
    async with httpx.AsyncClient(base_url=POLYMARKET_GAMMA_BASE_URL, timeout=20.0) as client:

        async def run_symbol(symbol: str) -> tuple[pl.DataFrame, pl.DataFrame]:
            async with semaphore:
                queries = _polymarket_queries(config, symbol)
                if not queries:
                    missing = _context_manifest_rows(
                        (symbol,),
                        source="polymarket_markets",
                        status="missing",
                        warning="polymarket_alias_missing",
                    )
                    return missing, pl.DataFrame()
                results = []
                for query in queries:
                    results.append(
                        await fetch_polymarket_search_async(
                            client,
                            query,
                            symbol=symbol,
                            limit_per_type=polymarket.search_limit_per_symbol,
                        )
                    )
                manifest = pl.concat([result.manifest for result in results], how="vertical")
                symbol_frames = [result.frame for result in results if not result.frame.is_empty()]
                frame = (
                    pl.concat(symbol_frames, how="vertical") if symbol_frames else pl.DataFrame()
                )
                if not frame.is_empty() and polymarket.min_volume_usd > 0.0:
                    frame = frame.filter(
                        pl.col("volume_24h").fill_null(0.0) >= polymarket.min_volume_usd
                    )
                if not frame.is_empty():
                    frame = frame.head(polymarket.max_markets_per_symbol)
                return manifest, frame

        for manifest, frame in await asyncio.gather(*(run_symbol(symbol) for symbol in symbols)):
            manifests.append(manifest)
            if not frame.is_empty():
                frames.append(frame)
    return pl.concat(manifests, how="vertical") if manifests else manifest_frame([]), frames


def run_score(
    config: AccumulationConfig, symbols: tuple[str, ...], discovery: pl.DataFrame
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    bundle = read_source_bundle(config.output_dir)
    coverage = bundle.manifest
    prices = bundle.bars
    coverage_scores = {
        symbol: compute_source_coverage_score(coverage, symbol) for symbol in symbols
    }
    features = join_hourly_accumulation_features_batch(
        price_frame=prices,
        symbols=symbols,
        discovery=discovery,
        books_by_symbol=_split_by_symbol(bundle.books, symbols),
        trades_by_symbol=_split_by_symbol(bundle.trades, symbols),
        funding_by_symbol=_split_by_symbol(bundle.funding, symbols),
        open_interest_by_symbol=_split_by_symbol(bundle.open_interest, symbols),
        messages_by_symbol=_split_by_symbol(bundle.messages, symbols),
        classifications_by_symbol=_split_by_symbol(bundle.message_classifications, symbols),
        coverage_scores=coverage_scores,
        ma_hours=config.features.ma_hours,
        max_source_staleness_hours=config.features.max_source_staleness_hours,
    )
    scores = score_accumulation_features(features, config.scoring)
    alerts = scores.filter(pl.col("alert_level") != "none") if not scores.is_empty() else scores
    _maybe_store(config, "accumulation_features", features)
    _maybe_store(config, "accumulation_scores", scores)
    _maybe_store(config, "accumulation_alerts", alerts)
    write_csv_artifacts(
        config.output_dir,
        features=features,
        scores=scores,
        alerts=alerts,
        data_coverage=coverage,
        source_manifest=coverage,
    )
    return features, scores, coverage


def run_summarize(config: AccumulationConfig, *, top_n: int) -> tuple[pl.DataFrame, pl.DataFrame]:
    scores = read_artifact(config.output_dir, "scores")
    features = read_artifact(config.output_dir, "features")
    coverage = read_artifact(config.output_dir, "source_manifest")
    summary = build_candidate_summary(scores, coverage, top_n=top_n)
    detail = build_candidate_detail(
        scores,
        features,
        coverage,
        settings=CandidateReadoutSettings(top_n=top_n),
    )
    next_fetch = build_next_fetch_actions(scores, coverage)
    write_csv_artifacts(
        config.output_dir,
        candidate_detail=detail,
        candidate_summary=summary,
        next_fetch_actions=next_fetch,
    )
    write_text_artifact(
        config.output_dir, "candidate-rationale.md", render_candidate_rationale(summary)
    )
    _maybe_store(config, "candidate_summary", summary)
    return summary, next_fetch


def main() -> None:
    args = parse_args()
    config = load_accumulation_config(Path(args.config))
    concurrency = args.fetch_concurrency or config.sources.fetch_concurrency
    book_mode = args.book_mode or config.market.book_mode
    if args.phase == "summarize":
        summary, next_fetch = run_summarize(config, top_n=args.top_n or config.summary.top_n)
        print(
            f"wrote summary={summary.height} next_fetch={next_fetch.height} out={config.output_dir}"
        )
        return
    symbols, discovery, _manifest = resolve_symbols_or_discover(config, args)
    if args.phase == "discover":
        print(f"wrote discovery={discovery.height} selected={len(symbols)} out={config.output_dir}")
        return
    if args.phase in {"collect-market", "all"}:
        manifest = run_collect_market(
            config,
            symbols,
            discovery,
            concurrency=concurrency,
            book_mode=book_mode,
            refresh_bars=args.refresh_bars,
            refresh_trades=args.refresh_trades,
            refresh_context=args.refresh_context,
        )
        print(f"wrote source_manifest={manifest.height} out={config.output_dir}")
    if args.phase in {"collect-onchain", "all"} and not config.onchain.enabled:
        skipped = manifest_frame(
            [
                source_manifest_row(
                    symbol=symbol,
                    source="onchain",
                    phase="collect-onchain",
                    status="skipped",
                    warning="onchain_disabled",
                )
                for symbol in symbols
            ]
        )
        existing = read_artifact(config.output_dir, "source_manifest")
        combined = _concat_frames(existing, skipped)
        write_csv_artifacts(config.output_dir, source_manifest=combined, data_coverage=combined)
        print("collect-onchain skipped: onchain.enabled=false")
    if args.phase in {"collect-context", "all"}:
        manifest = run_collect_context(config, symbols, concurrency=concurrency)
        print(f"wrote context_manifest={manifest.height} out={config.output_dir}")
    if args.phase in {"score", "all"}:
        features, scores, _coverage = run_score(config, symbols, discovery)
        alert_count = (
            scores.filter(pl.col("alert_level") != "none").height if not scores.is_empty() else 0
        )
        print(
            f"wrote features={features.height} scores={scores.height} "
            f"alerts={alert_count} out={config.output_dir}"
        )
    if args.phase in {"summarize", "all"}:
        summary, next_fetch = run_summarize(config, top_n=args.top_n or config.summary.top_n)
        print(
            f"wrote summary={summary.height} next_fetch={next_fetch.height} out={config.output_dir}"
        )


def _concat_frames(*frames: pl.DataFrame) -> pl.DataFrame:
    frames = [frame for frame in frames if not frame.is_empty()]
    return pl.concat(frames, how="vertical") if frames else manifest_frame([])


def _context_manifest_rows(
    symbols: tuple[str, ...], *, source: str, status: str, warning: str
) -> pl.DataFrame:
    return manifest_frame(
        [
            source_manifest_row(
                symbol=symbol,
                source=source,
                phase="collect-context",
                status=status,
                warning=warning,
            )
            for symbol in symbols
        ]
    )


def _polymarket_queries(config: AccumulationConfig, symbol: str) -> tuple[str, ...]:
    for alias in config.sources.polymarket.aliases:
        if alias.symbol == symbol:
            return alias.queries
    return ()


def _local_file_manifest(symbol: str, source: str, path: Path) -> pl.DataFrame:
    frame = _load_local_frame(path, symbol=symbol)
    return manifest_frame(
        [
            source_manifest_row(
                symbol=symbol,
                source=source,
                phase="collect-market",
                status="ok" if path.exists() and not frame.is_empty() else "missing",
                backend="local",
                endpoint=str(path),
                rows=frame.height if not frame.is_empty() else 0,
                warning="" if path.exists() and not frame.is_empty() else f"{source}_missing",
            )
        ]
    )


def _load_local_frame(path: Path, *, symbol: str | None = None) -> pl.DataFrame:
    if not path.exists():
        return pl.DataFrame()
    try:
        if path.suffix != ".csv":
            return pl.DataFrame()
        frame = pl.read_csv(path)
        if symbol is not None and "symbol" in frame.columns:
            frame = frame.filter(pl.col("symbol") == symbol)
        return frame
    except Exception:
        return pl.DataFrame()
    return pl.DataFrame()


def _with_symbol(frame: pl.DataFrame, symbol: str) -> pl.DataFrame:
    if frame.is_empty():
        return frame
    return frame.with_columns(pl.lit(symbol).alias("symbol"))


def _split_by_symbol(frame: pl.DataFrame, symbols: tuple[str, ...]) -> dict[str, pl.DataFrame]:
    if frame.is_empty() or "symbol" not in frame.columns:
        return {}
    return {symbol: frame.filter(pl.col("symbol") == symbol) for symbol in symbols}


def _maybe_store(config: AccumulationConfig, table: str, frame: pl.DataFrame) -> None:
    maybe_store_frame(
        config.output_dir,
        enabled=config.database.enabled,
        relative_path=config.database.path,
        table=table,
        frame=frame,
    )


def _discovery_row(discovery: pl.DataFrame, symbol: str) -> dict[str, object] | None:
    if discovery.is_empty() or "symbol" not in discovery.columns:
        return None
    rows = discovery.filter(pl.col("symbol") == symbol).head(1)
    return rows.to_dicts()[0] if not rows.is_empty() else None


def _float_or_none(row: dict[str, object] | None, key: str) -> float | None:
    if row is None or row.get(key) is None:
        return None
    return float(row[key])


def _str_or_none(row: dict[str, object] | None, key: str) -> str | None:
    if row is None or row.get(key) is None:
        return None
    return str(row[key])


if __name__ == "__main__":
    main()
