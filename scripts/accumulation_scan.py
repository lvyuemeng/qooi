from __future__ import annotations

import argparse
import asyncio
import fnmatch
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
from qooi.accumulation.features import join_hourly_accumulation_features_batch
from qooi.accumulation.scoring import score_accumulation_features
from qooi.accumulation.summary import (
    CandidateReadoutSettings,
    NextFetchPolicy,
    build_candidate_detail,
    build_candidate_summary,
    build_next_fetch_actions,
    render_candidate_rationale,
    render_scan_feedback,
)
from qooi.exchange.context import collect_okx_context_batch
from qooi.exchange.discovery import discover_candidates, select_candidate_symbols
from qooi.exchange.store import (
    AsyncCacheStore,
    BooksRequest,
    CacheStore,
    HistoryRefreshRequest,
    _normalize_bars,
    _read_frame,
)
from qooi.exchange.universe import (
    collect_broad_sources,
    collect_potential_board_universe,
    map_broad_to_okx,
    select_deep_symbols,
    select_potential_board_symbols,
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
from qooi.sources.polymarket import (
    POLYMARKET_GAMMA_BASE_URL,
    fetch_polymarket_search_async,
)
from qooi.strategies.potential import (
    build_potential_board,
    compact_potential_sources,
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
            "discover-broad",
            "all-broad",
            "potential-broad",
            "potential-board",
            "all",
        ),
        default="score",
    )
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--symbols", default="")
    parser.add_argument("--scan-top-n", type=int, default=None)
    parser.add_argument("--availability-top-n", type=int, default=None)
    parser.add_argument("--broad-top-n", type=int, default=None)
    parser.add_argument("--deep-top-n", type=int, default=None)
    parser.add_argument("--summary-top-n", type=int, default=None)
    parser.add_argument("--min-volume-usd", type=float, default=None)
    parser.add_argument("--min-coverage-pct", type=float, default=None)
    parser.add_argument("--fetch-concurrency", type=int, default=None)
    parser.add_argument("--book-mode", choices=("snapshot", "sample", "off"), default=None)
    parser.add_argument("--refresh-discovery", action="store_true")
    parser.add_argument("--refresh-broad", action="store_true")
    parser.add_argument("--refresh-bars", action="store_true")
    parser.add_argument("--refresh-trades", action="store_true")
    parser.add_argument("--refresh-context", action="store_true")
    parser.add_argument(
        "--summary-latest-only", action=argparse.BooleanOptionalAction, default=True
    )
    return parser.parse_args()


def _parse_symbols(raw: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in raw.split(",") if part.strip())


def _scan_top_n(config: AccumulationConfig, args: argparse.Namespace) -> int:
    value = args.scan_top_n if args.scan_top_n is not None else config.discovery.top_n
    if value <= 0:
        raise SystemExit("--scan-top-n must be greater than 0")
    return value


def _summary_top_n(config: AccumulationConfig, args: argparse.Namespace) -> int:
    value = args.summary_top_n if args.summary_top_n is not None else config.summary.top_n
    if value <= 0:
        raise SystemExit("--summary-top-n must be greater than 0")
    return value


def _availability_top_n(args: argparse.Namespace) -> int | None:
    value = getattr(args, "availability_top_n", None)
    if value is not None and value <= 0:
        raise SystemExit("--availability-top-n must be greater than 0")
    return value


def _broad_top_n(config: AccumulationConfig, args: argparse.Namespace) -> int:
    arg_value = getattr(args, "broad_top_n", None)
    value = arg_value if arg_value is not None else config.broad_scan.output_top_n
    if value <= 0:
        raise SystemExit("--broad-top-n must be greater than 0")
    return value


def _deep_top_n(config: AccumulationConfig, args: argparse.Namespace) -> int:
    arg_value = getattr(args, "deep_top_n", None)
    value = arg_value if arg_value is not None else config.discovery.top_n
    if value <= 0:
        raise SystemExit("--deep-top-n must be greater than 0")
    return value


def _context_probe_symbols(
    scan_symbols: tuple[str, ...],
    discovery: pl.DataFrame,
    availability_top_n: int | None,
) -> tuple[str, ...]:
    if availability_top_n is None:
        return scan_symbols
    if discovery.is_empty() or "symbol" not in discovery.columns:
        return scan_symbols
    return select_candidate_symbols(discovery, top_n=availability_top_n)


def _next_fetch_policy(config: AccumulationConfig) -> NextFetchPolicy:
    sources = ["discovery", "trades", "funding", "open_interest"]
    if not _family_disabled(config, "messages") and config.sources.messages.path.strip():
        sources.append("messages")
    return NextFetchPolicy(
        yellow_threshold=config.scoring.yellow_threshold,
        actionable_sources=tuple(sources),
    )


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


def run_discover_broad(
    config: AccumulationConfig,
    *,
    broad_top_n: int,
    deep_top_n: int,
    refresh_broad: bool = False,
) -> tuple[tuple[str, ...], pl.DataFrame, pl.DataFrame]:
    candidates_path = artifact_path(config.output_dir, "broad_candidates")
    if candidates_path.exists() and not refresh_broad:
        okx_discovery = read_artifact(config.output_dir, "candidate_discovery")
        candidates = read_artifact(config.output_dir, "broad_candidates")
        return (
            select_deep_symbols(candidates, deep_top_n=deep_top_n),
            okx_discovery,
            read_artifact(config.output_dir, "source_manifest"),
        )
    broad = collect_broad_sources(config, broad_top_n=broad_top_n)
    okx_result = discover_candidates(
        config,
        top_n=max(deep_top_n, config.discovery.top_n),
        min_volume_usd=None,
    )
    candidates = map_broad_to_okx(broad.candidates, okx_result.discovery)
    manifest = _concat_frames(broad.manifest, okx_result.manifest)
    write_csv_artifacts(
        config.output_dir,
        discovery=okx_result.discovery,
        source_manifest=manifest,
        data_coverage=manifest,
        broad_market_snapshot=broad.market,
        broad_protocol_snapshot=broad.protocols,
        broad_news_snapshot=broad.news,
        broad_candidates=candidates,
    )
    _maybe_store(config, "candidate_discovery", okx_result.discovery)
    _maybe_store(config, "source_manifest", manifest)
    _maybe_store(config, "broad_candidates", candidates)
    return select_deep_symbols(candidates, deep_top_n=deep_top_n), okx_result.discovery, manifest


def run_discover_potential_board(
    config: AccumulationConfig,
    *,
    broad_top_n: int,
    deep_top_n: int,
    refresh_broad: bool = False,
) -> tuple[tuple[str, ...], pl.DataFrame, pl.DataFrame]:
    candidates_path = artifact_path(config.output_dir, "broad_candidates")
    if candidates_path.exists() and not refresh_broad:
        okx_discovery = read_artifact(config.output_dir, "candidate_discovery")
        candidates = read_artifact(config.output_dir, "broad_candidates")
        return (
            select_potential_board_symbols(candidates, deep_top_n=deep_top_n),
            okx_discovery,
            read_artifact(config.output_dir, "source_manifest"),
        )
    broad = collect_potential_board_universe(config, top_n=broad_top_n)
    okx_result = discover_candidates(
        config,
        top_n=max(deep_top_n, config.discovery.top_n),
        min_volume_usd=None,
    )
    candidates = map_broad_to_okx(broad.candidates, okx_result.discovery)
    manifest = _concat_frames(broad.manifest, okx_result.manifest)
    write_csv_artifacts(
        config.output_dir,
        discovery=okx_result.discovery,
        source_manifest=manifest,
        data_coverage=manifest,
        broad_candidates=candidates,
    )
    _maybe_store(config, "candidate_discovery", okx_result.discovery)
    _maybe_store(config, "source_manifest", manifest)
    _maybe_store(config, "broad_candidates", candidates)
    return (
        select_potential_board_symbols(candidates, deep_top_n=deep_top_n),
        okx_result.discovery,
        manifest,
    )


def resolve_symbols_or_discover(
    config: AccumulationConfig,
    args: argparse.Namespace,
    *,
    scan_top_n: int,
) -> tuple[tuple[str, ...], pl.DataFrame, pl.DataFrame]:
    manual_symbols = _parse_symbols(args.symbols)
    discovery_path = config.output_dir / "candidate-discovery.csv"
    if args.phase == "discover":
        return run_discover(
            config,
            top_n=scan_top_n,
            min_volume_usd=args.min_volume_usd,
            manual_symbols=manual_symbols,
        )
    if args.phase == "all" and not manual_symbols:
        return run_discover(
            config,
            top_n=scan_top_n,
            min_volume_usd=args.min_volume_usd,
        )
    if manual_symbols:
        discovery = read_artifact(config.output_dir, "candidate_discovery")
        return manual_symbols, discovery, read_artifact(config.output_dir, "source_manifest")
    if args.refresh_discovery and args.phase in {"collect-market", "all"}:
        return run_discover(config, top_n=scan_top_n, min_volume_usd=args.min_volume_usd)
    if not discovery_path.exists():
        raise SystemExit(
            "candidate-discovery.csv is missing; run --phase discover first or pass --symbols"
        )
    discovery = read_artifact(config.output_dir, "candidate_discovery")
    symbols = select_candidate_symbols(discovery, top_n=scan_top_n)
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
    source_availability = _build_source_availability_index(config.output_dir)
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
            collect_okx_context_batch(
                config,
                symbols,
                discovery,
                concurrency=concurrency,
                book_mode="off",
                refresh_trades=refresh_trades,
                refresh_context=refresh_context,
                collect_books=False,
                source_availability=source_availability,
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
            collect_okx_context_batch(
                config,
                symbols,
                discovery,
                concurrency=concurrency,
                book_mode=book_mode,
                refresh_trades=refresh_trades,
                refresh_context=refresh_context,
                source_availability=source_availability,
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
        taker_volume=public_frames.get("taker_volume"),
        long_short_ratios=public_frames.get("long_short_ratios"),
    )
    _maybe_store(config, "source_manifest", manifest)
    return manifest


def run_collect_context(
    config: AccumulationConfig,
    symbols: tuple[str, ...],
    discovery: pl.DataFrame | None = None,
    *,
    concurrency: int,
) -> pl.DataFrame:
    manifests = []
    market_frames = []
    messages = pl.DataFrame()
    message_classifications = pl.DataFrame()
    if _family_disabled(config, "polymarket"):
        manifests.append(
            _context_manifest_rows(
                symbols,
                source="polymarket_markets",
                status="skipped",
                warning="polymarket_disabled",
            )
        )
    else:
        context_manifest, market_frames = asyncio.run(
            _collect_polymarket_context(
                config,
                symbols,
                discovery if discovery is not None else pl.DataFrame(),
                concurrency=concurrency,
            )
        )
        manifests.append(context_manifest)
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
        polymarket_markets=pl.concat(market_frames, how="vertical_relaxed")
        if market_frames
        else pl.DataFrame(),
    )
    _maybe_store(config, "source_manifest", manifest)
    return manifest


def _collect_local_message_context(
    config: AccumulationConfig, symbols: tuple[str, ...]
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    messages_config = config.sources.messages
    if _family_disabled(config, "messages"):
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
    disabled_symbols = tuple(
        symbol for symbol in symbols if _source_disabled(config, "messages", symbol)
    )
    symbols = tuple(symbol for symbol in symbols if symbol not in set(disabled_symbols))
    disabled_manifest = (
        _context_manifest_rows(
            disabled_symbols,
            source="messages",
            status="skipped",
            warning="messages_disabled",
        )
        if disabled_symbols
        else pl.DataFrame()
    )
    if not messages_config.path.strip():
        return (
            _concat_frames(
                disabled_manifest,
                _context_manifest_rows(
                    symbols,
                    source="messages",
                    status="missing",
                    warning="local_messages_path_missing",
                ),
            ),
            pl.DataFrame(),
            pl.DataFrame(),
        )
    path = Path(messages_config.path)
    if not path.exists():
        return (
            _concat_frames(
                disabled_manifest,
                _context_manifest_rows(
                    symbols,
                    source="messages",
                    status="missing",
                    warning="local_messages_file_missing",
                ),
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
    return _concat_frames(disabled_manifest, manifest_frame(rows)), normalized, classifications


async def _collect_polymarket_context(
    config: AccumulationConfig,
    symbols: tuple[str, ...],
    discovery: pl.DataFrame,
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
                if _source_disabled(config, "polymarket", symbol):
                    return (
                        _context_manifest_rows(
                            (symbol,),
                            source="polymarket_markets",
                            status="skipped",
                            warning="polymarket_disabled",
                        ),
                        pl.DataFrame(),
                    )
                queries, disabled_query_count = _polymarket_queries(
                    config, symbol, _discovery_row(discovery, symbol)
                )
                if not queries:
                    warning = (
                        "polymarket_query_disabled"
                        if disabled_query_count
                        else "polymarket_query_missing"
                    )
                    missing = _context_manifest_rows(
                        (symbol,),
                        source="polymarket_markets",
                        status="skipped" if disabled_query_count else "missing",
                        warning=warning,
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
                    pl.concat(symbol_frames, how="vertical_relaxed")
                    if symbol_frames
                    else pl.DataFrame()
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
        books_by_symbol=_split_available_by_symbol(bundle.books, symbols, coverage, "books"),
        trades_by_symbol=_split_available_by_symbol(bundle.trades, symbols, coverage, "trades"),
        funding_by_symbol=_split_available_by_symbol(bundle.funding, symbols, coverage, "funding"),
        open_interest_by_symbol=_split_available_by_symbol(
            bundle.open_interest, symbols, coverage, "open_interest_history"
        ),
        taker_volume_by_symbol=_split_available_by_symbol(
            bundle.taker_volume, symbols, coverage, "taker_volume_contract"
        ),
        long_short_ratios_by_symbol=_split_available_by_symbol(
            bundle.long_short_ratios, symbols, coverage, "long_short_ratio_contract"
        ),
        messages_by_symbol=_split_available_by_symbol(
            bundle.messages, symbols, coverage, "messages"
        ),
        classifications_by_symbol=_split_available_by_symbol(
            bundle.message_classifications, symbols, coverage, "messages"
        ),
        coverage_scores=coverage_scores,
        flow_zscore_window_hours=config.features.flow_zscore_window_hours,
        large_trade_usd=config.features.large_trade_usd,
        resilience_minutes=config.features.resilience_minutes,
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
    broad_candidates = read_artifact(config.output_dir, "broad_candidates")
    policy = _next_fetch_policy(config)
    summary = build_candidate_summary(scores, coverage, top_n=top_n, policy=policy)
    detail = build_candidate_detail(
        scores,
        features,
        coverage,
        settings=CandidateReadoutSettings(top_n=top_n),
        policy=policy,
    )
    next_fetch = build_next_fetch_actions(scores, coverage, policy=policy)
    write_csv_artifacts(
        config.output_dir,
        candidate_detail=detail,
        candidate_summary=summary,
        next_fetch_actions=next_fetch,
    )
    write_text_artifact(
        config.output_dir, "candidate-rationale.md", render_candidate_rationale(summary)
    )
    write_text_artifact(
        config.output_dir,
        "scan-feedback.md",
        render_scan_feedback(summary, detail, next_fetch, coverage, broad_candidates),
    )
    _maybe_store(config, "candidate_summary", summary)
    return summary, next_fetch


def run_potential_board(config: AccumulationConfig) -> tuple[pl.DataFrame, pl.DataFrame]:
    bundle = read_source_bundle(config.output_dir)
    broad_candidates = read_artifact(config.output_dir, "broad_candidates")
    symbols = (
        tuple(bundle.bars["symbol"].drop_nulls().unique().to_list())
        if not bundle.bars.is_empty() and "symbol" in bundle.bars.columns
        else ()
    )
    coverage_scores = {
        symbol: compute_source_coverage_score(bundle.manifest, symbol) for symbol in symbols
    }
    context = join_hourly_accumulation_features_batch(
        price_frame=bundle.bars,
        symbols=symbols,
        discovery=bundle.discovery,
        books_by_symbol=_split_available_by_symbol(bundle.books, symbols, bundle.manifest, "books"),
        trades_by_symbol=_split_available_by_symbol(
            bundle.trades, symbols, bundle.manifest, "trades"
        ),
        funding_by_symbol=_split_available_by_symbol(
            bundle.funding, symbols, bundle.manifest, "funding"
        ),
        open_interest_by_symbol=_split_available_by_symbol(
            bundle.open_interest, symbols, bundle.manifest, "open_interest_history"
        ),
        taker_volume_by_symbol=_split_available_by_symbol(
            bundle.taker_volume, symbols, bundle.manifest, "taker_volume_contract"
        ),
        long_short_ratios_by_symbol=_split_available_by_symbol(
            bundle.long_short_ratios, symbols, bundle.manifest, "long_short_ratio_contract"
        ),
        messages_by_symbol=_split_available_by_symbol(
            bundle.messages, symbols, bundle.manifest, "messages"
        ),
        classifications_by_symbol=_split_available_by_symbol(
            bundle.message_classifications, symbols, bundle.manifest, "messages"
        ),
        coverage_scores=coverage_scores,
        flow_zscore_window_hours=config.features.flow_zscore_window_hours,
        large_trade_usd=config.features.large_trade_usd,
        resilience_minutes=config.features.resilience_minutes,
        ma_hours=config.features.ma_hours,
        max_source_staleness_hours=config.features.max_source_staleness_hours,
    )
    board, report = build_potential_board(
        bundle.bars,
        broad_candidates,
        bundle.discovery,
        context,
        config.potential_scan,
    )
    sources = compact_potential_sources(bundle.manifest)
    write_csv_artifacts(config.output_dir, potential_board=board, potential_sources=sources)
    write_text_artifact(config.output_dir, "potential/report.md", report)
    _maybe_store(config, "potential_board", board)
    return board, sources


def main() -> None:
    args = parse_args()
    config = load_accumulation_config(Path(args.config))
    scan_top_n = _scan_top_n(config, args)
    availability_top_n = _availability_top_n(args)
    broad_top_n = _broad_top_n(config, args)
    deep_top_n = _deep_top_n(config, args)
    summary_top_n = _summary_top_n(config, args)
    concurrency = args.fetch_concurrency or config.sources.fetch_concurrency
    book_mode = args.book_mode or config.market.book_mode
    if args.phase == "summarize":
        summary, next_fetch = run_summarize(config, top_n=summary_top_n)
        print(
            f"wrote summary={summary.height} next_fetch={next_fetch.height} out={config.output_dir}"
        )
        return
    if args.phase in {"potential-board", "potential-broad"}:
        symbols, discovery, _manifest = run_discover_potential_board(
            config,
            broad_top_n=broad_top_n,
            deep_top_n=deep_top_n,
            refresh_broad=args.refresh_broad,
        )
        broad_candidates = read_artifact(config.output_dir, "broad_candidates")
        print(
            f"wrote potential_universe={broad_candidates.height} selected={len(symbols)} "
            f"out={config.output_dir}"
        )
    elif args.phase in {"discover-broad", "all-broad"}:
        symbols, discovery, _manifest = run_discover_broad(
            config,
            broad_top_n=broad_top_n,
            deep_top_n=deep_top_n,
            refresh_broad=args.refresh_broad,
        )
        broad_candidates = read_artifact(config.output_dir, "broad_candidates")
        print(
            f"wrote broad_candidates={broad_candidates.height} selected={len(symbols)} "
            f"out={config.output_dir}"
        )
        if args.phase == "discover-broad":
            return
        if args.phase == "all-broad":
            args.phase = "all"
    else:
        symbols, discovery, _manifest = resolve_symbols_or_discover(
            config, args, scan_top_n=scan_top_n
        )
    if args.phase == "discover":
        print(f"wrote discovery={discovery.height} selected={len(symbols)} out={config.output_dir}")
        return
    if args.phase in {"collect-market", "all", "potential-broad", "potential-board"}:
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
    if args.phase in {"collect-onchain", "all"}:
        manifest = run_collect_onchain(config, symbols)
        print(f"wrote onchain_manifest={manifest.height} out={config.output_dir}")
    if args.phase in {"collect-context", "all"}:
        context_symbols = _context_probe_symbols(symbols, discovery, availability_top_n)
        manifest = run_collect_context(config, context_symbols, discovery, concurrency=concurrency)
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
        summary, next_fetch = run_summarize(config, top_n=summary_top_n)
        print(
            f"wrote summary={summary.height} next_fetch={next_fetch.height} "
            f"selected_symbols={len(symbols)} summary_rows={summary_top_n} out={config.output_dir}"
        )
    if args.phase in {"potential-board", "potential-broad"}:
        board, sources = run_potential_board(config)
        print(
            f"wrote potential_board={board.height} potential_sources={sources.height} "
            f"out={config.output_dir}"
        )


def _concat_frames(*frames: pl.DataFrame) -> pl.DataFrame:
    frames = [frame for frame in frames if not frame.is_empty()]
    return pl.concat(frames, how="vertical") if frames else manifest_frame([])


def run_collect_onchain(config: AccumulationConfig, symbols: tuple[str, ...]) -> pl.DataFrame:
    token_symbols = {token.symbol for token in config.onchain.tokens}
    rows = []
    for symbol in symbols:
        if _source_disabled(config, "onchain", symbol):
            rows.append(
                source_manifest_row(
                    symbol=symbol,
                    source="onchain",
                    phase="collect-onchain",
                    status="skipped",
                    warning="onchain_disabled",
                )
            )
            continue
        warning = (
            "onchain_provider_not_implemented"
            if symbol in token_symbols
            else "onchain_token_mapping_missing"
        )
        rows.append(
            source_manifest_row(
                symbol=symbol,
                source="onchain",
                phase="collect-onchain",
                status="missing",
                backend=config.onchain.provider,
                warning=warning,
            )
        )
    manifest = _concat_frames(
        read_artifact(config.output_dir, "source_manifest"), manifest_frame(rows)
    )
    write_csv_artifacts(config.output_dir, source_manifest=manifest, data_coverage=manifest)
    _maybe_store(config, "source_manifest", manifest)
    return manifest


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


def _market_manifest_rows(
    symbols: tuple[str, ...], *, source: str, status: str, warning: str
) -> pl.DataFrame:
    return manifest_frame(
        [
            source_manifest_row(
                symbol=symbol,
                source=source,
                phase="collect-market",
                status=status,
                warning=warning,
            )
            for symbol in symbols
        ]
    )


def _min_timestamp(frame: pl.DataFrame) -> int | None:
    return (
        int(frame["timestamp"].min())
        if "timestamp" in frame.columns and not frame.is_empty()
        else None
    )


def _max_timestamp(frame: pl.DataFrame) -> int | None:
    return (
        int(frame["timestamp"].max())
        if "timestamp" in frame.columns and not frame.is_empty()
        else None
    )


def _polymarket_queries(
    config: AccumulationConfig,
    symbol: str,
    discovery_row: dict[str, object] | None,
) -> tuple[tuple[str, ...], int]:
    queries = []
    base_ccy = _str_or_none(discovery_row, "base_ccy")
    if base_ccy and base_ccy.strip():
        queries.append(base_ccy.strip())
    fallback = _base_query_from_symbol(symbol)
    if fallback:
        queries.append(fallback)
    for alias in config.sources.polymarket.aliases:
        if alias.symbol == symbol:
            queries.extend(alias.queries)
    deduped = tuple(dict.fromkeys(query.strip() for query in queries if query.strip()))
    enabled_queries = tuple(
        query for query in deduped if not _query_disabled(config, "polymarket", query)
    )
    return enabled_queries, len(deduped) - len(enabled_queries)


def _base_query_from_symbol(symbol: str) -> str:
    text = symbol.strip()
    for suffix in ("-USDT-SWAP", "-USD-SWAP", "-USDC-SWAP"):
        if text.endswith(suffix):
            return text.removesuffix(suffix)
    return text.split("-")[0] if text else ""


def _family_disabled(config: AccumulationConfig, family: str) -> bool:
    return _matches_any(family, config.sources.disabled.families)


def _source_disabled(config: AccumulationConfig, family: str, symbol: str = "") -> bool:
    return _family_disabled(config, family) or bool(
        symbol and _matches_any(symbol, config.sources.disabled.symbols)
    )


def _query_disabled(config: AccumulationConfig, family: str, query: str) -> bool:
    if family != "polymarket":
        return False
    return _matches_any(query, config.sources.disabled.polymarket_queries)


def _matches_any(value: str, patterns: tuple[str, ...]) -> bool:
    normalized = value.strip().lower()
    for pattern in patterns:
        candidate = pattern.strip().lower()
        if candidate and fnmatch.fnmatchcase(normalized, candidate):
            return True
    return False


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


def _local_source_has_symbol_rows(path: Path, symbol: str) -> bool:
    return not _load_local_frame(path, symbol=symbol).is_empty()


def _build_source_availability_index(output_dir: Path) -> dict[str, set[str]]:
    artifact_names = (
        "source_trades",
        "source_funding",
        "source_open_interest",
        "source_taker_volume",
        "source_long_short_ratios",
    )
    availability: dict[str, set[str]] = {}
    for artifact_name in artifact_names:
        path = artifact_path(output_dir, artifact_name)
        frame = _load_local_frame(path)
        if frame.is_empty() or "symbol" not in frame.columns:
            availability[artifact_name] = set()
            continue
        availability[artifact_name] = {
            str(symbol) for symbol in frame["symbol"].drop_nulls().unique().to_list()
        }
    return availability


def _cached_source_has_symbol(
    availability: dict[str, set[str]] | None, artifact_name: str, symbol: str, path: Path
) -> bool:
    if availability is None:
        return _local_source_has_symbol_rows(path, symbol)
    return symbol in availability.get(artifact_name, set())


def _with_symbol(frame: pl.DataFrame, symbol: str) -> pl.DataFrame:
    if frame.is_empty():
        return frame
    return frame.with_columns(pl.lit(symbol).alias("symbol"))


def _split_by_symbol(frame: pl.DataFrame, symbols: tuple[str, ...]) -> dict[str, pl.DataFrame]:
    if frame.is_empty() or "symbol" not in frame.columns:
        return {}
    return {symbol: frame.filter(pl.col("symbol") == symbol) for symbol in symbols}


def _split_available_by_symbol(
    frame: pl.DataFrame, symbols: tuple[str, ...], manifest: pl.DataFrame, source: str
) -> dict[str, pl.DataFrame]:
    split = _split_by_symbol(frame, symbols)
    available: dict[str, pl.DataFrame] = {}
    for symbol in symbols:
        status = "missing"
        if not manifest.is_empty() and {"symbol", "source", "status"}.issubset(manifest.columns):
            rows = manifest.filter((pl.col("symbol") == symbol) & (pl.col("source") == source))
            if not rows.is_empty():
                if "timestamp" in rows.columns:
                    rows = rows.sort("timestamp")
                status = str(rows.tail(1)["status"][0] or "missing")
        available[symbol] = (
            split.get(symbol, pl.DataFrame()) if status in {"ok", "partial"} else pl.DataFrame()
        )
    return available


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


