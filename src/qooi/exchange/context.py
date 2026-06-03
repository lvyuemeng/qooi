"""OKX public context collection for scanner orchestration."""

from __future__ import annotations

import asyncio
import fnmatch
from pathlib import Path

import httpx
import polars as pl

from qooi.accumulation.config import AccumulationConfig
from qooi.accumulation.csv_io import artifact_path
from qooi.sources.coverage import manifest_frame, source_manifest_row
from qooi.sources.okx import (
    OKX_BASE_URL,
    fetch_okx_book_snapshot_async,
    fetch_okx_funding_history_async,
    fetch_okx_long_short_account_ratio_contract_async,
    fetch_okx_open_interest_history_async,
    fetch_okx_recent_trades_async,
    fetch_okx_taker_volume_contract_async,
    fetch_okx_top_trader_long_short_account_ratio_contract_async,
    fetch_okx_top_trader_long_short_position_ratio_contract_async,
)


async def collect_okx_context_batch(
    config: AccumulationConfig,
    symbols: tuple[str, ...],
    discovery: pl.DataFrame,
    *,
    concurrency: int,
    book_mode: str,
    refresh_trades: bool,
    refresh_context: bool,
    collect_books: bool = True,
    source_availability: dict[str, set[str]] | None = None,
) -> tuple[pl.DataFrame, dict[str, pl.DataFrame]]:
    semaphore = asyncio.Semaphore(max(1, concurrency))
    manifests = []
    frames_by_source: dict[str, list[pl.DataFrame]] = {
        "books": [],
        "trades": [],
        "funding": [],
        "open_interest": [],
        "taker_volume": [],
        "long_short_ratios": [],
    }
    async with httpx.AsyncClient(base_url=OKX_BASE_URL, timeout=20.0) as client:

        async def run_symbol(
            symbol: str,
        ) -> tuple[list[pl.DataFrame], dict[str, pl.DataFrame]]:
            async with semaphore:
                return await collect_okx_symbol_context(
                    client,
                    config,
                    symbol,
                    _discovery_row(discovery, symbol),
                    book_mode=book_mode,
                    refresh_trades=refresh_trades,
                    refresh_context=refresh_context,
                    collect_books=collect_books,
                    source_availability=source_availability,
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


async def collect_okx_symbol_context(
    client: httpx.AsyncClient,
    config: AccumulationConfig,
    symbol: str,
    discovery_row: dict[str, object] | None,
    *,
    book_mode: str,
    refresh_trades: bool,
    refresh_context: bool,
    collect_books: bool = True,
    source_availability: dict[str, set[str]] | None = None,
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
    pending: list[tuple[int, str, object]] = []
    local_manifests: list[tuple[int, pl.DataFrame]] = []
    order = 0
    trades_path = artifact_path(config.output_dir, "source_trades")
    if refresh_trades or not _cached_source_has_symbol(
        source_availability, "source_trades", symbol, trades_path
    ):
        pending.append(
            (
                order,
                "trades",
                fetch_okx_recent_trades_async(
                    client,
                    symbol,
                    limit=config.sources.trade_limit,
                    contract_value=contract_value,
                    contract_value_currency=contract_ccy,
                    contract_base_currency=contract_base,
                ),
            )
        )
    else:
        local_manifests.append((order, _local_file_manifest(symbol, "trades", trades_path)))
    order += 1
    funding_path = artifact_path(config.output_dir, "source_funding")
    if refresh_context or not _cached_source_has_symbol(
        source_availability, "source_funding", symbol, funding_path
    ):
        pending.append(
            (
                order,
                "funding",
                fetch_okx_funding_history_async(client, symbol, limit=config.sources.funding_limit),
            )
        )
    else:
        local_manifests.append((order, _local_file_manifest(symbol, "funding", funding_path)))
    order += 1
    if _source_disabled(config, "rubik", symbol):
        manifests.append(
            _market_manifest_rows(
                (symbol,),
                source="open_interest_history",
                status="skipped",
                warning="rubik_disabled",
            )
        )
        manifests.append(
            _market_manifest_rows(
                (symbol,),
                source="taker_volume_contract",
                status="skipped",
                warning="rubik_disabled",
            )
        )
        manifests.append(
            _market_manifest_rows(
                (symbol,),
                source="long_short_ratio_contract",
                status="skipped",
                warning="rubik_disabled",
            )
        )
    else:
        oi_path = artifact_path(config.output_dir, "source_open_interest")
        if refresh_context or not _cached_source_has_symbol(
            source_availability, "source_open_interest", symbol, oi_path
        ):
            pending.append(
                (
                    order,
                    "open_interest",
                    fetch_okx_open_interest_history_async(
                        client,
                        symbol,
                        period=config.sources.rubik_period,
                        limit=config.sources.rubik_limit,
                    ),
                )
            )
        else:
            local_manifests.append(
                (order, _local_file_manifest(symbol, "open_interest_history", oi_path))
            )
        order += 1
        taker_path = artifact_path(config.output_dir, "source_taker_volume")
        if refresh_context or not _cached_source_has_symbol(
            source_availability, "source_taker_volume", symbol, taker_path
        ):
            pending.append(
                (
                    order,
                    "taker_volume",
                    fetch_okx_taker_volume_contract_async(
                        client,
                        symbol,
                        period=config.sources.rubik_period,
                        unit=config.sources.rubik_taker_unit,
                        limit=config.sources.rubik_limit,
                    ),
                )
            )
        else:
            local_manifests.append(
                (order, _local_file_manifest(symbol, "taker_volume_contract", taker_path))
            )
        order += 1
        ratios_path = artifact_path(config.output_dir, "source_long_short_ratios")
        if refresh_context or not _cached_source_has_symbol(
            source_availability, "source_long_short_ratios", symbol, ratios_path
        ):
            pending.append(
                (order, "long_short_ratios", _fetch_long_short_ratios(client, config, symbol))
            )
        else:
            local_manifests.append(
                (order, _local_file_manifest(symbol, "long_short_ratio_contract", ratios_path))
            )
    gathered = await asyncio.gather(*(call for *_prefix, call in pending)) if pending else []
    manifest_by_order = dict(local_manifests)
    for (source_order, frame_key, _call), result in zip(pending, gathered, strict=True):
        if frame_key == "long_short_ratios":
            ratio_frame, ratio_manifest = result
            if not ratio_frame.is_empty():
                frames[frame_key] = _with_symbol(ratio_frame, symbol)
            manifest_by_order[source_order] = ratio_manifest
            continue
        if not result.frame.is_empty():
            frames[frame_key] = _with_symbol(result.frame, symbol)
        manifest_by_order[source_order] = result.manifest
    manifests.extend(frame for _idx, frame in sorted(manifest_by_order.items()))
    return manifests, frames


async def _fetch_long_short_ratios(
    client: httpx.AsyncClient, config: AccumulationConfig, symbol: str
) -> tuple[pl.DataFrame, pl.DataFrame]:
    calls = [
        fetch_okx_long_short_account_ratio_contract_async(
            client,
            symbol,
            period=config.sources.rubik_period,
            limit=config.sources.rubik_limit,
        ),
        fetch_okx_top_trader_long_short_account_ratio_contract_async(
            client,
            symbol,
            period=config.sources.rubik_period,
            limit=config.sources.rubik_limit,
        ),
        fetch_okx_top_trader_long_short_position_ratio_contract_async(
            client,
            symbol,
            period=config.sources.rubik_period,
            limit=config.sources.rubik_limit,
        ),
    ]
    results = await asyncio.gather(*calls)
    frames = [result.frame for result in results if not result.frame.is_empty()]
    combined = _join_ratio_frames(frames)
    endpoints = ";".join(
        dict.fromkeys(
            str(row.get("endpoint") or "")
            for result in results
            for row in result.manifest.to_dicts()
            if str(row.get("endpoint") or "")
        )
    )
    warnings = ";".join(
        dict.fromkeys(
            str(row.get("warning") or "")
            for result in results
            for row in result.manifest.to_dicts()
            if str(row.get("warning") or "")
        )
    )
    timestamp = (
        max(
            int(row.get("timestamp") or 0)
            for result in results
            for row in result.manifest.to_dicts()
        )
        + 1
    )
    manifest = manifest_frame(
        [
            source_manifest_row(
                symbol=symbol,
                source="long_short_ratio_contract",
                phase="collect-market",
                status="ok" if not combined.is_empty() else "missing",
                backend="okx_rest",
                endpoint=endpoints,
                rows=combined.height,
                range_start=_min_timestamp(combined),
                range_end=_max_timestamp(combined),
                warning=warnings
                if warnings
                else ("" if not combined.is_empty() else "long_short_ratio_contract_missing"),
                timestamp=timestamp,
            )
        ]
    )
    return combined, manifest


def _join_ratio_frames(frames: list[pl.DataFrame]) -> pl.DataFrame:
    frames = [frame for frame in frames if not frame.is_empty()]
    if not frames:
        return pl.DataFrame()
    out = frames[0]
    for frame in frames[1:]:
        out = out.join(frame, on="timestamp", how="full", coalesce=True)
    return out.sort("timestamp")


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


def _family_disabled(config: AccumulationConfig, family: str) -> bool:
    return _matches_any(family, config.sources.disabled.families)


def _source_disabled(config: AccumulationConfig, family: str, symbol: str = "") -> bool:
    return _family_disabled(config, family) or bool(
        symbol and _matches_any(symbol, config.sources.disabled.symbols)
    )


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


def _local_source_has_symbol_rows(path: Path, symbol: str) -> bool:
    return not _load_local_frame(path, symbol=symbol).is_empty()


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
