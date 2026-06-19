"""Discovery algorithm — pure polars, no I/O."""

# ruff: noqa: E501, E701, E702
from __future__ import annotations

from dataclasses import dataclass

import polars as pl

DISCOVERY_SCHEMA: dict[str, pl.DataType] = {
    "symbol": pl.String,
    "inst_id": pl.String,
    "inst_type": pl.String,
    "state": pl.String,
    "base_ccy": pl.String,
    "quote_ccy": pl.String,
    "settle_ccy": pl.String,
    "ct_val": pl.Float64,
    "ct_val_ccy": pl.String,
    "list_time": pl.Int64,
    "quote_volume_24h": pl.Float64,
    "last": pl.Float64,
    "bid_px": pl.Float64,
    "ask_px": pl.Float64,
    "spread_bps": pl.Float64,
    "history_coverage_pct": pl.Float64,
    "eligible": pl.Boolean,
    "exclude_reason": pl.String,
    "rank_score": pl.Float64,
}


@dataclass(frozen=True)
class DiscoveryResult:
    symbols: tuple[str, ...]
    frame: pl.DataFrame


def empty_discovery_frame() -> pl.DataFrame:
    return pl.DataFrame(schema=DISCOVERY_SCHEMA)


def rank_discovery(
    instruments: pl.DataFrame,
    tickers: pl.DataFrame,
    *,
    min_volume_usd: float = 1_000_000.0,
    max_spread_bps: float = 50.0,
    min_history_coverage_pct: float = 0.0,
    manual_symbols: tuple[str, ...] = (),
) -> pl.DataFrame:
    """Rank instruments by volume, spread, contract coverage. Pure polars."""
    if instruments.is_empty():
        return empty_discovery_frame()
    joined = (
        instruments.join(tickers, on="inst_id", how="left")
        if not tickers.is_empty()
        else instruments
    )
    if "history_coverage_pct" not in joined.columns:
        joined = joined.with_columns(pl.lit(None).cast(pl.Float64).alias("history_coverage_pct"))
    if "symbol" not in joined.columns:
        joined = joined.with_columns(pl.col("inst_id").alias("symbol"))

    manual_list = list(manual_symbols)
    live = pl.col("state").is_in(["live", "trading"])
    has_volume = pl.col("quote_volume_24h").fill_null(0.0) >= min_volume_usd
    has_contract = pl.col("ct_val").is_not_null()
    spread_ok = pl.col("spread_bps").is_null() | (pl.col("spread_bps") <= max_spread_bps)
    coverage_ok = pl.col("history_coverage_pct").is_null() | (
        pl.col("history_coverage_pct") >= min_history_coverage_pct
    )
    manual_expr = pl.col("inst_id").is_in(manual_list) if manual_list else pl.lit(False)
    eligible = live & has_volume & has_contract & spread_ok & coverage_ok

    joined = joined.with_columns(
        [
            (eligible | manual_expr).alias("eligible"),
            pl.when(~live)
            .then(pl.lit("not_live"))
            .when(~has_volume & ~manual_expr)
            .then(pl.lit("volume_below_min"))
            .when(~has_contract & ~manual_expr)
            .then(pl.lit("contract_metadata_missing"))
            .when(~spread_ok & ~manual_expr)
            .then(pl.lit("spread_above_max"))
            .when(~coverage_ok & ~manual_expr)
            .then(pl.lit("history_coverage_below_min"))
            .when(manual_expr & ~eligible)
            .then(pl.lit("manual_override"))
            .otherwise(pl.lit(""))
            .alias("exclude_reason"),
            (
                pl.col("quote_volume_24h").fill_null(0.0).clip(1.0).log10()
                - pl.col("spread_bps").fill_null(0.0) / 100.0
                + pl.col("history_coverage_pct").fill_null(0.0) / 100.0
                - pl.when(has_contract).then(0.0).otherwise(2.0)
            ).alias("rank_score"),
        ]
    )
    for col, dtype in DISCOVERY_SCHEMA.items():
        if col not in joined.columns:
            joined = joined.with_columns(pl.lit(None).cast(dtype).alias(col))
    return joined.select(DISCOVERY_SCHEMA.keys()).sort("rank_score", descending=True)


def select_symbols(
    frame: pl.DataFrame,
    *,
    top_n: int = 25,
    manual_symbols: tuple[str, ...] = (),
) -> tuple[str, ...]:
    """Pick top-N eligible symbols. Pure."""
    if frame.is_empty():
        return manual_symbols
    selected = list(dict.fromkeys(manual_symbols))
    for symbol in (
        frame.filter(pl.col("eligible"))
        .sort("rank_score", descending=True)
        .get_column("symbol")
        .to_list()
    ):
        if symbol not in selected:
            selected.append(symbol)
        if len(selected) >= top_n:
            break
    return tuple(selected)
