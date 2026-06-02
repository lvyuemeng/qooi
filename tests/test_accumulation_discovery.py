from __future__ import annotations

import polars as pl

from qooi.accumulation.config import DiscoveryConfig
from qooi.accumulation.discovery import rank_discovery_frame, select_candidate_symbols


def _instruments() -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "inst_id": "BTC-USDT-SWAP",
                "symbol": "BTC-USDT-SWAP",
                "inst_type": "SWAP",
                "state": "live",
                "ct_val": 0.01,
                "ct_val_ccy": "USDT",
            },
            {
                "inst_id": "ILLQ-USDT-SWAP",
                "symbol": "ILLQ-USDT-SWAP",
                "inst_type": "SWAP",
                "state": "live",
                "ct_val": 1.0,
                "ct_val_ccy": "USDT",
            },
            {
                "inst_id": "MISS-USDT-SWAP",
                "symbol": "MISS-USDT-SWAP",
                "inst_type": "SWAP",
                "state": "live",
                "ct_val": None,
                "ct_val_ccy": "",
            },
        ]
    )


def test_ranks_liquid_symbols_above_illiquid() -> None:
    tickers = pl.DataFrame(
        [
            {
                "inst_id": "BTC-USDT-SWAP",
                "quote_volume_24h": 1_000_000_000.0,
                "bid_px": 100.0,
                "ask_px": 100.1,
                "spread_bps": 10.0,
            },
            {
                "inst_id": "ILLQ-USDT-SWAP",
                "quote_volume_24h": 100.0,
                "bid_px": 1.0,
                "ask_px": 1.1,
                "spread_bps": 950.0,
            },
            {
                "inst_id": "MISS-USDT-SWAP",
                "quote_volume_24h": 2_000_000.0,
                "bid_px": 2.0,
                "ask_px": 2.01,
                "spread_bps": 50.0,
            },
        ]
    )

    out = rank_discovery_frame(
        _instruments(), tickers, pl.DataFrame(), DiscoveryConfig(min_volume_usd=1_000.0)
    )

    assert out["symbol"][0] == "BTC-USDT-SWAP"
    assert out.filter(pl.col("symbol") == "ILLQ-USDT-SWAP")["eligible"][0] is False


def test_excludes_missing_contract_metadata_unless_manual_override() -> None:
    tickers = pl.DataFrame(
        {"inst_id": ["MISS-USDT-SWAP"], "quote_volume_24h": [2_000_000.0], "spread_bps": [1.0]}
    )

    out = rank_discovery_frame(
        _instruments().filter(pl.col("inst_id") == "MISS-USDT-SWAP"),
        tickers,
        pl.DataFrame(),
        DiscoveryConfig(),
    )
    manual = rank_discovery_frame(
        _instruments().filter(pl.col("inst_id") == "MISS-USDT-SWAP"),
        tickers,
        pl.DataFrame(),
        DiscoveryConfig(),
        manual_symbols=("MISS-USDT-SWAP",),
    )

    assert out["eligible"][0] is False
    assert out["exclude_reason"][0] == "contract_metadata_missing"
    assert manual["eligible"][0] is True
    assert manual["exclude_reason"][0] == "manual_override"


def test_select_preserves_manual_symbols() -> None:
    discovery = pl.DataFrame({"symbol": ["BTC-USDT-SWAP"], "eligible": [True], "rank_score": [1.0]})

    assert select_candidate_symbols(discovery, top_n=2, manual_symbols=("MAN-USDT-SWAP",)) == (
        "MAN-USDT-SWAP",
        "BTC-USDT-SWAP",
    )
