from __future__ import annotations

import polars as pl

from qooi.strategies.potential import _ranked_rows_to_board, render_potential_board


def test_potential_board_keeps_latest_row_per_symbol() -> None:
    board = _ranked_rows_to_board(
        pl.DataFrame(
            [
                _candidate("BASE-USDT-SWAP", timestamp=1, rank=1, stage="falling_knife"),
                _candidate("BASE-USDT-SWAP", timestamp=2, rank=2, stage="base_ready"),
            ]
        ),
        _market("BASE-USDT-SWAP"),
    )

    assert board.height == 1
    assert board["timestamp"][0] == 2
    assert board["structure_state"][0] == "base_ready"


def test_potential_board_treats_trending_as_attention_annotation_only() -> None:
    board = _ranked_rows_to_board(
        pl.DataFrame(
            [
                _candidate(
                    "TREND-USDT-SWAP",
                    stage="cooldown_downtrend",
                    funds_state="funds_confirmed",
                    broad_reasons="coingecko_trending",
                )
            ]
        ),
        _market("TREND-USDT-SWAP", heat_source="coingecko_trending"),
    )

    assert board["attention_source"][0] == "coingecko_trending"
    assert board["board_bucket"][0] == "C_context"


def test_potential_board_accepts_market_without_provider_column() -> None:
    board = _ranked_rows_to_board(
        pl.DataFrame([_candidate("BASE-USDT-SWAP")]),
        _market("BASE-USDT-SWAP").drop("provider"),
    )

    assert board["market_data_provider"][0] == "unknown"
    assert board["price_change_pct_1h"][0] == 0.1


def test_potential_board_report_explains_empty_prepared_bucket() -> None:
    board = _ranked_rows_to_board(
        pl.DataFrame(
            [
                _candidate(
                    "KNIFE-USDT-SWAP",
                    stage="falling_knife",
                    structure_block_reason="falling_knife;active_lower_lows",
                )
            ]
        ),
        _market("KNIFE-USDT-SWAP"),
    )

    report = render_potential_board(board)

    assert "## Why No A_prepared" in report
    assert "falling_knife=1" in report
    assert "active_lower_lows=1" in report


def _candidate(
    symbol: str,
    *,
    timestamp: int = 1,
    rank: int = 1,
    stage: str = "base_ready",
    funds_state: str = "funds_building",
    broad_reasons: str = "",
    structure_block_reason: str = "",
) -> dict[str, object]:
    return {
        "rank": rank,
        "timestamp": timestamp,
        "symbol": symbol,
        "base_ccy": symbol.split("-")[0],
        "market_cap_usd": 100_000_000.0,
        "volume_24h_usd": 5_000_000.0,
        "potential_score": 42.0,
        "stage": stage,
        "setup_pass_reason": "near_low",
        "risk_state": "clean",
        "funds_state": funds_state,
        "funds_score_total": 35.0,
        "funds_positive_reasons": "taker_buying;oi_expanding",
        "funds_negative_reasons": "",
        "next_confirmation_needed": "need_onchain_flow",
        "evidence_families_present": "trade;derivatives",
        "base_duration_hours": 240,
        "new_low_count_30d": 0,
        "reclaim_state": "ma7_reclaim",
        "ma_7d_slope_7d": 0.01,
        "ma_30d_slope_14d": -0.01,
        "structure_block_reason": structure_block_reason,
        "broad_reasons": broad_reasons,
        "data_quality_warning": "",
    }


def _market(symbol: str, *, heat_source: str = "") -> pl.DataFrame:
    return pl.DataFrame(
        {
            "okx_symbol": [symbol],
            "price_change_pct_1h": [0.1],
            "price_change_pct_24h": [1.5],
            "provider": ["coingecko_markets"],
            "heat_source": [heat_source],
        }
    )


