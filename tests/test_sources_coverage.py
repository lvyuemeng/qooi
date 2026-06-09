from __future__ import annotations

import polars as pl

from qooi.sources.coverage import compute_source_coverage_score, missing_evidence_for_symbol


def test_manifest_rows_roll_up_to_coverage_score() -> None:
    coverage = pl.DataFrame(
        {
            "symbol": ["BTC", "BTC", "BTC"],
            "source": ["bars", "books", "trades"],
            "status": ["ok", "partial", "failed"],
        }
    )

    assert compute_source_coverage_score(coverage, "BTC") == 0.475


def test_missing_data_is_warning_not_neutral_evidence() -> None:
    coverage = pl.DataFrame({"symbol": ["BTC"], "source": ["bars"], "status": ["ok"]})

    missing = missing_evidence_for_symbol(coverage, "BTC")

    assert "books_missing" in missing
    assert "trades_missing" in missing


