from __future__ import annotations

import polars as pl

from qooi.accumulation.artifacts import ARTIFACT_SPECS
from qooi.accumulation.csv_io import assert_csv_catalog, write_artifact, write_csv_artifacts
from qooi.accumulation.database import AccumulationStore


def test_store_upserts_and_reads_scores(tmp_path) -> None:
    store = AccumulationStore(tmp_path / "accumulation.sqlite")
    scores = pl.DataFrame(
        {"timestamp": [1], "symbol": ["BTC-USDT-SWAP"], "alert_level": ["red"], "score_total": [45]}
    )

    store.upsert_frame("accumulation_scores", scores)
    out = store.read_table("accumulation_scores")

    assert out.height == 1
    assert out["symbol"][0] == "BTC-USDT-SWAP"


def test_csv_artifact_names_are_stable(tmp_path) -> None:
    frame = pl.DataFrame({"timestamp": [1], "symbol": ["BTC-USDT-SWAP"]})

    write_csv_artifacts(tmp_path, features=frame, scores=frame, alerts=frame, data_coverage=frame)

    assert (tmp_path / "accumulation-features.csv").exists()
    assert (tmp_path / "accumulation-scores.csv").exists()
    assert (tmp_path / "accumulation-alerts.csv").exists()
    assert (tmp_path / "accumulation-data-coverage.csv").exists()


def test_artifact_catalog_is_csv_only_and_unique() -> None:
    assert_csv_catalog()
    paths = [spec.relative_path for spec in ARTIFACT_SPECS.values()]

    assert len(paths) == len(set(paths))
    assert all(path.endswith(".csv") for path in paths)


def test_source_artifacts_use_family_paths(tmp_path) -> None:
    write_artifact(
        tmp_path,
        "source_funding",
        pl.DataFrame(
            {
                "symbol": ["BTC-USDT-SWAP"],
                "timestamp": [1],
                "funding_time": [1],
                "funding_rate": [0.01],
            }
        ),
    )

    assert (tmp_path / "sources" / "funding.csv").exists()
    assert not (tmp_path / "funding.parquet").exists()


def test_polymarket_artifacts_use_context_source_paths(tmp_path) -> None:
    write_artifact(
        tmp_path,
        "source_polymarket_markets",
        pl.DataFrame(
            {
                "symbol": ["BTC-USDT-SWAP"],
                "timestamp": [1],
                "query": ["Bitcoin"],
                "market_id": ["1"],
            }
        ),
    )

    assert (tmp_path / "sources" / "polymarket-markets.csv").exists()

