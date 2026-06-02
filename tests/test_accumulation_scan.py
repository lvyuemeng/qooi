from __future__ import annotations

import importlib.util
import os
from argparse import Namespace
from pathlib import Path

import polars as pl
import pytest

from qooi.accumulation.config import AccumulationConfig, load_accumulation_config
from qooi.accumulation.csv_io import read_artifact, write_artifact
from qooi.sources.coverage import manifest_frame, source_manifest_row

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "accumulation_scan.py"
_SPEC = importlib.util.spec_from_file_location("accumulation_scan_script", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
scan = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(scan)


def _config(tmp_path) -> AccumulationConfig:
    return AccumulationConfig.model_validate(
        {
            "run": {"out": str(tmp_path)},
            "market": {"book_samples": 1, "book_every_seconds": 0.0, "book_sample_symbols_max": 1},
        }
    )


def test_score_phase_requires_existing_discovery_or_symbols(tmp_path, monkeypatch) -> None:
    cfg = _config(tmp_path)
    args = Namespace(
        phase="score",
        symbols="",
        top_n=10,
        min_volume_usd=None,
        refresh_discovery=False,
    )

    def fail_discover(*_args, **_kwargs):
        raise AssertionError("score phase must not discover")

    monkeypatch.setattr(scan, "run_discover", fail_discover)

    with pytest.raises(SystemExit, match="candidate-discovery.csv is missing"):
        scan.resolve_symbols_or_discover(cfg, args)


def test_summarize_phase_does_not_resolve_symbols(tmp_path, monkeypatch) -> None:
    cfg = _config(tmp_path)
    args = Namespace(
        phase="summarize",
        config="unused.toml",
        fetch_concurrency=None,
        book_mode=None,
        top_n=10,
    )
    called = {"summary": False}

    monkeypatch.setattr(scan, "parse_args", lambda: args)
    monkeypatch.setattr(scan, "load_accumulation_config", lambda _path: cfg)
    monkeypatch.setattr(
        scan,
        "resolve_symbols_or_discover",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("unexpected resolve")),
    )

    def summarize(_config, *, top_n: int):
        called["summary"] = True
        return pl.DataFrame(), pl.DataFrame()

    monkeypatch.setattr(scan, "run_summarize", summarize)

    scan.main()

    assert called["summary"] is True


def test_local_file_manifest_counts_csv_family_rows_for_symbol(tmp_path) -> None:
    path = tmp_path / "sources" / "funding.csv"
    path.parent.mkdir(parents=True)
    pl.DataFrame(
        {
            "symbol": ["BTC-USDT-SWAP", "ETH-USDT-SWAP"],
            "timestamp": [1, 2],
            "funding_rate": [0.1, 0.2],
        }
    ).write_csv(path)

    manifest = scan._local_file_manifest("BTC-USDT-SWAP", "funding", path)

    assert manifest["status"][0] == "ok"
    assert manifest["rows"][0] == 1


def test_local_file_manifest_rejects_parquet_source_artifacts(tmp_path) -> None:
    path = tmp_path / "funding.parquet"
    path.write_bytes(b"not a scanner artifact")

    manifest = scan._local_file_manifest("BTC-USDT-SWAP", "funding", path)

    assert manifest["status"][0] == "missing"
    assert manifest["rows"][0] == 0


def test_sample_book_mode_keeps_other_source_manifests(tmp_path, monkeypatch) -> None:
    cfg = _config(tmp_path)
    symbols = ("BTC-USDT-SWAP", "ETH-USDT-SWAP")

    async def refresh_bars(*_args, **_kwargs):
        return manifest_frame([])

    async def collect_public(*_args, **_kwargs):
        return manifest_frame(
            [
                source_manifest_row(
                    symbol="BTC-USDT-SWAP",
                    source="trades",
                    phase="collect-market",
                    status="ok",
                )
            ]
        )

    class DummyStore:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def books(self, _request):
            return pl.DataFrame()

    monkeypatch.setattr(scan, "_refresh_bars", refresh_bars)
    monkeypatch.setattr(scan, "_collect_public_sources", collect_public)
    monkeypatch.setattr(scan, "CacheStore", lambda: DummyStore())

    manifest = scan.run_collect_market(
        cfg,
        symbols,
        pl.DataFrame(),
        concurrency=1,
        book_mode="sample",
        refresh_bars=False,
        refresh_trades=False,
        refresh_context=False,
    )

    books = manifest.filter(pl.col("source") == "books")
    assert books.height == 2
    assert books.filter(pl.col("symbol") == "ETH-USDT-SWAP")["status"][0] == "skipped"
    assert manifest.filter(pl.col("source") == "trades").height == 1


def test_collect_context_disabled_writes_skipped_manifest(tmp_path) -> None:
    cfg = _config(tmp_path)

    manifest = scan.run_collect_context(cfg, ("BTC-USDT-SWAP",), concurrency=1)

    context = manifest.filter(pl.col("source") == "polymarket_markets")
    assert context.height == 1
    assert context["status"][0] == "skipped"
    assert context["warning"][0] == "polymarket_disabled"


def test_collect_context_writes_local_message_artifacts(tmp_path) -> None:
    messages_path = tmp_path / "messages.csv"
    pl.DataFrame(
        {
            "symbol": ["BTC-USDT-SWAP", "ETH-USDT-SWAP"],
            "timestamp": [3_600_000, 7_200_000],
            "text": ["mainnet upgrade announced", "whale transfer noted"],
        }
    ).write_csv(messages_path)
    cfg = AccumulationConfig.model_validate(
        {
            "run": {"out": str(tmp_path / "out")},
            "sources": {
                "messages": {"enabled": True, "path": str(messages_path)},
            },
        }
    )

    manifest = scan.run_collect_context(cfg, ("BTC-USDT-SWAP",), concurrency=1)
    messages = read_artifact(cfg.output_dir, "source_messages")
    classifications = read_artifact(cfg.output_dir, "message_classifications")

    message_manifest = manifest.filter(pl.col("source") == "messages")
    assert message_manifest.height == 1
    assert message_manifest["status"][0] == "ok"
    assert message_manifest["backend"][0] == "local_csv"
    assert messages.height == 1
    assert messages["symbol"][0] == "BTC-USDT-SWAP"
    assert classifications.height == 1
    assert classifications["message_type"][0] == "fundamental"


def test_score_reads_csv_source_bundle_only(tmp_path, monkeypatch) -> None:
    cfg = _config(tmp_path)
    symbol = "BTC-USDT-SWAP"
    discovery = pl.DataFrame(
        {
            "symbol": [symbol],
            "inst_id": [symbol],
            "quote_volume_24h": [1_000_000.0],
            "spread_bps": [5.0],
        }
    )
    manifest = manifest_frame(
        [source_manifest_row(symbol=symbol, source="bars", phase="collect-market", status="ok")]
    )
    write_artifact(tmp_path, "candidate_discovery", discovery)
    write_artifact(tmp_path, "source_manifest", manifest)
    write_artifact(
        tmp_path,
        "source_bars",
        pl.DataFrame(
            {
                "symbol": [symbol, symbol],
                "timestamp": [1, 2],
                "open": [10.0, 11.0],
                "high": [11.0, 12.0],
                "low": [9.0, 10.0],
                "close": [10.0, 11.0],
                "vol": [100.0, 110.0],
            }
        ),
    )

    monkeypatch.setattr(
        scan,
        "_read_frame",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("cache read")),
    )

    features, scores, coverage = scan.run_score(cfg, (symbol,), discovery)

    assert features.height == 2
    assert scores.height == 2
    assert coverage.height == 1
    assert not (tmp_path / "accumulation.sqlite").exists()


def test_score_uses_message_source_bundle(tmp_path) -> None:
    cfg = _config(tmp_path)
    symbol = "BTC-USDT-SWAP"
    discovery = pl.DataFrame({"symbol": [symbol], "inst_id": [symbol]})
    write_artifact(tmp_path, "candidate_discovery", discovery)
    write_artifact(
        tmp_path,
        "source_manifest",
        manifest_frame(
            [
                source_manifest_row(
                    symbol=symbol, source="bars", phase="collect-market", status="ok"
                ),
                source_manifest_row(
                    symbol=symbol, source="messages", phase="collect-context", status="ok"
                ),
            ]
        ),
    )
    write_artifact(
        tmp_path,
        "source_bars",
        pl.DataFrame(
            {
                "symbol": [symbol, symbol],
                "timestamp": [3_600_000, 7_200_000],
                "open": [10.0, 10.0],
                "high": [11.0, 11.0],
                "low": [9.0, 9.0],
                "close": [10.0, 10.5],
                "vol": [100.0, 100.0],
            }
        ),
    )
    write_artifact(
        tmp_path,
        "source_messages",
        pl.DataFrame(
            {
                "symbol": [symbol, symbol],
                "timestamp": [3_600_000, 7_200_000],
                "source": ["local_csv", "local_csv"],
                "source_id": ["m1", "m2"],
                "text": ["mainnet upgrade", "ecosystem partnership"],
            }
        ),
    )
    write_artifact(
        tmp_path,
        "message_classifications",
        pl.DataFrame(
            {
                "symbol": [symbol, symbol],
                "timestamp": [3_600_000, 7_200_000],
                "message_id": ["m1", "m2"],
                "message_type": ["fundamental", "fundamental"],
            }
        ),
    )

    features, _scores, _coverage = scan.run_score(cfg, (symbol,), discovery)

    latest = features.sort("timestamp").tail(1)
    assert latest["fundamental_news_ratio"][0] == 1.0
    assert "messages_missing" not in latest["data_quality_warning"][0]


def test_summarize_writes_candidate_detail_artifact(tmp_path) -> None:
    cfg = _config(tmp_path)
    write_artifact(
        tmp_path,
        "scores",
        pl.DataFrame(
            {
                "timestamp": [1],
                "symbol": ["MEME-USDT-SWAP"],
                "score_total": [25],
                "alert_level": ["yellow"],
                "confidence_level": ["medium"],
                "suggestion_type": ["prepare_watch"],
                "missing_evidence": ["onchain_missing"],
            }
        ),
    )
    write_artifact(
        tmp_path,
        "features",
        pl.DataFrame(
            {
                "timestamp": [1],
                "symbol": ["MEME-USDT-SWAP"],
                "return_24h": [0.05],
            }
        ),
    )
    write_artifact(tmp_path, "source_manifest", manifest_frame([]))

    summary, next_fetch = scan.run_summarize(cfg, top_n=1)
    detail = read_artifact(tmp_path, "candidate_detail")

    assert summary.height == 1
    assert next_fetch.height == 1
    assert detail.height == 1
    assert detail["symbol"][0] == "MEME-USDT-SWAP"
    assert detail["return_24h"][0] == 0.05


def test_database_enabled_writes_under_db_directory(tmp_path) -> None:
    cfg = AccumulationConfig.model_validate(
        {
            "run": {"out": str(tmp_path)},
            "database": {"enabled": True, "path": "db/accumulation.sqlite"},
        }
    )
    scan._maybe_store(
        cfg,
        "accumulation_scores",
        pl.DataFrame(
            {
                "timestamp": [1],
                "symbol": ["BTC-USDT-SWAP"],
                "alert_level": ["none"],
                "score_total": [0],
            }
        ),
    )

    assert (tmp_path / "db" / "accumulation.sqlite").exists()
    assert not (tmp_path / "accumulation.sqlite").exists()


def test_accumulation_config_loads_dotenv(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("ETHERSCAN_API_KEY", raising=False)
    (tmp_path / ".env").write_text("ETHERSCAN_API_KEY=from-dotenv\n", encoding="utf-8")
    config_path = tmp_path / "accumulation.toml"
    config_path.write_text("[run]\nout = 'out'\n", encoding="utf-8")

    cfg = load_accumulation_config(config_path)

    assert str(cfg.output_dir) == "out"
    assert os.environ["ETHERSCAN_API_KEY"] == "from-dotenv"


def test_polymarket_source_config_accepts_aliases(tmp_path) -> None:
    config_path = tmp_path / "accumulation.toml"
    config_path.write_text(
        "\n".join(
            [
                "[run]",
                "out = 'out'",
                "[sources.polymarket]",
                "enabled = true",
                "[[sources.polymarket.aliases]]",
                "symbol = 'BTC-USDT-SWAP'",
                "queries = ['Bitcoin', 'BTC']",
            ]
        ),
        encoding="utf-8",
    )

    cfg = load_accumulation_config(config_path)

    assert cfg.sources.polymarket.enabled is True
    assert cfg.sources.polymarket.aliases[0].queries == ("Bitcoin", "BTC")
