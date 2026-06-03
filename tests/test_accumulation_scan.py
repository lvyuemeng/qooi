from __future__ import annotations

import importlib.util
import os
from argparse import Namespace
from pathlib import Path

import polars as pl
import pytest

from qooi.accumulation.config import AccumulationConfig, load_accumulation_config
from qooi.accumulation.csv_io import read_artifact, write_artifact, write_source_bundle
from qooi.sources.coverage import manifest_frame, source_manifest_row
from qooi.sources.models import SourceResult

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
        scan_top_n=None,
        min_volume_usd=None,
        refresh_discovery=False,
    )

    def fail_discover(*_args, **_kwargs):
        raise AssertionError("score phase must not discover")

    monkeypatch.setattr(scan, "run_discover", fail_discover)

    with pytest.raises(SystemExit, match="candidate-discovery.csv is missing"):
        scan.resolve_symbols_or_discover(cfg, args, scan_top_n=25)


def test_summarize_phase_does_not_resolve_symbols(tmp_path, monkeypatch) -> None:
    cfg = _config(tmp_path)
    args = Namespace(
        phase="summarize",
        config="unused.toml",
        fetch_concurrency=None,
        book_mode=None,
        scan_top_n=None,
        summary_top_n=10,
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


def test_local_source_requires_rows_for_selected_symbol(tmp_path) -> None:
    path = tmp_path / "sources" / "trades.csv"
    path.parent.mkdir(parents=True)
    pl.DataFrame(
        {
            "symbol": ["ETH-USDT-SWAP"],
            "timestamp": [1],
            "trade_id": ["1"],
        }
    ).write_csv(path)

    assert scan._local_source_has_symbol_rows(path, "ETH-USDT-SWAP") is True
    assert scan._local_source_has_symbol_rows(path, "BTC-USDT-SWAP") is False


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
    monkeypatch.setattr(scan, "collect_okx_context_batch", collect_public)
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


def test_collect_context_disabled_family_writes_skipped_manifest(tmp_path) -> None:
    cfg = AccumulationConfig.model_validate(
        {"run": {"out": str(tmp_path)}, "sources": {"disabled": {"families": ["polymarket"]}}}
    )

    manifest = scan.run_collect_context(cfg, ("BTC-USDT-SWAP",), concurrency=1)

    context = manifest.filter(pl.col("source") == "polymarket_markets")
    assert context.height == 1
    assert context["status"][0] == "skipped"
    assert context["warning"][0] == "polymarket_disabled"


def test_collect_onchain_does_not_require_address_book(tmp_path) -> None:
    cfg = AccumulationConfig.model_validate(
        {
            "run": {"out": str(tmp_path)},
            "onchain": {"exchange_address_book": ""},
        }
    )

    manifest = scan.run_collect_onchain(cfg, ("BTC-USDT-SWAP",))

    onchain = manifest.filter(pl.col("source") == "onchain")
    assert onchain.height == 1
    assert onchain["status"][0] == "missing"
    assert onchain["warning"][0] == "onchain_token_mapping_missing"


def test_collect_onchain_token_mapping_reports_unimplemented_provider(tmp_path) -> None:
    cfg = AccumulationConfig.model_validate(
        {
            "run": {"out": str(tmp_path)},
            "onchain": {
                "tokens": [
                    {
                        "symbol": "BTC-USDT-SWAP",
                        "chain": "ethereum",
                        "token_address": "0xbtc",
                    }
                ]
            },
        }
    )

    manifest = scan.run_collect_onchain(cfg, ("BTC-USDT-SWAP",))

    onchain = manifest.filter(pl.col("source") == "onchain")
    assert onchain["status"][0] == "missing"
    assert onchain["warning"][0] == "onchain_provider_not_implemented"


def test_collect_onchain_disabled_symbol_writes_skipped_manifest(tmp_path) -> None:
    cfg = AccumulationConfig.model_validate(
        {
            "run": {"out": str(tmp_path)},
            "sources": {"disabled": {"symbols": ["BTC-USDT-SWAP"]}},
        }
    )

    manifest = scan.run_collect_onchain(cfg, ("BTC-USDT-SWAP",))

    onchain = manifest.filter(pl.col("source") == "onchain")
    assert onchain["status"][0] == "skipped"
    assert onchain["warning"][0] == "onchain_disabled"


def test_collect_context_empty_message_path_reports_config_blocker(tmp_path) -> None:
    cfg = AccumulationConfig.model_validate(
        {"run": {"out": str(tmp_path)}, "sources": {"disabled": {"families": ["polymarket"]}}}
    )

    manifest = scan.run_collect_context(cfg, ("BTC-USDT-SWAP",), concurrency=1)

    messages = manifest.filter(pl.col("source") == "messages")
    assert messages["status"][0] == "missing"
    assert messages["warning"][0] == "local_messages_path_missing"


def test_polymarket_no_alias_symbol_fetches_generated_query(tmp_path, monkeypatch) -> None:
    cfg = _config(tmp_path)
    calls = []

    async def fetch(_client, query: str, *, symbol: str, limit_per_type: int):
        calls.append((query, symbol, limit_per_type))
        return SourceResult(
            pl.DataFrame(),
            manifest_frame(
                [
                    source_manifest_row(
                        symbol=symbol,
                        source="polymarket_markets",
                        phase="collect-context",
                        status="missing",
                        backend="polymarket_gamma",
                        endpoint="/public-search",
                        warning="polymarket_unmatched",
                    )
                ]
            ),
        )

    monkeypatch.setattr(scan, "fetch_polymarket_search_async", fetch)

    manifest = scan.run_collect_context(cfg, ("DOGE-USDT-SWAP",), concurrency=1)

    assert calls == [("DOGE", "DOGE-USDT-SWAP", cfg.sources.polymarket.search_limit_per_symbol)]
    polymarket = manifest.filter(pl.col("source") == "polymarket_markets")
    assert polymarket["warning"][0] == "polymarket_unmatched"
    assert polymarket["endpoint"][0] == "/public-search"


def test_polymarket_disabled_query_does_not_fetch(tmp_path, monkeypatch) -> None:
    cfg = AccumulationConfig.model_validate(
        {
            "run": {"out": str(tmp_path)},
            "sources": {"disabled": {"polymarket_queries": ["DOGE"]}},
        }
    )

    async def fetch(*_args, **_kwargs):
        raise AssertionError("disabled query must not fetch")

    monkeypatch.setattr(scan, "fetch_polymarket_search_async", fetch)

    manifest = scan.run_collect_context(cfg, ("DOGE-USDT-SWAP",), concurrency=1)

    polymarket = manifest.filter(pl.col("source") == "polymarket_markets")
    assert polymarket["status"][0] == "skipped"
    assert polymarket["warning"][0] == "polymarket_query_disabled"


def test_context_probe_symbols_can_be_wider_than_scan_symbols() -> None:
    discovery = pl.DataFrame(
        {
            "symbol": ["A-USDT-SWAP", "B-USDT-SWAP", "C-USDT-SWAP"],
            "eligible": [True, True, True],
            "rank_score": [3.0, 2.0, 1.0],
        }
    )

    symbols = scan._context_probe_symbols(("A-USDT-SWAP",), discovery, 3)

    assert symbols == ("A-USDT-SWAP", "B-USDT-SWAP", "C-USDT-SWAP")


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
                "disabled": {"families": ["polymarket"]},
                "messages": {"path": str(messages_path)},
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


def test_potential_board_writes_reduced_artifacts(tmp_path) -> None:
    cfg = AccumulationConfig.model_validate(
        {"run": {"out": str(tmp_path)}, "potential_scan": {"min_history_hours": 2}}
    )
    symbol = "PENGU-USDT-SWAP"
    write_artifact(
        tmp_path,
        "candidate_discovery",
        pl.DataFrame({"symbol": [symbol], "base_ccy": ["PENGU"], "eligible": [True]}),
    )
    write_artifact(
        tmp_path,
        "source_bars",
        pl.DataFrame(
            {
                "symbol": [symbol, symbol, symbol],
                "timestamp": [0, 3_600_000, 7_200_000],
                "open": [10.0, 10.0, 10.0],
                "high": [10.0, 10.0, 10.0],
                "low": [10.0, 10.0, 10.0],
                "close": [10.0, 10.1, 10.2],
                "vol": [100.0, 100.0, 400.0],
            }
        ),
    )
    write_artifact(
        tmp_path,
        "features",
        pl.DataFrame(
            {
                "symbol": [symbol],
                "timestamp": [7_200_000],
                "taker_buy_ratio": [0.7],
                "source_coverage_score": [0.9],
            }
        ),
    )
    write_artifact(
        tmp_path,
        "scores",
        pl.DataFrame(
            {
                "symbol": [symbol],
                "timestamp": [7_200_000],
                "score_total": [-25],
                "alert_level": ["none"],
                "missing_evidence": ["onchain_missing;messages_missing"],
            }
        ),
    )

    board, sources = scan.run_potential_board(cfg)

    assert board.height == 1
    assert sources.is_empty()
    assert read_artifact(tmp_path, "potential_board").height == 1
    assert (tmp_path / "potential" / "report.md").exists()


def test_potential_broad_phase_routes_to_board(tmp_path, monkeypatch) -> None:
    cfg = _config(tmp_path)
    args = Namespace(
        phase="potential-broad",
        config="unused.toml",
        symbols="",
        scan_top_n=None,
        availability_top_n=None,
        broad_top_n=None,
        deep_top_n=None,
        summary_top_n=None,
        min_volume_usd=None,
        min_coverage_pct=None,
        fetch_concurrency=None,
        book_mode=None,
        refresh_discovery=False,
        refresh_broad=False,
        refresh_bars=False,
        refresh_trades=False,
        refresh_context=False,
    )
    calls = []

    monkeypatch.setattr(scan, "parse_args", lambda: args)
    monkeypatch.setattr(scan, "load_accumulation_config", lambda _path: cfg)
    monkeypatch.setattr(
        scan,
        "run_discover_potential_board",
        lambda *_args, **_kwargs: (("PENGU-USDT-SWAP",), pl.DataFrame(), pl.DataFrame()),
    )
    monkeypatch.setattr(
        scan,
        "run_collect_market",
        lambda *_args, **_kwargs: calls.append("market") or pl.DataFrame(),
    )
    monkeypatch.setattr(
        scan,
        "run_collect_onchain",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("unexpected onchain")),
    )
    monkeypatch.setattr(
        scan,
        "run_collect_context",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("unexpected context")),
    )
    monkeypatch.setattr(
        scan,
        "run_score",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("unexpected score")),
    )
    monkeypatch.setattr(
        scan,
        "run_potential_board",
        lambda *_args, **_kwargs: calls.append("board") or (pl.DataFrame(), pl.DataFrame()),
    )

    scan.main()

    assert calls == ["market", "board"]


def test_potential_board_phase_skips_strict_score_path(tmp_path, monkeypatch) -> None:
    cfg = _config(tmp_path)
    args = Namespace(
        phase="potential-board",
        config="unused.toml",
        symbols="",
        scan_top_n=None,
        availability_top_n=None,
        broad_top_n=None,
        deep_top_n=None,
        summary_top_n=None,
        min_volume_usd=None,
        min_coverage_pct=None,
        fetch_concurrency=None,
        book_mode=None,
        refresh_discovery=False,
        refresh_broad=False,
        refresh_bars=False,
        refresh_trades=False,
        refresh_context=False,
    )
    calls = []

    monkeypatch.setattr(scan, "parse_args", lambda: args)
    monkeypatch.setattr(scan, "load_accumulation_config", lambda _path: cfg)
    monkeypatch.setattr(
        scan,
        "run_discover_broad",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("unexpected broad discovery")
        ),
    )
    monkeypatch.setattr(
        scan,
        "run_discover_potential_board",
        lambda *_args, **_kwargs: (("BASE-USDT-SWAP",), pl.DataFrame(), pl.DataFrame()),
    )
    monkeypatch.setattr(
        scan,
        "run_collect_market",
        lambda *_args, **_kwargs: calls.append("market") or pl.DataFrame(),
    )
    monkeypatch.setattr(
        scan,
        "run_score",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("unexpected score")),
    )
    monkeypatch.setattr(
        scan,
        "run_potential_board",
        lambda *_args, **_kwargs: calls.append("board") or (pl.DataFrame(), pl.DataFrame()),
    )

    scan.main()

    assert calls == ["market", "board"]


def test_potential_board_writes_reduced_artifacts_without_strict_scores(tmp_path) -> None:
    cfg = AccumulationConfig.model_validate(
        {
            "run": {"out": str(tmp_path)},
            "potential_scan": {
                "min_history_hours": 20,
                "full_history_hours": 120,
                "min_base_duration_hours": 10,
                "max_new_low_count_30d": 2,
            },
        }
    )
    symbol = "BASE-USDT-SWAP"
    bars = pl.DataFrame(
        [
            {"symbol": symbol, "timestamp": idx * 3_600_000, "close": 10.0, "vol": 100.0}
            for idx in range(160)
        ]
    )
    write_source_bundle(tmp_path, bars=bars)
    write_artifact(
        tmp_path,
        "candidate_discovery",
        pl.DataFrame(
            {
                "symbol": [symbol],
                "inst_id": [symbol],
                "base_ccy": ["BASE"],
                "eligible": [True],
                "rank_score": [1.0],
            }
        ),
    )
    write_artifact(
        tmp_path,
        "broad_candidates",
        pl.DataFrame(
            {
                "rank": [1],
                "timestamp": [1],
                "base_ccy": ["BASE"],
                "coin_id": ["base"],
                "name": ["Base"],
                "okx_symbol": [symbol],
                "okx_mapped": [True],
                "market_cap_usd": [100_000_000.0],
                "volume_24h_usd": [5_000_000.0],
                "price_change_pct_1h": [0.0],
                "price_change_pct_24h": [0.0],
                "trending_rank": [None],
                "trending_score": [None],
                "heat_source": [""],
                "broad_score": [1.0],
                "broad_reasons": [""],
                "exclude_reason": [""],
            }
        ),
    )

    board, sources = scan.run_potential_board(cfg)

    assert board.height == 1
    assert sources.is_empty()
    assert read_artifact(tmp_path, "potential_board").height == 1
    assert read_artifact(tmp_path, "potential_sources").is_empty()
    assert (tmp_path / "potential" / "report.md").exists()
    assert "CoinGecko search-trending is annotation only" in (
        tmp_path / "potential" / "report.md"
    ).read_text(encoding="utf-8")


def test_source_bundle_write_preserves_other_symbol_rows(tmp_path) -> None:
    write_source_bundle(
        tmp_path,
        funding=pl.DataFrame(
            {
                "symbol": ["BTC-USDT-SWAP", "ETH-USDT-SWAP"],
                "timestamp": [1, 1],
                "funding_time": [1, 1],
                "funding_rate": [0.1, 0.2],
            }
        ),
    )

    write_source_bundle(
        tmp_path,
        funding=pl.DataFrame(
            {
                "symbol": ["BTC-USDT-SWAP"],
                "timestamp": [2],
                "funding_time": [2],
                "funding_rate": [0.3],
            }
        ),
    )

    funding = read_artifact(tmp_path, "source_funding").sort(["symbol", "timestamp"])

    assert funding["symbol"].to_list() == ["BTC-USDT-SWAP", "ETH-USDT-SWAP"]
    assert funding.filter(pl.col("symbol") == "BTC-USDT-SWAP")["funding_rate"][0] == 0.3
    assert funding.filter(pl.col("symbol") == "ETH-USDT-SWAP")["funding_rate"][0] == 0.2


def test_collect_market_writes_rubik_source_artifacts(tmp_path, monkeypatch) -> None:
    cfg = _config(tmp_path)
    symbol = "BTC-USDT-SWAP"

    async def refresh_bars(*_args, **_kwargs):
        return manifest_frame([]), pl.DataFrame()

    async def collect_public(*_args, **_kwargs):
        return (
            manifest_frame(
                [
                    source_manifest_row(
                        symbol=symbol,
                        source="open_interest_history",
                        phase="collect-market",
                        status="ok",
                    ),
                    source_manifest_row(
                        symbol=symbol,
                        source="taker_volume_contract",
                        phase="collect-market",
                        status="ok",
                    ),
                    source_manifest_row(
                        symbol=symbol,
                        source="long_short_ratio_contract",
                        phase="collect-market",
                        status="ok",
                    ),
                ]
            ),
            {
                "open_interest": pl.DataFrame(
                    {
                        "symbol": [symbol],
                        "timestamp": [1],
                        "open_interest": [100.0],
                        "open_interest_usd": [1000.0],
                    }
                ),
                "taker_volume": pl.DataFrame(
                    {
                        "symbol": [symbol],
                        "timestamp": [1],
                        "taker_buy_volume": [60.0],
                        "taker_sell_volume": [40.0],
                        "taker_volume_unit": ["2"],
                    }
                ),
                "long_short_ratios": pl.DataFrame(
                    {
                        "symbol": [symbol],
                        "timestamp": [1],
                        "long_short_account_ratio": [1.2],
                    }
                ),
            },
        )

    monkeypatch.setattr(scan, "_refresh_bars", refresh_bars)
    monkeypatch.setattr(scan, "collect_okx_context_batch", collect_public)

    manifest = scan.run_collect_market(
        cfg,
        (symbol,),
        pl.DataFrame(),
        concurrency=1,
        book_mode="off",
        refresh_bars=False,
        refresh_trades=False,
        refresh_context=False,
    )

    assert manifest.filter(pl.col("source") == "taker_volume_contract").height == 1
    assert read_artifact(tmp_path, "source_open_interest")["open_interest_usd"][0] == 1000.0
    assert read_artifact(tmp_path, "source_taker_volume")["taker_buy_volume"][0] == 60.0
    assert read_artifact(tmp_path, "source_long_short_ratios")["long_short_account_ratio"][0] == 1.2


def test_collect_market_passes_cached_source_availability_index(tmp_path, monkeypatch) -> None:
    cfg = _config(tmp_path)
    symbol = "BTC-USDT-SWAP"
    write_artifact(
        tmp_path,
        "source_trades",
        pl.DataFrame({"symbol": [symbol], "timestamp": [1], "trade_id": ["1"]}),
    )
    captured = {}

    async def refresh_bars(*_args, **_kwargs):
        return manifest_frame([]), pl.DataFrame()

    async def collect_public(*_args, **kwargs):
        captured["availability"] = kwargs["source_availability"]
        return manifest_frame([]), {}

    monkeypatch.setattr(scan, "_refresh_bars", refresh_bars)
    monkeypatch.setattr(scan, "collect_okx_context_batch", collect_public)

    scan.run_collect_market(
        cfg,
        (symbol,),
        pl.DataFrame(),
        concurrency=1,
        book_mode="off",
        refresh_bars=False,
        refresh_trades=False,
        refresh_context=False,
    )

    assert captured["availability"]["source_trades"] == {symbol}


def test_score_uses_rubik_source_bundle(tmp_path) -> None:
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
                    symbol=symbol,
                    source="taker_volume_contract",
                    phase="collect-market",
                    status="ok",
                ),
                source_manifest_row(
                    symbol=symbol,
                    source="long_short_ratio_contract",
                    phase="collect-market",
                    status="ok",
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
        "source_taker_volume",
        pl.DataFrame(
            {
                "symbol": [symbol],
                "timestamp": [7_200_000],
                "taker_buy_volume": [70.0],
                "taker_sell_volume": [30.0],
                "taker_volume_unit": ["2"],
            }
        ),
    )
    write_artifact(
        tmp_path,
        "source_long_short_ratios",
        pl.DataFrame(
            {
                "symbol": [symbol],
                "timestamp": [7_200_000],
                "long_short_account_ratio": [1.2],
                "top_trader_long_short_position_ratio": [2.0],
            }
        ),
    )

    features, _scores, _coverage = scan.run_score(cfg, (symbol,), discovery)
    latest = features.sort("timestamp").tail(1)

    assert latest["taker_buy_ratio"][0] == 0.7
    assert latest["top_trader_long_short_position_ratio"][0] == 2.0


def test_score_ignores_source_artifact_when_latest_manifest_skipped(tmp_path) -> None:
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
                    symbol=symbol, source="books", phase="collect-market", status="skipped"
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
        "source_books",
        pl.DataFrame(
            {
                "symbol": [symbol],
                "timestamp": [7_200_000],
                "ob_bid_price": [10.0],
                "ob_ask_price": [10.1],
                "ob_bid_vol_5": [100.0],
                "ob_ask_vol_5": [50.0],
                "ob_bid_vol_10": [100.0],
                "ob_ask_vol_10": [50.0],
                "ob_bid_vol_25": [100.0],
                "ob_ask_vol_25": [50.0],
                "ob_bid_vol": [100.0],
                "ob_ask_vol": [50.0],
                "ob_imbalance_5": [0.33],
                "ob_imbalance_10": [0.33],
                "ob_imbalance_25": [0.33],
            }
        ),
    )

    features, _scores, _coverage = scan.run_score(cfg, (symbol,), discovery)

    latest = features.sort("timestamp").tail(1)
    assert latest["depth_imbalance_10_mean"][0] is None
    assert "book_missing" in latest["data_quality_warning"][0]


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
    feedback = (tmp_path / "scan-feedback.md").read_text(encoding="utf-8")

    assert summary.height == 1
    assert next_fetch.height == 0
    assert detail.height == 1
    assert detail["symbol"][0] == "MEME-USDT-SWAP"
    assert detail["return_24h"][0] == 0.05
    assert "MEME-USDT-SWAP" in feedback
    assert "collect-onchain" not in feedback


def test_summarize_reads_broad_candidates_for_feedback(tmp_path) -> None:
    cfg = _config(tmp_path)
    write_artifact(
        tmp_path,
        "scores",
        pl.DataFrame(
            {
                "timestamp": [1],
                "symbol": ["EDGE-USDT-SWAP"],
                "score_total": [20],
                "alert_level": ["yellow"],
                "confidence_level": ["medium"],
                "suggestion_type": ["prepare_watch"],
                "missing_evidence": ["messages_missing"],
                "positive_components": ["depth_support_on_down_day +20"],
            }
        ),
    )
    write_artifact(tmp_path, "features", pl.DataFrame())
    write_artifact(tmp_path, "source_manifest", manifest_frame([]))
    write_artifact(
        tmp_path,
        "broad_candidates",
        pl.DataFrame(
            {
                "rank": [1, 2],
                "base_ccy": ["EDGE", "BILL"],
                "okx_symbol": ["EDGE-USDT-SWAP", "BILL-USDT-SWAP"],
                "okx_mapped": [True, True],
                "broad_score": [12.0, 11.0],
                "broad_reasons": ["active_1h", "active_24h"],
                "exclude_reason": ["", ""],
            }
        ),
    )

    scan.run_summarize(cfg, top_n=1)
    feedback = (tmp_path / "scan-feedback.md").read_text(encoding="utf-8")

    assert "## Selection Readout" in feedback
    assert "EDGE-USDT-SWAP: verdict=watch_orderbook, bucket=strict_alert" in feedback
    assert "BILL-USDT-SWAP: verdict=data_blocked, bucket=data_blocked" in feedback


def test_resolve_uses_scan_top_n_not_summary_top_n(tmp_path, monkeypatch) -> None:
    cfg = _config(tmp_path)
    args = Namespace(
        phase="all",
        symbols="",
        scan_top_n=5,
        summary_top_n=1,
        min_volume_usd=None,
        refresh_discovery=False,
    )
    called = {}

    def discover(_config, *, top_n: int, min_volume_usd: float | None, manual_symbols=()):
        called["top_n"] = top_n
        return ("A", "B", "C", "D", "E"), pl.DataFrame(), pl.DataFrame()

    monkeypatch.setattr(scan, "run_discover", discover)

    symbols, _discovery, _manifest = scan.resolve_symbols_or_discover(cfg, args, scan_top_n=5)

    assert called["top_n"] == 5
    assert len(symbols) == 5


def test_all_broad_uses_mapped_broad_symbols_as_deep_universe(tmp_path, monkeypatch) -> None:
    cfg = _config(tmp_path)
    selected = ("DOGE-USDT-SWAP", "AAVE-USDT-SWAP")
    discovery = pl.DataFrame(
        {
            "symbol": list(selected),
            "eligible": [True, True],
            "rank_score": [2.0, 1.0],
        }
    )
    args = Namespace(
        phase="all-broad",
        config="unused.toml",
        scan_top_n=None,
        availability_top_n=None,
        broad_top_n=100,
        deep_top_n=25,
        summary_top_n=25,
        fetch_concurrency=1,
        book_mode=None,
        refresh_broad=True,
        refresh_bars=False,
        refresh_trades=False,
        refresh_context=True,
    )
    calls = {}

    write_artifact(
        tmp_path,
        "broad_candidates",
        pl.DataFrame(
            {
                "rank": [1, 2],
                "base_ccy": ["DOGE", "AAVE"],
                "okx_symbol": list(selected),
                "okx_mapped": [True, True],
                "broad_score": [12.0, 11.0],
                "broad_reasons": ["coingecko_trending", "active_1h"],
                "exclude_reason": ["", ""],
            }
        ),
    )

    monkeypatch.setattr(scan, "parse_args", lambda: args)
    monkeypatch.setattr(scan, "load_accumulation_config", lambda _path: cfg)
    monkeypatch.setattr(
        scan,
        "run_discover_broad",
        lambda *_args, **_kwargs: (selected, discovery, pl.DataFrame()),
    )
    monkeypatch.setattr(
        scan,
        "run_collect_market",
        lambda _cfg, symbols, _discovery, **kwargs: (
            calls.setdefault("collect_market", (symbols, kwargs)),
            pl.DataFrame(),
        )[1],
    )
    monkeypatch.setattr(
        scan,
        "run_collect_onchain",
        lambda _cfg, symbols: (calls.setdefault("onchain", symbols), pl.DataFrame())[1],
    )
    monkeypatch.setattr(
        scan,
        "run_collect_context",
        lambda _cfg, symbols, _discovery, **_kwargs: (
            calls.setdefault("context", symbols),
            pl.DataFrame(),
        )[1],
    )
    monkeypatch.setattr(
        scan,
        "run_score",
        lambda _cfg, symbols, _discovery: (
            calls.setdefault("score", symbols),
            (pl.DataFrame(), pl.DataFrame(), pl.DataFrame()),
        )[1],
    )
    monkeypatch.setattr(
        scan,
        "run_summarize",
        lambda _cfg, *, top_n: (
            calls.setdefault("summarize", top_n),
            (pl.DataFrame(), pl.DataFrame()),
        )[1],
    )

    scan.main()

    assert calls["collect_market"][0] == selected
    assert calls["collect_market"][1]["refresh_context"] is True
    assert calls["onchain"] == selected
    assert calls["score"] == selected
    assert calls["summarize"] == 25


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


def test_polymarket_source_config_accepts_additive_aliases(tmp_path) -> None:
    config_path = tmp_path / "accumulation.toml"
    config_path.write_text(
        "\n".join(
            [
                "[run]",
                "out = 'out'",
                "[sources.polymarket]",
                "[[sources.polymarket.aliases]]",
                "symbol = 'BTC-USDT-SWAP'",
                "queries = ['Bitcoin', 'BTC']",
            ]
        ),
        encoding="utf-8",
    )

    cfg = load_accumulation_config(config_path)

    assert cfg.sources.polymarket.aliases[0].queries == ("Bitcoin", "BTC")


def test_broad_coingecko_config_accepts_trending_fields(tmp_path) -> None:
    config_path = tmp_path / "accumulation.toml"
    config_path.write_text(
        "\n".join(
            [
                "[run]",
                "out = 'out'",
                "[broad_scan.coingecko]",
                "include_trending = true",
                "trending_weight = 6.5",
            ]
        ),
        encoding="utf-8",
    )

    cfg = load_accumulation_config(config_path)

    assert cfg.broad_scan.coingecko.include_trending is True
    assert cfg.broad_scan.coingecko.trending_weight == 6.5


def test_old_optional_source_enabled_fields_fail_strict_config(tmp_path) -> None:
    config_path = tmp_path / "accumulation.toml"
    config_path.write_text(
        "\n".join(
            [
                "[run]",
                "out = 'out'",
                "[sources.polymarket]",
                "enabled = true",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(Exception, match="enabled"):
        load_accumulation_config(config_path)


