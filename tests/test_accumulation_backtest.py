from __future__ import annotations

import importlib.util
from argparse import Namespace
from pathlib import Path

import polars as pl
import pytest

from qooi.accumulation.config import AccumulationConfig
from qooi.accumulation.csv_io import read_artifact, write_artifact
from qooi.core.event_backtest import build_backtest_events, summarize_backtest_events

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "accumulation_backtest.py"
_SPEC = importlib.util.spec_from_file_location("accumulation_backtest_script", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
backtest_script = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(backtest_script)


def test_backtest_events_compute_forward_returns_and_drawdown() -> None:
    scores = pl.DataFrame(
        {"timestamp": [0], "symbol": ["BTC-USDT-SWAP"], "alert_level": ["red"], "score_total": [45]}
    )
    prices = pl.DataFrame(
        {
            "timestamp": [idx * 3_600_000 for idx in range(169)],
            "close": [100.0 + idx for idx in range(169)],
            "high": [101.0 + idx for idx in range(169)],
            "low": [99.0 + idx for idx in range(169)],
        }
    )

    events = build_backtest_events(scores, prices)

    assert events.height == 1
    assert events["return_3h"][0] == pytest.approx(0.03)
    assert events["hit_take_profit_5pct_7d"][0] is True
    assert events["hit_stop_loss_5pct_7d"][0] is False


def test_backtest_summary_writes_empty_schema_when_no_alerts() -> None:
    summary = summarize_backtest_events(build_backtest_events(pl.DataFrame(), pl.DataFrame()))

    assert summary.is_empty()
    assert "signal_count" in summary.columns


def test_backtest_script_reads_source_bars_artifact(tmp_path, monkeypatch) -> None:
    cfg = AccumulationConfig.model_validate({"run": {"out": str(tmp_path)}})
    symbol = "BTC-USDT-SWAP"
    write_artifact(
        tmp_path,
        "scores",
        pl.DataFrame(
            {"timestamp": [0], "symbol": [symbol], "alert_level": ["red"], "score_total": [45]}
        ),
    )
    write_artifact(
        tmp_path,
        "source_bars",
        pl.DataFrame(
            {
                "symbol": [symbol for _ in range(169)],
                "timestamp": [idx * 3_600_000 for idx in range(169)],
                "open": [100.0 + idx for idx in range(169)],
                "high": [101.0 + idx for idx in range(169)],
                "low": [99.0 + idx for idx in range(169)],
                "close": [100.0 + idx for idx in range(169)],
                "vol": [1.0 for _ in range(169)],
            }
        ),
    )
    monkeypatch.setattr(
        backtest_script,
        "parse_args",
        lambda: Namespace(config="unused.toml", horizons=backtest_script.SUPPORTED_HORIZONS),
    )
    monkeypatch.setattr(backtest_script, "load_accumulation_config", lambda _path: cfg)

    backtest_script.main()

    events = read_artifact(tmp_path, "backtest_events")
    coverage = read_artifact(tmp_path, "data_coverage")
    assert events.height == 1
    assert coverage.filter(pl.col("status") == "ok").height == 1


def test_backtest_script_records_missing_source_bars(tmp_path, monkeypatch) -> None:
    cfg = AccumulationConfig.model_validate({"run": {"out": str(tmp_path)}})
    symbol = "BTC-USDT-SWAP"
    write_artifact(
        tmp_path,
        "scores",
        pl.DataFrame(
            {"timestamp": [0], "symbol": [symbol], "alert_level": ["red"], "score_total": [45]}
        ),
    )
    monkeypatch.setattr(
        backtest_script,
        "parse_args",
        lambda: Namespace(config="unused.toml", horizons=backtest_script.SUPPORTED_HORIZONS),
    )
    monkeypatch.setattr(backtest_script, "load_accumulation_config", lambda _path: cfg)

    backtest_script.main()

    events = read_artifact(tmp_path, "backtest_events")
    coverage = read_artifact(tmp_path, "data_coverage")
    assert events.is_empty()
    assert coverage["status"][0] == "missing"
    assert coverage["warning"][0] == "price_missing;backtest_skipped"


def test_backtest_script_does_not_import_scan_script() -> None:
    text = _SCRIPT_PATH.read_text(encoding="utf-8")

    assert "accumulation_scan" not in text
