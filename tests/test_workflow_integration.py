"""Workflow integration tests — full run() with real OKX.

Run with: RUN_INTEGRATION_TESTS=1 uv run pytest tests/test_workflow_integration.py -v
"""

from pathlib import Path

import pytest


def _minimal_config(tmp_path: Path) -> tuple[str, dict]:
    """Write a minimal TOML config to tmp_path, return (path, parsed)."""
    config = {
        "potential": {
            "bars": {"timeframes": ("1H",), "days": 1},
            "books": None,
            "trades": None,
            "funding": None,
            "open_interest": None,
            "taker_volume": None,
            "long_short": None,
            "symbols": ("BTC-USDT-SWAP",),
            "fetch_concurrency": 1,
            "max_staleness_hours": 876000,
            "output": str(tmp_path / "report.md"),
        }
    }
    config_path = tmp_path / "config.toml"
    lines = ["[potential]"]
    for k, v in config["potential"].items():
        if k == "bars":
            lines.append("[potential.bars]")
            for bk, bv in v.items():
                if isinstance(bv, tuple):
                    lines.append(f"{bk} = {list(bv)}")
                else:
                    lines.append(f"{bk} = {bv!r}")
        elif k in ("output", "symbols", "fetch_concurrency", "max_staleness_hours"):
            if isinstance(v, tuple):
                lines.append(f"{k} = {list(v)}")
            else:
                lines.append(f"{k} = {v!r}")
    config_path.write_text("\n".join(lines))
    return str(config_path), config


# ═══════════════════════════════════════════════════════════════════════════
# Full workflow
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.integration
def test_run_bars_only(tmp_path):
    """Full workflow: bars only, single symbol, single timeframe."""
    from qooi.scanner.workflow import run

    config_path, _ = _minimal_config(tmp_path)
    result = run(config_path)

    assert Path(result).exists()
    content = Path(result).read_text()
    assert "Bar Coverage" in content
    assert "BTC-USDT-SWAP" in content


@pytest.mark.integration
def test_run_bars_and_books(tmp_path):
    """Full workflow: bars + books."""
    from qooi.scanner.workflow import run

    config_path, config = _minimal_config(tmp_path)
    # enable books
    config["potential"]["books"] = {"limit": 5}
    _write_config(config_path, config)

    result = run(config_path)
    content = Path(result).read_text()
    assert "Bar Coverage" in content
    assert "Source" in content.lower()


@pytest.mark.integration
def test_run_all_products(tmp_path):
    """Full workflow: all 7 products."""
    from qooi.scanner.workflow import run

    config_path, config = _minimal_config(tmp_path)
    config["potential"]["books"] = {"limit": 5}
    config["potential"]["trades"] = {"limit": 10}
    config["potential"]["funding"] = {"limit": 10}
    config["potential"]["fetch_concurrency"] = 2
    _write_config(config_path, config)

    result = run(config_path)
    content = Path(result).read_text()
    assert "Bar Coverage" in content
    assert "Source" in content.lower()
    assert "Decision" in content.lower()


@pytest.mark.integration
def test_report_contains_frame_health(tmp_path):
    """Report reads FrameHealth fields directly from ProductResult."""
    from qooi.scanner.workflow import run

    config_path, _ = _minimal_config(tmp_path)
    result = run(config_path)
    content = Path(result).read_text()

    # FrameHealth fields appear as numbers in report
    assert "Rows:" in content
    assert "Coverage:" in content
    assert "Age:" in content


def _write_config(path: str, config: dict) -> None:
    """Rewrite config TOML with updated sections."""
    p = Path(path)
    lines = ["[potential]"]
    for k, v in config["potential"].items():
        if isinstance(v, dict):
            lines.append(f"\n[potential.{k}]")
            for sk, sv in v.items():
                if isinstance(sv, tuple):
                    lines.append(f"{sk} = {list(sv)}")
                else:
                    lines.append(f"{sk} = {sv!r}")
        elif isinstance(v, tuple):
            lines.append(f"{k} = {list(v)}")
        else:
            lines.append(f"{k} = {v!r}")
    p.write_text("\n".join(lines))
