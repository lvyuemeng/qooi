from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "qooi"


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                modules.add(node.module)
    return modules


def _python_files(package: str) -> list[Path]:
    return sorted((SRC / package).glob("*.py"))


def _assert_no_forbidden_imports(package: str, forbidden_prefixes: tuple[str, ...]) -> None:
    violations: list[str] = []
    for path in _python_files(package):
        for module in sorted(_imported_modules(path)):
            if module == "qooi" or not module.startswith("qooi."):
                continue
            if any(
                module == prefix or module.startswith(f"{prefix}.")
                for prefix in forbidden_prefixes
            ):
                rel = path.relative_to(ROOT).as_posix()
                violations.append(f"{rel} imports {module}")
    assert not violations, "Forbidden module-boundary imports:\n" + "\n".join(violations)


def test_scanner_does_not_cross_execution_or_ai_boundaries() -> None:
    _assert_no_forbidden_imports(
        "scanner",
        (
            "qooi.core.basket",
            "qooi.core.executor",
            "qooi.core.recovery",
            "qooi.dynamic",
            "qooi.exchange.trading",
        ),
    )


def test_dynamic_does_not_import_scanner_or_execution_boundaries() -> None:
    _assert_no_forbidden_imports(
        "dynamic",
        (
            "qooi.scanner",
            "qooi.core.basket",
            "qooi.core.executor",
            "qooi.core.recovery",
            "qooi.exchange",
            "qooi.sources",
            "qooi.strategies",
        ),
    )


def test_sources_do_not_import_research_scanner_strategy_or_execution() -> None:
    _assert_no_forbidden_imports(
        "sources",
        (
            "qooi.scanner",
            "qooi.research",
            "qooi.strategies",
            "qooi.core.basket",
            "qooi.core.executor",
            "qooi.core.recovery",
            "qooi.exchange.trading",
            "qooi.dynamic",
        ),
    )


def test_strategies_do_not_import_io_scanner_or_execution_boundaries() -> None:
    _assert_no_forbidden_imports(
        "strategies",
        (
            "qooi.scanner",
            "qooi.sources",
            "qooi.dynamic",
            "qooi.exchange",
            "qooi.core.basket",
            "qooi.core.executor",
            "qooi.core.recovery",
        ),
    )
