from __future__ import annotations

import ast
import importlib
from pathlib import Path

import polars as pl

import qooi.scanner as scanner
from qooi.scanner import frames, ladder

SCANNER_ROOT = Path(__file__).resolve().parents[1] / "src" / "qooi" / "scanner"


def test_scanner_contracts_live_at_package_root() -> None:
    assert scanner.ReportInputs.__module__ == "qooi.scanner"
    assert scanner.PotentialArtifacts.__module__ == "qooi.scanner"
    assert scanner.SourceStateRow.__module__ == "qooi.scanner"
    assert scanner.TransitionPattern.__module__ == "qooi.scanner"


def test_scanner_legacy_compatibility_modules_removed() -> None:
    for module_name in (
        "qooi.scanner.contracts",
        "qooi.scanner.evidence",
        "qooi.scanner.candidates",
    ):
        assert importlib.util.find_spec(module_name) is None


def test_resolved_scanner_modules_are_importable() -> None:
    for module_name in ["frames", "ladder", "tailrun", "rank", "events"]:
        module = importlib.import_module(f"qooi.scanner.{module_name}")
        assert module.__name__ == f"qooi.scanner.{module_name}"


def test_frames_and_ladder_own_public_functions() -> None:
    assert frames.potential_observation_frame.__module__ == "qooi.scanner.frames"
    assert frames.potential_outcome_frame.__module__ == "qooi.scanner.frames"
    assert ladder.potential_evidence_frame.__module__ == "qooi.scanner.ladder"
    assert ladder.add_potential_parent_gain.__module__ == "qooi.scanner.ladder"
    assert ladder.select_potential_evidence_level.__module__ == "qooi.scanner.ladder"


def test_rank_module_owns_candidate_ranking_entrypoint() -> None:
    rank = importlib.import_module("qooi.scanner.rank")
    frame = pl.DataFrame(
        {
            "symbol": ["AAA"],
            "evidence_level": ["symbol_state"],
            "probability_lift": [2.0],
            "information_gain_bits": [0.5],
            "candidate_score": [1.0],
            "N": [30],
            "selected_evidence_level": [True],
        }
    )

    expected = rank.rank_candidate_evidence(frame)
    actual = rank.rank_candidates(frame)

    assert actual.to_dicts() == expected.to_dicts()


def test_scanner_modules_do_not_import_strategies() -> None:
    offenders: list[str] = []
    for path in SCANNER_ROOT.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.ImportFrom)
                and node.module
                and node.module.startswith("qooi.strategies")
            ):
                offenders.append(f"{path.name}:{node.lineno}:{node.module}")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("qooi.strategies"):
                        offenders.append(f"{path.name}:{node.lineno}:{alias.name}")

    assert offenders == []


def test_diagnostics_imports_resolved_graph_boundaries() -> None:
    source = (SCANNER_ROOT / "diagnostics.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported_aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "qooi.scanner":
            for alias in node.names:
                imported_aliases[alias.asname or alias.name] = alias.name

    assert imported_aliases["candidate_eval"] == "rank"
    assert imported_aliases["source_eval"] == "events"
    assert imported_aliases["ladder_eval"] == "ladder"
    assert imported_aliases["rank_eval"] == "rank"
    assert imported_aliases["tailrun_eval"] == "tailrun"


def test_resolved_modules_do_not_reexport_old_owners() -> None:
    forbidden = {
        "frames.py": ("qooi.scanner.evidence",),
        "ladder.py": ("qooi.scanner.evidence",),
        "rank.py": ("qooi.scanner.candidates",),
        "tailrun.py": ("qooi.scanner.diagnostics",),
    }
    offenders: list[str] = []
    for filename, forbidden_modules in forbidden.items():
        source = (SCANNER_ROOT / filename).read_text(encoding="utf-8")
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module in forbidden_modules:
                offenders.append(f"{filename}:{node.lineno}:{node.module}")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in forbidden_modules:
                        offenders.append(f"{filename}:{node.lineno}:{alias.name}")
    assert offenders == []


def test_path_result_contracts_live_with_path_modules() -> None:
    ladder = importlib.import_module("qooi.scanner.ladder")
    tailrun = importlib.import_module("qooi.scanner.tailrun")
    diagnostics_source = (SCANNER_ROOT / "diagnostics.py").read_text(encoding="utf-8")
    diagnostics_tree = ast.parse(diagnostics_source)
    diagnostics_classes = {
        node.name for node in diagnostics_tree.body if isinstance(node, ast.ClassDef)
    }

    assert hasattr(ladder, "LadderResult")
    assert hasattr(tailrun, "TailtreeResult")
    assert hasattr(tailrun, "TailtreeEvidenceResult")
    assert "LadderResult" not in diagnostics_classes
    assert "TailtreeResult" not in diagnostics_classes
    assert "TailtreeEvidenceResult" not in diagnostics_classes
