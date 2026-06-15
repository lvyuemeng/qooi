from __future__ import annotations

import ast
import importlib
import importlib.util
from pathlib import Path

import qooi.scanner as scanner
from qooi.scanner import ladder, outcome, state

SCANNER_ROOT = Path(__file__).resolve().parents[1] / "src" / "qooi" / "scanner"


def test_scanner_contracts_live_at_package_root() -> None:
    assert scanner.ReportInputs.__module__ == "qooi.scanner"
    assert scanner.PotentialArtifacts.__module__ == "qooi.scanner"
    assert scanner.SourceStateRow.__module__ == "qooi.scanner"
    assert scanner.TransitionPattern.__module__ == "qooi.scanner"


def test_scanner_legacy_compatibility_modules_removed() -> None:
    for module_name in (
        "qooi.scanner.classifiers",
        "qooi.scanner.contracts",
        "qooi.scanner.decisions",
        "qooi.scanner.evidence",
        "qooi.scanner.candidates",
        "qooi.scanner.events",
        "qooi.scanner.features",
        "qooi.scanner.history",
        "qooi.scanner.source_events",
    ):
        assert importlib.util.find_spec(module_name) is None


def test_resolved_scanner_modules_are_importable() -> None:
    module_names = [
        "ladder",
        "tailrun",
        "rank",
        "feasibility",
        "state",
        "outcome",
    ]
    for module_name in module_names:
        module = importlib.import_module(f"qooi.scanner.{module_name}")
        assert module.__name__ == f"qooi.scanner.{module_name}"


def test_state_outcome_and_ladder_own_public_functions() -> None:
    assert state.KlineClassifier.__module__ == "qooi.scanner.state"
    assert state.extract_continuous_features.__module__ == "qooi.scanner.state"
    assert state.potential_observation_frame.__module__ == "qooi.scanner.state"
    assert outcome.kline_path_history_frame.__module__ == "qooi.scanner.outcome"
    assert outcome.potential_outcome_frame.__module__ == "qooi.scanner.outcome"
    assert outcome.realized_transition_frame.__module__ == "qooi.scanner.outcome"
    assert outcome.source_events_frame.__module__ == "qooi.scanner.outcome"
    assert outcome.source_outcomes_frame.__module__ == "qooi.scanner.outcome"
    assert outcome.source_timeliness_frame.__module__ == "qooi.scanner.outcome"
    assert outcome.source_state_predictability_frame.__module__ == "qooi.scanner.outcome"
    assert ladder.potential_evidence_frame.__module__ == "qooi.scanner.ladder"
    assert ladder.add_potential_parent_gain.__module__ == "qooi.scanner.ladder"
    assert ladder.select_potential_evidence_level.__module__ == "qooi.scanner.ladder"
    assert not hasattr(state, "ClassifierDiagnosticsBuilder")
    assert not hasattr(state, "evaluate_classifier_frame")
    assert not hasattr(importlib.import_module("qooi.scanner.rank"), "rank_candidates")


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
    assert imported_aliases["feasibility_eval"] == "feasibility"
    assert imported_aliases["state_eval"] == "state"
    assert imported_aliases["outcome_eval"] == "outcome"
    assert imported_aliases["ladder_eval"] == "ladder"
    assert imported_aliases["rank_eval"] == "rank"
    assert imported_aliases["tailrun_eval"] == "tailrun"


def test_resolved_modules_do_not_reexport_old_owners() -> None:
    forbidden = {
        "ladder.py": ("qooi.scanner.evidence",),
        "rank.py": ("qooi.scanner.candidates",),
        "tailrun/__init__.py": ("qooi.scanner.diagnostics",),
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


def test_workflow_does_not_own_scanner_config_models() -> None:
    source = (SCANNER_ROOT / "workflow.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    workflow_classes = {node.name for node in tree.body if isinstance(node, ast.ClassDef)}

    assert "PotentialConfig" not in workflow_classes
    assert "TailtreeConfig" not in workflow_classes
    assert "BaseModel" not in source
    assert "ConfigDict" not in source


def test_scanner_config_is_composed_by_domain_sections() -> None:
    config_module = importlib.import_module("qooi.scanner.config")
    config = config_module.PotentialConfig.model_validate(
        {
            "transition": {"horizon": 8, "scan_budget": 13},
            "evidence": {"kind": "tailtree", "tailtree": {"model_tag": "unit"}},
        }
    )

    assert config.transition.horizon == 8
    assert config.transition.scan_budget == 13
    assert config.evidence.kind == "tailtree"
    assert config.evidence.tailtree.model_tag == "unit"

    fields = set(config_module.PotentialConfig.model_fields)
    assert "transition_horizon" not in fields
    assert "transition_scan_budget" not in fields
    assert "tail_tree_num_leaves" not in fields
    assert "tail_threshold_pct" not in fields
