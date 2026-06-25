from pathlib import Path

import polars as pl
import pytest
from pydantic import ValidationError

from qooi.pipeline.load import LoadStats
from qooi.scanner.config import TailtreeConfig
from qooi.scanner.output import MarketReadiness, ReviewDecision, ScannerRunFrames, render_report
from qooi.scanner.tailrun.core import _local_model_ref


def test_predict_config_requires_complete_predict_profile() -> None:
    with pytest.raises(ValidationError, match="predict_profile"):
        TailtreeConfig.model_validate(
            {
                "lifecycle": "load_predict",
                "profiles": [],
                "models": [
                {"model_id": "tailtree-event-lift-current-frontier-t0001-f02_24_up"},
                {"model_id": "tailtree-event-lift-current-frontier-t0001-f02_24_down"},
                ],
            }
        )


def test_predict_profile_requires_matching_up_down_model_refs() -> None:
    with pytest.raises(ValidationError, match="opportunity_model_ids must match models"):
        TailtreeConfig.model_validate(
            {
                "lifecycle": "load_predict",
                "profiles": [],
                "models": [
                {"model_id": "tailtree-event-lift-current-frontier-t0001-f02_24_up"},
                {"model_id": "tailtree-event-lift-current-frontier-t0001-f02_24_down"},
                ],
                "predict_profile": {
                "profile_id": "tailtree-event-lift-current-frontier-t0001-f02",
                "horizon": 24,
                "opportunity_model_ids": [
                    "tailtree-event-lift-current-frontier-t0001-f02_24_up",
                    "tailtree-event-lift-current-frontier-t0001-f99_24_down",
                ],
                "candidate_model_roles": [
                    "promoter",
                    "opposite_guard",
                    "weak_path_guard",
                ],
                },
            }
        )


def test_old_selection_pipe_config_is_rejected() -> None:
    with pytest.raises(ValidationError, match="selection_pipe"):
        TailtreeConfig.model_validate(
            {
                "profiles": [
                {
                    "profile_id": "p",
                    "model_tag": "m",
                    "objective": "tail_event_lift",
                    "selection_pipe": {"pipe_id": "candidate_dual_guard"},
                }
                ]
            }
        )


def test_local_model_ref_path_is_deterministic(tmp_path: Path) -> None:
    ref = _local_model_ref(
        model_dir=tmp_path,
        parent_model_id="tailtree-event-lift-current-frontier-t0001-f02_24_up",
        role="opposite_guard",
        gate_id="score_pct_0.01",
    )

    assert ref.model_id == (
        "tailtree-event-lift-current-frontier-t0001-f02_24_up_"
        "opposite_guard_score_pct_0_01"
    )
    assert ref.model_path == tmp_path / f"{ref.model_id}.json"


def test_report_does_not_render_selection_pipe_section() -> None:
    frames = ScannerRunFrames(
        market=MarketReadiness(
            symbols=0,
            timeframes=0,
            target_days=0,
            source_products=0,
            before={},
            after={},
            stats=LoadStats(),
        ),
        products={},
        states={},
        transitions=None,
        histories=pl.DataFrame(),
        source_events=pl.DataFrame(),
        ladder=pl.DataFrame(),
        tailtree=pl.DataFrame(),
        ranked=pl.DataFrame(),
        horizon_consistency=pl.DataFrame(),
        action_surface=pl.DataFrame(),
        prediction_freshness=pl.DataFrame(),
        decisions=[
            ReviewDecision(
                symbol="ABC-USDT-SWAP",
                direction="up",
                horizon=24,
                action="watch",
                score=1.0,
                reason="candidate_dual_guard candidate",
            )
        ],
    )

    # Render with the shipped config to keep the test on the real report boundary.
    from qooi.scanner.workflow import load_config

    report = render_report(frames, load_config(Path("configs/potential-tailtree-train.toml")))

    assert "## Tailtree Model Selection Pipe" not in report
    assert "## Candidate Board" in report
