from qooi.scanner.tailrun.types import TailtreeSelectionEfficiencyRow


def test_tailtree_selection_efficiency_row_uses_model_dump() -> None:
    row = TailtreeSelectionEfficiencyRow(
        universe_snapshot_id="active",
        model_tag="tailtree-event-lift",
        objective="tail_event_lift",
        training_profile="advanced",
        trial_id="trial-1",
        trial_source="fixed",
        outcome_horizon=24,
        tree_direction="up",
        budget_family="score_bucket",
        budget_value=1.0,
        eligible_symbol_count=10,
        selected_symbol_count=3,
        observation_row_count=100,
        feature_count=20,
        train_exceedance_count=5,
        valid_observation_count=100,
        valid_tail_count=5,
        valid_tail_rate=0.05,
        selected_observation_count=20,
        selected_observation_rate=0.2,
        selected_tail_count=3,
        selected_tail_rate=0.15,
        selected_tail_per_1k_obs=150.0,
        valid_tail_lift=3.0,
        selected_profit_proxy_mean=2.0,
        selected_profit_proxy_p90=4.0,
        selected_utility_mean=2.0,
        selected_utility_p90=4.0,
        profit_proxy_per_selected_obs=2.0,
        profit_proxy_per_1k_observed=20.0,
        hpo_score=5.0,
        promotion_threshold_pass_int=1,
        trained_tree_count=2,
        selected_bucket_or_leaf_count=1,
        fit_seconds=1.5,
        score_seconds=0.25,
    )

    dumped = row.model_dump()

    assert dumped["objective"] == "tail_event_lift"
    assert dumped["selected_observation_count"] == 20
    assert dumped["profit_proxy_per_1k_observed"] == 20.0
