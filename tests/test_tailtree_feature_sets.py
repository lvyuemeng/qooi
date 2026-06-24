import polars as pl

from qooi.scanner.tailrun.core import train_features


def test_candidate_features_add_qualification_inputs() -> None:
    observations = pl.DataFrame(
        {
            "background_regime": ["trend"],
            "decision_core": ["markup"],
            "atr_percentile": [80.0],
            "bar_return_24h_pct": [3.0],
            "return_efficiency_24h": [0.7],
            "market_dispersion_24h": [4.2],
            "symbol_vs_market_return_24h": [1.1],
        }
    )

    opportunity = train_features(observations, role="opportunity")
    candidate = train_features(observations, role="candidate")

    assert opportunity.categorical == candidate.categorical
    assert "bar_return_24h_pct" in opportunity.continuous
    assert "return_efficiency_24h" not in opportunity.continuous
    assert "market_dispersion_24h" not in opportunity.continuous
    assert "symbol_vs_market_return_24h" not in opportunity.continuous

    assert "return_efficiency_24h" in candidate.continuous
    assert "market_dispersion_24h" in candidate.continuous
    assert "symbol_vs_market_return_24h" in candidate.continuous


def test_train_features_include_source_context_and_exclude_path_strings() -> None:
    observations = pl.DataFrame(
        {
            "background_regime": ["trend"],
            "decision_core": ["markup"],
            "atr_percentile": [80.0],
            "funding_level_state": ["funding_positive"],
            "funding_level_transition": ["funding_rising"],
            "funding_price_divergence_24h": ["funding_price_agree"],
            "lsr_level_state": ["lsr_short_crowding"],
            "lsr_level_transition": ["lsr_crowding_rising"],
            "lsr_price_divergence_24h": ["lsr_price_diverge"],
            "oi_flow_state": ["oi_build"],
            "oi_flow_transition": ["oi_building"],
            "taker_pressure_state": ["taker_balanced"],
            "taker_pressure_transition": ["taker_pressure_rising"],
            "funding_direction_run_length": [12.0],
            "lsr_direction_run_length": [8.0],
            "lsr_log_ratio_change_24h": [0.1],
            "oi_flow_run_length": [10.0],
            "oi_change_pct_24h": [2.1],
            "taker_pressure_run_length": [5.0],
            "taker_buy_pressure_24h_mean": [0.55],
            "funding_path_24h": ["high_card_path"],
            "lsr_path_24h": ["high_card_path"],
            "oi_price_flow_path_24h": ["high_card_path"],
            "taker_pressure_path_24h": ["high_card_path"],
        }
    )

    features = train_features(observations, role="opportunity")

    assert "funding_level_state" in features.categorical
    assert "lsr_level_state" in features.categorical
    assert "oi_flow_state" in features.categorical
    assert "taker_pressure_state" in features.categorical
    assert "funding_direction_run_length" in features.continuous
    assert "lsr_log_ratio_change_24h" in features.continuous
    assert "oi_change_pct_24h" in features.continuous
    assert "taker_buy_pressure_24h_mean" in features.continuous

    assert "funding_path_24h" not in features.categorical
    assert "lsr_path_24h" not in features.categorical
    assert "oi_price_flow_path_24h" not in features.categorical
    assert "taker_pressure_path_24h" not in features.categorical
