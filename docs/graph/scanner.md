# Scanner Graph

Implementation-facing graph for the current scanner code.

## CLI

```text
scripts/scanner_potential.py
  -> qooi.scanner.workflow.run(config_path: Path | str) -> Path
```

## Config

```text
qooi.scanner.workflow.load_config(path: Path) -> PotentialConfig
qooi.scanner.config.PotentialConfig
qooi.scanner.config.TailtreeConfig
qooi.scanner.config.TailtreeProfileConfig
```

Shipped scanner configs:

```text
configs/potential-daily-tailtree.toml
configs/potential-advanced-tailtree.toml
configs/potential-smoke.toml
configs/potential-optuna-direct-smoke.toml
```

## Market load boundary

```text
qooi.scanner.workflow.scanner_market_request(config, symbols) -> MarketLoadRequest
qooi.scanner.workflow.scanner_market_policy(config) -> MarketLoadPolicy
qooi.pipeline.load.load_market(request, policy, client) -> market result
```

`OkxClient` owns the exchange connection. Scanner builds requests; it does not own transport details.

## Workflow pipe

```text
qooi.scanner.workflow.run(config_path)
  -> load_config
  -> rank_discovery/select_symbols
  -> load_market
  -> state.classify_states
  -> state.extract_continuous_features
  -> state.potential_observation_frame
  -> outcome.path_histories
  -> outcome.realized_transition_frame
  -> outcome.source_outcomes_frame
  -> outcome.potential_outcome_frame
  -> tailrun.core.run_tailtree or ladder.evidence
  -> rank.tailtree_candidates / rank.ladder_candidates
  -> rank.candidate_metric_surface
  -> rank.rank_candidates
  -> workflow.prediction_freshness_frame
  -> output.review_decisions
  -> output.render_report
  -> profile.write
```

## State product

```text
qooi.scanner.state.classify_states(bars, timeframe) -> dict[str, pl.DataFrame]
qooi.scanner.state.extract_continuous_features(bars, state_frames, source_frames, decision_timeframe=...) -> pl.DataFrame
qooi.scanner.state.potential_observation_frame(...) -> pl.DataFrame
```

Output grain:

```text
symbol × decision_timeframe × decision_bar_close_ms
```

State owns known-at-close values only.

## Outcome product

```text
qooi.scanner.outcome.path_histories(...)
qooi.scanner.outcome.realized_transition_frame(...)
qooi.scanner.outcome.source_outcomes_frame(...)
qooi.scanner.outcome.potential_outcome_frame(...)
```

Output grain:

```text
symbol × bar_close_ms × outcome_horizon
```

Outcome owns future/path labels and source-outcome rows.

## Evidence dispatch

Ladder path:

```text
qooi.scanner.ladder.evidence(observations, outcomes, config) -> LadderResult
```

Tailtree path:

```text
qooi.scanner.tailrun.core.run_tailtree(
    TailtreeInputFrames(observations, source_outcomes, realized, histories),
    config=config,
    profile=profile,
) -> TailtreeRunOutput
```

`workflow.py` does not train models directly.

## Rank/review/report

```text
qooi.scanner.rank.tailtree_candidates(observations, evidence, models, latest_only=True) -> pl.DataFrame
qooi.scanner.rank.candidate_metric_surface(ladder=..., tailtree=...) -> pl.DataFrame
qooi.scanner.rank.rank_candidates(surface) -> pl.DataFrame
qooi.scanner.workflow.prediction_freshness_frame(ranked, config) -> pl.DataFrame
qooi.scanner.output.review_decisions(ranked, freshness, source_health, config) -> list[ReviewDecision]
qooi.scanner.output.render_report(...) -> str
```

Review gates:

```text
fresh prediction
no missing required source
support >= config.evidence.tailtree.selection.min_selected_observation_count
tail_lift >= config.evidence.tailtree.selection.min_valid_tail_lift
weaker opposite direction -> watch
promote cap from top_k
```

## Artifacts

Main output directory:

```text
data/output/potential/<run>/
```

Current artifacts:

```text
report.md
tailtree-profile-runs.csv
tailtree-selection-efficiency.csv
tailtree-action-surface.csv
models/*.json
models/tailtree-selection-efficiency.csv
profile/stages.csv
profile/frames.csv
profile/summary.md
```

## Current public scanner modules

```text
qooi.scanner.config
qooi.scanner.workflow
qooi.scanner.state
qooi.scanner.outcome
qooi.scanner.ladder
qooi.scanner.rank
qooi.scanner.output
qooi.scanner.transitions
qooi.scanner.tailtree.model
qooi.scanner.tailtree.evidence
qooi.scanner.tailrun.types
qooi.scanner.tailrun.core
qooi.scanner.tailrun.planning
qooi.scanner.tailrun.search
qooi.scanner.tailrun.selection
qooi.scanner.tailrun.artifacts
```

Removed/non-current module names must not be used as graph authority.
