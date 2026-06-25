# qooi Tailtree Scanner Research Introduction

## Abstract

`qooi` is a research workspace for finding altcoin perpetual-swap symbols whose known-at-close market state materially changes the distribution of future extreme behavior. The current scanner research path is the **tailtree** model: a LightGBM-based tail-event evidence system with a two-stage candidate-selection layer. It is designed to answer a narrow research question:

> Given only information known at a bar close, which symbols enter a state where future 24-hour extreme behavior becomes unusually concentrated, and which of those opportunities survive behavior-aware promotion and guard filters?

The scanner is not a live trading system. Its outputs are diagnostics, candidate boards, action surfaces, and benchmark artifacts for research review. No scanner artifact by itself authorizes allocation, execution, or strategy promotion.

## 1. Why

### 1.1 Research problem

Altcoin perpetual markets are sparse, regime-sensitive, and heavily path-dependent. A useful scanner cannot only ask whether a future return is positive or negative. It must ask whether the current state increases the chance of **material future tail behavior** under a bounded horizon, then reduce false or unsafe candidate promotion.

The active qooi scanner is therefore built around three principles:

1. **Known-at-close evidence only**  
   Features are built from observations available at the decision bar close. Future returns and path labels are outcome columns used only for training and evaluation.

2. **Extreme-event opportunity over smooth prediction**  
   The target is not a smooth market forecast. The scanner searches for concentrated future tail behavior: large forward excursions, path states, and opportunity proxies.

3. **Candidate safety over recall**  
   The current tailtree selection prioritizes avoiding false positives and wrong-direction candidates over maximizing recall. A missed candidate is acceptable; a promoted wrong candidate is more costly to the research loop.

### 1.2 Why tailtree

The tailtree path exists because fixed ladder rules and raw rank surfaces are not expressive enough to discover sparse state-dependent extreme behavior. Tailtree uses tree-based partitions to learn where future tail behavior concentrates, then reports those partitions as evidence and candidate-selection diagnostics.

The model is useful when it can answer:

```text
state/context at bar close
  -> future h24 tail-event concentration
  -> behavior-aware candidate promotion or rejection
```

It is not intended to solve:

```text
execution timing
position sizing
liquidity/cost/slippage modeling
wallet/account decisions
live trading authorization
```

Those remain downstream strategy/execution concerns.

## 2. What

### 2.1 Current scanner object

The current research object is the **h24 source-context candidate-dual-guard tailtree scanner**.

Current frontier:

```text
horizon: h24
stage-1 objective: tail_event_lift
final selection objective: candidate_dual_guard
validation: walkforward
search: Optuna
model family: LightGBM tree score buckets
source inputs: persistent known-at-close market/source state features
```

The public configs are reduced to two scanner workflows:

```text
configs/potential-tailtree-train.toml    # train + scan + persist model graph
configs/potential-tailtree-predict.toml  # load existing model graph + score fresh/current data
```

### 2.2 Two-stage model graph

The current model is best described as a two-stage graph.

#### Stage 1 — Opportunity evidence

Stage 1 trains or loads directional `tail_event_lift` models:

```text
opportunity_up
opportunity_down
```

These models find score buckets where future h24 tail events are concentrated. Current predict model ids:

```text
tailtree-event-lift-current-frontier-t0001-f02_24_up
tailtree-event-lift-current-frontier-t0001-f02_24_down
```

Stage-1 output contributes:

```text
tailtree_score
tail_lift
score-bucket support
candidate gates
```

#### Stage 2 — Candidate-local selection and guards

Stage 2 is a candidate-local model family trained per candidate gate:

```text
promoter
opposite_guard
weak_path_guard
```

The role of each model:

| Role | Purpose | Score column |
|---|---|---|
| `promoter` | promote candidates that remain useful after stage-1 gating | `promotion_score` |
| `opposite_guard` | reject candidates with wrong/opposite-side behavior | `opposite_guard_score` |
| `weak_path_guard` | reject weak/no-tail path behavior | `weak_path_guard_score` |

The final emitted objective is:

```text
candidate_dual_guard
```

This means the candidate passed the promoter and survived the opposite-direction and weak-path guard filters. The internal guard roles are not reported as competing final objectives.

### 2.3 Predict invariant

The predict config must not train models.

Current predict invariant:

```text
configs/potential-tailtree-predict.toml
  -> lifecycle = "load_predict"
  -> load existing opportunity up/down JSON models
  -> load existing candidate-local promoter/opposite_guard/weak_path_guard JSON models
  -> score fresh/current observations
  -> emit candidate_dual_guard candidate selection
```

If a required candidate-local model JSON is missing, predict fails loudly instead of silently falling back to stage-1-only scoring.

Current predict profile:

```toml
[potential.evidence.tailtree.predict_profile]
profile_id = "tailtree-event-lift-current-frontier-t0001-f02"
horizon = 24
opportunity_model_ids = [
  "tailtree-event-lift-current-frontier-t0001-f02_24_up",
  "tailtree-event-lift-current-frontier-t0001-f02_24_down",
]
candidate_model_roles = ["promoter", "opposite_guard", "weak_path_guard"]
candidate_model_side = "up"
```

## 3. How

### 3.1 Data construction

The scanner separates feature construction from outcome construction.

Known-at-close state:

```text
scanner.state
  -> classify_states
  -> extract_continuous_features
  -> potential_observation_frame
```

Future/path outcomes:

```text
scanner.outcome
  -> realized_transition_frame
  -> source_outcomes_frame
  -> potential_outcome_frame
```

Tailtree consumes:

```text
TailtreeInputFrames(
  observations,
  source_outcomes,
  realized,
  histories,
)
```

No future outcome column may enter known-at-close feature construction.

### 3.2 Feature family

The active tailtree feature set includes 17 persistent source-context columns. These are kline-like state/path/context features from funding, long-short ratio, open interest, and taker-pressure families.

Current source-context inputs:

```text
funding_level_state
funding_level_transition
funding_price_divergence_24h
funding_direction_run_length
lsr_level_state
lsr_level_transition
lsr_price_divergence_24h
lsr_direction_run_length
lsr_log_ratio_change_24h
oi_flow_state
oi_flow_transition
oi_flow_run_length
oi_change_pct_24h
taker_pressure_state
taker_pressure_transition
taker_pressure_run_length
taker_buy_pressure_24h_mean
```

Excluded from training:

```text
high-cardinality source path strings
current-only books/trades
execution/cost/slippage/wallet fields
```

The design avoids z-score/mean-reversion feature priors. Source features are treated as state, transition, run length, path, and divergence context.

### 3.3 Labels and path semantics

Tailtree labels summarize future h24 path behavior. Core semantics:

```text
tail_touch_up      = forward_max_return_pct > threshold_pct
tail_touch_down    = forward_min_return_pct < -threshold_pct
tail_touch_both    = both sides touched inside the same horizon
first_touch_side   = up | down | tie | none
path_state         = none | clean_up | clean_down | up_first_both | down_first_both | chop_both | late_up | late_down
path_actionability = tradable_up | tradable_down | reversal_watch | gray_zone | no_action
```

The active stage-1 objective, `tail_event_lift`, keeps a broad extreme-event opportunity target. Pure clean/actionable labels and standalone guard objectives were tested and did not replace the current frontier.

### 3.4 Training workflow

The train workflow is:

```text
load market/source data
  -> build known-at-close observations
  -> build future h24 outcome/path labels
  -> train tail_event_lift opportunity up/down models
  -> score observations into candidate gates
  -> train candidate-local promoter/opposite_guard/weak_path_guard models per gate
  -> emit candidate_dual_guard selection-efficiency rows
  -> persist stage-1 and candidate-local JSON models
  -> write Candidate Board, Action Surface, and benchmark artifacts
```

Candidate-local model artifacts are gate-specific. Current naming rule:

```text
models/<parent_model_id>_promoter_<gate_slug>.json
models/<parent_model_id>_opposite_guard_<gate_slug>.json
models/<parent_model_id>_weak_path_guard_<gate_slug>.json
```

### 3.5 Predict workflow

The predict workflow is:

```text
load fresh/current data
  -> build known-at-close observations
  -> load opportunity up/down JSON models
  -> score stage-1 tail_event_lift evidence
  -> construct the same candidate gates
  -> load gate-specific candidate-local JSON models
  -> compute promotion_score, opposite_guard_score, weak_path_guard_score
  -> emit candidate_dual_guard candidates
  -> write Candidate Board, Action Surface, and benchmark artifacts
```

Predict does not train either stage.

## 4. Objective and parameter choices

### 4.1 Current objective choice

The current active objective path is:

```text
tail_event_lift -> candidate_dual_guard
```

Reason:

```text
tail_event_lift keeps broad extreme-opportunity recall at stage 1;
candidate_dual_guard applies behavior-aware candidate promotion and guards;
pure clean/actionable objectives were too sparse or benchmarked worse;
standalone guard objectives were useful as filters, not final competing objectives;
source-context input improved the frontier and was folded into the single active feature set.
```

Removed or inactive objective surfaces include:

```text
source_blended
candidate_conditional_promoter
candidate_opposite_guard
continuous_guard_curve
two_model_guard
candidate_dual_guard_source_blended suffix variants
```

The suffix/source branch was normalized away by making source-context columns part of the active feature set.

### 4.2 Horizon and threshold policy

Current public configs use:

```text
horizon: 24h
threshold policy: hybrid material tail policy
material floor: 20% in current train/predict configs
quantile: 0.95
reference scope: universe_horizon
```

The scanner currently prefers h24 because daily prediction freshness often approaches a 24-hour decision boundary, and h24 has stronger tail-label support than shorter horizons in the current data surface.

### 4.3 Search and validation

Current train workflow uses:

```text
walkforward validation
Optuna search
selection-efficiency feedback
frontier benchmark feedback
```

The selection-efficiency artifact is the canonical opportunity-selection feedback surface. It is not realized PnL.

Core feedback columns include:

```text
objective
outcome_horizon
tree_direction
budget_family
budget_value
selected_observation_count
selected_tail_count
valid_tail_lift
selected_utility_mean
profit_proxy_per_1k_observed
hpo_score
behavior_hpo_score
paired_behavior_false_direction_rate
paired_behavior_utility_margin_mean
promotion_threshold_pass_int
fit_seconds
```

## 5. Benchmarks

### 5.1 Latest verified train smoke

Latest verified train run:

```text
uv run python -m scripts.scanner_potential --config configs/potential-tailtree-train.toml
```

Output report:

```text
data/output/potential/tailtree-train/report.md
```

Train artifacts:

```text
tailtree-selection-efficiency.csv shape: (3504, 78)
selection objectives: candidate_dual_guard=3456, tail_event_lift=48

tailtree-frontier-benchmark.csv shape: (2119, 86)
frontier objective: candidate_dual_guard only

candidate-local model JSONs:
promoter=36
opposite_guard=36
weak_path_guard=36
```

Top inspected frontier rows:

| objective | selected | precision | false-dir | utility | action |
|---|---:|---:|---:|---:|---|
| candidate_dual_guard | 50 | 0.600 | 0.340 | 5.297 | promote_candidate_frontier |
| candidate_dual_guard | 50 | 0.580 | 0.120 | 3.652 | promote_candidate_frontier |
| candidate_dual_guard | 50 | 0.580 | 0.160 | 3.574 | promote_candidate_frontier |
| candidate_dual_guard | 50 | 0.560 | 0.120 | 3.551 | promote_candidate_frontier |

### 5.2 Latest verified predict smoke

Latest verified predict run:

```text
uv run python -m scripts.scanner_potential --config configs/potential-tailtree-predict.toml
```

Output report:

```text
data/output/potential/tailtree-predict/report.md
```

Predict artifacts:

```text
tailtree-selection-efficiency.csv shape: (584, 78)
selection objectives: candidate_dual_guard=576, tail_event_lift=8

tailtree-frontier-benchmark.csv shape: (392, 86)
frontier objective: candidate_dual_guard only

tailtree-action-surface.csv exists
```

The predict run proves the current invariant:

```text
load existing two-stage model graph
score current data
emit candidate_dual_guard
```

### 5.3 Where to read candidate output

Human-facing current candidate report:

```text
data/output/potential/tailtree-predict/report.md
```

Primary sections:

```text
## Candidate Board
## Tailtree Action Surface
```

Dense diagnostic artifacts:

```text
data/output/potential/tailtree-predict/tailtree-action-surface.csv
data/output/potential/tailtree-predict/tailtree-selection-efficiency.csv
data/output/potential/tailtree-predict/tailtree-frontier-benchmark.csv
```

The frontier benchmark and the candidate report have different grains:

```text
tailtree-frontier-benchmark.csv
  = historical/evaluation feedback proving the candidate_dual_guard surface under fixed budgets

report.md Candidate Board
  = fresh/current symbols selected by the loaded two-stage candidate_dual_guard graph
```

The Candidate Board follows the benchmarked `candidate_dual_guard` family, then applies report-time top-k, freshness, source-health, conflict, and action-side gates. It is not a literal copy of the top frontier benchmark rows.

## 6. Interpretation

### 6.1 What a candidate means

A promoted candidate means:

```text
known-at-close state matched high tail_event_lift evidence;
the candidate-local promoter accepted it;
opposite-direction guard did not reject it;
weak/no-tail path guard did not reject it;
it survived current report action-gating.
```

It does not mean:

```text
buy now;
size a position;
ignore liquidity/cost/slippage;
strategy is production-ready;
future return is guaranteed.
```

### 6.2 Why reports focus on Candidate Board and Action Surface

The report intentionally does not include a separate model-pipe section. The useful research output is candidate selection, not a restatement of the pipeline. The pipeline is documented in architecture and graph docs; the report shows the current result surface.

Use:

```text
Candidate Board       # human-facing promoted/watch/skipped symbols
Tailtree Action Surface # dense candidate/action diagnostics
Model Evidence Appendix # model/profile evidence summary
```

## 7. Current limitations

1. **Research-only status**  
   Tailtree output is not allocation-ready and must not be treated as a live trading signal.

2. **h24 current frontier**  
   The current public frontier is h24-specific. Multi-horizon selection is not the default workflow.

3. **Candidate-local up-side focus**  
   Current predict profile uses `candidate_model_side = "up"`. Down-side behavior is used as market-state/opposite-risk context unless a future short-side workflow is explicitly promoted.

4. **Source coverage remains a gate**  
   Missing, stale, provider-bounded, or shallow source data must remain explicit in diagnostics.

5. **No execution model inside scanner promotion**  
   Liquidity, cost, slippage, funding carry, sizing, and execution are downstream concerns. They are not subtracted inside the scanner objective.

## 8. Documentation map

Durable context:

```text
docs/context.md
```

Scanner architecture:

```text
docs/architecture/scanner.md
```

Tailtree implementation graph:

```text
docs/graph/tailtree.md
```

Current generated candidate report:

```text
data/output/potential/tailtree-predict/report.md
```
