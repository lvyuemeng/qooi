# Tailtree Module Graph

`qooi.scanner.tailtree` is the public LightGBM + GPD extreme-value evidence API. It consumes scanner observation/outcome frames and emits tree models plus numeric evidence products.

Implementation layout:

```text
qooi.scanner.tailtree             # package export surface, no implementation sprawl
qooi.scanner.tailtree.model       # labels, training frame, LightGBM/GPD model
qooi.scanner.tailtree.evidence    # leaf and score-bucket evidence products
```

Callers should import from `qooi.scanner.tailtree`; graph docs may name the owner module
when documenting internals. Public compatibility is preserved while implementation code
stays grouped by product rather than by `_tailtree_*` prefixes.

Design doc: `docs/architecture/scanner.md`.

---

## Dependency contract

Optional dependencies only for this path:

```toml
[project.optional-dependencies]
tailtree = [
  "lightgbm>=4.0",
  "scipy>=1.11",
]
```

Rules:

- no pandas/pyarrow dependency is required for the tailtree path;
- use Polars → numpy matrices for LightGBM input;
- pydantic validates config/metadata serialization boundaries;
- model persistence is JSON: `booster.model_to_string()` + metadata JSON.

---

## Statistical role of GPD

GPD models threshold exceedance severity only.

```text
up tail:   forward_max_return_pct >= threshold_pct
up excess: forward_max_return_pct - threshold_pct

down tail:   forward_min_return_pct <= -threshold_pct
down excess: abs(forward_min_return_pct) - threshold_pct
```

Division of labor:

| Quantity | Population | Purpose |
|---|---|---|
| tree split training | tail rows only | partition extreme events by exceedance behavior |
| GPD fit | tail rows only | estimate tail shape/scale severity |
| `N_total` | all rows assigned to leaf | denominator for leaf frequency |
| `tail_rate` | all rows assigned to leaf | probability of entering tail |
| `tail_lift` | leaf rate vs global all-row rate | concentration of tail frequency |

GPD is not a full-return model and not an up/down/flat classifier.

Horizon semantics:

```text
bar = scanner config bar
outcome_horizon = N bars
label time scale = N * bar duration
```

For the daily-deep config, `bar = "1H"` and `transition.mae_mfe_horizon = 12`, so the implemented label is a 12-hour path-extreme touch. This label can count burst-then-fade paths as successful excursions. Continuation/exhaustion semantics require additional path-shape diagnostics, not a different interpretation of `tail_up`/`tail_down`.

---

## Public data products

### Feature, label, and output grains

Tailtree has three separate data products. Do not collapse them into one
"prediction" table:

```text
Feature row grain
  key: symbol, decision_timeframe, decision_bar_close_ms, optional source_family/source_state/source_known_at_ms
  meaning: known-at-close state available at the decision bar

Label row grain
  key: symbol, decision_bar_close_ms, outcome_horizon, direction
  meaning: future path excursion after the decision bar

Leaf evidence row grain
  key: tree_direction, leaf_id
  meaning: historical concentration/severity of tail labels for observations assigned to that leaf
```

Consistent historical source features are restricted to funding-like derivative families:

```text
funding_rate
oi_delta
taker_buy_sell_ratio
long_short_ratio
funding_age_ms
oi_age_ms
taker_age_ms
lsr_age_ms
```

Books/trades fields such as `imbalance_value`, `spread_bps`, `buy_sell_ratio`,
`book_age_ms`, and `trade_age_ms` are not training features unless a consistent
historical source contract exists for the active run. They may still be current-review
or cost/liquidity features outside tailtree training.

Tailtree feature selection is contract-based, not manifest-based. The trainable feature
set is the explicit list of columns from families whose persisted data contract is
historically consistent at the decision-bar grain:

```text
persistent source family + known-at aligned historical artifact -> allowed training columns
ephemeral/current-only source family -> review/cost/feasibility columns only
```

Implemented selector:

```text
qooi.scanner.tailrun._tailtree_training_features(
    observations: pl.DataFrame,
) -> tuple[list[str], list[str]]
```

It returns categorical and continuous training feature names from the explicit persistent
contract. It does not infer trainability from column presence alone; column presence only
filters the already-allowed list.

A future API becomes trainable by landing a persistent artifact contract and then adding
its columns to the tailtree training feature list. The model must not infer trainability
from column presence alone.

Tailtree outputs evidence, not execution decisions:

```text
TailTreeModel.predict_leaf(...)       -> observation row + leaf_id
TailTreeModel.predict_score(...)      -> observation row + tailtree_score
leaf_evidence_frame(...)              -> per-leaf tail frequency/severity evidence
score_bucket_evidence_frame(...)      -> per-score-bucket tail frequency/utility evidence
candidate_evidence_frame(...)         -> current observation matched to historical evidence
candidate_horizon_consistency_frame(...) -> candidate-level horizon agreement panel
rank/candidate selection              -> downstream inspection surface, not model-owned
```

Evidence bucket dispatch is objective-owned:

```text
tail_severity_gpd
  model output used for evidence: leaf_id
  persisted evidence artifact: potential-leaf-evidence-h{H}-{direction}.csv
  candidate match key: outcome_horizon, tree_direction, leaf_id

tail_utility_quantile
  model output used for evidence: tailtree_score
  persisted evidence artifact: potential-score-bucket-evidence-h{H}-{direction}.csv
  candidate match key: outcome_horizon, tree_direction, score_bucket
```

The score-bucket artifact is additive. It must not overwrite or reinterpret leaf evidence
Utility candidates may carry `leaf_id = null`; rank/report consumers use the
numeric score-bucket evidence columns.

Multi-horizon candidate ensemble is a consistency panel over calibrated candidate rows.
It does not average raw model scores and does not average opposite directions together.

```text
qooi.scanner.rank.candidate_horizon_consistency_frame(
    candidate_rank: pl.DataFrame,
) -> pl.DataFrame
```

Input grain:

```text
symbol × decision_timeframe × tree_direction × outcome_horizon
```

Output grain:

```text
symbol × decision_timeframe × tree_direction
```

Required output columns:

```text
horizon_count
strong_horizon_count          # horizons passing lift/score gates
horizon_span_bars             # max(horizon)-min(horizon)
best_outcome_horizon
best_rank_score
best_tail_lift
best_tail_utility_score       # max tail_lift × log1p(N_tail_exceedances)
direction_consistency_score   # count/strength score, not mean raw score
opposite_direction_count
opposite_direction_best_rank_score
conflict_penalty_score
consistency_rank_score
```

The artifact is written as `candidate-horizon-consistency.csv` beside
`candidate-rank.csv` and is report-only feedback. Candidate promotion still uses the
canonical ranked candidate rows.

Selection-efficiency/HPO feedback is the canonical model-selection artifact. It replaces
ad-hoc objective benchmark/HPO feedback surfaces; do not add parallel `tailtree-hpo.csv`,
`objective-benchmark.csv`, or `selection-summary.csv` files. It does not replace candidate
rank rows because candidate rank is the current inspection product, while selection
efficiency is the model/workflow evaluation product:

```text
tailtree-selection-efficiency.csv
```

Row grain:

```text
trial_id × fold_id × model_tag × objective × training_profile × outcome_horizon × tree_direction × budget_family × budget_value
```

Required columns:

```text
trial_id
trial_source
fold_id
evaluation_protocol
train_start_ms
train_end_ms
valid_start_ms
valid_end_ms
embargo_bars
universe_snapshot_id
eligible_symbol_count
selected_symbol_count
observation_row_count
feature_count
train_exceedance_count
valid_observation_count
valid_tail_count
valid_tail_rate
selected_observation_count
selected_observation_rate
selected_tail_count
selected_tail_rate
selected_tail_per_1k_obs
valid_tail_lift
selected_profit_proxy_mean
selected_profit_proxy_p90
selected_utility_mean
selected_utility_p90
profit_proxy_per_selected_obs
profit_proxy_per_1k_observed
hpo_score
promotion_threshold_pass_int
trained_tree_count
selected_bucket_or_leaf_count
fit_seconds
score_seconds
```

`selected_utility_*` columns are the current measurable proxy. The durable selection
contract is profit from selected extreme behavior/events, so future cost/slippage/replay
fields should feed `selected_profit_proxy_*` rather than optimizing mean-market
probability or raw utility alone.

Budget families compare selection ability at equal candidate budgets rather than comparing
raw objective gate widths:

```text
top_k:      1, 3, 5, 10
top_pct:    1, 5, 10, 20
score_gate: calibrated objective-native threshold
```

Small-grid HPO API graph, first implementation:

```text
qooi.scanner.tailrun.selection.tailtree_selection_efficiency_frame(
    candidates: pl.DataFrame,
    *,
    run_summary: pl.DataFrame,
    universe_snapshot_id: str,
    model_tag: str,
    objective: str,
    training_profile: str,
    budgets: TailtreeSelectionBudgets,
    feasibility: TailtreeSelectionFeasibilityPolicy,
) -> pl.DataFrame

qooi.scanner.tailrun.selection.write_tailtree_selection_efficiency(
    frame: pl.DataFrame,
    diagnostics_dir: Path,
    model_dir: Path,
) -> None
```

Decoupled tuning API graph, target shape:

```text
TailtreeTrialSpec
  trial_id: str
  trial_source: Literal["primary", "fixed", "search"]
  model_tag: str
  objective: str
  training_profile: str
  num_leaves: int
  min_data_in_leaf: int
  learning_rate: float
  num_iterations: int
  early_stopping_rounds: int

TailtreeEvaluationSpec
  evaluation_protocol: Literal["single_split", "walkforward"]
  fold_id: int
  train_start_ms: int | None
  train_end_ms: int | None
  valid_start_ms: int | None
  valid_end_ms: int | None
  embargo_bars: int

TailtreeSelectionBudgets
  top_k: tuple[int, ...] = (1, 3, 5, 10)
  top_pct: tuple[float, ...] = (0.01, 0.05, 0.10, 0.20)
  score_gate: tuple[float, ...] = ()

TailtreeSelectionFeasibilityPolicy
  min_selected_observation_count: int
  min_selected_symbol_count: int
  min_selected_tail_count: int
  min_valid_tail_lift: float
  min_profit_proxy_per_selected_obs: float
```

Budgets and feasibility stay separate despite sharing the ergonomic
`[potential.evidence.tailtree.selection]` config section. Budgets enumerate replay rows;
feasibility determines whether each replay row is allowed to win. Neither type owns trial
generation, walkforward fold construction, or artifact naming.

Coding guidance:

```text
1. Keep fixed trials, sampler/search, walkforward evaluation, and selection replay separate.
2. Keep `[[potential.evidence.tailtree.trials]]` as fixed trial specs only.
3. Do not put Optuna state, fold settings, or selection gates into the fixed-trial shape.
4. Walkforward must be available to one fixed parameter set without HPO search.
5. Optuna, if added, is an optional trial source only; it emits TailtreeTrialSpec rows.
6. Generate all tuning/objective feedback into `tailtree-selection-efficiency.csv` only.
7. Keep `tailtree-run-summary.csv` as structural trainability coverage only.
8. Do not add sampler-owned artifacts or another report section/table for HPO.
```

### Labeled outcome frame

```text
qooi.scanner.tailtree.label_tail_exceedances(
    outcome_frame: pl.DataFrame,
    *,
    threshold_pct: float,
) -> pl.DataFrame
```

Adds:

```text
tail_up: bool
tail_down: bool
tail_exceedance_value_up: float
tail_exceedance_value_down: float
tail_utility_up: float
tail_utility_down: float
```

Fixed-horizon path-shape columns preserve the excursion label but add retention/exhaustion diagnostics:

```text
time_to_max_bar: int
time_to_min_bar: int
close_retention_ratio: float
post_max_drawdown_pct: float
post_min_rebound_pct: float
path_efficiency: float
```

These numeric columns let downstream selection distinguish:

```text
up_touch          # high excursion only
up_continuation   # excursion retained into terminal close
up_exhaustion     # excursion unwound after peak
down_touch
down_continuation
down_exhaustion
volatility        # both directions show elevated tail lift
```

Multi-horizon extension should use rows keyed by `outcome_horizon` first:

```text
qooi.scanner.outcome.potential_outcome_frame(...)
    -> rows for horizon in configured_horizons

qooi.scanner.tailtree.label_tail_exceedances(...)
    -> preserves outcome_horizon and adds direction labels per row
```

Artifact identity must include horizon semantics:

```text
model_tag
bar
outcome_horizon
threshold_pct
feature_schema_hash
```

Start with separate per-horizon model artifacts. A shared multi-horizon model is allowed
only after per-horizon summaries show enough non-null labels and validation tail counts.

### Tailtree training frame

Public helper:

```text
qooi.scanner.tailtree.tailtree_training_frame(
    observations: pl.DataFrame,
    labeled_outcomes: pl.DataFrame,
    *,
    direction: Literal["up", "down"],
) -> TailtreeTrainingFrame
```

Contract:

```text
TailtreeTrainingFrame
├── direction
├── all_observations        # denominator population
├── tail_observations       # tree/GPD training population
├── exceedance_values       # positive threshold excesses
├── utility_values          # log/quantile target source for tail_utility objectives
└── global_tail_rate        # len(tail_observations) / len(all_observations)
```

Training uses `tail_observations`; diagnostics project `all_observations` through the trained model.
Validation-quality diagnostics use the same deterministic tailtree validation fraction
and stay numeric:

```text
train_tail_count
valid_observation_count
valid_tail_count
valid_tail_rate
valid_selected_observation_count
valid_selected_tail_count
valid_selected_tail_rate
valid_tail_lift
```

`valid_tail_lift` is selected-validation tail rate divided by validation baseline
tail rate. Zero valid baseline produces zero lift.

---

## Model classes

### `qooi.scanner.tailtree.TrainConfig`

```text
objective: Literal["tail_severity_gpd", "tail_utility_quantile"]
num_leaves: int
min_data_in_leaf: int
learning_rate: float
num_iterations: int
early_stopping_rounds: int
validation_fraction: float
random_seed: int
```

Objective behavior:

```text
tail_severity_gpd
  target rows = tail rows only
  target      = tail_exceedance_value_{direction}
  objective   = custom bounded-ξ GPD NLL surrogate
  GPD params  = post-hoc leaf exceedance fit

tail_utility_quantile
  target rows = tail rows only
  target      = log1p(tail_utility_{direction})
  objective   = LightGBM quantile loss, alpha=0.8
  evidence    = full-ensemble score buckets, not last-tree leaves
  GPD params  = raw exceedance descriptor; utility remains moment-based feedback
```

Baseline feedback columns are numeric and written into `tailtree-run-summary.csv`:

```text
objective
tail_utility_mean
tail_utility_p90
valid_selected_utility_mean
valid_selected_utility_p90
```

### `qooi.scanner.tailtree.score_bucket_evidence_frame`

```text
qooi.scanner.tailtree.score_bucket_evidence_frame(
    tree: TailTreeModel,
    observations: pl.DataFrame,
    outcomes: pl.DataFrame,
    *,
    score_quantiles: tuple[float, ...] = (0.99, 0.98, 0.95, 0.90),
) -> pl.DataFrame
```

Output grain:

```text
outcome_horizon × tree_direction × score_bucket
```

Required columns:

```text
score_bucket                  # top_1pct/top_2pct/top_5pct/top_10pct
score_quantile                # numeric cutoff quantile
score_min
score_max
N_total
N_tail_exceedances
leaf_tail_rate                # bucket tail rate, retained name for rank compatibility
global_tail_rate
tail_lift
tail_lift_stability
tail_utility_mean
tail_utility_p90
gpd_shape_xi                  # raw exceedance GPD basis only when available, else null
gpd_scale_sigma
```

Candidate matching for this artifact scores current rows with the full ensemble, assigns
the configured bucket, then joins on `(outcome_horizon, tree_direction, score_bucket)`.
It does not use LightGBM `pred_leaf=True` for boosted utility evidence.

### `qooi.scanner.tailtree.GPDParams`

```text
xi: float          # shape; bounded by validation
sigma: float       # scale; positive
tail_rate: float   # frequency diagnostic, not severity
```

### `qooi.scanner.tailtree.TreeMetadata`

```text
direction: "up" | "down"
num_leaves_actual: int
categorical_features: list[str]
continuous_features: list[str]
global_baseline: GPDParams
leaf_params: dict[int, GPDParams]
feature_importance: list[tuple[str, float]]
train_config: TrainConfig
train_timestamp: str
train_n_observations: int
train_n_exceedances: int
```

No `leaf_paths`: recursive path extraction is cosmetic and not part of the model/evidence contract.

### `qooi.scanner.tailtree.TailTreeModel`

```text
TailTreeModel.train(
    features: pl.DataFrame,              # tail rows for tree/GPD training
    exceedance_values: Sequence[float],  # positive exceedances, same length
    *,
    config: TrainConfig,
    categorical_features: list[str],
    continuous_features: list[str],
    direction: Literal["up", "down"],
    global_tail_rate: float,
    train_n_observations: int,
) -> TailTreeModel

TailTreeModel.predict_leaf(features: pl.DataFrame) -> pl.DataFrame
TailTreeModel.predict_leaf_params(features: pl.DataFrame) -> pl.DataFrame
TailTreeModel.to_json(path) -> None
TailTreeModel.from_json(path) -> TailTreeModel
```

`global_tail_rate` and `train_n_observations` come from the all-row training frame. The model is trained on tail rows, but its baseline frequency is not computed from tail rows alone.

---

## Training graph per direction

```text
for outcome_horizon in config.evidence.tailtree.outcome_horizon:
    labeled_horizon = labeled_outcomes.filter(pl.col("outcome_horizon") == outcome_horizon)
    training_frame = tailtree_training_frame(observations, labeled_horizon, direction)

    all_n       = training_frame.train_n_observations
    tail_train  = training_frame.tail_observations
    excess      = training_frame.exceedance_values
    global_rate = training_frame.global_tail_rate

    model = TailTreeModel.train(
        tail_train,
        excess,
        config=TrainConfig(...),
        categorical_features=present_categoricals,
        continuous_features=present_continuous,
        direction=direction,
        global_tail_rate=global_rate,
        train_n_observations=all_n,
    )

    summary = _tailtree_run_summary_frame(..., outcome_horizon=outcome_horizon)
    quality = TailtreeDirectionQuality.from_labeled_leaf_frame(...)
```

`tailrun/` keeps config/artifact access typed at the boundary via `ReportInputs`
and groups summary math in dataclass methods. It should not use untyped helper
arguments plus repeated `inputs.config...` attribute chains inside dataframe hot paths.

---

## Train/load-predict lifecycle

Training and current prediction are different scanner modes. The tailtree path exposes a lifecycle config instead of always training during every scan.

Implemented config shape:

```toml
[potential.evidence]
kind = "tailtree"

[potential.evidence.tailtree]
lifecycle = "train"        # "train" or "load_predict"
model_dir = "data/output/potential/daily-deep/models"
model_tag = "tailtree-1h-12h-v1"
outcome_horizon = [6, 12, 24]  # int or list[int]
# train writes one horizon-suffixed artifact set per configured horizon
```

Implemented lifecycle calls:

```text
qooi.scanner.tailrun.run(
    observations,
    source_outcomes,
    realized_transitions,
    inputs,
    *,
    source_event_row_count: int,
) -> TailtreeEvidenceResult
    if config.evidence.tailtree.lifecycle == "train":
        → qooi.scanner.tailrun.train_evaluate_predict(...)
        → remove stale direction artifacts for model_tag
        → write model_dir/model_tag/tailtree-run-summary.csv
        → write diagnostics/tailtree-run-summary.csv
        → summary rows include outcome_horizon
        → write model_dir/model_tag/tailtree-artifact-h{horizon}.json
        → conditionally write tail-tree-h{horizon}-{up,down}.json for trained directions
        → conditionally write potential-leaf-evidence-h{horizon}-{up,down}.csv for nonempty evidence
    if config.evidence.tailtree.lifecycle == "load_predict":
        → qooi.scanner.tailrun.load_predict(...)
        → for each configured outcome_horizon:
            → load tailtree-artifact-h{horizon}.json
            → validate metadata outcome_horizon/bar/threshold/model_tag
            → load tail-tree-h{horizon}-{up,down}.json where present
            → load matching potential-leaf-evidence-h{horizon}-{up,down}.csv
            → validate evidence rows carry the same outcome_horizon
            → copy frozen tailtree-run-summary.csv into current diagnostics for report feedback
        → return TailtreeEvidenceResult(models={(horizon, direction): tree}, evidence=concat(...))

qooi.scanner.tailtree
    → statistical/model code only: labels, training frame, TailTreeModel,
      leaf evidence, selection

qooi.scanner.tailrun
    → lifecycle code: artifact root, metadata hash, save/load, validation,
      train-vs-load dispatch, structural train-summary rows
```

Do not move lifecycle dispatch into report, candidate, or ranking modules.

Mode semantics:

| Lifecycle | Input contract | Model action | Output contract |
|---|---|---|---|
| `train` | observations + outcomes + source-event count | train eligible directions, replace artifact set, always write summary | evidence + optional models + candidates |
| `load_predict` | latest observations + frozen evidence/model artifacts for every configured horizon | load only; no outcome dependency for prediction | same horizon-dimensional candidate/rank/report grain as train |

`load_predict` must validate artifact compatibility before ranking:

```text
bar
outcome_horizon
threshold_pct
categorical_features
continuous_features
feature_schema_hash
training_window_ms
model_tag
```

A mismatch is a hard diagnostic error, not a silent retrain or fallback.

---

## Leaf diagnostics graph

```text
qooi.scanner.tailtree.leaf_evidence_frame(
    tree: TailTreeModel,
    observations: pl.DataFrame,
    outcomes: pl.DataFrame,
    *,
    recent_window_days: int = 30,
) -> pl.DataFrame
```

Implementation shape:

```text
1. with_leaf = tree.predict_leaf(observations)
2. collapse market/source duplicate outcomes by (symbol, decision_bar_close_ms), preserving any source tail labels
3. group by leaf_id over all rows:
     N_total
     N_tail_exceedances
     tail_rate = N_tail_exceedances / N_total
     tail_lift = tail_rate / tree.metadata.global_baseline.tail_rate
4. join per-leaf GPD params from tree metadata
5. add recent-window stability metrics
```

Leaf evidence and leaf context must use the same decision-key outcome aggregation so market baseline rows cannot erase source tail labels.

---

## Planned rolling validation graph

Rolling validation is time-blocked by `decision_bar_close_ms`; it must not randomly split rows across time.

Planned public data product:

```text
qooi.scanner.tailtree.rolling_leaf_validation_frame(
    observations: pl.DataFrame,
    labeled_outcomes: pl.DataFrame,
    *,
    direction: Literal["up", "down"],
    train_window_bars: int,
    validation_window_bars: int,
    step_bars: int,
) -> pl.DataFrame
```

Planned output columns:

```text
split_id
train_start_ms
train_end_ms
valid_start_ms
valid_end_ms
leaf_id
tree_direction
valid_N_total
valid_N_tail_exceedances
valid_tail_rate
valid_tail_lift
valid_close_retention_ratio
valid_post_peak_drawdown_pct
```

Planned leaf promotion aggregate:

```text
leaf_selected_window_count
median_valid_tail_lift
min_valid_tail_lift
valid_lift_decay
median_valid_close_retention_ratio
median_valid_post_peak_drawdown_pct
```

Selection should promote leaves that are stable across rolling windows, not merely strong in one in-sample tree.

---

## Selection graph

```text
qooi.scanner.tailtree.select_tail_leaves(leaf_evidence) -> pl.DataFrame
```

Default hard gate:

```text
N_tail_exceedances >= 30
tail_lift >= 1.5
tail_lift_stability numeric threshold when available
```

Output semantics:

```text
selection_mode = "hard_gate"       # at least one leaf passed the hard gate
selection_mode = "best_available"  # no leaf passed; top leaves are written for inspection
selected_evidence_level = true      # promoted evidence row
selected_evidence_level = false     # fallback row, not promoted
```

Selection is a research filter, not a trading authorization.

Planned promotion rule additions:

```text
selected_evidence_level
and rolling validation passes
and path-shape quality passes
and freshness/cost gates pass at candidate time
```

Multi-horizon output preserves one row per `(outcome_horizon, tree_direction, leaf_id)` so rank can compare short-burst, medium-continuation, and long-horizon setups quantitatively.

Candidate bridge:

```text
qooi.scanner.rank.candidate_evidence_frame(
    observations,
    evidence,
    tree_models={(outcome_horizon, direction): TailTreeModel, ...},
) -> rows keyed by symbol, decision_bar_close_ms, outcome_horizon, tree_direction
```

The rank bridge must not collapse to a single model pair. Current observations are
projected through every trained horizon/direction model, joined to evidence on
`(outcome_horizon, tree_direction, leaf_id)`, and then ranked as candidate-horizon
rows.

---

## Persistence graph

```text
model.to_json("tail-tree-up.json")
  → {
      "lightgbm_model": booster.model_to_string(),
      "metadata": TreeMetadata.model_dump(mode="json"),
    }

TailTreeModel.from_json("tail-tree-up.json")
  → TreeMetadata.model_validate(...)
  → lgb.Booster(model_str=...)
  → TailTreeModel(...)
```

---

## Removed/stale surfaces

Do not reintroduce:

```text
leaf_path_to_text(...)
TreeMetadata.leaf_paths
_walk_node(...)
Booster.trees_to_dataframe()        # pandas surface
features.to_pandas()                # pyarrow/pandas dependency path
lgb.train(..., fobj=...)            # LightGBM 4 uses params["objective"]
```
