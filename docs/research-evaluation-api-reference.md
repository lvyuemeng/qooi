# Research Evaluation API Reference

Date: 2026-05-22

## Purpose

This document defines the reduced `research-evaluation` API and the next extension point for behavior-driven dynamic state research.

It is a methodology reference, not a run report. Current empirical results are in `docs/research-evaluation-report.md`. The full architecture is in `docs/behavior-driven-state-research-architecture.md`.

## Public Diagnostics Surface

The diagnostics API has only two public modes:

```text
backtest
research-evaluation
```

Do not add new `diagnostics.mode` values for research subgraphs. New research artifacts should be added as outputs under `research-evaluation` when they are durable enough to keep.

Current retained `research-evaluation` outputs:

| Output | Status | Role |
|---|---|---|
| `timeframe-classifier` | Current | Classifier health and no-lookahead evidence |
| `dynamic-transition-discovery` | Current Stage 1 | Deterministic transition-pattern discovery artifacts |
| `pattern-quality` | Current scoring surface | Shared scored-pattern table across static, transition, learned-state, and policy-context patterns |
| `trade-record-modulation` | Current optional | Strategy-conditioned post-trade control evidence |

Removed modes and outputs are not compatibility surfaces. Removed names include `classifier`, `joint-forward-quality`, `tradability`, `market-state-forward`, `market-state-modulation`, `timeframe-tradability`, `timeframe-forward-quality`, `resonance-candidates`, `state`, `state-profitability`, `state-filter-delta`, and `modulation-effect`.

## Current Reduced Graph

Focused current graph:

```text
prepare_classifier_frame / prepared market frames
  -> timeframe-classifier
  -> dynamic-transition-discovery
       -> state-transition-graph.csv
       -> transition-information.csv
       -> transition-ngram-quality.csv
       -> none-event-context-quality.csv
  -> pattern-quality
       -> scored-patterns.csv
       -> promotion-candidates.csv
optional backtest branch
  -> trade-record-modulation
```

Focused config shape:

```toml
[diagnostics]
mode = "research-evaluation"
export_dir = "F:\\Stratum\\TEMP\\kilo\\qooi-research-evaluation-dynamic-transitions"

[research_evaluation]
outputs = ["timeframe-classifier", "dynamic-transition-discovery", "pattern-quality"]
include_backtest_report = false
write_exports = true
fail_fast = false

[research_evaluation.dynamic_transition_discovery]
enabled = true
min_rows = 30
ngram_lengths = [2, 3]
include_none_context = true
omega_threshold = 1.5
pwpr_threshold = 2.0
promotion_min_rows = 50
promotion_min_symbols = 3
promotion_min_time_splits = 2
promotion_symbol_agreement_pct = 67.0
promotion_time_agreement_pct = 100.0
```

State-leakage rule: classifier health, transition discovery, and pattern quality paths must not apply strategy signal filters. Strategy filters, entries, exits, basket lifecycle, and trade records belong only to backtest and trade-record branches.

## Extension Graph

Stage 1 uses retained outputs, not new modes:

```python
ResearchOutputName = Literal[
    "timeframe-classifier",
    "dynamic-transition-discovery",
    "pattern-quality",
    "trade-record-modulation",
]
```

Recommended Stage 1 output order:

```python
("timeframe-classifier", "dynamic-transition-discovery", "pattern-quality")
```

Target graph:

```text
prepare_classifier_frame / prepared market frames
  -> timeframe-classifier
  -> dynamic-transition-discovery
       -> state-transition-graph.csv
       -> transition-information.csv
       -> transition-ngram-quality.csv
       -> none-event-context-quality.csv
  -> pattern-quality
       -> scored-patterns.csv
       -> promotion-candidates.csv
optional backtest branch
  -> trade-record-modulation
```

`dynamic-transition-discovery` is the deterministic discovery layer for behavior-driven state changes. It should not create entries, exits, stops, targets, basket actions, or allocation decisions.

`pattern-quality` is the target scoring surface. It scores the shared `PatternTable`/`OutcomeTable`/`MetricTable` pipeline and can project temporary legacy CSVs only when migration requires them.

## Evaluation Order

Research diagnostics are interpreted in this order:

1. Classifier and feature validity: labels must be present, internally consistent, and known without lookahead.
2. Dynamic transition discovery: transition structure must be sufficiently supported before performance interpretation.
3. Pattern quality: candidate rows must pass side-normalized materiality gates through the shared scored-pattern contract.
4. Promotion stress gates: rows must pass cross-symbol and time-split stability.
5. Strategy promotion tests: hypotheses must become executable signals and survive backtests.

A state finding is not automatically a strategy filter. It must first become an explicit trading hypothesis and then pass strategy-level validation.

## Source Data

Diagnostics operate on OHLCV klines plus deterministic features derived from bars known at the evaluation timestamp.

| Field | Meaning |
|---|---|
| `timestamp` | Kline open timestamp in milliseconds |
| `open`, `high`, `low`, `close` | Candle prices |
| `vol` or `volume` | Candle volume when available |
| `atr_14` | 14-bar average true range after indicator enrichment |

Default research evaluation uses swap OHLCV unless a run explicitly selects another source policy. Default peer bars are `1H`, `4H`, and `1D`.

## No-Lookahead Contract

Classifier and grouping columns must be known by the source bar close.

Allowed classifier or transition inputs:

- Current closed H1 OHLCV.
- Prior H1 bars.
- Last fully closed H4/D1 bars.
- Rolling statistics whose window ends at or before the current bar.
- Current or previous deterministic classifier labels.
- Current or previous deterministic liquidity event labels.

Forbidden classifier or transition inputs:

- Future H1 bars.
- Current not-yet-closed H4/D1 bars.
- Forward outcome columns.
- Strategy trade outcomes.
- Executor fills, fees, equity, or basket state.

Forward outcome columns may use future OHLCV as labels. They are never allowed back into state construction, transition construction, or grouping construction.

## Classifier Frame Preparation

The strategy-independent classifier frame is prepared by `prepare_classifier_frame(...)`. It is timeframe-native: callers pass `FrameRequest.bar`, and no H4/D1 context is attached unless context frames are supplied explicitly.

Pipeline stages:

1. Load cache for `FrameRequest.bar` and the selected symbol.
2. Add indicators with `add_indicators(...)`.
3. Add MACD histogram with `add_macd_histogram(...)`.
4. Add structure and stage features with `add_price_structure_stage_features(...)`.
5. Add `timeframe` metadata.
6. Optionally attach explicit context frames and MTF state keys for research-evaluation paths.

The joint-quality and dynamic-transition paths additionally ensure liquidity and none-context columns are available:

- `add_liquidity_sweep_features(...)` for `liquidity_event_type` and related event columns.
- `add_none_context_diagnostics(...)` for `atr_percentile_bucket` and `key_level_proximity_bucket`.
- `add_mtf_state_keys(...)` so MTF context keys are available when configured.
- `add_market_state_reductions(...)` for semantic reduction columns such as `market_stage_reduced`.

Semantic reduction is not physical column aliasing. Raw classifier labels remain available for audit, while reduced labels provide lower-cardinality diagnostics.

## Higher-Timeframe Context Join

Higher-timeframe bars are only known after they close.

For each H4 or D1 row:

```text
known_ts = htf_timestamp + htf_step_ms
```

H1 rows join the latest HTF row whose `known_ts <= h1_timestamp`. This preserves the no-lookahead contract.

## Structure Classifier Definitions

The structure classifier labels each H1/H4/D1 bar using range, swing, and trend evidence.

Structure states:

| State | Meaning |
|---|---|
| `uptrend` | Higher-high/higher-low evidence dominates |
| `downtrend` | Lower-high/lower-low evidence dominates |
| `range` | Compressed range without dominant trend evidence |
| `unknown` | Warmup, conflict, data error, or unresolved structure |

Market stages:

| Stage | Meaning |
|---|---|
| `warmup` | Not enough prior range data |
| `data_error` | Required OHLC/ATR inputs missing or invalid |
| `markup` | Close breaks above prior range while uptrend evidence is present |
| `markdown` | Close breaks below prior range while downtrend evidence is present |
| `accumulation` | Compressed range near the range low |
| `distribution_or_reversal` | Compressed range near the range high |
| `range` | Compressed range away from extremes |
| `trend_continuation` | Trend evidence exists without a range breakout |
| `wide_range` | Range is not compressed and no clean trend stage is resolved |
| `transition` | Conflicting uptrend/downtrend evidence |
| `unknown` | Fallback for unhandled cases |

Invalid promotion labels are `warmup`, `data_error`, and `unknown`. Rows containing invalid state labels must not pass promotion.

## Liquidity Event Definitions

Liquidity features are calculated from prior liquidity levels, not future information.

Prior levels:

```text
prior_liquidity_high = rolling_max(high.shift(1), lookback)
prior_liquidity_low = rolling_min(low.shift(1), lookback)
```

Liquidity event labels:

| Label | Meaning |
|---|---|
| `bullish_reclaim` | Low sweep reclaimed with bullish rejection characteristics |
| `bearish_reclaim` | High sweep reclaimed with bearish rejection characteristics |
| `breakout_acceptance_high` | Upside level accepted by close |
| `breakout_acceptance_low` | Downside level accepted by close |
| `failed_breakout_high` | High sweep failed to accept above level |
| `failed_breakout_low` | Low sweep failed to accept below level |
| `none` | No recognized liquidity event |

## None-Context Diagnostics

`none` liquidity events can dominate market samples. None-context diagnostics add deterministic context for those residual rows.

Default context candidates:

| Context | Meaning |
|---|---|
| `atr_percentile_bucket` | Current ATR percentile regime |
| `key_level_proximity_bucket` | Current proximity to prior high/low levels |
| `market_stage_reduced` | Reduced current market stage |

`none-event-context-quality.csv` is diagnostic-only. It does not authorize trading on `none` rows.

## MTF State Keys

MTF keys compress H1/H4/D1 context into readable audit strings.

| Key | Formula |
|---|---|
| `mtf_state_key` | `d1_structure_trend_state|h4_market_stage|h1_market_stage` |
| `mtf_structure_key` | `d1_structure_trend_state|h4_structure_trend_state|h1_structure_trend_state` |
| `mtf_stage_key` | `d1_market_stage|h4_market_stage|h1_market_stage` |
| `mtf_event_state_key` | `d1_structure_trend_state|h4_market_stage|h1_market_stage|liquidity_event_type` |

MTF keys are diagnostic descriptors. They must pass count and stability requirements before becoming candidate hypotheses.

## Dynamic Transition Discovery

`dynamic-transition-discovery` is the proposed Stage 1 output. It consumes known-at-close state columns and emits deterministic transition artifacts.

Proposed config shape:

```toml
[research_evaluation]
outputs = ["timeframe-classifier", "dynamic-transition-discovery", "pattern-quality"]

[research_evaluation.dynamic_transition_discovery]
enabled = true
state_columns = [
  "market_stage_reduced",
  "d1_market_stage_reduced",
  "h4_market_stage_reduced",
  "structure_trend_state",
]
event_column = "liquidity_event_type"
ngram_lengths = [2, 3]
min_rows = 50
information_min_rows = 100
include_none_context = true
```

### Transition Columns

Transition columns use previous-to-current paths within each symbol:

```text
state_transition = state[t-1] + "->" + state[t]
state_ngram_3 = state[t-2] + "->" + state[t-1] + "->" + state[t]
```

All transition components must be known at or before the current bar close. Forward returns are labels only.

### `state-transition-graph.csv`

Required columns:

| Column | Meaning |
|---|---|
| `artifact` | Constant artifact family name |
| `symbol` | Symbol or aggregate scope |
| `timeframe` | Source timeframe |
| `state_column` | State column used for graph edges |
| `source_state` | Previous state |
| `target_state` | Current state |
| `rows` | Transition count |
| `source_rows` | Total rows from the source state |
| `transition_probability` | Empirical `source_state -> target_state` probability |
| `invalid_state_present` | Whether the edge includes invalid state labels |

Represent the graph as a DataFrame/CSV. Do not add NetworkX unless actual graph algorithms are required later.

### `transition-information.csv`

Required columns:

| Column | Meaning |
|---|---|
| `artifact` | Constant artifact family name |
| `symbol` | Symbol or aggregate scope |
| `timeframe` | Source timeframe |
| `state_column` | State column under test |
| `condition_column` | Optional conditioning column such as event type |
| `rows` | Eligible rows |
| `unique_prev_states` | Cardinality of `S_t-1` |
| `unique_current_states` | Cardinality of `S_t` |
| `unique_conditions` | Cardinality of the condition column |
| `transition_information` | `I(S_t; S_t-1)` |
| `conditional_transition_information` | `I(S_t; S_t-1 | E)` |
| `normalized_transition_information` | Normalized information ratio |
| `normalized_conditional_transition_information` | Normalized conditional information ratio |
| `sufficient` | Whether sample and cell counts pass config thresholds |
| `classification` | Deterministic interpretation bucket |
| `bias_warning` | Sparsity or high-cardinality warning |

Terminology:

- `transition_information` means mutual information between previous and current states.
- `conditional_transition_information` means mutual information between previous and current states conditioned on an event/context variable.
- These fields are not transfer entropy.

### `transition-ngram-quality.csv`

This artifact projects from the shared scored-pattern metric vocabulary.

Required columns include:

| Column | Meaning |
|---|---|
| `artifact` | Constant artifact family name |
| `symbol` | Symbol or aggregate scope |
| `horizon` | Forward return horizon |
| `state_column` | Source state column |
| `ngram_length` | Number of states in the path |
| `transition_ngram` | Transition path string |
| `liquidity_event_type` | Current event label |
| `side` | Event-implied side |
| `rows` | Complete forward windows |
| `positive_rate` | Side-normalized positive return rate |
| `omega_ratio` | Favorable return mass divided by adverse return mass |
| `pwpr` | Probability-weighted payoff ratio |
| `mean_side_return_pct` | Mean side-normalized forward return |
| `invalid_state_present` | Whether invalid labels appear in the path |
| `passes_candidate_gate` | Discovery gate result |
| `passes_promotion_gate` | Strict promotion gate result |
| `promotion_failure_reasons` | Failed strict gates |
| `sufficient_symbols` | Count of sufficient symbol rows |
| `symbol_direction_agreement_pct` | Direction agreement across sufficient symbols |
| `sufficient_time_splits` | Count of sufficient chronological splits |
| `time_split_sign_agreement_pct` | Direction agreement across sufficient splits |
| `time_stable` | Whether time-stability gate passes |

### `none-event-context-quality.csv`

Required columns:

| Column | Meaning |
|---|---|
| `artifact` | Constant artifact family name |
| `symbol` | Symbol or aggregate scope |
| `horizon` | Forward return horizon |
| `context_column` | Context field used for grouping |
| `none_context` | Context value when event is `none` |
| `rows` | Complete forward windows |
| `positive_rate` | Positive forward return rate |
| `omega_ratio` | Favorable return mass divided by adverse return mass |
| `pwpr` | Probability-weighted payoff ratio |
| `mean_side_return_pct` | Mean forward return under context |
| `invalid_state_present` | Whether invalid labels appear |
| `passes_candidate_gate` | Discovery gate result |
| `passes_promotion_gate` | Strict promotion gate result |
| `promotion_failure_reasons` | Failed strict gates |

## Historical Joint Forward Quality

The removed `joint-forward-quality` output answered:

```text
historically, for this known-at-close joint market context + liquidity event + side,
what was the side-normalized forward return distribution?
```

It was strategy-independent and did not create entries, exits, stops, targets, basket actions, or strategy filters. The retained implementation expresses this class of evidence through `pattern-quality` tables and promotion gates instead of a dedicated joint-quality output.

Event-to-side mapping:

| Event | Side |
|---|---|
| `failed_breakout_low` | `long` |
| `bullish_reclaim` | `long` |
| `breakout_acceptance_high` | `long` |
| `failed_breakout_high` | `short` |
| `bearish_reclaim` | `short` |
| `breakout_acceptance_low` | `short` |

Side-normalized return:

```text
side_return_pct = fwd_N_return_pct for long
side_return_pct = -fwd_N_return_pct for short
```

Initial diagnostic candidate gate:

```text
rows >= min_rows
omega_ratio > omega_threshold
pwpr > pwpr_threshold
directional_bias == up
invalid_state_present == false
```

Strict promotion gate:

```text
passes_candidate_gate == true
rows >= promotion_min_rows
sufficient_symbols >= promotion_min_symbols
symbol_direction_agreement_pct >= promotion_symbol_agreement_pct
sufficient_time_splits >= promotion_min_time_splits
time_split_sign_agreement_pct >= promotion_time_agreement_pct
```

`promotion-candidates.csv` contains only rows that pass `passes_promotion_gate`. It is written even when empty so rejection evidence is explicit.

## Trade-Record Modulation

`trade-record-modulation` uses executed trades. It is strategy-conditioned because a strategy's entry rules decide which states are sampled.

It is useful for diagnosing existing strategies but cannot measure states the strategy never entered. It remains optional and post-trade only.

## Dependency Policy

Stage 1 should use existing Polars and standard library code only.

Dependency decisions:

| Dependency | Decision |
|---|---|
| NumPy | Avoid in Stage 1 unless numeric kernels become clearer or materially faster |
| SciPy | Avoid in Stage 1; consider later for statistical tests or estimator validation |
| NetworkX | Avoid in Stage 1; transition graphs are CSV/DataFrame artifacts |
| scikit-learn | Optional Stage 2 baseline/utility dependency |
| PyTorch | Preferred optional Stage 2/3 deep learning dependency if learned states are approved |
| TensorFlow/Keras | Do not add by default |
| Gym | Do not use for new work |
| Gymnasium | Optional Stage 3 environment dependency |
| Stable-Baselines3 | Optional Stage 3 baseline policy toolkit |
| Dreamer/world-model stacks | Defer until environment and evaluation contracts exist |

Optional dependencies must not be imported at package import time by core/runtime modules.

## Promotion Rules

A diagnostic finding may become a strategy hypothesis only if all conditions hold:

- State features are no-lookahead and classifier diagnostics are consistent.
- Semantic aliases are collapsed before counting discoveries.
- Candidate rows pass aggregate, symbol, and time-split thresholds.
- Effects are economically material, not just statistically visible.
- Direction is stable across assets or intentionally scoped with a stated reason.
- The proposed rule has an ex-ante market-behavior rationale.
- The strategy implementation emits normal signal columns.
- The resulting backtest passes costs, slippage, stops, targets, sizing, basket constraints, risk gates, and comparability checks.

Until those gates pass, every finding remains diagnostic-only.
