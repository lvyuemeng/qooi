# Research Evaluation Report

Date: 2026-05-21

Implementation note: the diagnostics API has since been reduced to `backtest` and `research-evaluation` only. Current `research-evaluation` exports are `timeframe-classifier.csv`, `joint-forward-quality.csv`, and optional `trade-record-modulation.csv`; older output names referenced below are historical run artifacts, not retained public API surfaces.

## Executive Summary

This report evaluates the current `research-evaluation` evidence around joint multi-timeframe market structure. The intended research unit is not an independent timeframe score and not `1H` aided by `4H` or `1D`. The intended unit is an irreducible joint bucket:

```text
(joint multi-timeframe state, liquidity_event_type, side)
```

A valid `joint multi-timeframe state` may be `mtf_state_key`, `mtf_structure_key`, `mtf_stage_key`, `mtf_event_state_key`, or a custom label with the same meaning. The research question is: historically, when this joint structure group and liquidity event appeared with this side or direction, how did the forward-return distribution behave?

Current decision: diagnostic-only. No joint bucket, resonance candidate, MTF key, market-stage filter, timeframe-quality row, conditional dependency row, or strategy variant is authorized. The current artifacts validate classifier coverage, state-quality plumbing, forward-quality metric computation, candidate construction, and optional conditional dependency export, but the exact joint-group endogenous-quality table has not yet been produced.

| Layer | Observation Unit | Current Evidence | Decision |
|---|---|---|---|
| Classifier health | State labels by symbol and timeframe | `864` row artifacts, all `info`; `21,700` exported rows | Usable prerequisite evidence |
| Tradability / ETI | Structural quality of state partitions | `1,207` state-tradability rows; `0` high buckets | Supporting prior only |
| Forward quality | Current per-timeframe state/horizon summaries | `14,943` rows with Omega, Sortino-zero, PWPR, positive/negative rates, and directional bias | Metric plumbing validated; not final joint ranking |
| Resonance candidates | Trigger events aligned to closed confirming timeframe states | `119,036` broad candidates | Diagnostic queue only |
| Conditional dependency quality | State, transition, and continuous-structure conditions inside `market-state-forward` | Focused export produced `267,591` dependency rows; strongest sufficient CMI rows condition mostly on `liquidity_event_type` | Hypothesis-prior evidence only |
| Joint-group quality | `(joint_group, liquidity_event_type, side)` forward distribution | Not yet exported as a side-normalized dedicated table | Required downstream artifact |
| Strategy backtest | Executable entries, stops, targets, sizing, fees, slippage | Not run from qualified joint groups | No strategy change |

## Core Principle

The research should directly score the return distribution of joint multi-timeframe structure groups. It should not score each timeframe independently and then combine weights.

Candidate grouping tuple:

```text
joint_group = mtf_state_key or equivalent custom multi-timeframe label
event = liquidity_event_type
side = long | short, or direction implied by event
research_bucket = (joint_group, event, side)
```

The bucket itself is the analytical unit. A bucket is only interesting if its own historical forward-return distribution is sufficient, directional, economically plausible, and stable enough for later strategy testing.

High cardinality is not automatically a defect. It reflects market-state diversity. The filter is statistical sufficiency and stability, not rejection of MTF keys as a class. MTF keys are valid research grouping units when they are no-lookahead descriptors and their buckets pass count and stability gates.

## Metrics

The API reference defines the forward summary metrics needed for endogenous bucket quality. These metrics should be computed inside each `(joint_group, liquidity_event_type, side)` bucket, preferably on side-normalized returns so long and short opportunities have a common favorable direction.

| Metric | Formula / Meaning | Interpretation |
|---|---|---|
| `rows` | Complete forward windows in the group | Minimum sufficiency gate |
| `positive_rate` | Percent of side-normalized returns above zero | Win-rate proxy |
| `negative_rate` | Percent of side-normalized returns below zero | Loss-rate proxy |
| `positive_mean` | Mean positive return | Average favorable payoff |
| `negative_mean_abs` | Absolute mean negative return | Average adverse payoff |
| `omega_ratio` | `sum(max(r, 0)) / abs(sum(min(r, 0)))` | Total favorable return mass versus adverse mass |
| `pwpr` | `(positive_mean * positive_rate) / (negative_mean_abs * negative_rate)` | Probability-weighted payoff ratio |
| `sortino_zero` | `mean_return_pct / sqrt(mean(min(r, 0)^2))` | Return versus downside volatility around zero |
| `directional_bias` | `up`, `down`, `flat_or_mixed`, or `insufficient` | Directional clarity of the bucket |
| ETI / tradability | Transition, autocorrelation, and volatility-efficiency structure | Supporting structural prior, not a replacement for forward quality |
| Cross-asset stability | Direction agreement across sufficient symbol-level rows | Robustness gate when the bucket appears across assets |
| Time stability | Direction/materiality agreement across time splits | Walk-forward robustness gate |
| Conditional information | `I(Y; X | Z) = H(Y|Z) - H(Y|X,Z)` | Whether a state variable reduces forward-direction uncertainty after conditioning on another known state |

All return-distribution metrics are endogenous to the bucket. ETI can help prioritize structurally clean states, but a high ETI state without favorable forward distribution is not a good-to-trade group.

## Screening Rule

A research bucket can be considered a good-to-trade candidate only after passing endogenous-quality gates. These gates produce research candidates, not live authorization.

| Gate | Required Interpretation |
|---|---|
| No-lookahead labels | The joint group and event are known by the source bar close |
| Valid-state exclusion | Exclude `warmup`, `unknown`, and `data_error` from promotion screens |
| Sufficiency | `rows >= 30` for the joint bucket; stricter aggregate/symbol thresholds before promotion |
| Omega | `omega_ratio > 1.5` |
| PWPR | `pwpr > 2.0` |
| Directional clarity | `directional_bias` is not `flat_or_mixed` or `insufficient` |
| Cross-asset consistency | Sufficient symbol-level buckets mostly agree in direction when multiple symbols are present |
| Time stability | Time splits or walk-forward segments preserve the sign and materiality of the effect |
| Execution realism | Later backtest includes costs, slippage, stops, targets, sizing, basket constraints, and comparability gates |

A bucket that passes these gates is still only a candidate hypothesis until it is expressed as executable entry/exit mechanics and survives execution-aware tests.

## Current Run Status

Run command:

```bash
uv run python scripts/research.py --config configs/research/research-evaluation-expanded.toml
```

Export directory:

```text
F:\Stratum\TEMP\kilo\qooi-research-evaluation-expanded
```

Generated artifacts:

| Artifact | Rows | Current Role |
|---|---:|---|
| `timeframe-classifier.csv` | 21,700 | Classifier and row-health evidence |
| `timeframe-tradability.csv` | 1,639 | ETI/tradability and classifier-validity evidence |
| `timeframe-forward-quality.csv` | 14,943 | Per-timeframe forward metric evidence |
| `resonance-candidates.csv` | 119,036 | Broad event-candidate queue |
| `classifier.csv` | present | Legacy/contextual output |
| `tradability.csv` | present | Legacy/contextual output |

New conditional dependency artifacts are available through the `market-state-forward` branch when `[market_state.conditional_dependencies] enabled = true`. They are not listed in the current export directory because the expanded run used the peer-timeframe outputs and timed out before a conditional-dependency-enabled market-state-forward run was executed.

The full graph run exceeded the 15-minute tool timeout after the peer-timeframe CSVs were written. Cache warnings were logged for incomplete ETH and SOL history. The exported CSVs above are usable for current evidence, but the incomplete run tail and missing joint-group artifact block promotion conclusions.

Focused conditional dependency run:

```bash
uv run python scripts/research.py --config configs/research/research-evaluation-conditional.toml
```

Focused export directory:

```text
F:\Stratum\TEMP\kilo\qooi-research-evaluation-conditional
```

Focused generated artifacts:

| Artifact | Rows | Current Role |
|---|---:|---|
| `classifier.csv` | present | Classifier prerequisite evidence for the market-state branch |
| `market-state-forward.csv` | 322,731 | Forward summaries plus conditional dependency diagnostics |

Rows inside the focused `market-state-forward.csv`:

| Artifact | Rows |
|---|---:|
| `forward-summary` | 55,140 |
| `conditional-information` | 2,808 |
| `state-transition-quality` | 264,081 |
| `continuous-condition-quality` | 702 |

The first focused run exposed and fixed an export-summary formatter bug where null dependency `classification` values could fail `counts_text(...)`. The fix only changes summary formatting; it does not change diagnostic row generation.

## Classifier Evidence

Classifier row artifacts were produced across `15m`, `1H`, `4H`, and `1D`.

| Timeframe | Row Artifacts | Severity |
|---|---:|---|
| `15m` | 216 | all `info` |
| `1H` | 216 | all `info` |
| `4H` | 216 | all `info` |
| `1D` | 216 | all `info` |
| Total | 864 | all `info` |

Interpretation:

- The classifier layer completed across the configured bars without warning or failure row artifacts.
- Classifier health is a prerequisite for joint-group evaluation because each bucket depends on known state labels.
- Classifier health does not imply edge. It only says the state vocabulary is available for downstream measurement.

## Tradability Evidence

Tradability artifacts provide structural quality priors, not return-distribution authorization.

| Artifact Type | Rows |
|---|---:|
| `state-tradability` | 1,207 |
| `classifier-validity` | 432 |
| Total | 1,639 |

Tradability buckets by timeframe:

| Timeframe | Medium | Low | Insufficient | High |
|---|---:|---:|---:|---:|
| `15m` | 283 | 19 | 1 | 0 |
| `1H` | 286 | 15 | 1 | 0 |
| `4H` | 284 | 17 | 0 | 0 |
| `1D` | 252 | 19 | 30 | 0 |

Classifier-validity status by timeframe:

| Timeframe | Pass | Warn |
|---|---:|---:|
| `15m` | 104 | 4 |
| `1H` | 93 | 15 |
| `4H` | 92 | 16 |
| `1D` | 98 | 10 |

Interpretation:

- No timeframe produced a `high` tradability bucket.
- ETI and tradability can support structural triage, but they cannot replace direct forward-return distribution scoring for the joint bucket.
- Invalid states such as `warmup`, `unknown`, and `data_error` must be excluded from promotion screens even when they appear in high-count or structurally interesting rows.

## Forward-Quality Evidence

The current forward-quality artifact proves that the required return-distribution metrics are computed, but it scores per-timeframe state groups rather than the target joint group `(joint_group, liquidity_event_type, side)`.

| Timeframe | Summary Rows | Sufficient Rows | Directional Rows |
|---|---:|---:|---:|
| `15m` | 3,819 | 2,448 | 889 |
| `1H` | 3,810 | 3,528 | 998 |
| `4H` | 3,756 | 2,979 | 833 |
| `1D` | 3,558 | 1,662 | 467 |
| Total | 14,943 | 10,617 | 3,187 |

Representative current rows should be read only as examples of metric behavior. They are not final opportunities because they are not grouped by the desired joint multi-timeframe state, event, and side tuple.

| Timeframe | Horizon | Group | State | Rows | Mean Return % | Positive Rate % | Omega | PWPR | Bias |
|---|---:|---|---|---:|---:|---:|---:|---:|---|
| `15m` | 4 | `market_stage_reduced` | `markdown` | 2,121 | 0.36 | 57.38 | 2.23 | 2.23 | `up` |
| `1D` | 3 | `market_stage_reduced` | `markup` | 578 | 3.06 | 53.98 | 2.04 | 2.04 | `up` |
| `1D` | 3 | `liquidity_event_type` | `breakout_acceptance_high` | 1,261 | 2.73 | 52.10 | 1.98 | 1.98 | `up` |
| `15m` | 8 | `market_stage_reduced` | `markdown` | 2,121 | 0.37 | 58.79 | 1.96 | 1.96 | `up` |
| `15m` | 4 | `liquidity_event_type` | `breakout_acceptance_low` | 3,563 | 0.29 | 56.05 | 1.87 | 1.87 | `up` |

Interpretation:

- Omega, PWPR, positive/negative rates, Sortino-zero, and directional bias are the right metric vocabulary for endogenous quality.
- The current rows are not side-normalized joint buckets and should not be promoted as tradable signals.
- Counterintuitive rows, such as positive drift after `markdown`, are hypothesis prompts. They may reflect broad market drift, semantic inversion, overlapping windows, incomplete symbol robustness, or a genuine conditional effect that needs joint-group testing.

## Resonance Candidate Evidence

The resonance candidate table is a broad event queue. It is not a ranked table of good-to-trade buckets.

| Trigger Timeframe | Long | Short | Total |
|---|---:|---:|---:|
| `15m` | 8,145 | 7,637 | 15,782 |
| `1H` | 52,793 | 50,461 | 103,254 |
| Total | 60,938 | 58,098 | 119,036 |

Candidate metadata coverage:

| Metric | Rows |
|---|---:|
| Total candidates | 119,036 |
| Reward/risk present | 117,680 |
| Reward/risk >= 1 | 57,361 |
| Reward/risk >= 2 | 57,120 |
| Trigger ETI present | 119,036 |
| Trigger Omega present | 119,036 |
| Confirming ETI present | 0 |
| Confirming Omega present | 0 |

Confirmation alignment coverage:

| Trigger Timeframe | Confirming Timeframes | Rows |
|---|---|---:|
| `15m` | `1h,4h,1d` | 15,782 |
| `1H` | `4h,1d` | 103,254 |

Interpretation:

- The candidate count is intentionally broad and too large to imply signal discovery.
- Reward/risk is populated for most rows, but it is structural metadata, not realized executable PnL.
- Trigger ETI and trigger Omega are populated, but confirming ETI/Omega minima are not populated in the current artifact.
- The table can seed future joint buckets, but it must be aggregated into side-normalized `(joint_group, liquidity_event_type, side)` forward-quality summaries before promotion review.

## Joint-Group Evaluation Gap

The main gap is the absence of a dedicated joint-group endogenous-quality artifact.

Current limitation:

```text
timeframe-forward-quality.csv -> per-timeframe state summaries
resonance-candidates.csv -> broad candidate rows, no joint-group forward aggregation
missing artifact -> joint-forward-quality grouped by (mtf_state_key, liquidity_event_type, side)
```

The next artifact should group rows by:

```text
mtf_state_key or equivalent joint label
liquidity_event_type
side
horizon
optional: symbol, time split, trigger_timeframe
```

Required output fields should include:

| Field Family | Required Purpose |
|---|---|
| Count fields | `rows`, sufficient aggregate rows, sufficient symbol rows |
| Return-distribution fields | `positive_mean`, `negative_mean_abs`, `positive_rate`, `negative_rate`, `omega_ratio`, `pwpr`, `sortino_zero` |
| Direction fields | `directional_bias`, side-normalized mean return, long/short interpretation |
| Stability fields | Cross-asset direction agreement, time-split agreement, walk-forward stability |
| Structural fields | ETI/tradability support, invalid-state flags, no-lookahead provenance |
| Execution bridge fields | Candidate stop/target source, structural reward/risk, later backtest linkage |

This table is the correct place to answer whether a joint multi-timeframe structure/event/side bucket is good-to-trade. Existing per-timeframe quality and resonance rows are supporting inputs.

## Conditional Dependency Evaluation

The framework now supports optional conditional dependency evaluation inside the existing `market-state-forward` output. This intentionally does not add a separate public graph node; it keeps the API graph small while addressing the time-series dependency problem directly.

Artifact rows inside `market-state-forward.csv` when enabled:

| Artifact | Research Question |
|---|---|
| `conditional-information` | Does one state variable reduce uncertainty about forward direction after conditioning on another known state? |
| `state-transition-quality` | Do previous-to-current paths such as `range->markup` carry more forward quality than static labels? |
| `continuous-condition-quality` | Do deterministic structure buckets such as range compression depth or range position improve state resolution? |

Default activation is off to avoid expanding the already long-running research graph. The focused empirical run enabled:

```toml
[research_evaluation]
outputs = ["market-state-forward"]
include_backtest_report = false

[market_state.conditional_dependencies]
enabled = true
min_rows = 100
min_cell_rows = 20
max_cardinality = 200
```

Empirical export summary:

| Metric | Value |
|---|---:|
| Total rows in focused `market-state-forward.csv` | 322,731 |
| Dependency rows | 267,591 |
| Symbols represented, including `ALL` aggregate | 13 |
| Horizons represented | 3, 5, 10 |
| `conditional-information` rows | 2,808 |
| `state-transition-quality` rows | 264,081 |
| `continuous-condition-quality` rows | 702 |

Conditional-information classification:

| Classification | Rows |
|---|---:|
| `strong` | 44 |
| `moderate` | 37 |
| `weak` | 2,025 |
| `insufficient` | 702 |

Top sufficient conditional-information rows by normalized CMI:

| Symbol | Horizon | X Feature | Z Feature | Rows | Sufficient Cells | Normalized CMI | Class |
|---|---:|---|---|---:|---:|---:|---|
| `ALL` | 10 | `mtf_structure_key` | `liquidity_event_type` | 54,276 | 179 | 0.0959 | `strong` |
| `ALL` | 10 | `d1_market_stage_reduced` | `liquidity_event_type` | 54,276 | 53 | 0.0891 | `strong` |
| `ALL` | 10 | `h4_market_stage_reduced` | `liquidity_event_type` | 54,276 | 49 | 0.0887 | `strong` |
| `ALL` | 5 | `mtf_structure_key` | `liquidity_event_type` | 54,281 | 179 | 0.0868 | `strong` |
| `ALL` | 10 | `market_stage_reduced` | `liquidity_event_type` | 54,276 | 40 | 0.0863 | `strong` |
| `ALL` | 10 | `key_level_proximity_bucket` | `liquidity_event_type` | 54,276 | 10 | 0.0854 | `strong` |

Highest raw normalized CMI rows were broader MTF key relationships such as `mtf_stage_key | liquidity_event_type`, but those rows were marked `insufficient` because the dependency grid remained too sparse after cell gates. They are useful for semantic-reduction design, not promotion.

Clean transition rows after excluding `warmup`, `unknown`, and `data_error` labels and requiring `rows >= 100` produced `13,374` rows, including `2,935` directional rows. The strongest rows remain hypothesis prompts and are not cross-asset stable conclusions:

| Symbol | Horizon | Transition Feature | Transition | Rows | Mean Return % | Positive Rate % | Omega | Bias |
|---|---:|---|---|---:|---:|---:|---:|---|
| `LINK-USDT-SWAP` | 10 | `mtf_stage_key_transition` | `markdown|trend_continuation|trend_continuation->markdown|trend_continuation|trend_continuation` | 100 | 2.49 | 76.00 | 6.21 | `up` |
| `DOGE-USDT-SWAP` | 10 | `mtf_structure_key_transition` | `range|downtrend|uptrend->range|downtrend|uptrend` | 223 | 4.47 | 50.67 | 5.39 | `up` |
| `ALL` | 5 | `mtf_stage_key_transition` | `range|wide_range|accumulation->range|wide_range|accumulation` | 109 | 1.66 | 71.56 | 5.19 | `up` |
| `DOGE-USDT-SWAP` | 5 | `mtf_structure_key_transition` | `range|downtrend|uptrend->range|downtrend|uptrend` | 223 | 2.87 | 48.43 | 4.88 | `up` |
| `ALL` | 3 | `mtf_stage_key_transition` | `range|wide_range|accumulation->range|wide_range|accumulation` | 109 | 1.23 | 67.89 | 4.64 | `up` |

Clean continuous-condition rows after excluding invalid labels and requiring `rows >= 100` produced `546` rows, including `281` directional rows. The strongest clean continuous rows are weaker than the top transition rows:

| Symbol | Horizon | Condition | Bucket | Rows | Mean Return % | Positive Rate % | Omega | Bias |
|---|---:|---|---|---:|---:|---:|---:|---|
| `DOGE-USDT-SWAP` | 10 | `range_position_bucket` | `near_high` | 7,533 | 0.60 | 46.70 | 1.54 | `up` |
| `DOGE-USDT-SWAP` | 5 | `range_position_bucket` | `near_high` | 7,533 | 0.31 | 46.63 | 1.37 | `up` |
| `DOGE-USDT-SWAP` | 10 | `swing_low_distance_bucket` | `far` | 19,861 | 0.38 | 48.66 | 1.36 | `up` |
| `SOL-USDT-SWAP` | 5 | `range_position_bucket` | `near_low` | 7,055 | 0.25 | 54.60 | 1.28 | `up` |
| `SOL-USDT-SWAP` | 3 | `swing_low_distance_bucket` | `near` | 5,063 | 0.16 | 54.14 | 1.27 | `up` |

The key conditional-information formula is:

```text
I(Y; X | Z) = H(Y | Z) - H(Y | X,Z)
normalized_cmi = I(Y; X | Z) / max(H(Y | Z), eps)
```

Where `Y` is normally `fwd_N_direction`, `X` is a candidate state/transition/structure condition, and `Z` is the conditioning state. This answers a different question from static bucket quality: it asks whether a known structural variable contributes information after another known context is already accounted for.

Interpretation:

- `conditional-information` can prioritize dependency chains such as D1 state constraining H4/H1 event meaning.
- `state-transition-quality` uses paths such as `accumulation->trend_continuation` rather than static labels only.
- `continuous-condition-quality` brings structure geometry into the diagnostic through deterministic buckets such as range-width-to-threshold and range position.
- The strongest sufficient CMI rows show that context features can reduce forward-direction uncertainty after conditioning on `liquidity_event_type`, especially at the `ALL` aggregate level.
- Sparse high-cardinality MTF key rows still need semantic reduction before they can become reliable bucket definitions.
- Transition rows can show high Omega in isolated symbol/path cells, but they need cross-asset and time-split stability before use as strategy hypotheses.
- Continuous-condition rows are broad and sufficient, but their clean Omega values are modest and should be treated as weak structure priors.
- Conditional mutual information is association evidence, not causal proof.
- These rows are still hypothesis-discovery diagnostics and must be followed by cross-asset stability, time stability, and execution-aware backtests.

## Strategy Logic Implication

Future strategy logic should be framed as research hypotheses, not hand-weighted timeframe scores.

Entry consideration:

```text
if live_state in qualified (joint_group, liquidity_event_type, side) bucket:
    entry may be considered for strategy testing
```

Stop and target candidates should come from event structural extremes, nearby liquidity levels, ATR-aware risk limits, and higher-timeframe structure targets. The entry decision should be driven by the historical endogenous quality of the joint bucket, not by a manual combination of independent timeframe scores.

No live or simulated strategy should be authorized from the diagnostic bucket until the hypothesis is converted into explicit mechanics and backtested with fees, slippage, sizing, basket caps, stops, targets, and comparability checks.

## Promotion Assessment

Current status: not promoted.

| Gate | Current Result |
|---|---|
| Classifier health | Passes at row-artifact level across `15m`, `1H`, `4H`, and `1D` |
| Joint bucket sufficiency | Not evaluated because the dedicated joint-group table is missing |
| Conditional dependency evidence | Empirically exported: `267,591` dependency rows; sufficient strong CMI rows exist, but they are diagnostic priors only |
| Invalid-state exclusion | Required for promotion screens; not yet applied to a joint ranking table |
| Omega/PWPR thresholds | Metrics exist in forward-quality rows, but not for final joint buckets |
| Directional clarity | Available in per-timeframe rows, missing for side-normalized joint buckets |
| Cross-asset consistency | Not explicitly computed for the new joint bucket table |
| Time stability | Not explicitly computed for the new joint bucket table |
| Confirmation quality | `resonance-candidates.csv` has `0` populated confirming ETI/Omega minima |
| Execution-aware backtest | Not run from qualified joint buckets |

Promotion rule: a joint bucket may become a strategy hypothesis only after classifier health, joint-bucket counts, Omega/PWPR thresholds, directional clarity, cross-asset consistency, time or walk-forward stability, and execution-aware backtests all pass.

## Threats To Validity

| Threat | Impact | Mitigation |
|---|---|---|
| Missing joint artifact | Current evidence cannot rank the desired analytical unit | Generate `joint-forward-quality` grouped by joint state, event, side, and horizon |
| Conditional dependency sparsity | Highest-cardinality MTF dependency rows can fail cell sufficiency even with large aggregate row counts | Use semantic reductions and require cell coverage before interpreting CMI |
| Sparse high-cardinality buckets | Some joint labels will have too few observations | Use sufficiency filters and semantic reductions; do not reject MTF keys categorically |
| Overlapping forward windows | Adjacent rows share future bars | Add overlap-aware uncertainty, time splits, and walk-forward checks |
| Not fully side-normalized | Long and short quality can be misread | Normalize returns so favorable direction is positive for both sides |
| Cross-asset instability | Aggregate effects may be driven by one symbol | Compute symbol-level direction agreement and scoped hypotheses |
| Time instability | Effects may be regime-specific or overfit | Add chronological splits and walk-forward validation |
| Confirming quality missing | Resonance candidates lack populated confirming ETI/Omega minima | Join or recompute confirming timeframe quality before ranking candidates |
| Structural reward/risk only | Candidate reward/risk is not executable PnL | Backtest explicit stop/target/fill mechanics with costs and slippage |
| Strategy-conditioned controls | Executed trade records omit skipped states | Use trade-record modulation only as a negative-control or post-trade diagnostic |
| Incomplete graph tail | The full expanded run timed out after peer-timeframe CSV artifacts were written | Treat exported CSVs as current evidence and use focused configs for final report evidence |

## Final Decision

Current evidence validates the data path and metric vocabulary, but it does not authorize a strategy.

Decisions:

- Reinstate MTF keys as valid research grouping units under sufficiency and stability gates.
- Use `(joint multi-timeframe state, liquidity_event_type, side)` as the correct next research unit.
- Treat high cardinality as a sampling problem to be filtered, not as a reason to discard joint-state descriptors.
- Treat current `timeframe-forward-quality.csv` rows as metric-plumbing evidence, not final opportunities.
- Treat current `resonance-candidates.csv` rows as an unfiltered diagnostic queue, not a ranked signal table.
- Treat conditional dependency artifacts as the next evidence layer for dynamic structure relationships, not as causal proof or strategy authorization.
- Keep all findings diagnostic-only until joint-group quality, cross-asset stability, time stability, and execution-aware backtests are complete.

## Next Work

1. Use high-sufficiency conditional dependency rows to define candidate semantic reductions before building side-normalized joint buckets.
2. Generate a `joint-forward-quality` artifact grouped by `mtf_state_key` or reduced joint label, `liquidity_event_type`, `side`, and `horizon`, with optional `symbol`, `trigger_timeframe`, and time-split dimensions.
3. Side-normalize forward returns so long and short bucket quality can be compared directly.
4. Add invalid-state exclusions for `warmup`, `unknown`, and `data_error` before promotion screens.
5. Add cross-asset direction consistency fields and symbol-level sufficiency gates.
6. Add chronological time-split or walk-forward stability fields.
7. Join or compute confirming timeframe ETI/Omega minima for event candidates.
8. Backtest only dependency-informed joint buckets that pass endogenous-quality gates, with explicit timing, stops, targets, fees, slippage, sizing, and basket constraints.
