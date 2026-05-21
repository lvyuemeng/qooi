# Research Evaluation API Reference

Date: 2026-05-21

## Purpose

This document defines the reduced layered `research-evaluation` API centered on classifier health, `joint-forward-quality`, and optional trade-record control diagnostics.

It is a methodology reference, not a run report. For current empirical results, see `docs/research-evaluation-report.md`.

## Research-Evaluation API

The preferred orchestration surface for related diagnostics is `diagnostics.mode = "research-evaluation"`.

Example config shape:

```toml
[diagnostics]
mode = "research-evaluation"
export_dir = "F:\\Stratum\\TEMP\\kilo\\qooi-research-evaluation-expanded"

[research_evaluation]
outputs = [
  "timeframe-classifier",
  "joint-forward-quality",
]
include_backtest_report = false
write_exports = true
fail_fast = false

[research_evaluation.joint_forward_quality]
enabled = true
min_rows = 30
transition_min_rows = 50
omega_threshold = 1.5
pwpr_threshold = 2.0
```

Output dependencies are resolved as a reduced graph:

| Output | Required Upstream Evidence | Notes |
|---|---|---|
| `timeframe-classifier` | Cache-backed independent classifier frames per configured bar | Health and no-lookahead evidence only |
| `joint-forward-quality` | Prepared market classifier frames and internal forward outcome service | Core side-normalized joint state/event/side bucket quality, reduction diagnostics, and transition-event quality |
| `trade-record-modulation` | Strategy signal/backtest branch | Uses executed trades only, separate from classifier/market-state branches |

The old expanded `research-evaluation` outputs and former direct diagnostic modes were removed from the public API. The diagnostics API now has only two modes: `backtest` and `research-evaluation`.

State-leakage rule: classifier health and joint-quality paths must not apply strategy signal filters. Strategy filters, entries, exits, basket lifecycle, and trade records belong only to the backtest and trade-record modulation branch.

Focused joint-quality config shape:

```toml
[diagnostics]
mode = "research-evaluation"
export_dir = "F:\\Stratum\\TEMP\\kilo\\qooi-research-evaluation-joint-quality"

[research_evaluation]
outputs = ["timeframe-classifier", "joint-forward-quality"]
include_backtest_report = false
write_exports = true

[research_evaluation.joint_forward_quality]
enabled = true
min_rows = 30
transition_min_rows = 50
omega_threshold = 1.5
pwpr_threshold = 2.0
```

## Evaluation Layers

Diagnostics are interpreted in this order:

1. Classifier and feature validity: are the per-bar state labels internally consistent and known without lookahead?
2. Joint-forward-quality endogenous bucket ranking: which configured joint state/event/side buckets have stable side-normalized forward quality?
3. Dynamic and reduction support diagnostics: which configuration dimensions should be preserved, reduced, or rejected?
4. Strategy promotion tests: can a hypothesis be expressed as executable entries/exits and survive backtests with costs and risk controls?

A market-state finding is not automatically a strategy filter. It must first become an explicit trading hypothesis and then pass strategy-level validation.

## Source Data

The diagnostics operate on OHLCV klines.

| Field | Meaning |
|---|---|
| `timestamp` | Kline open timestamp in milliseconds |
| `open`, `high`, `low`, `close` | Candle prices |
| `vol` or `volume` | Candle volume when available |
| `atr_14` | 14-bar average true range after indicator enrichment |

Default research evaluation uses swap OHLCV unless a run explicitly selects another source policy. The default peer bars are `1H`, `4H`, and `1D`; configs may add `15m` through `[timeframes] bars = ["15m", "1H", "4H", "1D"]`. Contextual joins are internal preparation details for `joint-forward-quality`, not separate public outputs.

## No-Lookahead Contract

Classifier and grouping columns must be known by the source bar close.

Allowed for classifier features:

- Current closed H1 OHLCV.
- Prior H1 bars.
- Last fully closed H4/D1 bars.
- Rolling statistics whose window ends at or before the current H1 bar.

Forbidden for classifier features:

- Future H1 bars.
- Current not-yet-closed H4/D1 bars.
- Forward outcome columns.
- Strategy trade outcomes.

Forward outcome columns may use future OHLCV, but they are never allowed back into state construction or grouping.

## Classifier Frame Preparation

The strategy-independent classifier frame is prepared by `prepare_classifier_frame(...)`. It is timeframe-native: callers pass `FrameRequest.bar`, and no H4/D1 context is attached unless `contexts=DEFAULT_CONTEXTS` is supplied explicitly.

Pipeline stages:

1. Load cache for `FrameRequest.bar` and the selected symbol.
2. Add indicators with `add_indicators(...)`.
3. Add MACD histogram with `add_macd_histogram(...)`.
4. Add structure and stage features with `add_price_structure_stage_features(...)`.
5. Add `timeframe` metadata.
6. Optionally attach explicit context frames and MTF state keys for the internal joint-quality preparation path.

The joint-quality preparation path additionally ensures liquidity and none-context columns are available:

- `add_liquidity_sweep_features(...)` for `liquidity_event_type` and related event columns.
- `add_none_context_diagnostics(...)` for `atr_percentile_bucket` and `key_level_proximity_bucket`.
- `add_mtf_state_keys(...)` is then recomputed so `mtf_event_state_key` includes `liquidity_event_type`.
- `add_market_state_reductions(...)` adds semantic reduction columns such as `market_stage_reduced` without mutating raw classifier labels.

Semantic reduction is not physical column aliasing. Physical aliases resolve equivalent source columns such as `market_stage` and `h1_market_stage`; semantic reduction projects overlapping analytical labels such as `market_stage_reason=wide_range_no_stage` into one canonical state such as `market_stage_reduced=wide_range`. Raw `market_stage` and `market_stage_reason` remain available for audit and forensic runs.

## Higher-Timeframe Context Join

Higher-timeframe bars are only known after they close.

For each H4 or D1 row:

```text
known_ts = htf_timestamp + htf_step_ms
```

H1 rows join the latest HTF row whose `known_ts <= h1_timestamp`.

This means an H1 row at 10:00 can use an H4 bar only if that H4 bar has already fully closed by 10:00. This preserves the no-lookahead contract.

## Structure Classifier Definitions

The structure classifier labels each H1/H4/D1 bar using range, swing, and trend evidence.

Classifier shape parameters are CLI-configurable for diagnostics:

| Option | Meaning |
|---|---|
| `--classifier-swing-lookback` | Confirmed swing lookback |
| `--classifier-range-lookback` | Prior range window |
| `--classifier-trend-window` | Trend evidence rolling window |
| `--classifier-range-threshold-mode` | `rolling_quantile` or `fixed` range-width threshold |
| `--classifier-range-threshold-quantile` | Rolling quantile used for adaptive compression |
| `--classifier-range-threshold-window` | Rolling sample window for threshold estimation |
| `--classifier-range-threshold-min-samples` | Minimum rolling samples before adaptive threshold is ready |
| `--classifier-range-threshold-fallback` | `fixed` fallback or `data_error` while adaptive threshold is not ready |
| `--classifier-fixed-range-width-atr` | Fixed ATR-width threshold |
| `--classifier-level-proximity-atr` | ATR distance for near-range-high/low labels |

The default classifier remains rolling-quantile based. Explicit overrides are applied after `--classifier-profile` selection.

Important intermediate quantities:

| Quantity | Definition |
|---|---|
| `range_high` | Rolling max of prior highs over the range lookback |
| `range_low` | Rolling min of prior lows over the range lookback |
| `range_width_atr` | `(range_high - range_low) / atr_14` |
| `range_compression` | `range_width_atr <= range_width_atr_threshold` |
| `near_range_high` | Close is within configured ATR distance of `range_high` |
| `near_range_low` | Close is within configured ATR distance of `range_low` |
| `last_swing_high` | Last confirmed shifted swing high |
| `last_swing_low` | Last confirmed shifted swing low |

Trend evidence counts higher-high/higher-low and lower-high/lower-low events over a rolling trend window.

Structure states:

| State | Meaning |
|---|---|
| `uptrend` | Higher-high/higher-low evidence dominates |
| `downtrend` | Lower-high/lower-low evidence dominates |
| `range` | Compressed range without dominant trend evidence |
| `unknown` | Warmup, conflict, data error, or unresolved structure |

Market stages:

| Stage | Definition |
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

Market-stage reasons:

| Reason | Meaning |
|---|---|
| `warmup_range_not_ready` | Range window not ready |
| `data_error` | Invalid data inputs |
| `markup_breakout` | Markup stage caused by upside range breakout |
| `markdown_breakout` | Markdown stage caused by downside range breakout |
| `compressed_near_low` | Accumulation condition |
| `compressed_near_high` | Distribution/reversal condition |
| `compressed_mid_range` | Compressed range away from high/low |
| `trend_without_range_break` | Trend continuation condition |
| `wide_range_no_stage` | Wide range with no clean stage |
| `ambiguous_transition` | Conflicting trend evidence |
| `unknown_unhandled` | Fallback state |

Stage unknown reasons:

| Reason | Meaning |
|---|---|
| `warmup` | Warmup stage |
| `wide_range` | Wide range has no resolved stage |
| `transition` | Transition has conflicting evidence |
| `data_error` | Data error |
| `none` | Stage is resolved |

## Liquidity Event Definitions

Liquidity features are calculated from prior liquidity levels, not future information.

Prior levels:

```text
prior_liquidity_high = rolling_max(high.shift(1), lookback)
prior_liquidity_low = rolling_min(low.shift(1), lookback)
```

Event components:

| Component | Definition |
|---|---|
| `swept_high` | Current high exceeds `prior_liquidity_high` plus optional ATR buffer |
| `swept_low` | Current low falls below `prior_liquidity_low` minus optional ATR buffer |
| `reclaimed_high` | High is swept but close returns below prior high |
| `reclaimed_low` | Low is swept but close returns above prior low |
| `breakout_acceptance_high` | Close accepts above prior high plus buffer |
| `breakout_acceptance_low` | Close accepts below prior low minus buffer |
| `failed_breakout_high` | High was swept but upside acceptance did not occur |
| `failed_breakout_low` | Low was swept but downside acceptance did not occur |

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

`none` liquidity events can dominate trade or market samples. None-context diagnostics add context for those residual rows.

ATR percentile:

```text
atr_percentile_100[t] = percentile_rank(atr_14[t], atr_14[t-99:t])
```

ATR percentile bucket:

| Bucket | Rule |
|---|---|
| `unknown` | Rolling ATR sample not ready |
| `low` | Percentile < 25 |
| `normal` | 25 <= percentile < 75 |
| `high` | 75 <= percentile < 90 |
| `extreme` | Percentile >= 90 |

Key-level proximity bucket:

| Bucket | Rule |
|---|---|
| `near_prior_high_no_breach` | Close is near prior high and high did not breach it |
| `near_prior_low_no_breach` | Close is near prior low and low did not breach it |
| `mid_range_far_from_key_level` | Prior high/low and ATR exist, but neither near condition holds |
| `breached_or_unknown` | Prior levels missing, ATR missing, or level breached |

## MTF State Keys

MTF keys compress H1/H4/D1 context into readable audit strings.

| Key | Formula |
|---|---|
| `mtf_state_key` | `d1_structure_trend_state|h4_market_stage|h1_market_stage` |
| `mtf_structure_key` | `d1_structure_trend_state|h4_structure_trend_state|h1_structure_trend_state` |
| `mtf_stage_key` | `d1_market_stage|h4_market_stage|h1_market_stage` |
| `mtf_event_state_key` | `d1_structure_trend_state|h4_market_stage|h1_market_stage|liquidity_event_type` |

If a component is missing, it is rendered as `data_error`. `mtf_event_state_key` is only produced when `liquidity_event_type` exists.

MTF keys are not filters by themselves. They are diagnostic descriptors that must pass count and stability requirements before becoming candidate hypotheses.

## Configuration Column Sets

Configuration column sets are reusable research-only grouping specs. They describe how market-state context is projected before diagnostics compute intrinsic quality, conditional information, or side-normalized forward quality.

Configuration specs are shared; workflow adapters resolve each spec to columns available in the current dataframe. Missing columns are skipped schema-stably. Strategy code must not consume these specs directly.

Current configuration roles:

| Role | Meaning | Typical Use |
|---|---|---|
| `static_state` | Raw or audit-level state keys such as `mtf_structure_key` | Static baseline and sparsity comparison |
| `reduced_static_state` | Lower-cardinality semantic projections such as D1/H4/H1 reduced state | Configuration reduction and joint buckets |
| `inner_connection` | D1-to-H4 or H4-to-H1 relationships | Tests whether timeframe links should be preserved or merged |
| `transition_state` | Previous-to-current state paths | Dynamic state-transition diagnostics |
| `transition_inner_connection` | Previous-to-current inner-connection paths | Dynamic multi-timeframe relationship diagnostics |
| `event_side_joint` | Event-derived side plus state context | Side-normalized joint forward quality |

Initial named configuration specs:

| Configuration | Role | Source Columns |
|---|---|---|
| `config_static_raw_mtf` | `static_state` | `mtf_structure_key` |
| `config_reduced_d1_event` | `reduced_static_state` | `d1_market_stage_reduced` |
| `config_reduced_d1_structure_event` | `reduced_static_state` | `d1_structure_trend_state` |
| `config_reduced_d1_h4_h1_event` | `reduced_static_state` | `d1_structure_trend_state`, `h4_market_stage_reduced`, `market_stage_reduced` |
| `config_dynamic_stage_transition_event` | `transition_state` | `d1_market_stage_reduced_transition`, `h4_market_stage_reduced_transition`, `market_stage_reduced_transition` |
| `config_dynamic_inner_connection_event` | `transition_inner_connection` | `reduced_inner_connection_path_transition` |

Inner-connection columns are deterministic no-lookahead descriptions of relationships among already-known timeframe states:

| Column | Formula |
|---|---|
| `d1_to_h4_stage_connection` | `d1_market_stage_reduced->h4_market_stage_reduced` |
| `h4_to_h1_stage_connection` | `h4_market_stage_reduced->market_stage_reduced` |
| `d1_to_h4_structure_connection` | `d1_structure_trend_state->h4_structure_trend_state` |
| `h4_to_h1_structure_connection` | `h4_structure_trend_state->structure_trend_state` |
| `reduced_inner_connection_path` | `d1_to_h4_stage_connection|h4_to_h1_stage_connection` |

Transition columns use previous-to-current paths within each symbol:

```text
state_transition = state[t-1] + "->" + state[t]
connection_transition = connection[t-1] + "->" + connection[t]
```

Configuration sets feed the reduced workflow:

| Workflow | Reuse Rule |
|---|---|
| `joint-forward-quality` | Canonical owner for configured joint state/event/side buckets and support artifacts |
| `timeframe-classifier` | Health-only view of independent timeframe classifier outputs |
| Internal forward outcome service | Outcome labels used by `joint-forward-quality`; not a runnable diagnostic mode |
| Dynamic support rows | Configuration-design evidence embedded in joint-quality work |
| `trade-record-modulation` | Requires an `entry_*` adapter and remains post-trade control evidence only |

## Classifier Diagnostics

Classifier diagnostics evaluate label coverage, consistency, cardinality, transition behavior, and right-edge drift.

| Diagnostic | Calculation | Interpretation |
|---|---|---|
| Classifier coverage | Required classifier columns present divided by required column count | Missing columns mean classifier output is incomplete |
| Stage distribution | Count and percent by `market_stage` | Detects dominant or missing stages |
| Trend distribution | Count and percent by `structure_trend_state` | Detects structure-state concentration |
| Reason distribution | Count and percent by `stage_unknown_reason` and `structure_reason` | Explains unresolved labels |
| Unknown reason consistency | Matrix of stage, structure, and reason fields with contradiction checks | Flags impossible combinations |
| Resolved none audit | Counts `stage_unknown_reason=none` and contradictions | Confirms `none` means resolved rather than missing |
| Raw unknown attribution | Counts raw unknown structure with resolved stage reasons | Detects partially unresolved states |
| Threshold distribution | Quantiles and source of range-width threshold | Audits fixed vs rolling threshold behavior |
| Structure x stage matrix | Counts `structure_trend_state` by `market_stage` | Checks semantic alignment |
| Stage x reason matrix | Counts `market_stage` by `stage_unknown_reason` | Checks reason consistency |
| MTF state cardinality | Unique counts for MTF keys | High cardinality warns of sparse buckets |
| MTF transition summary | Counts changes between adjacent state keys | Measures churn |
| MTF dwell distribution | Run-length statistics for state keys | Measures persistence |
| MTF time distribution | State counts by time bucket | Detects temporal concentration |
| MTF right-edge drift | Recent-state instability check | Warns if latest rows churn excessively |

Classifier diagnostics answer whether state labels are usable. They do not answer whether a strategy is profitable.

Classifier exports also include a machine-readable health table with `health_check`, `status`, `value`, `threshold`, and `reason`. Initial checks cover required classifier columns, contradiction count, raw-unknown-with-resolved-none share, MTF key cardinality, transition churn, and right-edge drift. Health failures are diagnostic gates; they do not block report generation by default.

## Trade-Record State Diagnostics

Trade-record diagnostics use executed trades. They are strategy-conditioned because a strategy's entry rules decide which states are sampled.

Common entry-state fields in trade records use the `entry_` prefix:

| Trade Field | Source |
|---|---|
| `entry_market_stage` or `entry_market_stage_bucket` | H1 stage at entry |
| `entry_market_stage_reason` or `entry_market_stage_reason_bucket` | H1 stage reason at entry |
| `entry_liquidity_event_type` or `entry_liquidity_event_type_bucket` | H1 event at entry |
| `entry_structure_trend_state` or `entry_structure_bucket` | H1 structure at entry |
| `entry_h4_market_stage` | H4 stage known at entry |
| `entry_d1_market_stage` | D1 stage known at entry |
| `side` | Trade direction normalized to `long` or `short` |

State profitability diagnostics group actual trades and compute outcome statistics such as trade count, expectancy, net PnL, and risk flags. They identify where strategy PnL happened, not why the market state itself is predictive.

## Trade-Record Control

`trade-record-modulation` tests whether a modulator changes the outcome of a base feature among executed trades. It is optional control evidence under `research-evaluation`, not a direct diagnostic mode.

Default base features:

| Base Feature | Meaning |
|---|---|
| `entry_market_stage_bucket` | H1 stage at entry |
| `entry_market_stage_reason_bucket` | H1 stage reason at entry |
| `entry_liquidity_event_type_bucket` | H1 event at entry |
| `side` | Normalized direction |
| `entry_structure_bucket` | H1 structure at entry |

Default modulators:

| Modulator | Meaning |
|---|---|
| `entry_d1_structure_trend_state` | D1 structure at entry |
| `entry_d1_market_stage` | D1 stage at entry |
| `entry_h4_structure_trend_state` | H4 structure at entry |
| `entry_h4_market_stage` | H4 stage at entry |
| `entry_atr_percentile_bucket` | H1 ATR regime at entry |
| `entry_adx_bucket` | H1 ADX regime at entry |

For a base value `b` and modulator value `m`:

```text
base_group = trades where base_feature = b
conditional_group = trades where base_feature = b and modulator = m
base_expectancy = mean(net_pnl_usd in base_group)
conditional_expectancy = mean(net_pnl_usd in conditional_group)
delta_expectancy = conditional_expectancy - base_expectancy
```

Standard-error band:

```text
se(values) = sample_std(values) / sqrt(n)
delta_se = sqrt(se(base_group)^2 + se(conditional_group)^2)
delta_ci = delta_expectancy +/- 1.96 * delta_se
```

Significance:

```text
significant = base_trades >= min_base_trades
              and conditional_trades >= min_cell_trades
              and abs(delta_expectancy) >= practical_delta_threshold
              and delta_ci excludes zero
```

Classification:

| Classification | Meaning |
|---|---|
| `insufficient` | Base or conditional trade count below threshold |
| `unstable` | Counts pass but significance or symbol stability fails |
| `asset_specific` | Significant but not proven global |
| `global` | Aggregate significant with stable symbol signs |

Trade-record modulation is useful for diagnosing existing strategies but cannot measure states the strategy never entered.

Classifier version matters for trade-record modulation. Trade entry-state buckets are captured from the prepared strategy frame, so changing the classifier profile or threshold parameters can move trades between `entry_market_stage_bucket`, `entry_market_stage_reason_bucket`, and `entry_structure_bucket` groups without changing trade execution mechanics. A modulation run should therefore record the classifier profile and threshold settings alongside the CSV export.

Aggregate significant rows are not sufficient for authorization. A row can have `significant=true` but remain `classification=unstable` when symbol-level signs are sparse or inconsistent. Alias-equivalent rows, such as `side=long`, `entry_structure_bucket=uptrend`, and `entry_liquidity_event_type_bucket=failed_breakout_low` for a long-only failed-breakout pocket, should be interpreted as one underlying observation rather than multiple independent discoveries.

Trade-record modulation uses dollar expectancy by default. Until normalized trade return or R-multiple fields are available, `practical_delta_threshold` is dollar-based in this path and should not be compared directly with market forward percentage-point thresholds.

## Internal Forward Outcome Service

The reduced API does not expose `market-state-forward` as a runnable diagnostic mode. `joint-forward-quality` uses an internal forward outcome service to label every eligible classifier row before side-normalized joint buckets are computed.

For horizon `N`:

```text
future_close_N = close[t + N]
future_high_N = max(high[t + 1], ..., high[t + N])
future_low_N = min(low[t + 1], ..., low[t + N])
```

Forward outcome formulas:

| Outcome | Formula |
|---|---|
| `fwd_N_return_pct` | `(future_close_N / close[t] - 1) * 100` |
| `fwd_N_return_atr` | `(future_close_N - close[t]) / atr_14[t]` |
| `fwd_N_max_up_pct` | `(future_high_N / close[t] - 1) * 100` |
| `fwd_N_max_down_pct` | `(future_low_N / close[t] - 1) * 100` |
| `fwd_N_mfe_long_atr` | `(future_high_N - close[t]) / atr_14[t]` |
| `fwd_N_mae_long_atr` | `(close[t] - future_low_N) / atr_14[t]` |
| `fwd_N_mfe_short_atr` | `(close[t] - future_low_N) / atr_14[t]` |
| `fwd_N_mae_short_atr` | `(future_high_N - close[t]) / atr_14[t]` |

The last `N` rows per symbol do not have complete future windows and are excluded from summaries for that horizon.

Forward summary rows group by state fields and compute:

| Field | Meaning |
|---|---|
| `rows` | Complete future windows in the group |
| `up_count`, `down_count`, `flat_count` | Direction labels using return threshold |
| `mean_return_pct`, `median_return_pct` | Forward return central tendency |
| `positive_mean`, `negative_mean_abs` | Average positive return and absolute average negative return |
| `positive_rate`, `negative_rate` | Positive/negative outcome percentages |
| `omega_ratio` | `sum(max(return, 0)) / abs(sum(min(return, 0)))` |
| `sortino_zero` | `mean_return_pct / sqrt(mean(min(return, 0)^2))` |
| `pwpr` | Probability-weighted payoff ratio: `(positive_mean * positive_rate) / (negative_mean_abs * negative_rate)` |
| `mean_return_atr` | ATR-normalized close-to-close movement |
| `mean_mfe_*_atr`, `mean_mae_*_atr` | Long/short path opportunity and adversity |
| `return_ci_low`, `return_ci_high` | Exploratory standard-error band |
| `effective_rows` | Autocorrelation-shrunk sample size estimate |
| `overlap_lag` | Maximum lag used for overlap metadata, normally `horizon - 1` |
| `overlap_adjusted_ci_low`, `overlap_adjusted_ci_high` | Effective-sample adjusted exploratory band |
| `overlap_warning` | Warns when forward windows overlap |
| `directional_bias` | `up`, `down`, `flat_or_mixed`, or `insufficient` |

Overlap adjustment uses positive-autocorrelation shrinkage:

```text
n_eff = n / max(1, 1 + 2 * sum(max(rho_lag, 0)))
adjusted_se = sample_std / sqrt(n_eff)
```

This is deterministic metadata for robustness review, not a formal proof of independent observations.

Robust calculations are dataframe-first. Grouping, aggregation, FDR annotation, and summaries are represented as `pl.DataFrame` artifacts; NumPy is used only for numeric kernels such as Newey-West standard errors and standardized effect sizes.

### Removed Support Artifacts

Standalone conditional dependency, transition-quality, and continuous-condition exports were removed from the reduced API. Their useful role is now configuration design support inside `joint-forward-quality` artifacts such as `configuration-intrinsic-quality`, `transition-event-quality`, `joint-reduction-comparison`, and `inner-connection-reduction-quality`.

Artifact families:

| Artifact | Purpose |
|---|---|
| `conditional-information` | Measures empirical information gain from state variables about forward direction, optionally conditioned on another state variable |
| `state-transition-quality` | Reuses forward-summary metrics on previous-to-current state paths such as `range->markup` |
| `continuous-condition-quality` | Reuses forward-summary metrics on deterministic continuous-structure buckets such as range-width-to-threshold or range-position buckets |

Conditional information uses plug-in empirical entropy:

```text
H(Y) = -sum_y p(y) log2 p(y)
H(Y | Z) = sum_z p(z) H(Y | Z=z)
H(Y | X,Z) = sum_xz p(x,z) H(Y | X=x,Z=z)
I(Y; X) = H(Y) - H(Y | X)
I(Y; X | Z) = H(Y | Z) - H(Y | X,Z)
normalized_cmi = I(Y; X | Z) / max(H(Y | Z), eps)
```

Default outcome is `fwd_N_direction`. CMI is association evidence, not causal proof. Promotion review must still require sufficient cells, stability, materiality, and execution-aware backtests.

Dependency rows include support and sparsity controls: `rows`, `min_rows`, `min_cell_rows`, `unique_x`, `unique_z`, `cells`, `sufficient_cells`, `cell_coverage_pct`, `bias_warning`, `sufficient`, and `classification`.

Transition paths and continuous buckets obey the no-lookahead contract. They use current or previous known state/structure columns only; forward outcome columns are labels and are never fed back into grouping construction.

## Joint-Forward-Quality Diagnostics

`joint-forward-quality` builds the missing endogenous-quality table for side-normalized joint buckets. It answers:

```text
historically, for this known-at-close joint market context + liquidity event + side,
what was the side-normalized forward return distribution?
```

It is strategy-independent. It does not create entries, exits, stops, targets, basket actions, or strategy filters.

### Event Side Mapping

Only explicit directional liquidity events are converted into side-normalized rows. Ambiguous or `none` events are excluded from candidate side-normalized rows.

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

This lets long and short opportunities use the same favorable-is-positive metric vocabulary.

### Artifact Families

`joint-forward-quality.csv` can contain multiple artifact families under one output:

| Artifact | Purpose |
|---|---|
| `configuration-intrinsic-quality` | Scores whether a configuration is structurally usable before inspecting returns |
| `joint-forward-quality` | Scores static/reduced joint state + event + side buckets |
| `transition-event-quality` | Scores dynamic previous-to-current state/connection paths + event + side buckets |
| `joint-reduction-comparison` | Compares raw, reduced, and dynamic configuration families |
| `inner-connection-reduction-quality` | Diagnoses whether D1/H4/H1 inner connections should be preserved, merged, reduced, or rejected |

### Joint Bucket Metrics

Joint quality rows use return-distribution vocabulary on side-normalized returns:

| Field | Meaning |
|---|---|
| `bucket_family` | Configuration role such as `reduced_static_state` or `transition_state` |
| `configuration_name` | Named configuration spec used to form the bucket |
| `configuration_kind` | Static, reduced, transition, or inner-connection class |
| `joint_group` | Pipe-delimited state/configuration value |
| `joint_group_columns` | Source columns used to construct `joint_group` |
| `liquidity_event_type` | Directional event in the bucket |
| `side` | Event-implied `long` or `short` |
| `rows` | Complete forward windows in the bucket |
| `mean_side_return_pct` | Mean side-normalized forward return |
| `positive_rate`, `negative_rate` | Side-normalized positive/negative percentages |
| `positive_mean`, `negative_mean_abs` | Average favorable and adverse return magnitudes |
| `omega_ratio` | Favorable return mass divided by adverse return mass |
| `pwpr` | Probability-weighted payoff ratio on side-normalized returns |
| `sortino_zero` | Mean side return divided by downside deviation around zero |
| `directional_bias` | `up`, `flat_or_mixed`, or `insufficient` for side-normalized direction |
| `invalid_state_present` | Whether the group contains invalid state labels such as `warmup`, `unknown`, or `data_error` |
| `passes_candidate_gate` | Diagnostic gate result; not strategy authorization |
| `gate_failure_reasons` | Comma-separated failed gates such as `rows`, `omega`, `pwpr`, `direction`, or `invalid_state` |

Initial diagnostic candidate gate:

```text
rows >= min_rows
omega_ratio > omega_threshold
pwpr > pwpr_threshold
directional_bias == up
invalid_state_present == false
```

Default values in the focused config are `min_rows=30`, `omega_threshold=1.5`, and `pwpr_threshold=2.0`.

### Intrinsic Configuration Quality

Intrinsic rows are emitted before return quality is judged. They answer whether a configuration is usable as a partition of market states.

| Field | Meaning |
|---|---|
| `bucket_count` | Number of unique configuration buckets |
| `valid_bucket_count`, `invalid_bucket_count`, `invalid_bucket_pct` | Invalid-state leakage checks |
| `median_bucket_rows`, `p10_bucket_rows`, `p90_bucket_rows` | Bucket support distribution |
| `entropy`, `normalized_entropy` | Concentration or over-fragmentation of buckets |
| `compression_ratio_vs_raw_mtf` | Bucket count versus raw MTF baseline |
| `coverage_rows`, `coverage_pct` | Rows covered by the configuration |
| `dominant_bucket_pct` | Share of rows in the largest bucket |
| `transition_changed_rate`, `self_transition_pct` | Dynamic churn for transition-like configurations |
| `intrinsic_quality_bucket` | `high`, `medium`, `low`, or `insufficient` |
| `intrinsic_warnings` | Warnings such as `invalid_leakage`, `sparse_buckets`, `over_merged`, `high_entropy_sparse`, or `dominant_bucket` |

Interpretation:

- Low entropy can mean over-merged states.
- High entropy with low median bucket rows can mean sparse high-cardinality fragmentation.
- A strong forward-quality row from a poor intrinsic configuration should not be promoted.

### Transition-Event Quality

`transition-event-quality` is the preferred Phase 2 dynamic ranking surface. It uses previous-to-current state paths rather than static labels.

Examples:

```text
market_stage_reduced_transition + liquidity_event_type + side
d1_market_stage_reduced_transition + h4_market_stage_reduced_transition + market_stage_reduced_transition + liquidity_event_type + side
reduced_inner_connection_path_transition + liquidity_event_type + side
```

Transition rows include static-baseline comparison fields when available:

| Field | Meaning |
|---|---|
| `static_baseline_rows` | Rows in the corresponding static baseline, if computed |
| `static_baseline_omega_ratio` | Omega of the static baseline |
| `dynamic_lift_vs_static_omega` | Dynamic transition Omega minus static baseline Omega |
| `dynamic_lift_vs_static_mean_return_pct` | Dynamic transition mean side return minus static baseline mean side return |

Dynamic rows should be ranked with stability and lift ahead of raw Omega:

```text
1. invalid_state_present == false
2. rows >= transition_min_rows
3. symbol/time agreement when available
4. dynamic lift versus static baseline
5. omega_ratio
```

This prevents isolated static-like high-Omega pockets from dominating the dynamic candidate list.

### Reduction And Inner-Connection Decisions

`joint-reduction-comparison` summarizes each configuration family:

| Field | Meaning |
|---|---|
| `total_buckets` | Number of emitted quality buckets |
| `sufficient_buckets` | Buckets passing row-count sufficiency |
| `candidate_gate_buckets` | Buckets passing the diagnostic candidate gate |
| `median_bucket_rows`, `p90_bucket_rows` | Support distribution |
| `median_omega`, `p90_omega` | Omega distribution |
| `time_stable_buckets` | Time-stable buckets when available |
| `cross_asset_consistent_buckets` | Cross-asset-consistent buckets when available |
| `invalid_bucket_count` | Buckets containing invalid labels |

`inner-connection-reduction-quality` compares preserving, merging, reducing, or rejecting multi-timeframe inner connections.

| Field | Meaning |
|---|---|
| `merge_policy` | Evaluated policy such as `merge_adjacent` |
| `connection_family` | Connection family such as stage or structure connection |
| `raw_bucket_count`, `reduced_bucket_count` | Cardinality before and after reduction |
| `compression_ratio` | Reduced/raw bucket count |
| `information_retention_proxy` | Deterministic proxy for retained detail |
| `merge_decision` | `preserve`, `merge`, `reduce`, or `reject` |
| `decision_reason` | Diagnostic reason for the decision |

These decisions choose what to test next. They do not authorize strategy logic.

### Empirical Shrinkage Fields

The MVP uses deterministic empirical shrinkage, not full hierarchical Bayesian posterior modeling.

| Field | Meaning |
|---|---|
| `global_mean_side_return_pct` | Mean side-normalized return across the scope |
| `bucket_mean_side_return_pct` | Raw bucket mean side-normalized return |
| `shrinkage_weight` | `rows / (rows + prior_strength)` |
| `shrunk_mean_side_return_pct` | Bucket mean shrunk toward global mean |
| `shrunk_positive_rate` | Positive rate shrunk toward a neutral prior |
| `shrunk_omega_proxy` | Ranking proxy derived from shrunk mean |
| `rank_raw_omega`, `rank_shrunk_omega_proxy`, `rank_delta` | Raw-vs-shrunk ranking diagnostics |

Shrinkage is a ranking regularizer for sparse buckets. It is not proof of edge.

## Removed Outputs

Former public outputs and modes were removed rather than kept as compatibility aliases. Removed names include `classifier`, `tradability`, `market-state-forward`, `market-state-modulation`, `timeframe-tradability`, `timeframe-forward-quality`, `resonance-candidates`, `state`, `state-profitability`, `state-filter-delta`, and `modulation-effect`.

Replacement mapping:

| Removed surface | Replacement |
|---|---|
| Standalone classifier diagnostics | `timeframe-classifier` health rows |
| Tradability / ETI | Support concepts folded into classifier health and configuration intrinsic quality |
| Market-state forward/modulation | Internal forward labels plus `joint-forward-quality` rows |
| Timeframe forward quality | Side-normalized joint-quality rows |
| Resonance candidates | Configuration scanning and joint bucket quality |
| State/profitability/filter-delta modes | Backtest reports and explicit strategy analysis outside research diagnostics |
| Modulation-effect mode | Optional `trade-record-modulation` control output inside `research-evaluation` |

## Interpretation Rules

Use these rules when evaluating diagnostic outputs:

1. Missing classifier columns invalidate downstream state conclusions.
2. High-cardinality MTF keys are audit evidence, not immediate filters.
3. Trade-record modulation is strategy-conditioned and cannot discover skipped-state behavior.
4. Internal forward labels are strategy-independent but not executable PnL.
5. Semantic aliases, such as `market_stage=wide_range` and `market_stage_reason=wide_range_no_stage`, should not be counted as independent discoveries.
6. Joint-forward-quality candidate gates are diagnostic gates only; they are not strategy authorization.
7. Dynamic transition-event rows should be compared against static-state baselines before being treated as candidate signal-generation patterns.
8. Empirical shrinkage is a sparse-bucket ranking regularizer, not a Bayesian proof of edge.
9. Any candidate filter must be converted into explicit entry/exit mechanics and backtested with costs, slippage, stops, targets, basket constraints, and risk gates.

## Promotion Gate

A diagnostic finding may become a strategy hypothesis only if all conditions hold:

- State features are no-lookahead and classifier diagnostics are consistent.
- Semantic aliases are collapsed before counting discoveries or promotion candidates.
- Joint buckets use side-normalized returns and explicit event-to-side mapping.
- Configuration intrinsic quality is acceptable; poor-coverage, over-merged, or high-entropy sparse configurations are rejected before performance review.
- Dynamic transition rows beat their static baseline or have a stated reason for being evaluated independently.
- Counts pass thresholds at both aggregate and symbol levels, with market-state conditional rows above `100` before promotion.
- The forward effect is economically material, such as `abs(delta_return_pct) > 0.5` percentage points and `abs(delta_cohens_d) > 0.3` before execution testing.
- Direction or delta is stable across assets or intentionally scoped to one asset with a stated reason.
- The effect survives time segmentation or walk-forward validation.
- FDR/global status is treated as a screening gate, not an authorization gate.
- The proposed rule has an ex-ante market-behavior rationale.
- The strategy implementation includes executable timing, stop/target handling, fees, and slippage.
- The resulting backtest passes the normal risk and comparability gates.

Until those gates pass, the finding remains diagnostic-only.
