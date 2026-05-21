# Research Evaluation API Reference

Date: 2026-05-18

## Purpose

This document defines the layered `research-evaluation` API and the classifier, tradability, market-state, and trade-record diagnostics used for qooi strategy evaluation.

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
  "classifier",
  "tradability",
  "market-state-forward",
  "market-state-modulation",
  "trade-record-modulation",
]
include_backtest_report = true
write_exports = true
fail_fast = false
```

Output dependencies are resolved as a graph:

| Output | Required Upstream Evidence | Notes |
|---|---|---|
| `classifier` | Cache-backed prepared classifier frame | Validates state labels and supporting tables |
| `tradability` | `classifier` | Adds market-state reductions and state-prior scores |
| `market-state-forward` | `classifier` | Computes future market outcomes after every eligible bar |
| `market-state-modulation` | `market-state-forward` | Tests conditional forward-outcome deltas |
| `trade-record-modulation` | Strategy signal/backtest branch | Uses executed trades only, separate from classifier/market-state branches |

Dependency expansion is automatic. Requesting `market-state-modulation` also requests `market-state-forward`; requesting `tradability`, `market-state-forward`, or `market-state-modulation` also requests `classifier`.

State-leakage rule: classifier, tradability, market-state-forward, and market-state-modulation paths must not apply strategy signal filters. Strategy filters, entries, exits, basket lifecycle, and trade records belong only to the backtest and trade-record modulation branch.

## Evaluation Layers

Diagnostics are interpreted in this order:

1. Classifier and feature validity: are the per-bar state labels internally consistent and known without lookahead?
2. Strategy-sampled trade diagnostics: did the existing strategy perform differently in the states it actually entered?
3. Strategy-independent market diagnostics: what happened after every eligible market state, even when no strategy entered?
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

Default research evaluation uses swap OHLCV unless a run explicitly selects another source policy. The base classifier timeframe is H1. Higher-timeframe classifier context uses H4 and D1.

## No-Lookahead Contract

Classifier and grouping columns must be known by the H1 bar close.

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

The strategy-independent classifier frame is prepared by `prepare_classifier_frame(...)`.

Pipeline stages:

1. Load H1 cache for the selected symbol.
2. Add indicators with `add_indicators(...)`.
3. Add MACD histogram with `add_macd_histogram(...)`.
4. Add structure and stage features with `add_price_structure_stage_features(...)`.
5. Attach H1 aliases such as `h1_market_stage` and `h1_structure_trend_state`.
6. Load H4 and D1 context caches.
7. Build compact H4/D1 classifier context frames.
8. Attach higher-timeframe context with an as-of backward join.
9. Add compressed MTF state keys with `add_mtf_state_keys(...)`.

`market-state-forward` additionally ensures liquidity and none-context diagnostics are available:

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

## Trade-Record Modulation Effect

`modulation-effect` tests whether a modulator changes the outcome of a base feature among executed trades.

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

Trade-record modulation uses dollar expectancy by default. Until normalized trade return or R-multiple fields are available, `practical_delta_threshold` is dollar-based in this path and should not be compared directly with market-state-forward percentage-point thresholds.

## Market-State-Forward Diagnostics

`market-state-forward` uses every eligible H1 classifier row instead of trade rows.

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

## Market-State Modulation Effect

Market-state modulation compares a base H1 state against a higher-timeframe context using forward market returns.

Default base columns:

| Base Column | Meaning |
|---|---|
| `market_stage` | H1 stage |
| `market_stage_reason` | H1 stage reason |
| `liquidity_event_type` | H1 event |
| `structure_trend_state` | H1 structure |
| `atr_percentile_bucket` | H1 volatility bucket |

Default modulator columns:

| Modulator Column | Meaning |
|---|---|
| `d1_structure_trend_state` | D1 structure known at H1 close |
| `d1_market_stage` | D1 stage known at H1 close |
| `h4_structure_trend_state` | H4 structure known at H1 close |
| `h4_market_stage` | H4 stage known at H1 close |
| `h4_market_stage_reason` | H4 stage reason known at H1 close |

For outcome `fwd_10_return_pct`:

```text
base_rows = rows where base_column = base_value and fwd_10_return_pct is complete
conditional_rows = rows where base_column = base_value and modulator_column = modulator_value
base_mean_return_pct = mean(fwd_10_return_pct in base_rows)
conditional_mean_return_pct = mean(fwd_10_return_pct in conditional_rows)
delta_return_pct = conditional_mean_return_pct - base_mean_return_pct
```

The standard-error band uses the same deterministic formula as trade-record modulation, but the outcome is forward return percent rather than trade PnL dollars.

Default smoke thresholds:

| Threshold | Value |
|---|---:|
| `market_state_min_rows` | 30 for forward summaries |
| `market_state_min_base_rows` | 100 for modulation base groups |
| `market_state_min_cell_rows` | 30 for modulation conditional cells |
| `market_state_delta_threshold_pct` | 0.15 percentage points |

The practical delta threshold can be fixed or cost-linked:

```text
--market-state-delta-mode fixed
--market-state-delta-mode cost_multiple
practical_delta_threshold_pct = market_state_cost_pct * market_state_cost_multiple
```

Defaults preserve the fixed `0.15` percentage-point behavior. Cost-linked mode is opt-in and does not infer spread or slippage from data.

Custom market-state modulation columns can be supplied with `--market-state-base-columns` and `--market-state-modulator-columns`. Trade-record modulation has analogous `--modulation-base-columns` and `--modulation-modulator-columns`. Missing custom columns resolve through existing aliases when possible and otherwise produce schema-stable empty or partial outputs.

Cross-asset classification:

| Scope | Rule |
|---|---|
| `ALL` row | Uses all symbols together |
| Symbol row | Uses one symbol only |
| Stable across symbols | Sufficient symbol-level rows mostly agree in delta sign |
| `global` | Aggregate row significant and stable across symbols |
| `asset_specific` | Significant effect not established as global |

Market-state modulation rows include overlap-aware delta metadata: `effective_base_rows`, `effective_conditional_rows`, `overlap_adjusted_delta_ci_low`, `overlap_adjusted_delta_ci_high`, and `overlap_warning`. Existing `delta_ci_low` and `delta_ci_high` remain for compatibility.

Robust market-state modulation exports append stricter fields without removing existing columns:

| Field | Meaning |
|---|---|
| `outcome_column`, `outcome_kind` | Active forward outcome, such as `fwd_10_return_pct` / `return_pct` |
| `se_method` | Robustness method: `iid`, `effective_n`, `newey_west`, or `bootstrap` |
| `robust_delta_ci_low`, `robust_delta_ci_high` | Delta band from the selected standard-error method |
| `robust_significant` | Count, effective-count, practical-delta, and robust-band gate |
| `delta_return_atr` | Companion ATR-normalized mean delta when available |
| `delta_cohens_d` | Standardized effect size for the active outcome |
| `base_q10`, `conditional_q10`, `delta_q10` | Lower-tail quantile comparison |
| `base_cvar10`, `conditional_cvar10`, `delta_cvar10` | Lower-tail conditional mean comparison |
| `p_value`, `fdr_alpha`, `fdr_significant` | Normal-approximation p-value and optional Benjamini-Hochberg result |
| `time_stable` | Time-split sign agreement plus segment materiality/significance check |
| `cross_asset_homogeneous`, `partially_replicable` | Symbol-level effect consistency metadata |

Rows also include time-split stability fields: `time_splits`, `sufficient_time_splits`, `time_split_sign_agreement_pct`, and `meta_stable`. The first implementation splits complete forward rows into chronological quantile segments and checks whether sufficient segments agree in delta sign. `meta_stable` is a promotion-gate aid, not an automatic strategy authorization.

Visualization is generated from exported dataframe artifacts through `src/qooi/core/plot.py`. Plot functions do not recompute diagnostics; they render already-exported modulation fields such as `delta_cohens_d`, `delta_return_pct`, `fdr_significant`, `time_stable`, and `horizon`.

## Interpretation Rules

Use these rules when evaluating diagnostic outputs:

1. Missing classifier columns invalidate downstream state conclusions.
2. High-cardinality MTF keys are audit evidence, not immediate filters.
3. Trade-record modulation is strategy-conditioned and cannot discover skipped-state behavior.
4. Market-state-forward is strategy-independent but not executable PnL.
5. Confidence bands on forward summaries are exploratory because forward windows overlap.
6. Prefer robust CI, FDR, standardized effect, overlap-adjusted, cross-asset, and time-split stability fields when judging promotion readiness.
7. Semantic aliases, such as `market_stage=wide_range` and `market_stage_reason=wide_range_no_stage`, should not be counted as independent discoveries.
8. Any candidate filter must be converted into explicit entry/exit mechanics and backtested with costs, slippage, stops, targets, basket constraints, and risk gates.

## Tradability And Validity Artifacts

The `tradability` diagnostic mode emits strategy-independent artifacts:

- `state-tradability`: endogenous state scores built from transition entropy, return autocorrelation, and volatility efficiency. The resulting ETI is a hypothesis-prior diagnostic, not a strategy signal.
- `classifier-validity`: classifier audit rows for label coverage, transition stability, regime separation, and liquidity-event enrichment.

These artifacts are dependency-ordered evidence layers rather than independent proofs. Forward rows answer what happened after states, modulation rows test conditional deltas, tradability rows score state structure, and validity rows audit the classifier partition itself.

## Promotion Gate

A diagnostic finding may become a strategy hypothesis only if all conditions hold:

- State features are no-lookahead and classifier diagnostics are consistent.
- Semantic aliases are collapsed before counting discoveries or promotion candidates.
- Counts pass thresholds at both aggregate and symbol levels, with market-state conditional rows above `100` before promotion.
- The forward effect is economically material, such as `abs(delta_return_pct) > 0.5` percentage points and `abs(delta_cohens_d) > 0.3` before execution testing.
- Direction or delta is stable across assets or intentionally scoped to one asset with a stated reason.
- The effect survives time segmentation or walk-forward validation.
- FDR/global status is treated as a screening gate, not an authorization gate.
- The proposed rule has an ex-ante market-behavior rationale.
- The strategy implementation includes executable timing, stop/target handling, fees, and slippage.
- The resulting backtest passes the normal risk and comparability gates.

Until those gates pass, the finding remains diagnostic-only.
