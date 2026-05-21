# Research Evaluation Report

Date: 2026-05-21

## Executive Summary

This report records the current layered research-evaluation evidence for higher-timeframe and regime context. The research direction is market-first: validate classifier state health, rank tradable state priors, measure forward market behavior, then use executed trades only as a post-trade control.

Current decision: diagnostic-only. No modulation gate, MTF key, market-stage filter, or strategy variant is authorized.

| Layer | Observation Unit | Current Evidence | Decision |
|---|---|---|---|
| Classifier | Prepared H1 classifier rows with H4/D1 context | Expanded graph produced `216` classifier row artifacts, all `info` severity | Usable for downstream diagnostics, with validity warnings noted |
| Tradability | Prepared H1 market-state rows | `2,287` medium, `15` low, `2,787` insufficient, `0` high buckets | Hypothesis-prior only |
| Market-state forward | Every eligible H1 market bar | Robust completed smoke found `12,395` forward summaries and `11,832` modulation rows | Candidate generation only |
| Market-state modulation | Forward market outcomes conditional on context | `46` FDR-significant rows, only `9` global and FDR-significant; effects are mostly small | Not promotion-ready |
| Trade-record modulation | Executed strategy trades | Current-code recheck found no `global` or `asset_specific` post-trade effects | Negative-control only |
| Strategy backtest | Executable entries and basket lifecycle | Deferred until a market-derived hypothesis passes robustness and materiality gates | No strategy change |

## Full Graph Run Status

The expanded layered API was run with:

```bash
uv run python scripts/research.py --config configs/research/research-evaluation-expanded.toml
```

The config writes generated artifacts outside the repository:

```text
F:\Stratum\TEMP\kilo\qooi-research-evaluation-expanded
```

The run exceeded the 15-minute tool timeout after writing classifier and tradability exports. It did not complete the expanded market-state-forward, market-state-modulation, or trade-record-modulation branches within that execution window. Therefore, this report uses:

| Evidence Source | Status | Artifact |
|---|---|---|
| Expanded `research-evaluation` classifier | Completed | `F:\Stratum\TEMP\kilo\qooi-research-evaluation-expanded\classifier.csv` |
| Expanded `research-evaluation` tradability | Completed | `F:\Stratum\TEMP\kilo\qooi-research-evaluation-expanded\tradability.csv` |
| Robust market-state smoke | Latest completed market-state evidence | `F:\Stratum\TEMP\kilo\qooi-market-state-forward-robust.csv` |
| Current-code trade modulation recheck | Latest completed post-trade evidence | `F:\Stratum\TEMP\kilo\qooi-*-modulation-effect-current.csv` |

This distinction matters: the full API path is now the intended orchestration surface, but the expanded market-state branch still needs a longer run or reduced workload to produce a complete expanded graph artifact.

## Evidence Graph

The API dependency graph is ordered, not a flat set of independent diagnostics.

| Output | Upstream Evidence | Role |
|---|---|---|
| `classifier` | Cache history to prepared classifier frame | Validates labels and feature health |
| `tradability` | Classifier frame to market-state reductions | Ranks persistent/separable states as priors |
| `market-state-forward` | Classifier frame to future market outcomes | Measures what happened after every eligible state |
| `market-state-modulation` | Market-state-forward outcomes | Tests context-conditioned forward deltas |
| `trade-record-modulation` | Strategy signal and backtest branch to trades | Checks sampled strategy trades as a post-trade control |

The graph prevents state leakage: classifier, tradability, and market-state paths do not apply strategy signal filters. Strategy signal filters remain confined to signal/backtest and trade-record paths.

## Classifier Evidence

Expanded graph classifier export:

| Artifact | Rows |
|---|---:|
| `row` | 216 |
| `table` | 1,149,772 |

Classifier row severity:

| Severity | Rows |
|---|---:|
| `info` | 216 |

Interpretation:

- The completed expanded classifier branch did not emit warning or error severity rows in the summary row artifact.
- Classifier health remains a prerequisite, not a proof of strategy edge.
- The large table artifact count reflects supporting distribution and audit tables; it should not be interpreted as independent discoveries.

## Tradability Evidence

Expanded graph tradability export:

| Artifact | Rows |
|---|---:|
| `state-tradability` | 5,089 |
| `classifier-validity` | 108 |

Tradability buckets:

| Bucket | Rows |
|---|---:|
| `medium` | 2,287 |
| `low` | 15 |
| `insufficient` | 2,787 |
| `high` | 0 |

State-family row counts:

| State Column | Rows |
|---|---:|
| `mtf_stage_key` | 4,871 |
| `market_stage_reduced` | 110 |
| `atr_percentile_bucket` | 60 |
| `structure_trend_state` | 48 |

Classifier-validity status counts:

| Status | Rows |
|---|---:|
| `pass` | 93 |
| `warn` | 15 |

Representative usable non-error priors:

| Symbol | State | Rows | ETI | Entropy Norm | Bucket |
|---|---|---:|---:|---:|---|
| BNB-USDT-SWAP | `atr_percentile_bucket=low` | 8,788 | 0.599 | 0.203 | `medium` |
| BNB-USDT-SWAP | `structure_trend_state=uptrend` | 14,557 | 0.597 | 0.208 | `medium` |
| AVAX-USDT-SWAP | `structure_trend_state=uptrend` | 22,040 | 0.597 | 0.209 | `medium` |
| OP-USDT-SWAP | `atr_percentile_bucket=low` | 10,131 | 0.595 | 0.216 | `medium` |
| DOGE-USDT-SWAP | `atr_percentile_bucket=low` | 15,812 | 0.594 | 0.212 | `medium` |
| XRP-USDT-SWAP | `atr_percentile_bucket=low` | 17,082 | 0.594 | 0.214 | `medium` |

Interpretation:

- No state reached the `high` tradability bucket.
- The best usable priors are broad, low-cardinality states such as low ATR buckets and uptrend structure states.
- High-cardinality `mtf_stage_key` rows still dominate the artifact count and remain prone to sparse-cell fragmentation.
- Tradability ranks where to inspect market-state-forward evidence; it does not authorize a filter.

## Market-State Forward Evidence

Latest completed robust market-state run:

```bash
uv run python scripts/research.py --config configs/research/market-state-forward-robust.toml
```

Artifacts:

```text
F:\Stratum\TEMP\kilo\qooi-market-state-forward-robust.csv
F:\Stratum\TEMP\kilo\qooi-market-state-forward-plots\market-state-modulation-heatmap.svg
F:\Stratum\TEMP\kilo\qooi-market-state-forward-plots\market-state-horizon-decay.svg
```

Completed market-state counts:

| Artifact | Rows |
|---|---:|
| Forward summary rows | 12,395 |
| Market-state modulation rows | 11,832 |
| Sufficient forward groups | 3,720 |
| Directional forward groups | 1,279 |

Market-state modulation classifications:

| Classification | Rows |
|---|---:|
| `global` | 266 |
| `asset_specific` | 566 |
| `unstable` | 6,743 |
| `insufficient` | 4,257 |

Robustness gates:

| Gate | Rows |
|---|---:|
| `robust_significant=true` | 262 |
| `fdr_significant=true` | 46 |
| `time_stable=true` | 3,472 |
| `cross_asset_homogeneous=true` | 1,900 |
| `classification=global and fdr_significant=true` | 9 |

Interpretation:

- Market-state-forward is the only source in this report that measures skipped states, because it uses every eligible market row rather than executed trades.
- Only `46` of `11,832` modulation rows survived FDR, and only `9` were both `global` and FDR-significant.
- The strongest global rows are statistically detectable but economically weak; many deltas are near `0.3%` to `0.5%` and standardized effects remain small to moderate.
- Sparse MTF-key rows can look attractive but are not reliable without semantic reduction, stronger counts, cross-asset agreement, and time stability.

## Trade-Record Modulation Evidence

Current-code post-trade modulation recheck artifacts:

```text
F:\Stratum\TEMP\kilo\qooi-baseline-modulation-effect-current.csv
F:\Stratum\TEMP\kilo\qooi-no-range-modulation-effect-current.csv
F:\Stratum\TEMP\kilo\qooi-no-range-longs-modulation-effect-current.csv
```

Refreshed trade-record counts:

| Run | Strategy | Trades | Rows | Insufficient | Unstable | Significant | Global | Asset-Specific |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline | `structure_event_trend_aligned_v1` | 124 | 1,189 | 954 | 235 | 0 | 0 | 0 |
| Full no-range | `structure_event_trend_aligned_no_range_v1` | 60 | 953 | 882 | 71 | 3 | 0 | 0 |
| No-range-longs | `structure_event_trend_aligned_no_range_longs_v1` | 93 | 1,141 | 1,008 | 133 | 3 | 0 | 0 |

The only significant rows in the no-range runs collapse to one semantic condition: long failed-breakout-low/uptrend trades when H4 structure is also uptrend. The condition has only `20` conditional trades, fails symbol stability, and is classified `unstable`.

Interpretation:

- No `global` or `asset_specific` post-trade modulation effect is present.
- Trade-record modulation remains a diagnostic of sampled strategy behavior, not a market-state discovery mechanism.
- Strategy-conditioned evidence must not be used to create a market-state rule for states the strategy never entered.

## Promotion Assessment

No candidate currently passes the promotion gate.

| Gate | Current Result |
|---|---|
| Classifier health | Expanded classifier branch completed, but validity warnings remain in tradability artifact |
| Tradability strength | No `high` bucket states |
| Count quality | MTF keys fragment samples; several interesting forward rows have low conditional counts |
| FDR and robustness | Only `46` FDR rows and `9` global/FDR rows in completed robust smoke |
| Economic materiality | Global effects are mostly too small for execution authorization |
| Cross-asset stability | Asset-specific rows do not justify universal filters |
| Time stability | Required for promotion; current results remain exploratory |
| Execution realism | Forward returns exclude fees, spread, slippage, stops, targets, sizing, and basket lifecycle |
| Post-trade confirmation | No stable trade-record modulation effect |

## Threats To Validity

| Threat | Impact | Mitigation |
|---|---|---|
| Expanded graph timeout | Full expanded market-state evidence is incomplete | Rerun with longer timeout or split graph outputs |
| Overlapping forward windows | Adjacent rows share future bars | Prefer Newey-West/effective-sample metadata and treat bands as exploratory |
| Multiple comparisons | Thousands of state/context rows inflate false discovery risk | Require FDR, semantic reduction, time split, and cross-asset checks |
| Sparse conditional cells | Large deltas can come from fragile pockets | Raise conditional threshold before promotion |
| Alias duplication | Related state/reason columns can represent the same phenomenon | Collapse semantic aliases before counting discoveries |
| High-cardinality MTF keys | Fragment samples and dominate insufficient rows | Prefer broad state priors before MTF combinations |
| Non-executable outcomes | Forward returns are not trade PnL | Convert only robust observations into explicit strategy hypotheses |
| Strategy-conditioned sampling | Executed trades omit skipped states | Use trade-record modulation only as a post-trade control |

## Final Decision

Current status remains diagnostic-only.

Decisions:

- Do not create a modulation-gated strategy variant.
- Do not wire tradability, market-state-forward, or market-state-modulation outputs directly into entry filters.
- Do not treat MTF state keys as authorization features.
- Treat tradability as a hypothesis-prior layer only.
- Treat market-state-forward as candidate generation only.
- Treat trade-record modulation as a negative-control/post-trade diagnostic only.
- Require semantic reduction, stronger counts, FDR, standardized effect size, time stability, cross-asset homogeneity, and execution-aware tests before strategy work.

## Next Work

1. Rerun `research-evaluation-expanded.toml` with a longer execution window or split outputs into market-state-only and trade-record-only graph runs.
2. Collapse semantic aliases before interpreting discovery counts.
3. Review the `9` global/FDR rows against effect-size materiality and symbol-level homogeneity.
4. Raise promotion thresholds for conditional market-state rows below `100`.
5. Convert only stable market observations into explicit hypotheses with direction, trigger, stop, target, fees, slippage, sizing, and basket constraints.
6. Backtest only hypotheses that pass classifier, tradability, forward, modulation, time-split, and cross-asset evidence gates.
