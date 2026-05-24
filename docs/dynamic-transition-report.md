# Dynamic Transition Discovery Report

Date: 2026-05-23

## Abstract

This report isolates the Stage 1 dynamic-transition evaluation from the broader research architecture. The experiment applies deterministic transition analysis to the handcrafted structure classifier and evaluates two questions:

- Is the handcrafted classifier structurally healthy enough to support transition diagnostics?
- Do deterministic state transitions expose measurable persistence, information content, or candidate return patterns worth further investigation?

The answer is yes for diagnostics and no for promotion. All classifier health checks passed, transition persistence is measurable, and 306 candidate-gated transition patterns were found. However, most candidates are sparse, strict promotion support is not fully wired, and no transition pattern is authorized for strategy conversion.

## Experiment

Run command:

```bash
uv run python scripts/classifier_states.py --config configs/research/research-evaluation-dynamic-transitions.toml
```

Export directory:

```text
F:\Stratum\TEMP\kilo\qooi-research-evaluation-dynamic-transitions
```

Requested outputs:

| Output | Purpose |
|---|---|
| `timeframe-classifier` | Validate handcrafted classifier health across symbol/timeframe slices. |
| `dynamic-transition-discovery` | Export transition graph and transition-information artifacts. |
| `pattern-quality` | Score transition and none-event context patterns against forward returns. |

Research universe:

- 12 swap instruments after excluding `XAU-USDT-SWAP`.
- 1H base transition frame.
- Timeframe classifier checks for 1H, 4H, and 1D.
- Cache target in config: 2600 days, 62000 bars, no minimum coverage failure threshold.

Default Stage 1 state columns:

| State Column | Meaning |
|---|---|
| `market_stage_reduced` | Base-timeframe handcrafted market stage. |
| `h4_market_stage_reduced` | 4H context market stage attached as known-at-close context. |
| `d1_market_stage_reduced` | 1D context market stage attached as known-at-close context. |
| `structure_trend_state` | Base-timeframe trend structure state. |

Forward-return horizons:

| Horizon | Meaning |
|---:|---|
| 3 | 3-bar forward return label. |
| 5 | 5-bar forward return label. |
| 10 | 10-bar forward return label. |

## Artifact Inventory

| Artifact | Rows | Role |
|---|---:|---|
| `timeframe-classifier.csv` | 144 | Classifier structural health checks. |
| `state-transition-graph.csv` | 2,075 | Directed empirical transition edges. |
| `transition-information.csv` | 48 | Transition-information rows by symbol and state column. |
| `transition-ngram-quality.csv` | 89,040 | Return-quality rows for 2-step and 3-step transition paths. |
| `none-event-context-quality.csv` | 582 | Return-quality rows for `liquidity_event_type = none` contexts. |
| `scored-patterns.csv` | 89,622 | Full candidate-gated pattern metric table. |
| `promotion-candidates.csv` | 0 | Strict-promotion export; currently conservative because strict support wiring is incomplete. |

## Classifier Health

All 144 handcrafted classifier checks passed.

| Timeframe | Checks | Passes | Minimum Health Value |
|---|---:|---:|---:|
| 1H | 48 | 48 | 4.0 |
| 4H | 48 | 48 | 4.0 |
| 1D | 48 | 48 | 4.0 |

Classifier-health interpretation:

- Required columns were present for every symbol/timeframe slice.
- `market_stage` cardinality was stable at 9 to 10 states.
- `structure_trend_state` cardinality was stable at 4 states.
- `liquidity_event_type` cardinality was stable at 7 labels.
- The handcrafted classifier is structurally suitable for Stage 1 transition diagnostics.

Classifier-health limitation:

- The current health check is structural, not behavioral. It does not yet measure state balance, persistence calibration, unknown/warmup share, label churn, or cross-timeframe agreement.

## Transition Information

Transition information measures how much the current state is explained by the previous state. Normalized transition information divides this by state entropy, giving a persistence-oriented scale where higher values indicate stronger dependence on prior state.

| State Column | Symbols | Rows | Avg TI | Min TI | Max TI | Avg Normalized TI | Min Normalized TI | Max Normalized TI |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `d1_market_stage_reduced` | 12 | 573,096 | 2.1496 | 1.9758 | 2.3139 | 0.9549 | 0.9470 | 0.9612 |
| `h4_market_stage_reduced` | 12 | 573,096 | 1.7195 | 1.6485 | 1.8013 | 0.7890 | 0.7827 | 0.7981 |
| `structure_trend_state` | 12 | 573,096 | 1.0068 | 0.9720 | 1.0442 | 0.6304 | 0.6215 | 0.6386 |
| `market_stage_reduced` | 12 | 573,096 | 0.9337 | 0.8665 | 1.0450 | 0.4261 | 0.4126 | 0.4575 |

Interpretation:

- 1D reduced stage has very high normalized transition information. It is a slow-moving context label and mostly persists from one base bar to the next.
- 4H reduced stage is also persistent, but less than 1D.
- `structure_trend_state` has moderate persistence.
- Base `market_stage_reduced` has the lowest normalized transition information and the most transition activity. It is the most useful Stage 1 surface for studying state changes rather than state persistence.
- Conditional transition information is currently reported as `0.0` in this run. The current implementation is not yet showing added transition information from `liquidity_event_type` in that metric surface.

## Transition Graph

Self-transition share captures how many transition rows remained in the same state. It is different from the maximum edge probability, which can be high for rare source states.

| State Column | Rows | Edges | Self Edges | Self Rows | Max Edge Probability | Self-Transition Row Share |
|---|---:|---:|---:|---:|---:|---:|
| `d1_market_stage_reduced` | 573,012 | 563 | 109 | 566,880 | 0.9991 | 98.93% |
| `h4_market_stage_reduced` | 573,054 | 635 | 109 | 535,549 | 0.9990 | 93.46% |
| `structure_trend_state` | 573,084 | 192 | 48 | 507,091 | 0.9363 | 88.48% |
| `market_stage_reduced` | 573,084 | 685 | 110 | 422,577 | 0.9982 | 73.74% |

Most active non-self transitions by empirical probability:

| State Column | Symbol | Source State | Target State | Rows | Source Rows | Probability |
|---|---|---|---|---:|---:|---:|
| `market_stage_reduced` | BTC-USDT-SWAP | markdown | accumulation | 245 | 612 | 0.4003 |
| `market_stage_reduced` | BNB-USDT-SWAP | accumulation | range | 562 | 1,537 | 0.3656 |
| `market_stage_reduced` | XRP-USDT-SWAP | accumulation | range | 982 | 2,763 | 0.3554 |
| `market_stage_reduced` | LTC-USDT-SWAP | markup | distribution_or_reversal | 405 | 1,153 | 0.3513 |
| `market_stage_reduced` | ETH-USDT-SWAP | markdown | accumulation | 213 | 610 | 0.3492 |
| `market_stage_reduced` | ETH-USDT-SWAP | accumulation | range | 1,015 | 2,923 | 0.3472 |
| `market_stage_reduced` | DOGE-USDT-SWAP | accumulation | range | 916 | 2,644 | 0.3464 |
| `market_stage_reduced` | SOL-USDT-SWAP | markdown | accumulation | 221 | 640 | 0.3453 |
| `market_stage_reduced` | BTC-USDT-SWAP | accumulation | range | 1,073 | 3,147 | 0.3410 |
| `market_stage_reduced` | SOL-USDT-SWAP | markup | distribution_or_reversal | 284 | 839 | 0.3385 |

Interpretation:

- The dominant active transition surface is base `market_stage_reduced`.
- Common transitions align with intuitive handcrafted stage progression: `markdown -> accumulation`, `accumulation -> range`, and `markup -> distribution_or_reversal`.
- Higher-timeframe labels mostly act as context regime anchors rather than fast transition signals.

## Pattern Quality

The pattern-quality table scored 89,622 patterns and found 306 candidate-gated rows.

| Pattern Family | Rows | Candidate-Gated Rows | Median Pattern Rows | Median Omega | Median PWPR |
|---|---:|---:|---:|---:|---:|
| `none_event_context` | 582 | 0 | 17,642 | 1.037 | 1.028 |
| `transition` | 26,067 | 127 | 12 | 0.988 | 0.985 |
| `transition_ngram` | 62,973 | 179 | 5 | 0.997 | 0.994 |

Candidate distribution by symbol:

| Symbol | Candidates | Median Rows | Median Omega | Median Side Return % |
|---|---:|---:|---:|---:|
| DOGE-USDT-SWAP | 79 | 53.0 | 2.390 | 1.236 |
| XRP-USDT-SWAP | 42 | 65.0 | 2.429 | 0.763 |
| BTC-USDT-SWAP | 31 | 45.0 | 2.567 | 0.529 |
| SOL-USDT-SWAP | 25 | 36.0 | 2.103 | 1.152 |
| LINK-USDT-SWAP | 22 | 35.0 | 3.091 | 0.983 |
| OP-USDT-SWAP | 19 | 36.0 | 2.049 | 0.844 |
| ETH-USDT-SWAP | 16 | 39.5 | 2.539 | 0.485 |
| ADA-USDT-SWAP | 16 | 43.0 | 2.309 | 0.832 |
| BNB-USDT-SWAP | 15 | 40.0 | 2.124 | 0.326 |
| LTC-USDT-SWAP | 14 | 39.0 | 2.188 | 0.559 |
| ARB-USDT-SWAP | 14 | 38.5 | 2.618 | 1.232 |
| AVAX-USDT-SWAP | 13 | 38.0 | 2.744 | 0.850 |

Candidate distribution by state column:

| Pattern Family | State Column | Candidates | Median Rows | Median Omega | Median Return % |
|---|---|---:|---:|---:|---:|
| `transition` | `market_stage_reduced` | 61 | 40.0 | 2.308 | 0.696 |
| `transition` | `d1_market_stage_reduced` | 35 | 42.0 | 2.914 | 1.137 |
| `transition` | `h4_market_stage_reduced` | 26 | 59.5 | 2.642 | 1.232 |
| `transition` | `structure_trend_state` | 5 | 38.0 | 2.227 | 0.519 |
| `transition_ngram` | `market_stage_reduced` | 88 | 45.0 | 2.313 | 0.719 |
| `transition_ngram` | `h4_market_stage_reduced` | 45 | 48.0 | 2.133 | 0.805 |
| `transition_ngram` | `d1_market_stage_reduced` | 42 | 40.0 | 2.887 | 1.121 |
| `transition_ngram` | `structure_trend_state` | 4 | 33.0 | 2.639 | 0.841 |

Candidate distribution by horizon and side:

| Horizon | Side | Candidates | Median Rows | Median Omega | Median Return % |
|---:|---|---:|---:|---:|---:|
| 3 | none | 50 | 41.5 | 2.603 | 0.561 |
| 3 | long | 24 | 38.0 | 2.231 | 0.625 |
| 3 | short | 16 | 51.0 | 2.102 | 0.717 |
| 5 | none | 44 | 42.5 | 2.607 | 0.696 |
| 5 | long | 39 | 52.0 | 2.041 | 0.680 |
| 5 | short | 14 | 42.5 | 2.280 | 0.768 |
| 10 | none | 58 | 43.0 | 2.484 | 1.269 |
| 10 | long | 41 | 63.0 | 2.773 | 1.147 |
| 10 | short | 20 | 40.0 | 2.413 | 1.282 |

Candidate row-count bands:

| Row Band | Candidates |
|---|---:|
| `<50` | 185 |
| `50-99` | 82 |
| `100-249` | 29 |
| `250+` | 10 |

Interpretation:

- Most candidate-gated rows are sparse. 185 of 306 candidates have fewer than 50 rows.
- 10-bar patterns show larger median side returns than 3-bar or 5-bar patterns, but this may reflect lower sample robustness.
- Candidate-gated rows are not evenly distributed across symbols; DOGE, XRP, and BTC dominate the count.
- `none_event_context` did not produce valid candidate-gated rows, despite some high raw ratios, because high rows were invalid-state contaminated or failed PWPR/direction gates.

## Top Candidate Rows

The highest Omega candidate-gated rows are discovery leads, not promotion evidence.

| Family | Symbol | Horizon | Side | Rows | Positive Rate % | Omega | PWPR | Mean Side Return % | Pattern |
|---|---|---:|---|---:|---:|---:|---:|---:|---|
| `transition_ngram` | DOGE-USDT-SWAP | 5 | none | 30 | 70.00 | 9.326 | 3.997 | 3.944 | `h4_market_stage_reduced: markup->trend_continuation->trend_continuation; event=none` |
| `transition` | ARB-USDT-SWAP | 10 | short | 46 | 67.39 | 7.222 | 3.494 | 1.774 | `d1_market_stage_reduced: accumulation->accumulation; event=bearish_reclaim` |
| `transition_ngram` | ARB-USDT-SWAP | 10 | short | 46 | 67.39 | 7.222 | 3.494 | 1.774 | `d1_market_stage_reduced: accumulation->accumulation->accumulation; event=bearish_reclaim` |
| `transition` | DOGE-USDT-SWAP | 10 | none | 30 | 66.67 | 5.189 | 2.595 | 3.170 | `market_stage_reduced: trend_continuation->accumulation; event=none` |
| `transition` | DOGE-USDT-SWAP | 10 | none | 33 | 66.67 | 5.142 | 2.571 | 2.524 | `d1_market_stage_reduced: range->distribution_or_reversal; event=none` |
| `transition_ngram` | DOGE-USDT-SWAP | 10 | none | 33 | 66.67 | 5.142 | 2.571 | 2.524 | `d1_market_stage_reduced: range->range->distribution_or_reversal; event=none` |
| `transition` | DOGE-USDT-SWAP | 10 | long | 31 | 51.61 | 4.992 | 4.680 | 2.559 | `market_stage_reduced: trend_continuation->transition; event=breakout_acceptance_high` |
| `transition_ngram` | BTC-USDT-SWAP | 10 | none | 32 | 68.75 | 4.949 | 2.250 | 0.539 | `market_stage_reduced: distribution_or_reversal->range->transition; event=none` |
| `transition` | BTC-USDT-SWAP | 3 | none | 69 | 60.87 | 4.908 | 3.155 | 0.696 | `market_stage_reduced: wide_range->range; event=none` |
| `transition_ngram` | DOGE-USDT-SWAP | 10 | none | 30 | 63.33 | 4.794 | 2.775 | 0.913 | `market_stage_reduced: distribution_or_reversal->transition->transition; event=none` |

## Gate Failure Structure

The main failure mode is sparse rows combined with weak return-quality metrics.

| Gate Failure Reasons | Rows |
|---|---:|
| `rows,omega,pwpr,direction` | 30,605 |
| `rows` | 18,739 |
| `omega,pwpr,direction` | 9,057 |
| `omega,pwpr` | 7,756 |
| `rows,pwpr` | 5,918 |
| `rows,omega,pwpr` | 4,544 |
| `rows,invalid_state,omega,pwpr,direction` | 3,724 |
| `rows,invalid_state` | 1,931 |
| `pwpr` | 1,890 |
| `invalid_state,omega,pwpr,direction` | 1,278 |

Interpretation:

- The diagnostic search space is large and sparse by construction.
- Candidate gates are doing useful filtering, but they are not enough for promotion.
- Invalid labels mostly affect none-event context rows and should remain an exclusion gate.

## Promotion Assessment

`promotion-candidates.csv` contains 0 rows.

This should not be interpreted as a final rejection of all transition patterns because strict promotion support is not fully wired into the Stage 1 bundle. Current code applies candidate gates, but symbol support, time-split support, and final promotion gates are not yet fully applied to the transition discovery path.

Operational decision:

- No pattern is promoted.
- No strategy rule should be created from this report alone.
- The next engineering step is strict-promotion wiring, not threshold relaxation.

## Conclusions

1. The handcrafted classifier passes structural Stage 1 health checks.
2. Higher-timeframe reduced states are highly persistent and act as stable context labels.
3. Base `market_stage_reduced` is the most active transition surface and is the best initial target for transition-behavior analysis.
4. Transition and transition-ngram patterns produce 306 candidate-gated rows, but most are sparse.
5. None-event context patterns do not pass current candidate gates once invalid-state and PWPR filters are considered.
6. The result supports continued deterministic transition research, not strategy conversion.

## Next Work

1. Wire strict symbol/time-split support into `transition_discovery.py`.
2. Recompute `promotion-candidates.csv` with meaningful `passes_promotion_gate` semantics.
3. Add classifier-behavior diagnostics for persistence calibration, unknown/warmup share, state balance, churn, and timeframe agreement.
4. Re-run this report after strict promotion wiring.
5. Only after robust cross-symbol/time-split evidence appears, write explicit strategy hypotheses and test them through execution-aware backtests.
