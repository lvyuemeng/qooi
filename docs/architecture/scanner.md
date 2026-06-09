# Potential Scanner Architecture

## Purpose

The scanner is the active deterministic workflow for finding potential altcoins for research review. Its output is trading information aid: probability diagnostics, path-risk context, missing-data coverage, and source context that help a human decide what deserves deeper trading research.

It is research-only decision support. It does not place orders, mutate baskets, call the executor, or authorize allocation.

## Domain thesis

The scanner is built around a diagnostics probability framework, not a market-imitation model.

Working assumptions:

- Profit opportunities come from extreme path behavior, not the average market distribution.
- The whole market can be treated as alternating or mixing between consolidation and trend behavior.
- Potential candidates are symbols whose current known-at-close state vector materially changes the probability of future extreme behavior.
- Classification should preserve interpretable geometric/state roles so statistical tests can ask whether a state changes future tails, transitions, or path risk.
- Source context is conditional information and data-quality evidence; it is not trading authorization.

The practical theory base is geometric information theory plus large-deviation style tail diagnostics:

- geometry: describe market state by regime, structure, range, volatility, transition path, alignment, and source roles;
- information: ask whether child state vectors reduce uncertainty or add information beyond parent market context;
- large deviation: emphasize tail rates, excursion distributions, returned-to-origin behavior, and recent instability rather than fitting the whole return distribution.

## Owned modules

```text
src/qooi/scanner/workflow.py       # config, orchestration, artifact/report writes
src/qooi/scanner/contracts.py      # scanner contracts and ranking helpers
src/qooi/scanner/classifiers.py    # deterministic kline state classifier
src/qooi/scanner/transitions.py    # transition/path evidence discovery
src/qooi/scanner/decisions.py      # latest research-review decisions
src/qooi/scanner/diagnostics.py    # diagnostic/state artifact construction
src/qooi/scanner/history.py        # kline path history and realized transitions
src/qooi/scanner/source_events.py  # source-native event/outcome diagnostics
src/qooi/scanner/evidence.py       # unified observation and evidence ladder
src/qooi/scanner/report.py         # Markdown report rendering
```

## Architecture workflow

The scanner architecture is a reproducible computation from known-at-close observations to evidence-backed research candidates. The workflow has three layers:

```text
operational IO
  -> known-at-close state and outcome surfaces
  -> current candidate / backtest review surfaces
```

Operational IO:

```text
load config
  -> resolve bounded universe
  -> load/cache OHLCV efficiently across symbols/timeframes
  -> select source-context symbols
  -> load/cache source context
```

Research computation:

```text
OHLCV + source context
  -> classify known-at-close states
  -> build observation vector O_t
  -> build future outcome Y_{t,h}
  -> calculate parent-gated evidence E(O_t,h)
  -> select useful evidence rows
```

Current review and validation:

```text
latest O_t + selected E(O_t,h)
  -> candidate row C_t
  -> candidate rank components and coverage caveats
  -> Markdown research report

frozen train E_train(O,h) + holdout O_t
  -> holdout candidate rows
  -> out-of-sample evidence validation
  -> optional later strategy hypothesis
```

The current code already implements the operational IO, state classification, observation/outcome/evidence artifacts, diagnostics, and report display. The candidate join, ranking contract, and train/holdout evidence backtest are the next implementation surfaces.

The workflow separates expensive IO from Polars-native computation:

- exchange/source modules fetch and cache data;
- classifier/history/evidence modules compute frame transforms;
- diagnostics/report modules write or summarize artifacts;
- candidate/backtest modules, when added, should consume artifacts or frames instead of refetching data;
- no scanner computation should depend on executor/backtest side effects.

## Probability framework

The scanner should answer this question for each candidate row:

```text
Given this known-at-close observation vector, how different is the future path distribution from its parent market context, especially in the extreme tails?
```

Primary probability diagnostics:

- `p_up`, `p_down`, `p_flat`: directional posterior mass after thresholding forward returns.
- `lift_up`, `lift_down`, `lift_flat`: posterior change versus parent/baseline context.
- `information_gain_bits`: uncertainty reduction versus the parent level.
- `transition_information_gain_bits`: transition-structure information beyond parent context.
- `tail_up_rate`, `tail_down_rate`: large-move frequency, not average-return mimicry.
- `avg_forward_max_return_pct`, `avg_forward_min_return_pct`, `avg_path_range_pct`: favorable/adverse excursion and path amplitude.
- `path_skew`: directional asymmetry of the future path.
- `returned_to_origin_rate`: failure-to-trend or mean-reversion diagnostic.
- `information_stability` and `transition_information_stability`: recent-vs-long evidence stability.

The report should prefer candidates with enough count/symbol support, parent improvement, stable information, and favorable path-risk diagnostics. It should reject or down-rank unsupported rare states even when their point estimate looks attractive.

## Evidence ladder

```text
market_background
  -> market_swing
  -> market_decision
  -> market_decision_source
  -> market_decision_source_risk
```

Rules:

- Use one decision clock first, normally `1H`.
- Attach slower states by as-of join: latest closed `4H`/`1D` where timestamp <= decision time.
- Attach source rows with known-at timestamp <= decision time.
- Keep vector roles as real columns; do not collapse into an opaque key.
- Future returns/transitions are outcome columns only.
- Let empirical outcomes decide statistical direction.
- Child levels are useful only when they beat their parent by support, information, stability, and path quality.

## Computable research workflow

The scanner workflow should be expressed as one reproducible computation, not a loose list of diagnostics.

Goal:

```text
Find symbols whose current state vector has statistically useful conditional evidence for future extreme/path behavior, then present them as research-review candidates with explicit coverage and risk caveats.
```

Definitions:

- `O_t`: known-at-close observation vector at decision close `t`.
- `P(O_t)`: parent context for `O_t`, such as background or swing/decision level.
- `Y_{t,h}`: future outcome over horizon `h`, stored only as an outcome column.
- `E(O_t, h)`: evidence row comparing `Y_{t,h} | O_t` against `Y_{t,h} | P(O_t)`.
- `C_t`: current candidate row for a symbol, built by joining latest `O_t` to the best matching evidence row.

Procedure and owned object contracts:

| Step | Object | Computes | Input boundary | Consumer |
|---|---|---|---|---|
| 1 | Universe `U` | Bounded symbol set and eligibility notes. | Config and exchange discovery/cache only. | OHLCV/source loaders. |
| 2 | Kline state `K_t` | Closed-bar geometric roles: regime, structure, range, volatility, event, transition, direction hint. | OHLCV up to bar close. | Observation/outcome builders and latest bundles. |
| 3 | Source event `S_t` | Source-family state, direction, known-at timestamp, availability/freshness. | Provider/cache rows known at or before decision close. | Observation builder and source diagnostics. |
| 4 | Observation `O_t` | Decision/swing/background/source/risk vector. | `K_t`, `S_t`, as-of joins where timestamp <= decision close. | Evidence, current candidate join, holdout replay. |
| 5 | Outcome `Y_{t,h}` | Forward return, MFE/MAE, path range, return-to-origin, semantic transition. | Future bars only; never classifier input. | Evidence and backtest scoring. |
| 6 | Evidence `E(O_t,h)` | Posterior probabilities, lift, information, tail/path diagnostics versus parent context. | Historical `O_t` joined to `Y_{t,h}`. | Evidence selector, report, candidate join. |
| 7 | Selected evidence `E*` | Parent-gated useful evidence rows. | `E(O_t,h)` support, information, stability, path-quality gates. | Current candidate join and train/holdout freeze. |
| 8 | Candidate `C_t` | Latest/holdout observation matched to best selected evidence with caveats. | `O_t`, `E*`, coverage/freshness diagnostics. | Ranking, report, research review. |
| 9 | Ranked candidate `R_t` | Explicit score components for information, transition, tail/path, stability, data quality. | `C_t` only; no executor state. | Human research review and backtest summaries. |
| 10 | Backtest row `B_t` | Holdout candidate plus realized `Y_{t,h}` and baseline comparisons. | Frozen train `E*`, holdout `O_t`, holdout outcomes. | Promotion decision for a future strategy hypothesis. |

This makes every build task answer: which object does it compute, from which known-at-close inputs, and which downstream decision-review step consumes it?

## Architecture to module graph transformation

The module graph should be derived directly from the object contracts above:

```text
U  -> workflow.resolve_universe(...)
K  -> workflow.load_bars(...), KlineClassifier.classify(...), history.kline_path_history_frame(...)
S  -> sources.context.load_source_context(...), source_events.source_events_frame(...)
O  -> evidence.potential_observation_frame(...)
Y  -> history.realized_transition_frame(...), source_events.source_outcomes_frame(...), evidence.potential_outcome_frame(...)
E  -> evidence.potential_evidence_frame(...), evidence.add_potential_parent_gain(...)
E* -> evidence.select_potential_evidence_level(...)
C  -> planned candidate_evidence_frame(...)
R  -> planned rank_candidate_evidence(...)
B  -> planned backtest_candidate_evidence(...)
```

Rules for transforming architecture to graph docs:

- only public functions/classes that compute one of the objects above should be listed as supported graph surfaces;
- helper functions stay private unless another module imports them as a boundary;
- module graph docs must show both the current implemented edge and the next planned edge when the architecture object exists but code does not;
- artifacts are graph surfaces only when they are written/read across module boundaries;
- report sections are consumers of graph objects, not primary architecture objects.

## Backtest and validation workflow

The first backtest is a research backtest of the computation, not an executor trade simulation.

1. Split history by time: train/calibration window first, holdout/evaluation window second.
2. On train only, compute selected evidence rows `E_train(O, h)` and freeze their grouping columns and gates.
3. On holdout, build observations `O_holdout` using only known-at-close fields.
4. Match each holdout observation to the frozen train evidence rows.
5. Produce candidate rows only when the train evidence row passed parent gates and the holdout observation has acceptable coverage/freshness.
6. Score holdout outcomes with no refitting: tail hit rate, directional hit rate, lift over parent baseline, information stability, max adverse excursion, max favorable excursion, returned-to-origin rate, and candidate frequency.
7. Compare against baselines: parent context only, random same-universe same-frequency rows, and simple momentum/range heuristics.
8. Promote only if holdout evidence remains stable across time splits and symbols. Promotion still means research hypothesis, then later normal signal columns and execution-aware backtests.

Executor backtests happen only after the research backtest defines explicit signal columns, sizing assumptions, and exits. Until then, scanner validation is about whether the probability computation finds useful extreme-behavior candidates.

## Candidate semantics

Candidate labels must remain information-aid labels, not trading commands.

Allowed readout style:

- `rapid_trend_watch`: posterior/tail/path diagnostics favor trend continuation or expansion.
- `mean_reversion_watch`: path diagnostics favor return-to-origin or relief behavior.
- `volatility_expansion_watch`: extreme movement rate or path range rises without clean direction.
- `chop_avoid`: evidence points to noisy, low-quality, or adverse path behavior.
- `insufficient_evidence`: support, coverage, source freshness, or parent lift is not enough.

Do not hardcode a bullish or bearish conclusion from a state name. `statistical_direction` comes from outcome distributions and path diagnostics.

## Data scale and efficiency

The framework requires large history, many symbols, and multiple source families. Keep computation/fetching efficient:

- use bounded universe discovery and scan budgets before expensive source context;
- fetch OHLCV/source data through cache-aware APIs with explicit coverage rows;
- classify symbol/timeframe frames in parallel where safe;
- use Polars-native joins, groups, and expressions instead of row loops;
- write diagnostic Parquet artifacts so later report/research steps can reuse them;
- make shallow/missing/stale data explicit rather than silently dropping symbols.

## Boundary from AI and execution

Scanner decisions use deterministic classifier/source states only. `qooi.dynamic` learned states are not scanner inputs unless a future explicit promotion rewrites this architecture.

Scanner artifacts are not strategy signals. A potential evidence row can only motivate research follow-up. Promotion to trading requires normal signal columns plus execution-aware validation under the project promotion policy.
