# Research Evaluation Report

Date: 2026-05-23

## Executive Decision

The canonical reduced `research-evaluation` run rejects the current manually configured static/reduced bucket design for strategy conversion. The run produced valid diagnostic artifacts, but the strict promotion export is empty.

A Stage 1 dynamic-transition evaluation has now been applied to the handcrafted classifier. It confirms that the deterministic classifier is structurally healthy enough for transition diagnostics and that the transition pipe produces meaningful empirical artifacts, but it does not yet authorize strategy conversion.

Run command:

```bash
uv run python scripts/research.py --config configs/research/research-evaluation-joint-quality.toml
```

Export directory:

```text
F:\Stratum\TEMP\kilo\qooi-research-evaluation-joint-quality
```

Decision:

- Keep the result as diagnostic evidence.
- Do not lower thresholds to force survivors.
- Do not convert any current bucket into strategy logic.
- Continue deterministic dynamic transition discovery as the current research workstream.
- Treat current `promotion-candidates.csv` as conservative/non-authoritative until strict symbol/time-split promotion support is wired into the Stage 1 transition path.

## Current Reduced API Evidence

The current public diagnostics API has only two modes:

```text
backtest
research-evaluation
```

The retained `research-evaluation` outputs for the static-bucket run were:

| Output | Role | Result |
|---|---|---|
| `timeframe-classifier` | Classifier health prerequisite | Passed generated health checks |
| `joint-forward-quality` | Removed side-normalized joint bucket quality surface | Produced `166,151` historical rows |
| `trade-record-modulation` | Optional post-trade control evidence | Not requested |

Historical artifacts from that run:

| Artifact | Rows | Interpretation |
|---|---:|---|
| `timeframe-classifier.csv` | 144 | Health rows for 12 symbols x 3 timeframes x 4 checks |
| `joint-forward-quality.csv` | 166,151 | Full diagnostic table for joint buckets and support rows |
| `joint-promotion-candidates.csv` | 0 | Explicit rejection evidence under strict promotion gates |

Candidate summary:

| Gate | Rows |
|---|---:|
| Diagnostic candidates | 363 |
| Aggregate `ALL` diagnostic candidates | 47 |
| Strict promotion candidates | 0 |

## Why Static Buckets Did Not Promote

The current bucket families can produce local high-Omega diagnostic rows, but those rows do not survive cross-symbol and chronological stress gates.

Strict promotion settings:

| Setting | Value |
|---|---:|
| `promotion_min_rows` | 50 |
| `promotion_min_symbols` | 3 |
| `promotion_min_time_splits` | 2 |
| `promotion_symbol_agreement_pct` | 67.0 |
| `promotion_time_agreement_pct` | 100.0 |

Interpretation:

- Local buckets can look strong without proving repeatable behavior.
- Aggregate buckets can have enough rows but fail symbol support or time-split sign agreement.
- Sparse dynamic inner-connection rows can rank highly while remaining structurally weak.
- Invalid labels such as `warmup`, `unknown`, and `data_error` remain exclusion gates.

The empty `joint-promotion-candidates.csv` was therefore correct behavior. It was not a missing artifact.

## Architecture Pivot

The rejected hypothesis is not that market behavior has no structure. The rejected hypothesis is that the current manually configured static/reduced bucket graph is strong enough to promote.

The next architecture should discover behavior from state changes first:

```text
timeframe-classifier
  -> dynamic-transition-discovery
  -> transition quality / pattern-quality support
  -> promotion candidates
```

This keeps the reduced API small while moving the discovery unit from static labels to transition patterns.

## Stage 1 Dynamic Transition Evaluation

Stage 1 was run against the handcrafted classifier with the dynamic-transition config.

Run command:

```bash
uv run python scripts/research.py --config configs/research/research-evaluation-dynamic-transitions.toml
```

Export directory:

```text
F:\Stratum\TEMP\kilo\qooi-research-evaluation-dynamic-transitions
```

Outputs requested:

| Output | Role | Result |
|---|---|---|
| `timeframe-classifier` | Handcrafted classifier health prerequisite | 144 health checks, all passed |
| `dynamic-transition-discovery` | Deterministic transition graph and information artifacts | Produced graph and information artifacts |
| `pattern-quality` | Shared scored-pattern surface | 89,622 scored rows; 306 candidate-gated rows |

Artifact summary:

| Artifact | Rows | Interpretation |
|---|---:|---|
| `timeframe-classifier.csv` | 144 | 12 symbols x 3 timeframes x 4 checks; all `pass` |
| `state-transition-graph.csv` | 2,075 | Empirical state edge counts and transition probabilities |
| `transition-information.csv` | 48 | 12 symbols x 4 state columns; all sufficient under current 100-row threshold |
| `transition-ngram-quality.csv` | 89,040 | Return-quality rows for 2-step and 3-step transition paths |
| `none-event-context-quality.csv` | 582 | Return-quality rows for `liquidity_event_type = none` context patterns |
| `scored-patterns.csv` | 89,622 | Full candidate-gated pattern surface |
| `promotion-candidates.csv` | 0 | Expected empty placeholder until strict promotion support is fully wired |

Candidate-gate summary:

| Pattern Family | Rows | Candidate-Gated Rows |
|---|---:|---:|
| `transition` | 26,067 | 127 |
| `transition_ngram` | 62,973 | 179 |
| `none_event_context` | 582 | 0 |
| **Total** | **89,622** | **306** |

Classifier health interpretation:

- Required handcrafted classifier columns were present for every symbol/timeframe check.
- Cardinality was stable and small enough for deterministic diagnostics:
- `market_stage`: mostly 9 unique states, with ADA 1H/4H/1D and LINK 1H showing 10.
- `structure_trend_state`: 4 unique states.
- `liquidity_event_type`: 7 unique event labels.

This is sufficient for Stage 1 transition diagnostics. It is not yet a complete classifier-quality proof because the current health check does not measure state persistence, unknown/warmup rates, state balance, or timeframe agreement.

Transition-property interpretation:

| State Column | Rows | Self-Transition Row Share | Average Transition Information | Average Normalized Transition Information |
|---|---:|---:|---:|---:|
| `d1_market_stage_reduced` | 573,096 | 98.93% | 2.1496 | 0.9549 |
| `h4_market_stage_reduced` | 573,096 | 93.46% | 1.7195 | 0.7890 |
| `structure_trend_state` | 573,096 | 88.48% | 1.0068 | 0.6304 |
| `market_stage_reduced` | 573,096 | 73.74% | 0.9337 | 0.4261 |

Readout:

- The higher-timeframe reduced labels are highly persistent, especially `d1_market_stage_reduced`.
- `market_stage_reduced` changes more often at the base timeframe and is therefore the most active transition surface.
- `structure_trend_state` sits between higher-timeframe persistence and base market-stage mobility.
- Conditional transition information is currently reported as `0.0`, so the present implementation is not yet showing added information from `liquidity_event_type` in that metric surface.

Candidate readout:

- Candidate-gated rows are concentrated in transition and transition-ngram families.
- `DOGE-USDT-SWAP` and `XRP-USDT-SWAP` contributed the largest candidate counts in both transition and transition-ngram families.
- Several high-ratio rows are sparse, often around 30 to 46 rows, so they are useful discovery leads but not promotion evidence.
- `none_event_context` produced no candidate-gated rows under current thresholds.

Decision from Stage 1:

- The handcrafted classifier is acceptable for deterministic transition diagnostics.
- Transition artifacts show measurable state persistence and candidate-gated transition patterns.
- No transition pattern is promoted to strategy logic.
- Next implementation work should wire strict symbol/time-split promotion into `transition_discovery.py` before interpreting the empty promotion export as a final rejection.

## Stage 1 Immediate Research: Dynamic Transitions

Stage 1 is deterministic, diagnostic-only, and dependency-light. It consumes only known-at-close classifier/context labels and produces transition artifacts.

Output name:

```text
dynamic-transition-discovery
```

Artifacts:

| Artifact | Purpose |
|---|---|
| `state-transition-graph.csv` | Directed previous-state to current-state counts and probabilities |
| `transition-information.csv` | Transition information and conditional transition information summaries |
| `transition-ngram-quality.csv` | Side-normalized forward quality for transition paths |
| `none-event-context-quality.csv` | Contextual diagnostics for rows where liquidity event is `none` |

Terminology:

- Use `transition information` for `I(S_t; S_{t-1})`.
- Use `conditional transition information` for `I(S_t; S_{t-1} | E)`.
- Do not call this transfer entropy unless a future estimator actually implements transfer entropy.

Stage 1 non-goals:

- No strategy authorization.
- No live trading path.
- No executor, basket, recovery, or exchange changes.
- No new ML/RL dependency.
- No promotion decision until strict symbol/time-split support is wired and reviewed.

## Stage 2 Gated Research: Endogenous States

Stage 2 is only justified if Stage 1 finds robust transition patterns. It may introduce learned state encoders that produce research labels such as `behavior_state_id`.

Constraints:

- Inputs must be known at or before the decision bar close.
- Future returns may be labels for training heads, not encoder inputs.
- Chronological train/validation/test splits are mandatory.
- Learned state IDs remain research labels until promoted through the normal strategy contract.

Recommended optional dependency direction if Stage 2 is approved:

```toml
[dependency-groups]
ml = [
    "scikit-learn>=1.4",
    "torch>=2.3",
]
```

## Stage 3 Long-Term Research: Policy Learning

Stage 3 is only justified after stable endogenous states exist. Policy learners and world models remain simulation/research artifacts.

Constraints:

- A policy learner must not own basket lifecycle.
- A policy learner must not calculate fills, fees, account state, or recovery mutation.
- A world model must not call exchange clients.
- Any deployable policy must adapt into normal signal columns before execution.

Deployable signal columns:

| Column | Meaning |
|---|---|
| `raw_entry_signal` | Signed raw rule output after filters |
| `entry_signal` | Event-like signed signal that may open a basket |
| `position_signal` | Held directional thesis state |
| `exit_signal` | Strategy-owned exit intent |
| `signal_strength` | Numeric quality or confidence scalar |
| `signal_id` | Stable module or rule identifier |

Recommended optional dependency direction if Stage 3 is approved:

```toml
[dependency-groups]
rl = [
    "gymnasium>=0.29",
    "stable-baselines3>=2.0",
]
```

Do not select Dreamer or another world-model stack until there is a concrete environment contract, observation schema, reward definition, and train/eval split.

## Promotion Contract

No diagnostic artifact becomes strategy logic directly.

A transition pattern or learned state can become a strategy hypothesis only after all of these hold:

- Classifier or encoder health passes.
- State construction is no-lookahead.
- Row count, symbol support, and time-split support pass strict gates.
- Omega/PWPR and mean side return are material.
- Invalid states are excluded.
- The pattern has an ex-ante market-behavior rationale.
- The hypothesis is converted into explicit signal columns.
- Execution-aware backtests pass with costs, slippage, stops, targets, sizing, basket caps, and comparability checks.

## Final Decision

The current reduced static/joint run is complete and valid. It rejects promotion under strict gates and should not be tuned to manufacture candidates.

The Stage 1 dynamic-transition run is also complete as an initial diagnostic pass. It validates that the handcrafted classifier can feed transition evaluation and that transition properties can be measured, but it does not produce strategy-authorized candidates.

## Next Work

1. Wire `promotion.symbol_support()`, `promotion.time_split_support()`, and `promotion.apply_promotion_gate()` into `transition_discovery.py`.
2. Pass `information_min_rows` into transition-information sufficiency instead of relying on the hard-coded 100-row threshold.
3. Add classifier-quality diagnostics for state persistence, unknown/warmup rates, state balance, and timeframe agreement.
4. Re-run `configs/research/research-evaluation-dynamic-transitions.toml` after strict promotion wiring.
5. Promote nothing until transition artifacts pass strict evidence discipline and execution-aware backtests.
