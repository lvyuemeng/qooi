# Research Reports

Date: 2026-06-01

## Static Bucket Rejection

The reduced static/joint bucket research run produced valid diagnostics but no strict promotion candidates.

Evidence:

| Artifact | Result |
|---|---:|
| `joint-forward-quality.csv` | 166,151 rows |
| Diagnostic candidates | 363 |
| Aggregate `ALL` diagnostic candidates | 47 |
| `joint-promotion-candidates.csv` | 0 |

Decision:

- Keep as rejection evidence.
- Do not relax thresholds to manufacture survivors.
- Move discovery focus from static buckets to state transitions.

## Dynamic Transition Run

Command:

```bash
uv run python scripts/classifier_states.py --config configs/dyn-trans.toml
```

Result:

| Artifact | Rows |
|---|---:|
| `timeframe-classifier.csv` | 144 |
| `state-transition-graph.csv` | 2,075 |
| `transition-information.csv` | 48 |
| `transition-ngram-quality.csv` | 89,040 |
| `none-event-context-quality.csv` | 582 |
| `scored-patterns.csv` | 89,622 |
| `promotion-candidates.csv` | 0 |

Readout:

- Handcrafted classifier passed structural health checks.
- Higher-timeframe reduced states are highly persistent context labels.
- Base `market_stage_reduced` is the most active transition surface.
- Candidate-gated transition rows exist, but most are sparse.
- No transition pattern is promoted.

Next work:

- Wire symbol/time-split promotion gates.
- Pass `information_min_rows` through transition-information sufficiency.
- Add classifier behavior diagnostics: persistence, unknown/warmup share, balance, churn, timeframe agreement.

## Learned State Run

Command:

```bash
uv run python scripts/learned_states.py --config configs/learn-vq.toml
```

Result:

| Metric | Value |
|---|---:|
| Labeled windows | 146,435 |
| Emitted states | 45 |
| Final active training codes | 44 / 128 |
| Candidate-gated state rows | 7 |
| Candidate-gated transition rows | 0 |

Readout:

- VQ-RSSM produced non-collapsed learned states.
- Learned-state transition information is measurable but weak.
- Candidate rows are sparse discovery leads only.
- Runtime remains the main iteration bottleneck.
- XAU coverage is too low for equal cross-symbol comparison.

Next work:

- Add structured timing exports.
- Run batch and inference profiling.
- Add code-quality artifacts and seed-repeat stability checks.
- Re-run after strict promotion support is complete.

## Accumulation Scanner MVP

Command:

```bash
uv run python scripts/accumulation_scan.py --config configs/potential.toml
```

Initial offline smoke result:

| Artifact Family | Result |
|---|---:|
| Feature rows | 576,363 |
| Score rows | 576,363 |
| Alerts | 0 |
| Backtest events | 0 |

Readout:

- Offline score/backtest path works.
- Current run is orderbook/onchain-missing and should be treated as coverage smoke, not strategy evidence.
- Next useful evidence requires real market collection plus optional verified on-chain labels.
