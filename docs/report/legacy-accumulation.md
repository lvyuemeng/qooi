# Legacy Accumulation Scanner Notes

This document preserves historical accumulation-scanner vocabulary that is no longer the canonical architecture. The active architecture is the potential trading-change scanner in `docs/architecture/scanner.md`; concrete implementation surfaces live in `docs/graph/scanner.md`.

## Historical framing

The original accumulation scanner was an offline-first research workflow that produced explainable artifacts for accumulation-like market conditions. It asked:

```text
Which symbols show accumulation-like conditions, what evidence supports the hypothesis, what evidence is missing, and what happened after similar scored alert events?
```

Those artifacts were always research hypotheses only. They did not authorize trading, allocation, strategy promotion, or live execution.

## Superseded by potential scanner

The current scanner goal is broader and more neutral:

```text
Find swap symbols whose current known-at-close state vector materially changes future transition/path distributions enough to deserve trading research review.
```

The active scanner does not try to confirm accumulation, oversold relief, a bullish thesis, or a bearish thesis. It computes parent-gated evidence over known-at-close observations and future outcome columns, then reports probability, information, tail/path, stability, and coverage diagnostics.

## Historical two-pass loop

The old topic-specific loop was:

1. Run a broad, cheap public-market/source pass.
2. Review ranked candidates, confidence, missing evidence, coverage warnings, and suggested follow-up actions.
3. Deepen selected symbols only when readouts justified extra collection cost or secret-backed access.
4. Rescore and summarize selected symbols.
5. Replay or inspect scored alert events when enough historical price rows existed.

The current architecture keeps the useful part of this loop as a general scanner principle: separate discovery, evidence collection, scoring, human review, and replay from strategy promotion.

## Source and evidence notes retained

- Missing source evidence is confidence/coverage information, not neutral evidence.
- Derivative context such as open interest, taker volume, and long/short ratios may support positioning-pressure diagnostics but does not prove accumulation or wallet behavior.
- Exchange address books and wallet labels are partial research labels only; do not hardcode them as truth.
- Local messages and provider-backed context are source observations, not strategy signals.
- Historical event replay extracts outcomes after scored events without invoking basket execution.

## Historical exchange-flow sign convention

When old flow features are encountered, the retained sign convention is:

```text
net_exchange_flow = inflow - outflow
positive = exchange inflow = distribution-risk proxy
negative = exchange outflow = accumulation-like proxy
```

This convention is historical context only. Current candidate direction must come from empirical posterior/path diagnostics, not from the flow sign alone.

## Canonical replacements

| Legacy topic | Current canonical doc |
|---|---|
| Accumulation scanner architecture | `docs/architecture/scanner.md` |
| Scanner module/function surfaces | `docs/graph/scanner.md` |
| Source families and missing-data policy | `docs/architecture/sources.md` |
| Source implementation graph | `docs/graph/sources.md` |
| Exchange/cache/universe boundaries | `docs/architecture/data.md` |
| Exchange/cache implementation graph | `docs/graph/data.md` |
| Empirical run summaries | `docs/report/research.md` |
