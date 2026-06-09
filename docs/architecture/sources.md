# Source Collector Architecture

## Purpose

The source layer collects, normalizes, stores, and audits provider/context data for scanner and research workflows. It is ingestion and artifact infrastructure only.

## Owned modules

```text
src/qooi/sources/artifacts.py   # generic artifact path/read/write/coercion
src/qooi/sources/bundle.py      # source bundle IO and keyed merge behavior
src/qooi/sources/context.py     # source context availability/loading/merging
src/qooi/sources/coverage.py    # manifests, coverage, freshness, missing evidence
src/qooi/sources/manifest.py    # source manifest helpers
src/qooi/sources/models.py      # SourceResult model
src/qooi/sources/schema.py      # shared schemas
src/qooi/sources/http.py        # shared HTTP helpers and sanitized errors
src/qooi/sources/okx.py         # OKX public source helpers
src/qooi/sources/okx_ws.py      # OKX websocket public source helpers
src/qooi/sources/coingecko.py   # CoinGecko helpers
src/qooi/sources/coinpaprika.py # CoinPaprika helpers
src/qooi/sources/defillama.py   # DeFiLlama helpers
src/qooi/sources/cryptopanic.py # CryptoPanic helpers
src/qooi/sources/polymarket.py  # Polymarket helpers
src/qooi/sources/messages.py    # local message normalization/classification
```

## Responsibilities

- Fetch provider/source payloads when configured.
- Normalize provider JSON before the Polars/DataFrame boundary.
- Produce source frames, source manifests, and source bundles.
- Report missing API keys, missing rows, stale rows, freshness, and coverage status explicitly.
- Keep source-family, availability, and known-at/fetched-at semantics visible to downstream scanner/research consumers.

## Non-responsibilities

- No trading signals.
- No scanner candidate ranking or suggestion policy.
- No strategy promotion.
- No basket lifecycle, executor, recovery, or allocation logic.
- No hardcoded provider secrets, API keys, wallet labels, or exchange-wallet truth.

## Allowed dependencies

- Standard library, HTTP/runtime helpers, Polars/Pydantic-style data models where needed.
- Other `qooi.sources` modules.
- Data-only exchange cache/coverage helpers only where the current source coverage path requires them.

## Forbidden dependencies

- `qooi.scanner`
- `qooi.research`
- `qooi.strategies`
- `qooi.core.basket`
- `qooi.core.executor`
- `qooi.core.recovery`
- `qooi.exchange.trading`
- `qooi.dynamic`

## Missing-data policy

Missing or stale source evidence is not neutral evidence. It must appear as manifest/coverage/freshness diagnostics so scanner and research reports can show confidence caveats.

Derivative source context such as open interest, taker volume, funding, long/short ratios, news/context heat, or local messages can describe conditional source state. It does not authorize trading and does not replace empirical posterior/path diagnostics.

Exchange address books and wallet labels are partial research labels only. Do not hardcode them as complete truth.

## Integration boundary

Scanner and research modules consume normalized source frames, source bundles, and manifest/coverage rows. Concrete source implementation mappings live in `docs/graph/sources.md`.
