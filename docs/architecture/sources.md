# Sources Architecture

## Purpose

`qooi.sources` owns scanner source demand, provider source normalization, source artifact contracts, source fetch planning, materialization, and quantitative source availability.

It does not own OHLCV exchange cache, scanner evidence/ranking, strategy computation, research promotion, basket lifecycle, executor accounting, dynamic/learned-state logic, or trading IO.

## Demand and availability pipeline

```text
scanner config + symbols
  -> SourceNeed
  -> source collection execution
  -> provider SourceResult(frame, manifest)
  -> SourceBundle materialization
  -> SourceAvailability diagnostics
```

`SourceNeed` construction is pure. Provider calls and writes are side-effect boundaries.

Availability is computed from materialized frames first, not from the latest fetch manifest alone:

```text
frame rows + latest event/known-at timestamp + configured threshold
  -> frame_freshness
latest manifest row
  -> fetch provenance/status
frame_freshness + fetch provenance
  -> review usability diagnostics
```

A latest empty or failed incremental provider fetch must not overwrite usable cached frame rows. The manifest explains the latest fetch attempt; the frame determines observed rows, age, and whether the family is currently usable.

## Module layout

| Module | Ownership |
|---|---|
| `qooi.sources.models` | Minimal result contracts shared by provider wrappers. |
| `qooi.sources.schema` | Persisted source CSV schemas. |
| `qooi.sources.artifacts` | Artifact paths, schema coercion, and small family/artifact table. |
| `qooi.sources.bundle` | Source bundle read/write, source-frame lookup, keyed merge. |
| `qooi.sources.coverage` | Pure availability, staleness, shallow/deep coverage, fetch eligibility. |
| `qooi.sources.context` | Scanner-facing `load_source_context(...)` boundary. |
| `qooi.sources.collect` | Source needs, collection request/result contracts, and source fetch execution. |
| `qooi.sources.http` | Sanitized HTTP JSON helpers. |
| `qooi.sources.okx` | OKX provider endpoint wrappers and payload normalizers. |
| provider modules | CoinGecko, CoinPaprika, DeFiLlama, CryptoPanic, Polymarket wrappers. |
| `qooi.sources.messages` | Local message normalization/classification. |
| `qooi.sources.__init__` | Shared source primitives used by sibling modules. |

Removed target: source collection must not live in `qooi.exchange.context`.

## Core abstractions

| Abstraction | Owner | Meaning |
|---|---|---|
| `SourceNeed` | `sources.collect` | Scanner demand for family/symbol/time/depth/freshness. |
| `SourceFetchPlan` | `sources.collect` | Value contract for one provider/raw-source fetch decision. |
| `SourceCollectRequest` | `sources.collect` | Source collection execution inputs. |
| `SourceCollectResult` | `sources.collect` | Provider-collected manifest and family frames. |
| `SourceResult` | `sources.models` | Provider-normalized frame plus manifest. |
| `SourceFamily` | `sources.artifacts` | Small family/artifact/raw-source/timestamp/merge-key contract. |
| `ArtifactSpec` | `sources.artifacts` | Persisted CSV artifact path and schema. |
| `SourceBundle` | `sources.bundle` | Loaded source artifacts. |
| `SourceAvailability` | `sources.context` or `sources.coverage` | Quantitative per-family/per-symbol observed frame state: rows, latest timestamp, age, freshness threshold, and usability. |
| `SourceFetchObservation` | `sources.coverage` | Latest provider/raw-source attempt state from the manifest: status, warning, endpoint, transport, and optional HTTP diagnostics. |
| `SourceContextResult` | `sources.context` | Scanner-facing source frames, manifest, availability. |

Do not add fields or modules that do not answer a current demand question. Availability fields should be numeric/evaluable before qualitative: rows, latest timestamp, age hours, threshold hours, fresh bit, usable bit, and missing bit.

## Family/artifact contract

One small family table should derive:

```text
family -> artifact
raw source -> family
family -> timestamp column
family -> merge key
family -> row-kind semantics when needed
```

Current scanner families:

| Family | Raw source examples | Artifact |
|---|---|---|
| `books` | `books` | `sources/books.csv` |
| `trades` | `trades` | `sources/trades.csv` |
| `funding` | `funding`, `funding_rate` | `sources/funding.csv` |
| `open_interest` | `open_interest_history` | `sources/open-interest.csv` |
| `taker_volume` | `taker_volume_contract` | `sources/taker-volume-contract.csv` |
| `long_short_ratios` | OKX long/short raw sources | `sources/long-short-ratios.csv` |
| `messages` | local message rows | local message artifacts |

Funding semantics:

```text
funding_source_kind=history -> settled historical funding rows; count for depth
funding_source_kind=current -> current known-at snapshot; count for freshness
```

## Dependency policy

Target dependencies:

```text
sources.artifacts -> sources.schema
sources.bundle    -> sources.artifacts
sources.coverage  -> sources.artifacts + sources.manifest
sources.context   -> sources.collect + sources.bundle + sources.coverage
sources.collect   -> sources.artifacts + sources.bundle + sources.coverage + sources.okx
sources.okx       -> sources.http + sources.manifest + sources.models + exchange.market helpers
provider modules  -> sources.http + sources.manifest + sources.models
```

Forbidden dependencies:

```text
sources -> scanner evidence/rank/report modules
sources -> research policy/report interpretation
sources -> strategies
sources -> core basket/executor/recovery
sources -> exchange.trading
sources.context -> exchange.context
```

## Responsibilities

- Convert scanner/source demand into explicit `SourceNeed` rows.
- Execute source collection through source-owned request/result contracts.
- Fetch provider source rows through provider wrappers.
- Normalize provider payloads into `SourceResult` frames and manifests.
- Merge/write source artifacts by declared keys and schemas.
- Derive frame-level source availability from persisted rows, latest timestamps, and configured staleness thresholds.
- Preserve latest fetch-attempt provenance separately from frame usability.
- Emit explicit missing/stale/shallow availability for diagnostics.

## Non-responsibilities

- No exchange OHLCV cache ownership.
- No scanner evidence, tailtree, rank, ladder, or report policy.
- No strategy signal semantics.
- No research promotion policy.
- No live trading or account IO.

Concrete target APIs live in `docs/graph/sources.md`.
