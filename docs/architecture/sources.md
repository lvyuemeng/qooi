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

Availability is computed from materialized frames and source capability first, not from the latest fetch manifest alone:

```text
provider capability + scanner demand
  -> expected capability window
frame rows + latest event/known-at timestamp + configured threshold
  -> frame_freshness
frame rows + provider capability window
  -> capability-adjusted coverage
latest manifest row
  -> fetch provenance/status
frame_freshness + capability coverage + fetch provenance
  -> review usability diagnostics
```

A latest empty or failed incremental provider fetch must not overwrite usable cached frame rows. A provider-bounded history window must not be treated as missing just because it cannot satisfy a longer scanner evidence horizon. The manifest explains the latest fetch attempt; provider capability defines the feasible expectation; the frame determines observed rows, age, and whether the family is currently usable.

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

## Source config workflow

`SourceConfig` is a source-family demand override, not a second scanner run config.

```text
PotentialConfig.refresh_mode
  -> default refresh mode for both bar cache and source collection

SourceConfig.refresh_mode="inherit"
  -> source collection uses PotentialConfig.refresh_mode

SourceConfig.refresh_mode in {"incremental", "cache_only", "force"}
  -> source collection override only
```

`SourceConfig` owns provider-family limits and freshness:

```text
book_mode, book_depth, trade_limit, funding_limit, rubik_period,
rubik_limit, rubik_taker_unit, max_staleness_hours,
disabled_sources, disabled_symbols
```

It must not own OHLCV bar cache refresh, evidence path, tailtree lifecycle, candidate ranking, or report section selection. Old aliases that blur these roles should be removed, not preserved.

## Core abstractions

| Abstraction | Owner | Meaning |
|---|---|---|
| `SourceNeed` | `sources.collect` | Scanner demand for family/symbol/time/depth/freshness. |
| `SourceCapability` | `sources.artifacts` or `sources.coverage` | Provider/source-family capability: scope, period, max rows/lookback, earliest provider timestamp, latest-refresh support, backfill support, review/evidence role, and rank penalty weight. |
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

## Capability and requirement policy

Source status is capability-aware. Do not collapse provider limits, stale rows, missing rows, and optional absent sources into one `missing` bucket.

```text
fresh                  rows exist and latest age is inside the freshness threshold
stale                  rows exist but latest age exceeds the freshness threshold
missing                required source has no usable rows inside its provider capability
provider_bounded       rows fill the provider capability window but the provider cannot satisfy the longer scanner target
optional_absent        optional/unimplemented source has no rows and no main-rank penalty
fetch_failed_frame_fresh latest fetch failed/empty but materialized frame rows remain fresh
```

Evaluation uses two coverage denominators:

```text
coverage_target_pct     = actual_rows / scanner_target_rows
coverage_capability_pct = actual_rows / min(scanner_target_rows, provider_cap_rows)
```

Rank and current review should use `coverage_capability_pct`, latest age, and required-family role. `coverage_target_pct` remains visible for research/evidence context but does not create a full penalty when the provider cannot supply that depth.

Family role policy:

```text
required market families: books, trades, funding, open_interest, taker_volume, long_short_ratios
optional context families: messages until a real provider/source is enabled
```

Provider-bounded Rubik examples:

```text
contract long/short 1H: max 1,440 rows ~= 60 days
ccy-level Rubik 1H: max 30 days; 1D: max 180 days
```

These sources are current/review context first. They must not be penalized as failed 730-day evidence sources when they are fresh and near their capability window.

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
- Derive frame-level source availability from persisted rows, latest timestamps, provider capabilities, and configured staleness thresholds.
- Preserve latest fetch-attempt provenance separately from frame usability.
- Emit explicit fresh/stale/missing/provider-bounded/optional-absent availability for diagnostics and rank inputs.
- Keep current-review refresh separate from historical backfill for provider-bounded Rubik-style sources.
- Treat latest/current snapshots, such as current funding and current open interest, as freshness inputs distinct from historical-depth rows.

## Non-responsibilities

- No exchange OHLCV cache ownership.
- No scanner evidence, tailtree, rank, ladder, or report policy.
- No strategy signal semantics.
- No research promotion policy.
- No live trading or account IO.

Concrete target APIs live in `docs/graph/sources.md`.
