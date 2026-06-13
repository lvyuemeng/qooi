# Sources API Graph

Target API graph for demand-first source modules.

## Pipeline

```text
scanner config + symbols
  -> sources.context.load_source_context
  -> sources.collect.source_needs_from_config
  -> sources.collect.collect_source_context
  -> sources.bundle.merge_source_frames
  -> SourceContextResult
```

## Dependency graph

```text
sources.models

sources.schema

sources.artifacts
  -> sources.schema

sources.bundle
  -> sources.artifacts

sources.manifest
  -> sources.schema

sources.coverage
  -> sources.artifacts
  -> sources.manifest

sources.collect
  -> sources.artifacts
  -> sources.bundle
  -> sources.coverage
  -> sources.models
  -> sources.okx

sources.context
  -> sources.collect
  -> sources.bundle
  -> sources.coverage
  -> sources.artifacts
  -> sources.manifest

sources.okx
  -> sources.http
  -> sources.manifest
  -> sources.models
  -> exchange.market helpers

provider modules
  -> sources.http
  -> sources.manifest
  -> sources.models
```

No target edge:

```text
sources.context -> exchange.context
sources.coverage -> concrete exchange.store
sources -> scanner/research/strategy policy
```

## Core contracts

```text
sources.models.SourceResult(frame, manifest)

sources.collect.SourceNeed(
  family,
  symbols,
  start_ms,
  end_ms,
  min_rows,
  freshness_ms,
  mode,
)

sources.artifacts.SourceCapability(
  family,
  raw_source,
  scope,
  period,
  max_rows,
  max_lookback_days,
  earliest_provider_ms,
  supports_latest_refresh_int,
  supports_backfill_int,
  required_for_review_int,
  required_for_evidence_int,
  optional_int,
  rank_penalty_weight,
)

sources.collect.SourceFetchPlan(
  family,
  raw_source,
  symbol,
  start_ms,
  end_ms,
  limit,
  reason,
)

sources.collect.SourceCollectRequest(...)
sources.collect.SourceCollectResult(manifest, frames)

sources.artifacts.SourceFamily(
  name,
  artifact,
  timestamp_col,
  raw_sources,
  merge_key,
  row_kind_col=None,
  history_kind=None,
  known_at_col=None,
)

sources.artifacts.ArtifactSpec(name, relative_path, schema, required)

sources.bundle.SourceBundle(...)

sources.context.SourceAvailability(
  family,
  symbol,
  rows,
  latest_timestamp,
  latest_age_hours,
  freshness_threshold_hours,
  provider_cap_rows,
  provider_cap_lookback_days,
  coverage_target_pct,
  coverage_capability_pct,
  frame_fresh_int,
  frame_stale_int,
  frame_missing_int,
  provider_bounded_int,
  optional_absent_int,
  fetch_failed_frame_fresh_int,
  usable_int,
  required_for_review_int,
  required_for_evidence_int,
  rank_penalty_weight,
  latest_fetch_status,
  latest_fetch_warning,
)

sources.coverage.SourceFetchObservation(
  family,
  raw_source,
  symbol,
  status,
  warning,
  endpoint,
  status_code=None,
  provider_code=None,
  response_body_prefix=None,
)

sources.context.SourceContextResult(manifest, frames, availability)
```

Keep these contracts small. Add fields only when a current source decision requires them.

## `qooi.sources.context`

Scanner-facing API:

```text
resolve_source_refresh_mode(
    top_level_refresh_mode: RefreshMode,
    source_refresh_mode: SourceRefreshMode,
) -> RefreshMode

load_source_context(config, *, symbols, context_symbols, discovery) -> SourceContextResult
```

Refresh resolution:

```text
source.refresh_mode="inherit" -> top_level_refresh_mode
source.refresh_mode=<mode>    -> <mode>
```

No other source API should inspect both config locations. There is no compatibility alias where `[potential.source].refresh_mode` controls bar cache refresh.

Flow:

```text
load_source_context
  -> read_source_bundle
  -> source_needs_from_config
  -> collect_source_context
  -> merge/write source artifacts
  -> source_availability
       -> summarize materialized frame rows by family/symbol
       -> compute latest_age_hours vs freshness_threshold_hours
       -> join latest fetch observation from manifest for provenance only
  -> SourceContextResult
```

Availability rule:

```text
frame_missing_int          = required_for_review_int and rows == 0
frame_stale_int            = rows > 0 and latest_age_hours > freshness_threshold_hours
frame_fresh_int            = rows > 0 and latest_age_hours <= freshness_threshold_hours
provider_bounded_int       = scanner_target_rows > provider_cap_rows and coverage_capability_pct is high
actionable_missing_int     = frame_missing_int and not optional_absent_int
fetch_failed_frame_fresh_int = latest_fetch_status in {failed, missing} and frame_fresh_int
usable_int                 = frame_fresh_int or fetch_failed_frame_fresh_int
```

Coverage denominators:

```text
coverage_target_pct     = rows / scanner_target_rows
coverage_capability_pct = rows / min(scanner_target_rows, provider_cap_rows)
```

Rank and current-review gates consume `coverage_capability_pct`, `frame_stale_int`, `actionable_missing_int`, and required-family role. Long-horizon `coverage_target_pct` remains visible for research but does not create a full penalty when `provider_bounded_int=1`.

The latest manifest row can set `latest_fetch_status` and `latest_fetch_warning`; it must not set frame rows, latest timestamp, age, or usability by itself.

Allowed side effects:

```text
read source artifacts
write source artifacts
write source manifest
```

No provider endpoint logic here.

## `qooi.sources.collect`

Demand and collection API:

```text
source_needs_from_config(config, *, symbols, context_symbols, start_ms, end_ms) -> tuple[SourceNeed, ...]
collect_source_context(request: SourceCollectRequest) -> SourceCollectResult
```

Rules:

```text
source_needs_from_config: pure
collect_source_context: side effects allowed
```

Funding collection rules:

```text
history depth uses funding_source_kind=history
family freshness may use funding_rate/current rows
current funding fetch is independent from history depth
```

Capability-aware collection rules:

```text
collect_latest_context(request, families, symbols) -> SourceCollectResult
collect_historical_backfill(request, families, symbols) -> SourceCollectResult
```

Current review freshness comes before deep backfill:

```text
books/trades snapshots
funding current
open_interest current/latest snapshot
Rubik latest page for taker_volume and long_short_ratios
```

Historical backfill is separate and provider-bounded:

```text
funding history can page deeper through funding history
Rubik contract long/short 1H is capped by provider rows/lookback
ccy-level Rubik 1H is capped at 30 days; 1D at 180 days
```

An empty latest Rubik page records fetch provenance. It must not erase fresh cached rows or convert a provider-bounded source into missing.

Snapshot planning rules:

```text
books: snapshot only
trades: recent/snapshot only
```

Historical planning rules:

```text
funding history: page backward through funding history
open_interest/taker_volume/long_short_ratios: Rubik historical windows
```

## `qooi.sources.artifacts`

Artifact/family API:

```text
SOURCE_FAMILIES
SOURCE_CAPABILITIES
SOURCE_ARTIFACT_SPECS
source_family(name) -> SourceFamily
source_capability(family_or_raw_source, *, period) -> SourceCapability
source_manifest_family(raw_source) -> family
artifact_path(output_dir, spec) -> Path
coerce_frame(frame, schema) -> DataFrame
read_frame_artifact(output_dir, spec) -> DataFrame
write_frame_artifact(output_dir, spec, frame) -> None
```

`SOURCE_FAMILIES` is a small constant table, not a plugin framework.

It derives:

```text
family -> artifact
raw source -> family
family -> timestamp column
family -> merge key
family -> row-kind semantics
```

## `qooi.sources.bundle`

Bundle/materialization API:

```text
read_source_bundle(output_dir, specs) -> SourceBundle
write_source_bundle(output_dir, bundle, specs) -> None
source_frame(bundle, family) -> DataFrame
source_symbols(bundle, family) -> tuple[str, ...]
missing_symbols(bundle, family, symbols) -> tuple[str, ...]
latest_timestamp(bundle, family, symbol) -> int | None
merge_source_frames(family_or_artifact, cached, incoming) -> DataFrame
replace_symbol_rows(frame, symbol, replacement) -> DataFrame
```

Merge keys come from `SourceFamily`, not local duplicate maps.

## `qooi.sources.coverage`

Pure observation/availability API:

```text
source_frame_observations(
    frames: dict[str, DataFrame],
    families: tuple[SourceFamily, ...],
    capabilities: tuple[SourceCapability, ...],
    symbols: tuple[str, ...],
    *,
    now_ms: int,
    scanner_target_rows_by_family: dict[str, int],
    freshness_ms_by_family: dict[str, int],
) -> tuple[SourceAvailability, ...]

source_capability_windows(
    capabilities: tuple[SourceCapability, ...],
    *,
    scanner_target_rows_by_family: dict[str, int],
) -> DataFrame

latest_fetch_observations(manifest: DataFrame) -> tuple[SourceFetchObservation, ...]

join_fetch_provenance(
    availability: tuple[SourceAvailability, ...],
    fetches: tuple[SourceFetchObservation, ...],
) -> tuple[SourceAvailability, ...]

stale_symbols(availability, freshness_ms) -> tuple[str, ...]
eligible_fetch_symbols(availability, need) -> tuple[str, ...]
missing_evidence_for_symbol(availability, symbol) -> dict[str, int]
compute_source_coverage_score(availability) -> float
```

Frame observation owns quantitative state:

```text
rows
latest_timestamp
latest_age_hours
freshness_threshold_hours
provider_cap_rows
provider_cap_lookback_days
coverage_target_pct
coverage_capability_pct
frame_fresh_int
frame_stale_int
frame_missing_int
provider_bounded_int
optional_absent_int
usable_int
required_for_review_int
rank_penalty_weight
```

Fetch observation owns provider provenance:

```text
latest_fetch_status
latest_fetch_warning
endpoint
status_code
provider_code
response_body_prefix
```

`latest_fetch_status="missing"` after an empty incremental page does not make `frame_missing_int=1` when cached frame rows exist.

No concrete `exchange.store` import in target state.

## `qooi.sources.okx`

Provider endpoint API:

```text
fetch_okx_instruments(...)
fetch_okx_tickers(...)
fetch_okx_book_snapshot(...)
fetch_okx_recent_trades(...)
fetch_okx_funding_history(...)
fetch_okx_funding_rate(...)
fetch_okx_open_interest(...)
fetch_okx_open_interest_history(...)
fetch_okx_taker_volume_contract(...)
fetch_okx_long_short_account_ratio_contract(...)
fetch_okx_top_trader_long_short_account_ratio_contract(...)
fetch_okx_top_trader_long_short_position_ratio_contract(...)
fetch_okx_margin_loan_ratio(...)
fetch_okx_ccy_long_short_account_ratio(...)
fetch_okx_ccy_open_interest_volume(...)
```

Provider wrappers return:

```text
SourceResult(frame, manifest)
```

Provider manifest rows must include sanitized transport diagnostics when available:

```text
status_code              # HTTP status, e.g. 403
provider_code            # provider payload/code/body token, e.g. 1010
response_body_prefix     # short sanitized body prefix, no secrets
endpoint
params_shape             # sanitized parameter shape when useful
```

Historical pagers should preserve page-level stop evidence:

```text
fetch_stop
fetch_page_index
fetch_pages
fetch_cursor
fetch_oldest_ts
fetch_status_code
fetch_provider_code
```

They do not decide scanner demand, artifact merge, or source availability.

## Other provider modules

Provider wrapper shape:

```text
fetch_<provider>_<resource>(...) -> SourceResult
normalize_<provider>_<resource>(payload) -> DataFrame
empty_<provider_family>_frame() -> DataFrame where needed
```

Current providers:

```text
coingecko
coinpaprika
defillama
cryptopanic
polymarket
messages
okx_ws
```

## Source family table

```text
books
  raw_sources: books
  artifact: source_books
  timestamp: timestamp
  merge_key: symbol,timestamp
  mode: snapshot

trades
  raw_sources: trades
  artifact: source_trades
  timestamp: timestamp
  merge_key priority: (symbol,trade_id), then (symbol,timestamp,price,size,side)
  mode: snapshot/recent

funding
  raw_sources: funding,funding_rate
  artifact: source_funding
  timestamp: known_at_ms for availability, funding_time for history depth
  merge_key priority: (symbol,funding_time), then (symbol,timestamp)
  row_kind: funding_source_kind
  history_kind: history
  mode: both

open_interest
  raw_sources: open_interest_history, open_interest
  artifact: source_open_interest
  timestamp: timestamp
  mode: current + provider-bounded history
  current: /api/v5/public/open-interest by instId
  history: /api/v5/rubik/stat/contracts/open-interest-history by instId

taker_volume
  raw_sources: taker_volume_contract
  artifact: source_taker_volume
  timestamp: timestamp
  mode: latest + provider-bounded history

long_short_ratios
  raw_sources: long_short_ratio_contract, top_trader_long_short_account_ratio_contract, top_trader_long_short_position_ratio_contract
  artifact: source_long_short_ratios
  timestamp: timestamp
  mode: latest + provider-bounded history
  contract_1h_cap: 1440 rows ~= 60 days

ccy_rubik_context
  raw_sources: margin_loan_ratio, ccy_long_short_account_ratio, ccy_open_interest_volume
  artifact: source_ccy_rubik_context
  timestamp: timestamp
  mode: base-currency context, not instrument replacement
  ccy_1h_cap: 30 days
  ccy_1d_cap: 180 days

messages
  raw_sources: local messages
  artifact: local message artifacts
  timestamp: timestamp
  mode: optional context until provider-enabled
  rank_penalty_weight: 0 while optional_absent_int=1
```

## Target downstream callers

```text
scanner.workflow   -> sources.context.load_source_context
scanner.decisions  -> SourceContextResult.frames + availability
scanner.diagnostics-> SourceContextResult.frames + availability + manifest
exchange.discovery -> sources.okx instruments/tickers only
exchange.universe  -> broad provider wrappers only
```
