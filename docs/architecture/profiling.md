# Profiling Architecture

## Purpose

`qooi.profiling` is a cross-cutting diagnostics layer. It measures workflow stages and data-product shapes without owning exchange, source, scanner, research, strategy, or execution behavior.

Profiling is evidence for engineering decisions only. It does not affect trading, ranking, model output, or provider acquisition policy.

## Owned surface

```text
qooi.profiling
  ProfileConfig
  ProfileContext
  StageRecord
  FrameRecord
  NativeProfileRecord
```

## Responsibilities

- inject a `ProfileContext` into any workflow or module;
- record stage durations with `time.perf_counter()`;
- record dataframe shape/cardinality as numeric facts;
- summarize stdlib `cProfile`/`pstats` output when native function profiling is requested;
- reserve a native Polars `LazyFrame.profile()` hook for selected lazy dataframe stages;
- write one profile artifact set per run root.

## Non-responsibilities

- no scanner/source/exchange/research strategy logic;
- no repeated CSV-writing helper surface per module;
- no external profiler dependency by default;
- no global singleton context;
- no performance thresholds in normal correctness tests.

## Native-only profiler policy

Default profiler stack:

```text
time.perf_counter        -> stage timing
DataFrame metadata       -> rows, cols, cardinalities
cProfile + pstats        -> Python call hotpath
Polars LazyFrame.profile -> target hook for dataframe node hotpath where a lazy plan exists
```

External profilers are not recommended unless these native artifacts fail to identify the bottleneck.

## Context injection rule

```text
workflow/config owner creates ProfileContext
module receives ProfileContext | None
module records only its own stage/frame facts
qooi.profiling writes artifacts through one sink
```

Allowed dependency:

```python
from qooi.profiling import ProfileContext
```

Forbidden dependency inside `qooi.profiling`:

```python
from qooi.scanner.workflow import run
from qooi.sources.context import load_source_context
from qooi.exchange.store import refresh_history
```

## Record contracts

`StageRecord` grain: one timed stage.

```text
run_id, layer, component, stage, seconds, status
```

`FrameRecord` grain: one dataframe-like product.

```text
run_id, layer, component, frame, rows, cols,
symbol_count, timeframe_count, horizon_count,
source_family_count, decision_timeframe_count
```

`NativeProfileRecord` grain: one native profiler row.

```text
cProfile: run_id, rank, function, file, line, ncalls, tottime_s, cumtime_s
Polars:   run_id, layer, component, stage, node, start_us, end_us, duration_us
```

## Artifact contract

A workflow supplies the root path. Profiling writes a single profile directory under that root:

```text
profile/stages.csv
profile/frames.csv
profile/native.csv
profile/polars.csv
profile/summary.md
```

Modules do not own CSV writer variants. They emit typed records; `qooi.profiling` owns persistence.
