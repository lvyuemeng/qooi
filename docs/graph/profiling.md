# Profiling Module Graph

Architecture: `docs/architecture/profiling.md`.

## Public API

```text
qooi.profiling.ProfileConfig(
    mode: "off" | "stage" | "hotpath" | "native" = "off",
    top_n: int = 30,
)

qooi.profiling.ProfileContext(
    config: ProfileConfig,
    root: Path,
)

ProfileContext.stage(layer: str, component: str, stage: str) -> context manager
ProfileContext.frame(layer: str, component: str, frame: str, data: pl.DataFrame) -> None
ProfileContext.native(label: str, callable) -> result
ProfileContext.write() -> None
```

Native Polars node profiling remains a target hook for lazy plans; current code
ships stage/frame artifacts plus stdlib `cProfile`/`pstats` rows.

## Modes

```text
off     -> no-op
stage   -> stages.csv + frames.csv + summary.md
hotpath -> stage artifacts + cProfile/pstats native.csv
native  -> hotpath artifacts; reserved for selected Polars node profile rows
```

## Injection graph

```text
workflow root config
  -> ProfileConfig section
  -> ProfileContext(root=<workflow artifact root>/profile)
  -> injected into module calls
  -> modules record StageRecord / FrameRecord
  -> ProfileContext.write()
```

Consumers may include:

```text
qooi.exchange.*
qooi.sources.*
qooi.scanner.*
qooi.research.*
qooi.strategies.*
qooi.core.*
```

Dependency direction:

```text
domain module -> qooi.profiling
qooi.profiling -> stdlib + polars only
```

## Scanner use

```text
qooi.scanner.workflow.run(config_path)
  -> ProfileContext from config.profile
  -> stage(resolve_universe)
  -> stage(load_bars)
  -> stage(load_source_context)
  -> qooi.scanner.diagnostics.build_diagnostic_frames(inputs, profile)
       -> frame(kline_history)
       -> frame(realized_transitions)
       -> frame(potential_observations)
       -> frame(potential_outcomes)
  -> stage(render_report)
  -> profile.write()
```

## Artifact schemas

```text
stages.csv
  run_id, layer, component, stage, seconds, status

frames.csv
  run_id, layer, component, frame, rows, cols,
  symbol_count, timeframe_count, horizon_count,
  source_family_count, decision_timeframe_count

native.csv
  run_id, rank, function, file, line, ncalls, tottime_s, cumtime_s

polars.csv target
  run_id, layer, component, stage, node, start_us, end_us, duration_us
```

No module-specific CSV writer APIs are exposed. Modules emit records; `qooi.profiling` persists them.
