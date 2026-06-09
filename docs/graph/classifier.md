# Classifier Module Graph

Classifier graph surfaces are deterministic, known-at-close feature/state builders. They are allowed to create research/scanner input columns, but they do not rank symbols, make review decisions, place orders, or call executor/core code.

## Strategy structure classifier

```text
qooi.strategies.structure
  RangeWidthThresholdConfig
    mode: fixed | rolling_quantile
    fixed_atr_max
    quantile/window/min_samples/fallback

  StructureClassifierConfig
    default()
    fixed(range_width_atr_max=8.0)
    rolling_quantile(...)

  add_price_structure_stage_features(...)
    -> returns FeatureFn: pl.DataFrame -> pl.DataFrame
    -> adds no-lookahead structure/stage/audit columns

  add_liquidity_sweep_features(...)
    -> returns FeatureFn
    -> adds shifted liquidity sweep/reclaim/event-quality columns

  add_none_context_diagnostics(...)
    -> returns FeatureFn
    -> adds ATR/key-level/Z-pressure context for ambiguous/none states
```

Required public classifier output columns:

```text
structure_trend_state
market_stage
structure_reason
market_stage_reason
stage_unknown_reason
range_width_atr_threshold
range_width_threshold_mode
range_width_threshold_ready
range_width_threshold_source
```

Important retained compatibility edge:

```text
add_price_structure_stage_features(range_width_atr_max=...)
  -> accepted legacy keyword path
```

Removed legacy facade symbols, guarded by tests:

```text
qooi.strategies.structure.StructureClassifier        # removed
qooi.strategies.structure.classify_price_structure_frame  # removed
```

## Scanner kline state classifier

```text
qooi.scanner.classifiers
  KlineClassifier(scale)
    classify(frame)
      -> accepts volume or vol
      -> validates required OHLCV fields and minimum rows
      -> returns scanner state frame

  validate_state_frame(frame)
    -> checks/casts/selects STATE_FRAME_COLUMNS

  classifier_health(frame, label="")
    -> ClassifierHealthResult(frame, text)
    -> checks required classifier columns and basic cardinality health
```

Scanner state-frame contract:

```text
symbol: string
timestamp: int64
source_family: string
scale: string
state_key: string
context_event: string
direction_hint: bullish | bearish | neutral | blocked | missing
quality_weight: float
missing_flag: bool
stale_flag: bool
```

Missing/shallow kline data returns an explicit missing state frame; it must not silently drop a symbol.

Removed scanner diagnostic builder symbols, guarded by tests:

```text
qooi.scanner.classifiers.ClassifierDiagnosticsBuilder  # removed
qooi.scanner.classifiers.evaluate_classifier_frame     # removed
```

## Consumers

```text
scripts/classifier_states.py
  -> qooi.research.reports.run_reports(...)
  -> uses strategy structure classifier columns in deterministic research artifacts

qooi.scanner.workflow.load_bars(...)
  -> _classify_kline_frames(...)
  -> qooi.scanner.classifiers.KlineClassifier(scale).classify(frame)

qooi.scanner.decisions.compute_kline_states(...)
  -> consumes scanner state frames
  -> builds SymbolStateBundle inputs for review decisions

qooi.scanner.diagnostics.write_diagnostics(...)
  -> writes classifier/state diagnostic artifacts
```

## Boundary rules

- Classifiers may depend on `polars` and shared semantic enums.
- Scanner classifiers may format health text through evaluation table formatting only; they must not call executor/backtest/live trading APIs.
- Future returns and realized transitions are outcome columns only; they are never classifier inputs.
- Optional learned-state/dynamic classifiers are not scanner decision inputs unless architecture is explicitly rewritten.
