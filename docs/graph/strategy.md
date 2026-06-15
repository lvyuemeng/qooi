# Strategy / Signal Module Graph

```text
qooi.strategies.catalog
  strategy_selection()
  strategy_metadata()
    -> qooi.strategies.specs strategy constructors

qooi.strategies.specs
  StrategySpec
  SignalRule
  HoldPolicy
  compute_signal_frame(df, spec)
    -> add_indicators()
    -> spec.features[]
    -> spec.filters[]
    -> spec.entries[]
    -> required signal columns

qooi.strategies.indicators
  add_indicators()
  add_macd_histogram()
  compute_flow_pipeline_frame()
  predicate builders

qooi.strategies.structure
  add_price_structure()
  add_price_structure_stage_features()
  add_liquidity_sweep_features()
  add_none_context_diagnostics()

  required classifier columns:
    structure_trend_state
    market_stage
    structure_reason
    market_stage_reason
    stage_unknown_reason
    range_width_atr_threshold
    range_width_threshold_mode
    range_width_threshold_ready
    range_width_threshold_source

qooi.strategies.semantics
  shared labels/enums

qooi.strategies.portfolio
  qualify_asset()
  allocate_portfolio_weights()
```

Core integration:

```text
BacktestExecutor.run()
  -> compute_signal_frame() unless precomputed_signal=True
  -> BarSignal per row
  -> process_bar()
```
