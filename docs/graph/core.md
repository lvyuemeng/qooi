# Core Module Graph

```text
qooi.core.__init__
  BarMarket.from_frame()
  BarSignal
  PipelinePolicy
  PipelineContext
  process_bar(df, baskets/book, pair, exit_cfg, recovery_cfg, context)
    -> _evaluate_hold_thesis()
    -> evaluate_hard_exits()
    -> recovery.evaluate()
    -> evaluate_exits()
    -> _build_flip_and_entry_actions()
    -> list[BasketAction]

qooi.core.basket
  Basket
  BasketBook
    active()
    active_for_strategy()
    snapshot()
    apply_action()
    apply_actions()
    advance_bar()
  BasketManager
    can_open()
    create()
  evaluate_hard_exits()
  evaluate_exits()

qooi.core.recovery
  NoRecovery
  GridRecovery
  MartingaleRecovery
  ReverseRecovery
  evaluate()
    -> list[BasketAction] proposals

qooi.core.executor
  BacktestExecutor.run()
    -> compute_signal_frame()
    -> process_bar()
    -> account fills/fees/trades
    -> BasketBook.apply_actions()
    -> BasketBook.advance_bar()
    -> equity/diagnostics
  BacktestExecutor.run_report()
  LiveExecutor.execute()

qooi.core.evaluate
  Report / diagnostics dataclasses
  compare()
  format_*()

qooi.core.metrics
  compute_metrics()
```
