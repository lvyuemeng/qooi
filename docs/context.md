# qooi — domain context

## Domain

`qooi` is a quantitative strategy research and execution system for OKX crypto perpetual swaps.

Execution is swap/perp-first. Default research data is swap OHLCV because fills and contract sizing happen on swap instruments. Spot data may be used as an explicit signal source policy only when a strategy intentionally needs underlying spot prices. Spot-vs-swap is orchestration metadata, not a strategy identity.

The core domain object is a basket: an independent vehicle opened by an entry event and managed until basket-owned exits, strategy thesis failure, or bounded recovery closes or transforms it.

## Stack

- Python management: `uv`
- Python: 3.12
- DataFrame library: Polars
- Exchange access: OKX SDK and CCXT
- Static checks: Ruff and Ty
- Test runner: Pytest

## Architectural Rules

The system is layered. Lower layers must not import higher-layer policy.

Forbidden dependencies:

- Data layer must not know strategies, baskets, execution, or reports.
- Strategy layer must not know basket caps, fills, fees, account equity, or recovery state.
- Basket lifecycle must not fetch data, compute indicators, format reports, or account cash.
- Recovery evaluators must not mutate baskets or open/close positions directly.
- Executor must not compute strategy indicators or choose spot/swap source policy.
- Evaluation must not mutate execution state or silently compare incompatible runs.

Mutation boundary:

- `process_bar()` returns proposals only.
- `BasketBook` owns basket lifecycle mutation.
- Executor owns cash, fills, fees, trade rows, and equity accounting.
- Executor accounts accepted proposals before applying terminal lifecycle actions.

## 1. Data Layer

Files:

- `src/qooi/exchange/market.py`
- `src/qooi/exchange/store.py`

Responsibilities:

- Fetch bars, book, funding, and archive metadata from exchange backends.
- Normalize exchange schemas.
- Store and load Parquet caches.
- Plan historical fetch horizon from `days`, `min_bars`, and bar size.
- Validate actual cache coverage against the planned horizon.
- Return data-only metadata.

Exchange API vocabulary is resource-first and async-first:

- `bars()` and `bars_since()` replace public `ohlcv` / `candles` naming.
- `book()` and `books()` replace public `order_book` / `ob_*` naming.
- `funding()` returns funding-rate frames.
- `archives()` returns OKX downloadable historical-market-data metadata.
- Store APIs use uniform resource methods: `bars()`, `funding()`, `books()`, and `many()` for batch bar refreshes.
- Exchange capability is expressed with `SyncExchange` and `AsyncExchange` protocols, not a facade or base-class hierarchy.
- Concrete exchange clients are split by execution model: `OkxSyncExchange`, `OkxAsyncExchange`, `CcxtSyncExchange`, and `CcxtBooksStream`.
- Preferred lifecycle management is context-manager based: `with CacheStore() as store: ...` and `async with AsyncCacheStore() as store: ...`.

Canonical history types:

- `HistoryRequest`: one instrument/timeframe/history request.
- `HistoryTarget`: planned target bars, days, and since timestamp.
- `HistoryCoverage`: actual bars, range, gaps, duplicates, freshness, refresh flag, and coverage percent.

Source policy is selected outside the data layer:

- `swap`: swap signal and swap execution, default.
- `spot_signal_swap_exec`: spot signal with swap execution.
- `spot`: spot-only research approximation.

Removed old APIs:

- `CacheValidation`
- `CacheSummary`
- `validate_ohlcv()`
- `describe_ohlcv()`

Only planned-history vocabulary should remain for cache validity.

## 2. Strategy/Signal Layer

Files:

- `src/qooi/strategies/specs.py`
- `src/qooi/strategies/features.py`
- `src/qooi/strategies/indicators.py`

Responsibilities:

- Compute indicators and reusable features.
- Keep predicate expression builders with indicators and frame transforms with features.
- Apply filters and entry rules.
- Emit explicit signal columns.
- Express strategy-owned exit intent.
- Provide signal diagnostics by rule/module.

Required signal columns:

- `raw_entry_signal`: signed raw rule output after filters.
- `entry_signal`: event-like signed signal that may open a new basket.
- `position_signal`: held directional thesis state.
- `exit_signal`: strategy-owned exit intent.
- `signal_strength`: numeric confidence or quality scalar for sizing and diagnostics.
- `signal_id`: stable module/rule identifier for dedupe and reporting.
- `signal`: compatibility alias, preferably equal to `position_signal` until all callers migrate.

Signal semantics:

- Entry signals decide when a new basket may be opened.
- Position signals describe whether the strategy thesis is currently long, short, or neutral.
- Exit signals describe explicit strategy-owned exit intent.
- Signal strength should not encode direction; direction stays signed in signal columns.
- Signal ID should identify the module/rule, not the symbol or basket ID.

Strategies do not decide how many baskets may be active. Basket caps and lifecycle rules decide that.

Strategy improvement rules:

- Strategy improvements must be ex-ante thesis changes, not performance-selected side or symbol filters.
- A regime filter is legitimate only if it uses information available before entry and has a stated market-behavior rationale.
- Long-only, short-only, include-signal, and exclude-signal runs are diagnostic strata unless the strategy declares an asymmetric thesis before testing.
- Symbol-aware changes are execution and microstructure normalization only, not performance-aware tuning.
- Signal strength should encode quality or confidence only after base rule quality is stable under comparable mechanics.

## 3. Pipeline Context

The per-bar pipeline is configured by domain objects instead of many scalar arguments.

Core concepts:

- `BarMarket`: close, high, low, ATR, timestamp, and bar index.
- `BarSignal`: position, entry, exit, strength, and signal ID.
- `PipelinePolicy`: neutral-close compatibility, flip policy, and thesis-continuation mode.
- `PipelineContext`: strategy ID plus market, signal, and policy.

The pipeline contract:

- Consume one market bar and one explicit signal context.
- Evaluate existing baskets independently.
- Return fully populated action proposals.
- Do not mutate baskets.
- Do not mutate recovery action proposals in place.
- Do not apply lifecycle actions.
- Do not update trailing state or bars-held counters.

This keeps signal interpretation, lifecycle mutation, and executor accounting separated.

## 4. Basket Layer

Files:

- `src/qooi/core/basket.py`
- `src/qooi/core/__init__.py`

Responsibilities:

- Manage independent baskets created from entry events.
- Enforce basket caps.
- Store basket entry, size, stop, target, trailing state, recovery state, and lifecycle state.
- Provide immutable snapshots for executor accounting.
- Apply accepted lifecycle actions.
- Advance active basket bar state after the executor has accounted fills.

Correct basket semantics:

- A symbol may have multiple independent baskets.
- Multiple signals on the same symbol may create independent baskets up to explicit caps.
- Caps prevent signal storms; they should not force one basket per symbol by default.
- Basket IDs include symbol, strategy, direction, timestamp or sequence, and a uniqueness suffix.

Basket caps:

- `max_total`
- `max_per_symbol`
- `max_per_strategy_symbol`

Lifecycle mutation belongs in `BasketBook`:

- `apply_action()` applies accepted `ENTER`, `EXIT`, `ADD_GRID`, and `HEDGE` actions.
- `apply_actions()` applies a batch in executor order.
- `advance_bar()` updates trail state, target-hit state, and bars-held for still-active baskets that were not touched by an accepted action.
- `snapshot()` captures pre-mutation state for accounting and reports.

## 5. Hold And Thesis Continuation

A basket is held while its management rules say the thesis remains valid.

There are two supported hold modes:

- Basket-owned hold mode, default: a basket remains active until basket-owned stop, target, trailing, time stop, strategy `exit_signal`, or recovery terminal action fires.
- Strict thesis-continuation mode: a basket also exits when the held strategy thesis no longer confirms the basket direction.

Default rules:

- Neutral `position_signal == 0` does not close active baskets by default.
- Opposite held `position_signal` does not close active baskets by default.
- `exit_signal=True` closes matching active baskets as `strategy_exit`.
- `close_on_neutral_signal=True` is compatibility mode only.
- `require_thesis_continuation=True` exits as `thesis_failed` when the position thesis no longer supports the basket direction.

Flip policy applies only to opposite entry events, not passive held state:

- `ignore`: existing baskets continue; opposite entries may open another basket if caps allow.
- `close_same_strategy_opposite`: close same-strategy opposite-direction baskets when an opposite entry event appears.
- `reverse`: reserved for explicit close-and-open reversal semantics.

## 6. Recovery Layer

File:

- `src/qooi/core/recovery.py`

Responsibilities:

- Generate pure recovery proposals for grid, hedge, martingale, and reverse recovery.
- Respect recovery level and loss controls.
- Leave all basket mutation to `BasketBook`.

Recovery rules:

- Recovery policies return `BasketAction` proposals only.
- Recovery policies must not mutate `Basket`.
- Recovery level increments only after an accepted lifecycle action is applied.
- Grid recovery adds to the same basket after acceptance.
- Hedge recovery opens a separate hedge basket after acceptance.
- Reverse recovery closes the old basket and opens an opposite basket only when adverse movement criteria are met and, by default, an opposite signal thesis confirms the reversal.

Reverse recovery is not a rescue for weak entries. It should only be tested on strategies with positive base expectancy and bounded exposure.

Recovery acceptance rules:

- Candidate strategy evaluation uses `NoRecovery` by default.
- Recovery modes are mechanics stress tests until every exposure-creating recovery action has a risk sizing decision or equivalent accepted-risk metadata.
- Unsized recovery exposure is blocked under research defaults.
- Martingale and reverse recovery are atomic transformations: the original basket must not be closed if the paired reversal entry is blocked.
- A hard stop outranks recovery by default.

## 7. Exit And Fill Layer

Files:

- `src/qooi/core/basket.py`
- `src/qooi/core/executor.py`

Exit responsibilities:

- Stored basket stop and target levels are canonical.
- Intrabar OHLC stop/target checks are used in backtests.
- Trailing and breakeven logic read basket state but return proposals.
- Same-bar ambiguity should use a documented conservative policy.

Current conservative fill policy:

- Hard stops are checked before recovery.
- If a stop and target are both touched before target state is active, the backtest keeps stop-first behavior and reports a target-first counterfactual.
- Target activation enables trailing or breakeven management after the target-touch bar, not earlier within that same OHLC bar.
- Trailing and breakeven exits are disabled while recovery is active.

Fill/accounting responsibilities:

- Fees are notional based and charged on every fill.
- Trade PnL is computed from immutable pre-close `BasketSnapshot` fields.
- Mark-to-market equity includes unrealized PnL by default.
- Lifecycle close is applied after accounting so reset-in-place cannot erase accounting inputs.

Executor order per bar:

1. Build `PipelineContext` from signal columns and market bar.
2. Call `process_bar()` for proposals.
3. Account accepted fills, fees, and trade rows using action snapshots.
4. Apply accepted actions through `BasketBook`.
5. Advance untouched active baskets with `BasketBook.advance_bar()`.
6. Mark portfolio value and drawdown state.

## 8. Sizing And Investment Per Basket

Sizing must explain risk and notional constraints.

Current formula:

- `risk_per_contract = abs(entry_px - stop_px) * ct_val`
- `risk_budget_usd = capital * max_risk_pct * signal_strength * lot_multiplier`
- `risk_sized_contracts = floor(risk_budget_usd / risk_per_contract)`
- `max_notional_usd = capital * leverage * max_notional_pct_per_basket`
- `notional_sized_contracts = floor(max_notional_usd / (entry_px * ct_val))`
- `contracts = min(risk_sized_contracts, notional_sized_contracts)` when at least `min_contracts`

If the calculated size is below `min_contracts`, the trade is blocked rather than forcing a minimum that violates risk or notional caps.

`SizingDecision` records:

- selected contracts
- risk per contract
- risk budget
- risk-sized contracts
- max notional budget
- notional-sized contracts
- binding cap
- blocked reason

## 9. Backtest Engine Layer

File:

- `src/qooi/core/executor.py`

Responsibilities:

- Simulate one prepared OHLCV/signal frame.
- Consume explicit signal columns.
- Invoke the basket pipeline.
- Simulate fills, fees, mark-to-market equity, and diagnostics.
- Return raw trades/equity or structured `BacktestResult`.

The engine does not fetch data, choose spot/swap policy, parse CLI, run temporal validation styles directly, or format reports.

## 10. Backtest Styles Layer

File:

- `src/qooi/core/styles.py`

Responsibilities:

- Walk-forward validation.
- Rolling-window validation.
- Cross-validation.
- Stability statistics.

Styles are strategy-independent and repeatedly call a backtest function over slices.

## 11. Evaluation Layer

Files:

- `src/qooi/core/metrics.py`
- `src/qooi/core/evaluate.py`

Responsibilities:

- Compute metrics from trades and equity.
- Format reports.
- Compare compatible runs.
- Warn when runs are not comparable.

Metric priority:

- Primary: trade count, win rate, profit factor, expectancy, average win/loss, median trade, exit reason mix.
- Secondary: exposure-normalized return, drawdown, active bars, mark-to-market equity metrics.
- Tertiary: calendar Sharpe, Sortino, annual return, annual volatility.

Calendar annualized metrics are diagnostics only for sparse systems.

Required report sections:

1. Run metadata.
2. Data coverage.
3. Signal funnel.
4. Basket lifecycle.
5. Trade metrics.
6. Exposure metrics.
7. Equity metrics.
8. Annualized diagnostics.

Comparability warnings should appear when data horizon, source policy, strategy arguments, basket caps, fill policy, fee model, drawdown stop, or mark-to-market settings differ.

## 12. Diagnostics Principle

Diagnostics are part of the strategy correctness model, not a flat collection of counters.

Diagnostic data structures must provide information for four-layer strategy consistency, correctness, debugging, and improvement:

1. Feature diagnostics prove indicators and input features are valid before judging signals.
2. Signal/thesis diagnostics prove strategy rules express the intended directional thesis and entry/exit events.
3. Basket/execution lifecycle diagnostics prove signals produce consistent basket and order behavior, including accepted entries, blocked entries, duplicate suppression, exit reasons, same-bar sequencing, multiple baskets, and recovery actions.
4. Portfolio/sizing/risk diagnostics prove PnL, drawdown, exposure, fee drag, stop effectiveness, sizing caps, notional asymmetry, and recovery impact are acceptable.

Every new diagnostic field should belong to one of these layers. Avoid adding unrelated top-level counters when a layer-owned structure can explain the behavior more clearly.

Interpretation order:

1. If feature diagnostics fail, fix data/features before tuning signals.
2. If signal diagnostics fail, fix strategy rules before executor or risk changes.
3. If lifecycle diagnostics fail, fix basket/order behavior before judging metrics.
4. If portfolio/sizing/risk diagnostics fail while earlier layers pass, treat the problem as sizing, exposure, risk control, or recovery behavior rather than signal quality.

Reports should summarize layer status first and show supporting counters only when they help explain a behavior or debugging path.

### Research-Evaluation Diagnostics

The exposed diagnostics API has two modes: `backtest` and `research-evaluation`.

`research-evaluation` exposes only:

- `timeframe-classifier`: classifier health evidence.
- `dynamic-transition-discovery`: deterministic transition-pattern discovery artifacts.
- `pattern-quality`: shared scored-pattern surface for static, transition, learned-state, and policy-context diagnostics.
- `trade-record-modulation`: optional post-trade control evidence.

Dynamic transition discovery is an output under `research-evaluation`, not a new diagnostics mode.

Behavior-driven state research uses the shared `ResearchFrame -> PatternTable -> OutcomeTable -> MetricTable -> ScoredPatternTable -> ArtifactBundle` data pipe.

Former direct diagnostic modes and outputs such as `modulation-effect`, `market-state-forward`, `tradability`, `joint-forward-quality`, `timeframe-forward-quality`, `resonance-candidates`, `state`, `state-profitability`, and `state-filter-delta` were removed. Their useful concepts are represented by shared pattern metrics, promotion gates, transition artifacts, or normal backtest reports.

Forward labels may use future OHLCV only as outcome columns. Grouping and state columns must remain known at bar close. Pattern-quality artifacts are diagnostic-only: no entry filter or strategy variant is authorized until an explicit strategy hypothesis passes execution-aware backtests.

Detailed formulas and the reduced graph contract are documented in `docs/research-evaluation-api-reference.md`.

## 13. Behavior-Driven State Research

Behavior-driven state research is an exploratory research layer above data preparation and below strategy promotion. It discovers market-state structure from known-at-close labels, then subjects any candidate pattern to the normal promotion and backtest contract.

Research stages:

- Stage 1 dynamic transition diagnostics consume known-at-close classifier labels and deterministic context columns.
- Stage 2 learned state encoders may produce `behavior_state_id` research labels only after Stage 1 evidence justifies model complexity.
- Stage 3 policy learners and world models remain simulation/research artifacts until adapted into normal qooi signal columns.

Boundary rules:

- Dynamic transition discovery must not import executor, basket, recovery, or exchange trading clients.
- Learned state IDs are research labels, not strategy identities.
- Policy learners must not own basket lifecycle, fills, fees, cash, exchange calls, or recovery mutation.
- Forward returns may score transition patterns but must not feed state construction.
- No diagnostic artifact authorizes live trading or allocation.

Deployable outputs must eventually adapt into normal signal columns:

- `raw_entry_signal`
- `entry_signal`
- `position_signal`
- `exit_signal`
- `signal_strength`
- `signal_id`

Stage 1 artifacts are:

- `state-transition-graph.csv`
- `transition-information.csv`
- `transition-ngram-quality.csv`
- `none-event-context-quality.csv`

Dependency policy:

- Stage 1 uses Polars and the standard library only.
- Stage 2 ML dependencies are optional research dependencies, not core runtime dependencies.
- Stage 3 RL dependencies are optional research dependencies and require a concrete simulation environment contract first.

## 14. Backtest Orchestration

File:

- `scripts/classifier_states.py`
- `scripts/learned_states.py`
- `src/qooi/research/data.py`
- `src/qooi/research/reports.py`

Responsibilities:

- Parse CLI.
- Select pairs.
- Select explicit strategy specs from `qooi.strategies.catalog`.
- Choose data-source policy.
- Translate CLI-derived classifier settings and debug filters outside the strategy layer.
- Plan context frames and cache-backed classifier/signal frame preparation outside the strategy layer.
- Ask the data layer for requested histories.
- Prepare the final execution frame.
- Call the engine or styles layer.
- Call evaluation formatting.

## 15. Dynamic Statistics Research

The first adaptive-indicator slice is dependency-free and intentionally precedes HMM, PCA, Kalman, GARCH package fitting, or ML classifiers.

Available composable features:

- `add_ewma_z_score()` for exponentially weighted price normalization.
- `add_robust_z_score()` for rolling median/MAD normalization.
- `add_volatility_regime()` for short/long realized volatility ratio and numeric regime.
- `add_garch_like_volatility()` for recursive conditional-volatility approximation and `garch_z_return`.
- `add_dynamic_z_blend()` for inspectable blended fixed, EWMA, and robust Z-score columns.

Research strategy variants:

- `adaptive_zscore_mean_reversion` compares blended dynamic Z-score with ADX and volatility-ratio gates.
- `robust_zscore_mean_reversion` isolates MAD normalization against the fixed rolling Z-score baseline.

Signal strength remains `1.0` for entries in this slice. Strength-based sizing should not be enabled until base signal quality is measured under comparable data settings.

## 16. Data Depth Research

Backtest research defaults target `730` days and `12,000` bars. These are requested targets, not guaranteed exchange availability.

Cache coverage must be interpreted with `HistoryCoverage.notes`, which may include fetch audit details such as backend, endpoint, page count, page limit, cursor direction, duplicate count, and pagination stop reason. Coverage below target is a comparability warning unless the CLI is run with a positive `--min-coverage-pct` threshold.

Asset universes are explicit:

- `CORE_UNIVERSE` remains the live/core universe in `qooi.research.instruments`.
- `RESEARCH_UNIVERSE` expands liquid swap research assets without changing live trading defaults.
- research scripts select the orchestration universe through config `run.universe = "core"|"research"`.

## 17. Current Strategy Recommendation

Latest mechanics-normalized evaluation changed the interpretation of prior cached strategy results.

Current recommendation:

- No strategy is ready for allocation from the current evidence.
- `zscore_mean_reversion` remains the primary research subject because it has enough trades to diagnose, not because it is a deployable candidate.
- `rsi_bounce_reversion` remains a sparse quality baseline for comparison, not a deployable strategy.
- `rsi_macd_trend` remains rejected in current form.
- Recovery modes are stress tests only and must not be ranked as candidate strategy improvements.
- BTC safe-profile runs with low acceptance and dominant min-contract blocks are execution-infeasible under the current intended account/risk profile; they should not be treated as signal-quality evidence.
- Next strategy work is regime and thesis robustness diagnostics, not side-specific selection, symbol-aware performance tuning, or recovery.

Candidate evaluation baseline:

- Use swap execution data unless a strategy has an explicit spot-signal thesis.
- Use `NoRecovery`.
- Use safe-profile sizing and `max_per_strategy_symbol=1` unless explicitly testing basket stacking.
- Treat `EXECUTION_INFEASIBLE`, `RECOVERY_EXPERIMENTAL`, `INTRABAR_AMBIGUITY`, and high stop-loss concentration as evidence gates before any strategy promotion.
- Validate any candidate with rolling or walk-forward tests before increasing basket caps, notional budgets, or enabling recovery.

## 18. Dead-Code Policy

Removed or obsolete APIs should not be reintroduced unless a current caller requires them.

Known removed branches/modules:

- old cache validation vocabulary: `CacheValidation`, `CacheSummary`, `validate_ohlcv()`, `describe_ohlcv()`
- old strategy composition modules: `compose.py`, `flow_pipeline.py`
- old misplaced indicator module: `core/indicators.py`
- old exchange evaluation module: `exchange/eval.py`
- old standalone walk-forward script: `scripts/backtest_walkforward.py`

Active code should use:

- `HistoryRequest`, `HistoryTarget`, `HistoryCoverage`
- `strategies/specs.py`, `features.py`, `indicators.py`
- `core/styles.py` for temporal validation styles
- `scripts/classifier_states.py` for classifier-state research
- `scripts/learned_states.py` for learned behavior-state research

## Validation Commands

```bash
uv run ruff check src/qooi/core src/qooi/strategies src/qooi/exchange scripts tests
uv run ty check src/qooi/core
uv run ty check src/qooi/strategies
uv run ty check src/qooi/exchange/market.py src/qooi/exchange/store.py src/qooi/exchange/trading.py scripts/classifier_states.py scripts/learned_states.py scripts/trade.py
uv run pytest
uv run python scripts/classifier_states.py --config configs/research/research-evaluation-dynamic-transitions.toml
```

## Glossary

| Term | Definition |
|------|------------|
| bar | OHLCV candle granularity such as 1H, 4H, 1D |
| swap | Perpetual derivative instrument used for execution |
| spot signal | Optional use of spot OHLCV for indicator/signal generation |
| entry signal | Event-like signed signal that may open a new basket |
| position signal | Held directional thesis state |
| exit signal | Strategy-owned exit intent |
| signal strength | Numeric quality/confidence scalar used for sizing and diagnostics |
| signal ID | Stable module/rule identifier |
| basket | Independent vehicle containing initial and recovery positions plus lifecycle state |
| basket cap | Limit preventing too many concurrent baskets |
| thesis continuation | Rule that confirms an active basket's original directional idea remains valid |
| thesis failed | Exit reason when strict continuation mode no longer confirms the basket direction |
| recovery | Grid, hedge, martingale, or reverse logic attached to one basket |
| reverse recovery | Bounded adverse-move recovery that closes a basket and opens the opposite direction when confirmed |
| basket snapshot | Immutable pre-mutation basket state used for executor accounting |
| mark-to-market | Equity valuation including unrealized PnL each bar |
| fill policy | Rules for stop/target/trailing fills and ambiguity handling |
| comparability | Whether two reports share enough metadata to compare metrics |
| behavior state encoder | Research model mapping known-at-close OHLCV windows to discrete market-state labels |
| endogenous state | Data-discovered discrete market state used for diagnostics before strategy promotion |
| state-transition graph | Directed graph of known-at-close state changes with empirical transition probabilities |
| transition information | Mutual information between previous and current known-at-close states |
| transition n-gram | Consecutive known-at-close state path used as a diagnostic grouping unit |
| policy learner | Research model that proposes actions in simulation but must be adapted to qooi signal columns before execution |
| world model | Research model of market dynamics used for simulation or stress testing, not an executor |
