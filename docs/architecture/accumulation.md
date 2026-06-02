# Accumulation Detection Architecture

Date: 2026-06-02

## Purpose

The accumulation scanner is an offline-first research path that produces explainable hourly feature, score, alert, coverage, candidate-readout, and event-replay artifacts. It is an artifact-producing decision-support workflow, not a strategy, executor path, trading signal, or live-trading authorization.

User-facing outcome:

```text
Rank candidate symbols, explain why each symbol is worth attention, show what evidence is missing, and suggest the next fetch actions that would improve confidence.
```

Primary research question:

```text
Which symbols show accumulation-like conditions, what components fired, what evidence is missing, and what happened after similar scored alert events?
```

## User Workflow

The scanner is script-orchestrated by `scripts/accumulation_scan.py`. The common workflow is a two-pass loop:

1. Broad cheap scan: `discover -> collect-market -> collect-onchain if enabled -> collect-context -> score -> summarize`.
2. Review current readouts: `candidate-summary.csv`, `candidate-detail.csv`, `candidate-rationale.md`, and `next-fetch-actions.csv`.
3. Deepen selected symbols manually from `next-fetch-actions.csv` using `--symbols` and the relevant phase.
4. Rescore and summarize the selected symbols.
5. Run `scripts/accumulation_backtest.py` to replay scored alert events when alert rows exist.

Current behavior notes:

- `--phase all` runs available phases in script order.
- `collect-onchain` writes skipped manifest rows when `[onchain].enabled=false`; full enabled collection is still pending.
- `collect-context` currently orchestrates Polymarket context and local CSV message normalization/classification when `[sources.messages].enabled=true`.
- Public market collection is dependency-light and does not require secrets.
- Secret-backed or provider-plan-limited sources must surface coverage or manifest diagnostics instead of crashing unrelated symbols.
- `--once` is accepted for workflow compatibility. The current scanner does not implement a scheduler loop, so `--once` is not a repeated-run control.

Manual symbol selection remains supported through `--symbols` for focused research or reproduction. The automatic path reduces the user's burden from choosing coins manually to reviewing a ranked candidate list and deciding whether deeper data collection is warranted.

## Command Reference

Common commands:

```bash
uv run python scripts/accumulation_scan.py --config configs/research/accumulation-mvp.toml --phase all --top-n 25 --book-mode snapshot --fetch-concurrency 3 --once
uv run python scripts/accumulation_scan.py --config configs/research/accumulation-mvp.toml --phase summarize --top-n 25
uv run python scripts/accumulation_scan.py --config configs/research/accumulation-mvp.toml --phase collect-market --symbols BILL-USDT-SWAP,AI-USDT-SWAP --refresh-trades --refresh-context
uv run python scripts/accumulation_scan.py --config configs/research/accumulation-mvp.toml --phase collect-onchain --symbols BILL-USDT-SWAP,AI-USDT-SWAP
uv run python scripts/accumulation_scan.py --config configs/research/accumulation-mvp.toml --phase collect-context --symbols BILL-USDT-SWAP,AI-USDT-SWAP
uv run python scripts/accumulation_scan.py --config configs/research/accumulation-mvp.toml --phase score --symbols BILL-USDT-SWAP,AI-USDT-SWAP
uv run python scripts/accumulation_backtest.py --config configs/research/accumulation-mvp.toml
```

Current scan phases:

| Phase | Purpose |
|---|---|
| `discover` | Build and rank feasible candidate symbols from public metadata, liquidity signals, and cache coverage. |
| `collect-market` | Fetch candles, recent books, recent trades, funding, and open-interest context where supported. |
| `collect-onchain` | Optional on-chain phase. Current script behavior is disabled-only manifest handling when `[onchain].enabled=false`; enabled collection remains pending. |
| `collect-context` | Fetch optional context sources. Current script wiring handles Polymarket and local CSV messages. |
| `score` | Build feature and score rows from collected sources. |
| `summarize` | Produce candidate detail, candidate summary, rationale text, and next-fetch actions. |
| `all` | Run the available pipeline with explicit coverage artifacts. |

Current scan flags:

| Flag | Purpose | Notes |
|---|---|---|
| `--config` | TOML config path. | Defaults to `configs/research/accumulation-mvp.toml`. |
| `--phase` | Select `discover`, `collect-market`, `collect-onchain`, `collect-context`, `score`, `summarize`, or `all`. | Orchestration is owned by `scripts/accumulation_scan.py`. |
| `--once` | Compatibility flag. | Accepted but not a scheduler-loop control in current code. |
| `--symbols` | Comma-separated focused symbol list. | Example: `BTC-USDT-SWAP,ETH-USDT-SWAP`. |
| `--top-n` | Candidate/readout row limit. | Used by discovery and summarize. |
| `--min-volume-usd` | Override discovery minimum volume. | Applies to discovery. |
| `--min-coverage-pct` | Override discovery minimum history coverage. | Applies to symbol selection thresholds. |
| `--fetch-concurrency` | Override public source concurrency. | Falls back to `[sources].fetch_concurrency`. |
| `--book-mode` | Select `snapshot`, `sample`, or `off`. | Falls back to `[market].book_mode`. |
| `--refresh-discovery` | Refresh discovery before eligible collection phases. | Used by collection path when applicable. |
| `--refresh-bars` | Refresh cached bars in market collection. | Public market source. |
| `--refresh-trades` | Refresh recent trades in market collection. | Public market source. |
| `--refresh-context` | Refresh context-like public market data in market collection. | Used for funding/open-interest style context. |
| `--summary-latest-only` / `--no-summary-latest-only` | Accepted summary-latest option. | Defaults to latest-only behavior. |

## Artifacts

The artifact path/schema graph is centralized in `src/qooi/accumulation/artifacts.py`. CSV read/write behavior is centralized in `src/qooi/accumulation/csv_io.py`.

Current artifact contract:

| Artifact | Current Path | Purpose |
|---|---|---|
| `candidate_discovery` | `candidate-discovery.csv` | Ranked discovered candidate universe with explicit eligibility and exclusion diagnostics. |
| `source_manifest` | `source-manifest.csv` | Source status, backend, endpoint, rows, warnings, and stop reasons. |
| `source_bars` | `sources/bars.csv` | Hourly source bars used for feature generation and event replay. |
| `source_books` | `sources/books.csv` | Order-book snapshots or samples. |
| `source_trades` | `sources/trades.csv` | Recent trade rows and notional metadata when available. |
| `source_funding` | `sources/funding.csv` | Funding-rate source rows. |
| `source_open_interest` | `sources/open-interest.csv` | Open-interest source rows. |
| `source_onchain_flows` | `sources/onchain-flows.csv` | On-chain exchange-flow rows when enabled collection is implemented and configured. |
| `source_messages` | `sources/messages-normalized.csv` | Normalized local message rows collected from `[sources.messages].path`. |
| `message_classifications` | `sources/message-classifications.csv` | Deterministic message classifications for normalized local messages. |
| `source_polymarket_markets` | `sources/polymarket-markets.csv` | Polymarket context markets. |
| `source_polymarket_events` | `sources/polymarket-events.csv` | Polymarket event/context rows when available. |
| `features` | `accumulation-features.csv` | Hourly feature rows by symbol. |
| `scores` | `accumulation-scores.csv` | Scores, component contributions, explanations, states, suggestions, and warnings. |
| `alerts` | `accumulation-alerts.csv` | Alert-level rows selected from scores. |
| `candidate_detail` | `candidate-detail.csv` | One-row-per-symbol latest score plus latest feature snapshot. |
| `candidate_summary` | `candidate-summary.csv` | Compact top-N ranked decision surface. |
| `next_fetch_actions` | `next-fetch-actions.csv` | Machine-readable targeted follow-up queue. |
| `backtest_events` | `accumulation-backtest-events.csv` | Forward outcomes after historical scored alert events. |
| `backtest_summary` | `accumulation-backtest-summary.csv` | Aggregated alert replay results. |
| `data_coverage` | `accumulation-data-coverage.csv` | Source coverage and missing-data diagnostics. |

`candidate-rationale.md` is written by `run_summarize()` as a text report. `scan-feedback.md` exists in some output folders but is not currently written by `run_summarize()`, so it is not part of the current generated artifact contract.

## Package And Boundary Graph

Implemented surfaces:

```text
scripts/accumulation_scan.py       # CLI phase orchestration owner
scripts/accumulation_backtest.py   # event replay artifact script

src/qooi/accumulation/config.py    # strict TOML config and ETHERSCAN_API_KEY env loading
src/qooi/accumulation/schema.py    # canonical artifact schemas
src/qooi/accumulation/artifacts.py # artifact name/path/schema catalog
src/qooi/accumulation/csv_io.py    # artifact IO APIs
src/qooi/accumulation/discovery.py # public candidate discovery primitives
src/qooi/accumulation/features.py  # pure feature functions
src/qooi/accumulation/scoring.py   # rule-based score contributions
src/qooi/accumulation/summary.py   # candidate readouts and next-fetch policy
src/qooi/accumulation/onchain.py   # Etherscan V2 primitives and exchange-flow helpers
src/qooi/accumulation/database.py  # optional artifact persistence support

src/qooi/sources/coverage.py       # source manifest and coverage scoring helpers
src/qooi/sources/http.py           # HTTP helper surface
src/qooi/sources/okx.py            # public OKX source collectors
src/qooi/sources/polymarket.py     # public Polymarket source collectors
src/qooi/sources/messages.py       # local message normalization/classification helpers

src/qooi/core/event_backtest.py    # generic scored-event outcome extraction
```

Boundary rules:

- Package modules expose pure source, feature, scoring, summary, artifact, on-chain, config, persistence, or event primitives.
- Script modules own `run_*` workflow functions, CLI parsing, phase sequencing, artifact write timing, and user-facing print output.
- Do not add `qooi.accumulation.scan`, `source_artifacts.py`, `market_sources.py`, or package-level workflow `run_*` APIs.
- Do not add a parallel artifact graph. Use `artifacts.py` for names/paths/schemas and `csv_io.py` for reads/writes.
- Accumulation alert replay uses `qooi.core.event_backtest`, not `qooi.core.executor.BacktestExecutor`.

## Source Families

Every source family must define its source, historical availability, and missing-data behavior. Missing sources are warnings or coverage rows, not neutral scores.

| Family | Examples | Source | Historical Availability | Missing-Data Behavior |
|---|---|---|---|---|
| Price and structure | Returns, MA200, drawdown, volatility compression, range position, breakout or reclaim context. | Swap candles and history candles. | Historical hourly prices are supported from instrument listing. | Block price-derived scores when bars are missing or coverage is below threshold. |
| Order book | Depth imbalance, spread, depth slope, depth persistence, bid refill after sells. | Public books or downloaded book archives. | REST books are current/recent samples; archives are needed for heavy replay. | Emit book coverage warnings; do not treat missing book support as weak support. |
| Trades and orders | Recent trade buy ratio, large sell absorption, sell-event resilience, large trade clustering, taker-side skew. | Public recent trades, history trades, or historical trade archives. | Recent REST trades are limited; deeper history requires supported endpoints or archives. | Emit trades coverage warnings; absorption/resilience components stay unavailable. |
| Exchange flow and on-chain | Net exchange flow, flow z-score, negative outflow streak, whale ratio, coverage count. | Optional Etherscan V2 token-transfer classification selected by `chainid` and exchange address book. | Provider/account-plan dependent; full enabled script collection remains pending. | Emit on-chain coverage warnings, including unsupported chain-plan diagnostics; exchange address books remain partial labels. |
| Derivatives public data | Funding rate, funding z-score, open-interest trend, liquidation-proxy placeholders. | Public funding history and open interest. | Funding has limited history; open interest is current unless separately sampled. | Emit derivatives coverage warnings and keep unavailable placeholders scoring-neutral with explanation. |
| Message and news | Mention growth, fundamental-vs-emotion ratio, overheated filter. | Local CSV collector first, then future provider-backed collectors if needed. | Local CSV depends on curated input; live providers are not required for the current design. | Emit message coverage warnings; missing message evidence is not silence. |
| Polymarket context | Related prediction-market context. | Public Polymarket Gamma API. | Public-context availability depends on market search results. | Emit context manifest rows and warnings when no configured alias/query exists. |

Feature semantics retained from the MVP:

```text
net_exchange_flow = inflow - outflow
positive = exchange inflow = distribution risk
negative = exchange outflow = accumulation-like
```

Top-10 depth uses raw top-10 fields if present. Otherwise it falls back to current cached top-25 imbalance and should be interpreted as a proxy.

## On-Chain Model

Current config uses one provider literal, `provider="etherscan"`, with Etherscan V2 chain selection by `chainid`. The environment key is `ETHERSCAN_API_KEY` and is loaded through `.env` in `load_accumulation_config()` when available.

Do not document or require `BSCSCAN_API_KEY`. BSC support uses the Etherscan V2 model, not a separate BscScan fallback key.

Unsupported chain/account-plan responses are diagnostics:

- Record sanitized coverage or manifest warnings such as unsupported chain plan, missing key, token config missing, or exchange address book missing.
- Do not include API keys, raw secret values, or hardcoded exchange-wallet labels in errors or artifacts.
- Do not crash unrelated symbols because one chain, token, account plan, or address book entry is unavailable.

## Message Source Model

The current message source design starts with local CSV and deterministic keyword classification. `src/qooi/sources/messages.py` provides helper APIs, and `scripts/accumulation_scan.py --phase collect-context` wires them when `[sources.messages].enabled=true` and `path` is set.

Config fields:

- `[sources.messages].enabled`
- `[sources.messages].path`
- `[sources.messages].default_source`

Accepted local CSV columns:

- Required: `symbol`, `timestamp`, `text`.
- Optional: `source`, `source_id`, `author_id_hash`, `text_hash`, `lang`, `url`, `engagement_count`, `reply_count`, `repost_count`.

Normalized output and classification output:

- `sources/messages-normalized.csv`
- `sources/message-classifications.csv`

Message classes:

- `fundamental`
- `trading_funds`
- `community_emotion`
- `unknown_or_noise`

This is source/type classification, not a generic bullish/bearish sentiment shortcut. Message evidence should modify confidence and risk context, not directly become a trading signal.

## API Reading Guide

Map scanner needs to public OKX APIs before adding collectors. Prefer dependency-light REST for current or recent evidence and downloadable historical-market-data archives for heavy replay. Secrets are not required for the public market, public data, and archive paths listed here.

See `docs/api/okx.md` for SDK method names and historical market-data module IDs.

| Need | OKX API | Historical? | Secret? | Scanner Use |
|---|---|---|---|---|
| Universe | `/api/v5/public/instruments` | Current | No | Discover swap symbols and contract metadata. |
| Liquidity | `/api/v5/market/tickers` | Current | No | Rank candidates before deeper collection. |
| Price structure | `/api/v5/market/candles`, `/api/v5/market/history-candles` | Yes | No | Returns, MA, range, volatility, drawdown, and structure context. |
| Book support | `/api/v5/market/books` | Current or replay only if sampled | No | Depth imbalance, spread, slope, and bid-support snapshots. |
| Recent tape | `/api/v5/market/trades`, `/api/v5/market/history-trades` | Recent or limited endpoint history | No | Buy ratio, large-sell absorption, and resilience context. |
| Funding | `/api/v5/public/funding-rate-history` | Limited history | No | Derivative crowding and funding z-score context. |
| Open interest | `/api/v5/public/open-interest` | Current unless sampled | No | Positioning context and open-interest trend when sampled. |
| Deep history | Historical market data modules | Downloadable archives | No | Optional heavy replay source for trades, candles, books, and funding. |

Archive guidance:

- Use candle history for normal hourly structure before archive ingestion.
- Use trade or book archives only when recent REST sampling cannot answer the research question.
- Record backend, endpoint or archive module, page count, date range, and stop reason in coverage notes.

## Feature And Score Rows

Score rows include total score, alert level, component scores, boolean rule flags, explanation, state labels, suggestion type, missing evidence, and data-quality warning.

Alert thresholds:

```text
red:    score_total > red_threshold
orange: score_total > orange_threshold
none:   otherwise
```

Default scoring rules:

| Rule | Score |
|---|---:|
| Exchange outflow z-score streak | `+25` |
| Whale accumulation high | `+15` |
| Bid-depth support on down day | `+20` |
| Large-sell resilience high | `+15` |
| Message not overheated | `+10` |
| Exchange inflow spike | `-30` |
| Message overheated | `-20` |
| Below MA200 and weak depth | `-25` |

Candidate decision fields:

| Field | Meaning |
|---|---|
| `suggestion_type` | Decision label such as `reject_or_deprioritize`, `prepare_watch`, `trend_active_review`, or `alert`. |
| `confidence_level` | Confidence bucket that is separate from raw score. |
| `structure_state` | Price-structure interpretation. |
| `preparation_state` | Whether setup-like preparation evidence exists. |
| `flow_state` | Exchange-flow interpretation when available. |
| `attention_state` | Message/news/context attention interpretation when available. |
| `activation_state` | Whether alert-like activation evidence exists. |
| `risk_state` | Risk or blocking-state interpretation. |
| `missing_evidence` | Explicit missing source/evidence tokens. |
| `next_fetch_action` | Human-facing rendering of the first recommended follow-up action. |

Summary rows should distinguish confidence from score. A high score with missing trades or on-chain coverage is a high-scoring partial-evidence candidate, not a confirmed accumulation event.

## Candidate Readout Semantics

The current readout prioritizes decision support over raw counters:

- `candidate-summary.csv`: compact top-N ranked decision surface.
- `candidate-detail.csv`: one-row-per-symbol latest score plus latest feature snapshot.
- `candidate-rationale.md`: Markdown explanation from summary rows.
- `next-fetch-actions.csv`: machine-readable targeted follow-up queue.

Readouts include:

- Ranked current candidates.
- Alert counts by level through score and summary fields.
- Top positive components.
- Top negative filters.
- Data-quality warnings.
- Missing evidence that would materially change confidence.
- Next fetch actions for high-value missing sources.

Next-fetch queue rules:

- Contract metadata or trade-notional metadata gaps produce `discover`, priority `1`.
- Missing trades on supported/watchlist candidates produce `collect-market`, priority `1`.
- Missing on-chain evidence on supported/watchlist candidates produces `collect-onchain`, priority `2`, `requires_secret=true`.
- Missing messages or classifications on supported/watchlist candidates produce `collect-context`, priority `3`.
- Funding or open-interest gaps produce `collect-market`, priority `3`.
- Rejected or deprioritized symbols only get metadata-gap actions by default, not expensive secret-backed fetches.

The first action is rendered as `next_fetch_action` in summary/detail outputs.

Example explanation strings:

```text
BTC-USDT-SWAP yellow: bid-depth support +20; trades_missing; onchain_missing; no exchange-flow confirmation
ETH-USDT-SWAP orange: exchange outflow streak +25; resilience +15; funding neutral; book coverage partial
```

## Config Reference

Current relevant config groups:

| Group | Fields |
|---|---|
| `[run]` | `universe`, `out` |
| `[market]` | `ds`, `bar`, `days`, `refresh`, `book_mode`, `book_depth`, `book_samples`, `book_every_seconds`, `book_snapshot_count`, `book_sample_symbols_max`, `book_sample_total_seconds` |
| `[discovery]` | `min_volume_usd`, `max_spread_bps`, `min_history_coverage_pct`, `missing_contract_penalty`, `spread_bps_penalty_scale`, `coverage_bonus_scale` |
| `[sources]` | `fetch_concurrency`, `trade_limit`, `funding_limit`, `open_interest_refresh` |
| `[sources.polymarket]` | `enabled`, `provider`, `search_limit_per_symbol`, `max_markets_per_symbol`, `min_volume_usd`, `include_closed`, `lookback_hours`, `fetch_concurrency`, `aliases` |
| `[sources.messages]` | `enabled`, `path`, `default_source` |
| `[onchain]` | `enabled`, `provider="etherscan"`, `lookback_days`, `poll_minutes`, `exchange_address_book`, `tokens` |
| `[summary]` | `top_n`, `latest_only` |
| `[database]` | `enabled`, `path` |

Environment key:

- `ETHERSCAN_API_KEY` is used for Etherscan V2 when `[onchain].enabled=true` and enabled collection is implemented/configured.
- `BSCSCAN_API_KEY` is not part of the current config model.

## Backtest Boundary

`scripts/accumulation_backtest.py` is an artifact-based event replay script:

- It reads `scores`, `candidate_discovery`, and `source_bars` artifacts.
- It uses `qooi.core.event_backtest.build_backtest_events()` and `summarize_backtest_events()`.
- It writes `accumulation-backtest-events.csv`, `accumulation-backtest-summary.csv`, and `accumulation-data-coverage.csv`.
- It does not import `accumulation_scan`.
- It does not import or instantiate `BacktestExecutor`.
- Missing bars are explicit coverage rows with `price_missing;backtest_skipped`.

This boundary matters because accumulation alert replay is scored-event outcome extraction. `BacktestExecutor` is the strategy/basket lifecycle simulator.

## Boundaries

- No imports from live trading clients, executor, basket lifecycle, or recovery modules in accumulation scanner paths.
- Scanner artifacts are research hypotheses until converted into normal signal columns and validated through execution-aware backtests.
- Features at hour `t` may use only source rows timestamped `<= t`.
- Missing data is a warning or coverage row, not a neutral signal.
- API keys are read only when a gated optional fetch is attempted; public discovery and public market collectors must not hardcode secrets.
- API keys and exchange-wallet labels must not be hardcoded.
- Exchange address books are partial labels, not complete exchange-wallet truth.
- Discovery and scoring must not allocate capital, create baskets, mutate execution state, or authorize live trading.
- Research artifacts do not authorize live trading.

## Implemented And Pending Work

Implemented:

- Universe discovery from OKX instruments, tickers, and cache coverage.
- Public market source manifest and coverage diagnostics.
- Source coverage score for candidate rows.
- Candidate summary, detail, rationale, and next-fetch artifacts.
- Funding and open-interest public-source collection/features.
- Local message normalization/classification collection through `collect-context`.
- Message feature computation and score-phase source bundle wiring.
- Artifact-based accumulation event replay script.

Pending or partial:

- Full enabled `collect-onchain` flow aggregation and script write path.
- Optional script-only `deepen --from-next-fetch` phase.
- Optional `scan-feedback.md` generation if a future slice explicitly wires it.
- Archive-based historical trades and book ingestion when recent REST evidence is insufficient.

## Validation

Accumulation subset validation:

```bash
uv run pytest tests/test_accumulation_coverage.py tests/test_accumulation_features.py tests/test_accumulation_scoring.py tests/test_accumulation_summary.py tests/test_accumulation_sources.py tests/test_accumulation_backtest.py tests/test_accumulation_scan.py
uv run ruff check src/qooi/accumulation src/qooi/sources scripts/accumulation_scan.py scripts/accumulation_backtest.py tests/test_accumulation_*.py
```

Smoke examples:

```bash
uv run python scripts/accumulation_scan.py --config configs/research/accumulation-mvp.toml --phase all --top-n 25 --book-mode snapshot --fetch-concurrency 3 --once
uv run python scripts/accumulation_scan.py --config configs/research/accumulation-mvp.toml --phase summarize --top-n 25
uv run python scripts/accumulation_backtest.py --config configs/research/accumulation-mvp.toml
```
