# Pipeline Architecture

## Purpose

`qooi.pipeline` provides the scanner with market-data loading, cache IO, coverage planning, and universe discovery primitives. It is not a scanner, strategy, or execution layer.

## Owned modules

```text
src/qooi/pipeline/
├── __init__.py    # now_ms
├── types.py       # FrameHealth, ProductResult
├── io.py          # load/save/merge frame helpers
├── coverage.py    # coverage states/jobs/plans/summaries
├── discovery.py   # universe ranking/selecting
└── load.py        # market load request/policy/result types and load_market
```

## Responsibilities

- read/write/merge cached frames;
- plan bounded coverage work;
- classify coverage state (`complete`, `stale`, `provider_bounded`, `coin_too_new`, `deferred_by_budget`, etc.);
- rank/select scanner universes;
- load bar/source market frames through a caller-provided transport client.

## Non-responsibilities

- no scanner state/outcome/evidence/rank logic;
- no trading/execution decisions;
- no wallet/account concepts;
- no model training;
- no hidden retry policy outside the transport/client boundary.

## Dependency direction

```text
scanner.workflow -> pipeline.load / pipeline.coverage / pipeline.discovery
pipeline.load    -> transport client methods supplied by caller
pipeline         -> no scanner imports
```

`OkxClient` owns exchange connection details. Pipeline request objects describe what to load; scanner config decides why.

## Scanner boundary

Scanner converts `PotentialConfig` into:

```text
MarketLoadRequest
MarketLoadPolicy
```

Pipeline returns loaded frames and health/stats. Scanner then builds:

```text
state rows
outcome rows
evidence rows
rank/review/report artifacts
```

Pipeline does not know tailtree, ladder evidence, candidate promotion, or report markdown.
