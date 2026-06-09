# Dynamic / AI Research Architecture

## Purpose

`qooi.dynamic` contains optional learned-state and AI research experiments. This area is isolated because current AI-related work is not the active promotion-ready path and must not block or contaminate deterministic scanner, research, strategy, or execution modules.

## Owned modules

```text
src/qooi/dynamic/contracts.py  # Torch-free sequence/window contracts
src/qooi/dynamic/states.py     # learned-state configs and evaluation helpers
src/qooi/dynamic/state.py      # prepared state-discovery lifecycle container
src/qooi/dynamic/training.py   # training runtime config
src/qooi/dynamic/vq_rssm.py    # optional VQ-RSSM model/train/predict lifecycle
```

## Responsibilities

- Define sequence/window/code contracts for learned-state experiments.
- Prepare learned-state discovery and evaluation outputs from already-prepared frames.
- Keep optional ML dependencies lazy so normal scanner/research imports remain lightweight.
- Produce checkpoints, learned labels, inference diagnostics, and stability/code-quality artifacts for research review.

## Non-responsibilities

- No exchange calls.
- No source collection.
- No cache mutation.
- No scanner decision ownership.
- No strategy signal ownership.
- No basket lifecycle, recovery, executor, sizing, or live-trading authorization.

## Allowed dependencies

- Prepared research frames and known-at-close feature windows.
- `qooi.dynamic` contracts/training/model modules.
- `qooi.core.config`-style configuration primitives where needed.

## Forbidden dependencies

- `qooi.scanner`
- `qooi.exchange`
- `qooi.sources`
- `qooi.strategies`
- `qooi.core.basket`
- `qooi.core.executor`
- `qooi.core.recovery`

## Allowed inputs and outputs

Allowed inputs:

- prepared research frames;
- known-at-close feature windows;
- chronological train/valid/test splits.

Allowed outputs:

- learned research labels;
- checkpoints;
- inference diagnostics;
- stability/code-quality artifacts.

Forbidden output use:

- direct scanner states;
- direct strategy identities;
- direct signal columns;
- sizing/risk/recovery controls;
- live-trading authorization.

## Promotion / integration boundary

A learned state can only leave this sandbox if:

1. It is no-lookahead.
2. Active codes do not collapse.
3. Code/state behavior is stable by seed, time split, and symbol.
4. Candidate rows survive deterministic promotion gates.
5. There is an ex-ante behavior rationale.
6. It is adapted into normal strategy signal columns and tested through core execution.

Concrete dynamic implementation surfaces live in `docs/graph/dynamic.md`.
