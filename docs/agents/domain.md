# Domain docs — single-context

This repo has a single stable domain context for quantitative trading strategies.

## Layout

```text
/
├── docs/context.md        ← canonical domain glossary and layering rules
├── docs/graph/            ← module graphs
├── docs/architecture/     ← architecture notes and scanner designs
└── docs/report/           ← empirical summaries and decisions
```

## Consumer rules

1. Before any skill that reads domain context runs, it reads `docs/context.md` to understand the project's domain language.
2. Before any architectural change, it checks `docs/architecture/` for prior decisions or designs that might constrain the change.
3. `docs/context.md` is updated inline whenever a new domain term is resolved during a session.
4. New architecture documents are created lazily — only when a decision is hard to reverse, surprising without context, or needed to coordinate implementation.
