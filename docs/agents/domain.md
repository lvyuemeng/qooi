# Domain docs — single-context

This repo has a single domain context for quantitative trading strategies on China A-shares using BigQuant.

## Layout

```
/
├── CONTEXT.md           ← domain glossary (all terms)
└── docs/adr/            ← architecture decision records (repo-wide)
```

## Consumer rules

1. Before any skill that reads domain context runs, it reads `CONTEXT.md` at the repo root to understand the project's domain language.
2. Before any architectural change, it checks `docs/adr/` for prior decisions that might constrain the change.
3. `CONTEXT.md` is updated inline whenever a new domain term is resolved during a session.
4. ADRs are created lazily — only when a decision is hard to reverse, surprising without context, and the result of a real trade-off.
