# Tailtree ponytail prune + HPO objective wiring plan

## Scope

Apply a second ponytail pass to `src/qooi/scanner/tailrun/` and `src/qooi/scanner/tailtree/model.py` after the first reduction pass.

## Current problems

1. `core.py` still carries small helper/wrapper leftovers around the new flat job/fold path.
2. `model.py` repeats LightGBM feature matrix preparation in prediction methods and keeps single-implementation Protocol types mostly for typing workaround.
3. Optuna still records `result.score` from raw evidence score. Selection metrics already compute `objective_hpo_score`, but HPO does not use it.

## Change plan

### 1. Delete stale wrappers only after grep

- Check live callers for old `run_frame_split`, `load_predict`, `_build_tail_tree_evidence` before deleting.
- If live callers/tests still use them, leave public compatibility for now and only prune internal helpers in this pass.
- Close with anti-forgetfulness grep for removed private helper names.

### 2. Make HPO consume selection objective

- In `run_tailtree_fold`, build `selection_efficiency` once.
- Set fold `score` from `objective_hpo_score.max()` when available; fallback to raw evidence score for empty/old surfaces.
- Keep `hpo_score/base_hpo_score/objective_hpo_score` artifact columns so diagnostics still compare base vs objective.

### 3. Prune simple helpers by putting behavior on real product shapes

- Move fold score choice into `TailtreeFoldRunResult`/near construction only if it reduces loose helper calls.
- Replace `_profile_feedback(...)` with direct `TailtreeProfileFeedback(...)` at the only call site.
- Replace `_tailtree_score(...)` if the logic is now only fallback for the fold result.

### 4. Prune `model.py` duplication without creating managers

- Remove single-implementation Protocol classes if `Any` is shorter and keeps `ty` green.
- Collapse repeated feature-matrix preparation for `predict_leaf` and `predict_score` into one method on `TailTreeModel`; this is behavior of a real model object, not a utility pile.
- Keep `TailTreePayload`; it is the real Pydantic artifact shape.

## Verification

Run:

```bash
grep -R --exclude='*.pyc' --exclude-dir='__pycache__' "_profile_feedback\|def _tailtree_score\|LightGbmBooster\|LightGbmDataset" -n src/qooi/scanner/tailrun src/qooi/scanner/tailtree || true
uv run python -m ruff check src/qooi/scanner/tailrun src/qooi/scanner/tailtree/model.py tests/test_scanner_workflow_migration.py tests/test_state.py
uv run python -m ty check src/qooi/scanner/tailrun src/qooi/scanner/tailtree/model.py tests/test_scanner_workflow_migration.py tests/test_state.py
uv run python -m pytest tests/test_scanner_workflow_migration.py tests/test_state.py -q
uv run python -m scripts.scanner_potential --config configs/potential-paired-replay-test-tailtree.toml
```

Then inspect `tailtree-selection-efficiency.csv`; expect `objective_hpo_score` to differ from `base_hpo_score`, and Optuna fold score to use objective-aware score.
