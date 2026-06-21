# Tailtree paired candidate replay implementation result

## Goal

Transform the workflow redesign into concrete ponytail code while preserving the architecture boundary:

```text
direction = label/objective slice
threshold = label/objective slice
utility = replay/HPO/selection quality term
```

No feature columns were added:

```text
candidate_direction_code ❌
return_threshold_pct ❌
```

## Code change

Touched:

```text
src/qooi/scanner/tailrun/core.py
configs/potential-paired-replay-test-tailtree.toml
```

Did not modify `configs/potential-advanced-tailtree.toml` for this pass.

Added flat workflow pieces:

```text
_score_bucket_candidate_frame()
_paired_candidate_replay_frame()
_candidate_replay_metrics()
```

The run loop now produces:

```text
model -> scored observations -> evidence summary -> paired candidate replay -> selection metrics
```

Training remains unchanged:

```text
binary up/down threshold-event concentration
```

## Test config

New reversible config:

```text
configs/potential-paired-replay-test-tailtree.toml
```

Purpose:

```text
safe test-oriented run, separate output/model dir, no advanced config mutation
```

Differences from advanced profile:

```text
output = data/output/potential/paired-replay-test/report.md
model_dir = data/output/potential/paired-replay-test/models
max_symbols = 80
max_trials = 1
max_folds = 1
```

## Verification

Before scan:

```text
ruff: pass
ty: pass
pytest tests/test_scanner_workflow_migration.py tests/test_state.py -q: 14 passed
```

Scan command:

```bash
uv run --no-sync python -u scripts/scanner_potential.py --config configs/potential-paired-replay-test-tailtree.toml
```

Runtime:

```text
PAIRED_REPLAY_TEST_SECONDS=210
```

## Artifacts

Primary run artifacts:

```text
data/output/potential/paired-replay-test/report.md
data/output/potential/paired-replay-test/tailtree-selection-efficiency.csv
data/output/potential/paired-replay-test/profile/frames.csv
```

Benchmark snapshots:

```text
data/output/potential/benchmarks/paired-replay-test-report.md
data/output/potential/benchmarks/paired-replay-test-tailtree-selection-efficiency.csv
data/output/potential/benchmarks/paired-replay-test-profile-frames.csv
```

## Frame output

Profile frames confirm the new flat workflow surface exists:

```text
scores_paired-replay-diagnostic-test-t0000.h24.up       30,393 rows
scores_paired-replay-diagnostic-test-t0000.h24.down     30,393 rows
scores_paired-replay-diagnostic-test-t0000.h48.up       30,393 rows
scores_paired-replay-diagnostic-test-t0000.h48.down     30,393 rows
candidate_replay_paired-replay-diagnostic-test-t0000   407,624 rows
tailtree_selection_efficiency                               16 rows / 44 cols
```

So the workflow now exposes the missing bridge:

```text
scored observations -> paired candidate replay
```

## Selection-efficiency diagnostics

New columns:

```text
base_hpo_score
objective_hpo_score
candidate_pair_count
paired_opposite_rate
paired_gray_zone_rate
paired_false_direction_rate
paired_false_direction_cost_mean
paired_directional_margin_mean
```

Metric comparability preserved:

```text
max(abs(hpo_score - base_hpo_score)) = 0.0
```

So this pass does not silently redefine HPO behavior.

## Best rows

Best by `hpo_score` / `base_hpo_score`:

```text
h24 down top_1pct
hpo_score=57.231896
base_hpo_score=57.231896
objective_hpo_score=54.172250
valid_tail_lift=54.120486
utility=1.086564
candidate_pair_count=3565
paired_opposite_rate=0.657784
paired_gray_zone_rate=0.103787
paired_false_direction_rate=0.413745
paired_false_direction_cost_mean=2.540712
paired_directional_margin_mean=-0.121536
```

Interesting asymmetry:

```text
h24 up top_1pct
hpo_score=36.892772
objective_hpo_score=35.522991
valid_tail_lift=30.913876
utility=2.713930
paired_gray_zone_rate=0.103816
paired_false_direction_rate=0.035073
paired_false_direction_cost_mean=0.850701
paired_directional_margin_mean=0.449590
```

Interpretation:

```text
down buckets have stronger lift but much worse false-direction exposure;
up buckets have lower lift but cleaner direction diagnostics.
```

This is exactly the kind of signal the aggregate score-bucket penalty could not show.

## Report comparison caveat

This test config is not one-to-one comparable with the advanced baseline because it uses:

```text
80 symbols, 1 trial, 1 fold
```

Baseline conflict-abstain snapshot used a larger advanced run. Still, the report surface is useful as smoke feasibility.

Test report:

```text
promoted=3
conflict_watches=2
promoted_conflicts=0
```

Baseline snapshot:

```text
promoted=3
conflict_watches=8
promoted_conflicts=0
```

Do not overclaim the conflict-watch reduction because the config differs. The reliable claim is:

```text
candidate-level paired diagnostics are produced, HPO remains comparable, and promotion stays conflict-free.
```

## Feasibility verdict

Feasible to keep as diagnostics and workflow restructuring.

Why:

```text
- no training target change
- no direction/threshold feature leakage
- no advanced config mutation
- hpo_score remains comparable to base_hpo_score
- candidate-level paired replay avoids the previous aggregate-evidence mistake
- diagnostics expose useful directional asymmetry
```

Not yet proven as model-quality improvement because:

```text
objective_hpo_score is diagnostic only
hpo_score still equals base_hpo_score
config is smaller than the advanced baseline
```

## Next ponytail step

Do not switch HPO optimization yet.

Next pass should run an advanced-equivalent config or a second test config with the same trial/fold/symbol shape as baseline but separate output dir:

```text
configs/potential-paired-replay-advanced-test-tailtree.toml
```

Then compare:

```text
base_hpo_score
objective_hpo_score
valid_tail_lift
selected_profit_proxy_mean
paired_gray_zone_rate
paired_false_direction_rate
conflict_watches
```

Only after paired diagnostics are stable should we allow:

```text
hpo_score = objective_hpo_score
```

and benchmark that switch separately.
