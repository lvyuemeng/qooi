# Paired direction-threshold utility implementation pass

## Goal

Implement the smallest ponytail version of the revised design:

```text
one shared artifact per horizon/fold/trial
training rows = observation × candidate_direction × return_threshold_pct
prediction emits normal up/down evidence rows from the same artifact
objective includes utility via target = event * log1p(direction_utility)
```

## Scope

Touch only:

```text
src/qooi/scanner/config.py
src/qooi/scanner/tailtree/model.py
src/qooi/scanner/tailtree/evidence.py
src/qooi/scanner/tailrun/core.py
src/qooi/scanner/rank.py
configs/potential-advanced-tailtree.toml
```

Do not add a parallel framework.

## Ponytail compromise

Downstream ranking already expects up/down evidence rows. The shared model will therefore train once per horizon and score twice:

```text
score_up   = shared_model(observation, candidate_direction_code=+1, return_threshold_pct=threshold)
score_down = shared_model(observation, candidate_direction_code=-1, return_threshold_pct=threshold)
```

The same tree object may be inserted under both downstream keys so candidate matching remains unchanged, but only one model file/training call is produced per horizon/run.

## Acceptance

Compare against:

```text
data/output/potential/benchmarks/conflict-abstain-tailtree-selection-efficiency.csv
```

Keep only if:

```text
runtime materially improves or stays acceptable
HPO/lift are flat or better
conflict watches do not increase materially
promoted_conflicts == 0
```

Otherwise revert implementation and keep the benchmark artifact/report.
