# Dynamic / AI Graph

`qooi.dynamic` is an isolated learned-state research sandbox. The scanner stage does not expose a dynamic script entry point.

Current package graph:

```text
qooi.dynamic.contracts
  WindowDataset
  AssetFeatureSequence
  SequenceDataset
  CodeSequence

qooi.dynamic.states
  LearnedStateRunConfig
  FeatureColumns
  VolatilityScalingConfig
  LearnedObjectiveConfig
  summarize_hidden
  project_codebook
  summarize_state_stability
  filter_short_state_runs

qooi.dynamic.training
  TrainingConfig

qooi.dynamic.vq_rssm
  VqRssmSpec
  train
  train_sequences
  save_checkpoint
  InferenceDiagnostics

qooi.dynamic.state
  PreparedStateDiscovery
```

Forbidden graph edges:

```text
qooi.dynamic -X-> qooi.scanner promotion/review decisions
qooi.dynamic -X-> qooi.core executor/basket/recovery
qooi.dynamic -X-> qooi.transport provider credentials
qooi.dynamic -X-> live trading or allocation
```

If learned-state work returns later, add a new explicit script and config in the same change. Do not keep stale script names as graph authority.
