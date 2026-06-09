# Dynamic / AI Module Graph

```text
scripts/learned_states.py
  main()
    -> train_phase()
    -> predict_phase()
    -> evaluate_phase()

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
  summarize_hidden()
  project_codebook()
  summarize_state_stability()
  filter_short_state_runs()

qooi.dynamic.training
  TrainingConfig

qooi.dynamic.vq_rssm
  VqRssmSpec
  train()
  train_sequences()
  save_checkpoint()
  InferenceDiagnostics

qooi.dynamic.state
  PreparedStateDiscovery
```

Isolation graph:

```text
prepared research/cache frames
  -> scripts/learned_states.py
  -> qooi.dynamic contracts/states/vq_rssm
  -> learned labels/checkpoints/diagnostics
  -> research reports only
```

Forbidden graph edges:

```text
qooi.dynamic -X-> qooi.scanner decisions
qooi.dynamic -X-> qooi.core executor/basket/recovery
qooi.dynamic -X-> qooi.exchange provider fetch
qooi.dynamic -X-> qooi.strategies signal ownership
```
