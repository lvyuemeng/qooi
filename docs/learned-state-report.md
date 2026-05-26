# Learned State Discovery Report

Date: 2026-05-25

## Abstract

This report isolates the Stage 2 learned-state evaluation from the broader research architecture. The experiment trains a VQ-RSSM encoder on causal OHLCV shape windows and evaluates three questions:

- Can the model learn a non-collapsed discrete `behavior_state_id` vocabulary from known-at-close market windows?
- Do the learned states expose measurable persistence, information content, or candidate return patterns worth further investigation?
- Is the current model runtime efficient enough for repeated research iteration?

The answer is yes for diagnostics, no for promotion, and not yet for fast iteration. The completed run produced 146,435 labeled windows, emitted 45 distinct states, and ended with 44 active training codes from a 128-code codebook. Candidate gates found 7 single-state pattern rows, all sparse and not promotion evidence. Runtime improved after CUDA enablement and model-loop cleanup, but VQ-RSSM remains expensive enough that the next work should prioritize profiling, tensor-backed window construction, and a controlled batch/compile/precision sweep before deeper model changes.

## Experiment

Run command:

```bash
uv run python scripts/learned_states.py --config configs/research/learn-vq.toml
```

Export directory:

```text
data/output/learned-states/vq-rssm
```

Requested outputs:

| Output | Purpose |
|---|---|
| `behavior-state-model.pt` | Trained VQ-RSSM checkpoint. |
| `behavior-state-training-metrics.csv` | Epoch loss, split rows, active-code count, and codebook utilization. |
| `behavior-state-sequence.csv` | Per-window `behavior_state_id` sequence with provenance alignment. |
| `behavior-state-distribution-by-symbol.csv` | State-count and state-share diagnostics by symbol. |
| `behavior-state-information-metrics.csv` | State entropy and transition-information diagnostics by symbol. |
| `behavior-state-scored-patterns.csv` | Shared pattern-quality surface for state, transition, and transition-ngram patterns. |

Research universe:

- 13 swap instruments in the `research` universe.
- 1H learned-state frame.
- Cache target in config: 730 days, 12,000 minimum bars, 12,000 row cap, trim enabled.
- Source policy: `run.ds = "swap"`.

Model and runtime configuration:

| Component | Setting |
|---|---|
| Window length | 64 |
| Window stride | 1 |
| Feature columns | `open_rel`, `high_rel`, `low_rel`, `close_rel`, `volume_log_rel` |
| Hidden dimension | 128 |
| Latent dimension | 16 |
| Codebook size | 128 |
| KL annealing steps | 5,000 |
| Epochs | 40 |
| Current configured batch | 512 |
| Current configured validation batch | 4,096 |
| Current configured prediction batch | 4,096 |
| Device policy | `auto` |

## Architecture Boundary

Stage 2 learned states are research diagnostics, not strategy identities. Forward returns are never encoder inputs; they are used only later as evaluation labels.

The learned-state path keeps the same known-at-close discipline as Stage 1:

1. `LearnedStateConfig.prepare()` computes causal relative OHLCV features from current bar values and prior close/volume.
2. `LearnedStateConfig.prepare_many()` applies preparation independently per asset.
3. `WindowConfig.assign_split()` applies chronological `train`, `valid`, and `test` labels per asset.
4. `WindowConfig.windows()` emits numeric windows plus separate provenance.
5. `vq_rssm.train()` updates model parameters only from train windows.
6. `vq_rssm.predict_codes()` emits codes for all windows.
7. `WindowProvenance.states_from_codes()` reconstructs a `StateSequence` with symbol, timestamp, row, split, code, and distance.
8. `StateSequence.research_frame()` adapts learned labels into the shared research-evaluation pipe.

Configuration ownership is intentionally narrow:

| Config Section | Ownership |
|---|---|
| `[learn.vq]` | VQ-RSSM model shape and objective settings. |
| `[learn.train]` | General training/runtime settings such as batches, device, threads, and logging. |
| `[learn.win]` | Window length, stride, and split fractions. |
| `run.ds` | Data-source policy. |

No model registry, flow dispatcher, callback protocol, or progress abstraction is involved in this workflow. The current model uses explicit `training_loss()` and `final_outputs()` methods rather than string-mode dispatch.

## Artifact Inventory

| Artifact | Rows | Role |
|---|---:|---|
| `behavior-state-data-provenance.csv` | 13 | Cache/source coverage by symbol. |
| `behavior-state-training-metrics.csv` | 40 | Training and validation metrics by epoch. |
| `behavior-state-sequence.csv` | 146,435 | Learned state sequence for all windows. |
| `behavior-state-distribution-by-symbol.csv` | 554 | State distribution by symbol. |
| `behavior-state-information-metrics.csv` | 13 | Entropy and information diagnostics by symbol. |
| `behavior-state-transition-information.csv` | 13 | Transition-information pattern rows. |
| `behavior-state-transition-graph.csv` | 21,475 | Directed empirical learned-state transition edges. |
| `behavior-state-transition-matrix.csv` | 21,475 | Transition matrix export. |
| `behavior-state-transition-paths.csv` | 126 | Projected transition paths. |
| `behavior-state-forward-returns.csv` | 1,560 | State return-quality rows. |
| `behavior-state-transition-forward-returns.csv` | 21,597 | Transition return-quality rows. |
| `behavior-state-forward-quality.csv` | 64,686 | Pattern-quality rows by family. |
| `behavior-state-scored-patterns.csv` | 64,686 | Candidate-gated pattern metric table. |
| `behavior-state-cross-asset-stability.csv` | 14,373 | Cross-symbol stability projection. |
| `behavior-state-hidden-summary.csv` | 45 | Hidden-state means by emitted state. |
| `behavior-state-codebook-reconstructions.csv` | 256 | Codebook reconstruction probes. |
| `behavior-state-morphology.csv` | 2,880 | Average input-window morphology by state and step. |

## Data Coverage

The run consumed 147,267 output bars from 576,363 cached raw rows.

| Coverage Group | Symbols | Output Rows | Coverage |
|---|---:|---:|---:|
| Standard-depth swap instruments | 12 | 144,000 | 100.000% |
| `XAU-USDT-SWAP` | 1 | 3,267 | 27.225% |

Interpretation:

- Twelve instruments met the 12,000-row research target and were trimmed.
- `XAU-USDT-SWAP` has materially lower depth and should not be treated as equally comparable.
- No provenance row reports cache refresh during this artifact generation; outputs came from existing cache.

## Training Result

The model completed 40 epochs.

| Metric | Value |
|---|---:|
| Train windows | 102,255 |
| Validation windows | 29,441 |
| Test windows | 14,739 |
| Final train loss | 2.350922 |
| Final validation loss | 1.513869 |
| Best validation epoch | 39 |
| Best validation loss | 1.511402 |
| Final active training codes | 44 / 128 |
| Final codebook utilization | 34.375% |

Interpretation:

- The model did not collapse to a single code or tiny state set.
- Validation loss was still close to its best value at epoch 40, so the run does not show obvious late-epoch divergence.
- Only about one third of the codebook is active, suggesting either excess code capacity for this feature set or underuse caused by optimization/objective dynamics.

## State Distribution

`behavior-state-sequence.csv` contains 146,435 labeled windows and 45 unique emitted states.

Top global states:

| State | Rows |
|---:|---:|
| 63 | 11,331 |
| 19 | 10,873 |
| 105 | 8,881 |
| 126 | 8,416 |
| 67 | 7,697 |
| 109 | 6,920 |
| 0 | 6,391 |
| 1 | 6,319 |
| 51 | 6,027 |
| 22 | 5,532 |

Per-symbol concentration:

| Group | Active States | Top-State Share |
|---|---:|---:|
| Standard-depth symbols | 41-44 | 7.22%-9.56% |
| `XAU-USDT-SWAP` | 41 | 12.52% |

Interpretation:

- State usage is broad enough for diagnostic analysis.
- The shared codebook is not dominated by one global state.
- XAU is more concentrated, likely because it has much less data.

## Learned-State Information

Normalized mutual information measures how much current state is explained by previous state, divided by state entropy.

| Metric | Min | Mean | Max |
|---|---:|---:|---:|
| Active states per symbol | 41 | 41.92 | 44 |
| Normalized mutual information | 0.0316 | 0.0450 | 0.1483 |
| Normalized transition information | 0.0316 | 0.0450 | 0.1482 |

Interpretation:

- Learned-state transition dependence is measurable but much weaker than the handcrafted Stage 1 higher-timeframe labels.
- XAU has the maximum normalized mutual information, but low coverage makes that result fragile.
- The learned states are less persistent than handcrafted higher-timeframe classifier labels, which may be useful for event-like diagnostics but weakens direct transition-persistence narratives.

## Pattern Quality

The shared pattern-quality table scored 64,686 rows and found 7 candidate-gated rows.

| Pattern Family | Rows | Candidate-Gated Rows |
|---|---:|---:|
| `state` | 1,560 | 7 |
| `transition` | 21,597 | 0 |
| `transition_ngram` | 41,529 | 0 |

Candidate distribution:

| Symbol | Candidates | Row Range | Horizons |
|---|---:|---:|---|
| `ARB-USDT-SWAP` | 4 | 33-50 | 3, 6 |
| `ADA-USDT-SWAP` | 2 | 44 | 3, 6 |
| `AVAX-USDT-SWAP` | 1 | 31 | 3 |

Candidate rows:

| Symbol | State | Horizon | Rows | Positive Rate % | Omega | PWPR | Mean Side Return % |
|---|---:|---:|---:|---:|---:|---:|---:|
| `ADA-USDT-SWAP` | 101 | 3 | 44 | 45.45 | 1.710 | 2.052 | 0.195 |
| `ADA-USDT-SWAP` | 101 | 6 | 44 | 38.64 | 1.512 | 2.401 | 0.244 |
| `ARB-USDT-SWAP` | 1 | 3 | 33 | 51.52 | 2.226 | 2.095 | 0.432 |
| `ARB-USDT-SWAP` | 101 | 3 | 39 | 66.67 | 4.415 | 2.207 | 0.881 |
| `ARB-USDT-SWAP` | 38 | 6 | 50 | 50.00 | 2.044 | 2.044 | 0.604 |
| `ARB-USDT-SWAP` | 101 | 6 | 39 | 64.10 | 4.325 | 2.422 | 1.125 |
| `AVAX-USDT-SWAP` | 89 | 3 | 31 | 48.39 | 2.025 | 2.160 | 0.225 |

Interpretation:

- Candidate-gated learned-state patterns are rare.
- All candidate-gated rows are single-state patterns; transition and transition-ngram learned-state patterns produced no candidate-gated rows.
- Sample sizes are small, between 31 and 50 rows, so these rows are discovery leads only.
- State 101 appears in several candidate rows, but this is not enough to define a strategy rule.

## Runtime Performance

Observed runtime behavior shows that the model is GPU-capable but still expensive.

| Phase / Setting | Observation |
|---|---|
| CPU-only PyTorch build | Initial torch build was `+cpu`; GPU was invisible to torch. |
| CUDA verification | `torch=2.11.0+cu128`, CUDA 12.8, RTX 4060 Laptop GPU visible. |
| CPU epoch timing before CUDA | Roughly 600-790 seconds per epoch in observed logs. |
| CUDA epoch timing before second-pass optimizations | Roughly 360-480 seconds per epoch in observed logs. |
| Current configured batch | 512, up from 64 and 256. |
| Current model loop | Explicit `training_loss()` and `final_outputs()` avoid unneeded full-sequence materialization. |

Performance interpretation:

- CUDA is active and necessary, but the model remains dominated by recurrent per-step work: `windows * seq_len * codes * latent_dim`.
- A plain `nn.GRU` replacement would change semantics because the next hidden state depends on the quantized latent from the current step.
- The completed artifact set does not include per-phase timing CSVs, so the next performance decision needs a short profiling run after the latest code and batch changes.
- Current preparation still materializes overlapping windows as nested Python tuples before torch conversion. This is likely the next CPU/data-structure bottleneck after model-loop cleanup.

## Improvement Hypotheses

The next learned-state work should separate model-quality experiments from runtime experiments. A faster model is only useful if code usage, predictive diagnostics, and stability do not degrade.

Model-spec improvements to test:

| Change | Rationale | Risk / Evaluation |
|---|---|---|
| Activation function in decoder and embeddings | Current decoder uses `ReLU`; `GELU` or `SiLU` may better model smooth market-shape variation. | Compare validation loss, reconstruction morphology, active-code utilization, and candidate drift. |
| Wider input embedding | Current embedding width is fixed at 32; larger embedding width may improve local feature encoding before recurrence. | More compute per step; require validation-loss and code-utilization improvement. |
| Dropout in decoder/posterior path | May reduce overfitting and improve state robustness across symbols. | Too much dropout may destabilize VQ assignments; evaluate code churn and cross-symbol stability. |
| LayerNorm on hidden/posterior inputs | May stabilize recurrent hidden scale and codebook assignment. | Could alter learned morphology; compare state distribution and information metrics. |
| Multi-layer transition cell | A deeper recurrent transition may model richer dynamics than one `GRUCell`. | Higher cost; only test after profiling and batch efficiency are under control. |
| Larger hidden dimension | `hidden=128` may underfit richer morphology. | Cost grows materially; require better validation loss and more stable candidate output. |
| Smaller/larger codebook | Only 44/128 codes were active; `codes=64` may be sufficient, while `codes=256` may help if capacity is limiting. | Smaller codebook may merge useful states; larger codebook may increase sparsity and runtime. |
| Shorter/longer windows | `len=64` may be too long for some regimes or too short for slower structure. | Must compare state persistence, candidate rows, and runtime; do not tune on return metrics alone. |

Preferred first model-spec experiments:

1. `codes=64` with the current architecture to test whether the 128-codebook is oversized.
2. `activation=gelu` or `activation=silu` with the current dimensions.
3. `dropout=0.05` after posterior/decoder hidden layers only, not inside the recurrent transition at first.
4. `LayerNorm` on posterior/prior inputs if code assignment remains unstable or validation loss is noisy.

Data-evaluation improvements needed before promotion:

| Evaluation | Purpose |
|---|---|
| Refresh or exclude XAU | XAU has only 27.225% target coverage and should not drive cross-symbol conclusions. |
| Repeat run with a different seed | Measure state-code stability and candidate drift under initialization noise. |
| Rolling or walk-forward splits | Check whether learned states and candidates survive time-split changes. |
| Cross-symbol holdout | Train on a subset of symbols and evaluate state morphology/candidates on held-out symbols. |
| Horizon sensitivity | Re-score horizons beyond `[1, 3, 6]` only after base diagnostics are stable. |
| Feature ablation | Compare price-only, volume-only, volatility-scaled, and unscaled feature sets. |
| State stability metrics | Track adjusted mutual information or nearest-centroid mapping between runs. |

Efficiency improvements to prioritize:

| Improvement | Expected Benefit | Notes |
|---|---|---|
| Structured timing CSV | Makes runtime comparison artifact-backed. | Record load, prepare, tensor build, epoch, validation, prediction, diagnostics, artifact write. |
| Tensor-backed window construction | Removes nested Python tuple duplication for overlapping windows. | Preserve per-asset boundaries so windows never cross symbols. |
| Full inference tensor reuse | Avoids rebuilding feature tensors per prediction batch. | Gate by estimated CUDA memory. |
| Batch sweep | Reduces optimizer-step overhead. | Test `512`, `1024`, and maybe `1536` only if memory allows. |
| Optional `torch.compile` | May reduce Python recurrent-loop overhead. | Must be explicit opt-in because Windows/CUDA compile warmup can be costly. |
| Optional AMP | May improve throughput. | Must compare validation loss, code utilization, and state/candidate drift against fp32. |

Prediction and code-quality improvements:

| Metric / Check | Purpose |
|---|---|
| Reconstruction error by feature and symbol | Identify whether states encode price shape, volume shock, or symbol-specific noise. |
| Code utilization by epoch | Detect late code collapse or unused capacity earlier than final summary. |
| Code transition churn | Distinguish stable state structure from noisy high-frequency code flipping. |
| State morphology separation | Confirm states map to distinct average window shapes, not arbitrary partitions. |
| Hidden-state cluster separation | Compare hidden summaries across states for redundancy. |
| Candidate stability across seeds | Reject candidates that appear only under one random initialization. |
| Candidate stability across time splits | Reject candidates that are period-specific artifacts. |
| Promotion-gate completeness | Apply symbol-support and time-split support before any strategy hypothesis. |

These checks should be reported before changing strategy rules. Learned states remain diagnostic labels until prediction quality and code stability survive seed, time, and symbol robustness tests.

## Promotion Assessment

No learned-state pattern is promoted.

Operational decision:

- Learned states remain diagnostic labels.
- No strategy rule should be created from this report alone.
- Candidate rows should be treated as hypothesis leads requiring cross-symbol, time-split, and execution-aware confirmation.
- The next engineering step is performance and robustness instrumentation, not threshold relaxation.

## Conclusions

1. Stage 2 successfully generated non-collapsed learned behavior states from causal OHLCV windows.
2. The model used 45 emitted states overall and 44 active training codes at the final epoch.
3. Learned-state transition information is measurable but weak compared with handcrafted Stage 1 higher-timeframe labels.
4. Candidate gates found 7 sparse single-state rows and no transition or transition-ngram candidates.
5. XAU coverage is too low for equal comparison with the rest of the research universe.
6. The result supports continued learned-state research, not strategy conversion.
7. Runtime remains the main blocker for iteration speed.

## Next Work

1. Add structured timing export for learned-state runs so runtime comparisons are artifact-backed rather than log-only.
2. Run a one-epoch profiling command with the current optimized code and `batch = 512`, recording load, prepare, tensor creation, epoch, validation, prediction, diagnostics, and artifact durations.
3. If GPU memory has headroom, run a controlled batch sweep: `batch = 512`, `1024`; `valid_batch = 4096`, `8192`; `pred_batch = 4096`, `8192`.
4. Implement tensor-backed window construction if preparation/tensor conversion remains material; preserve per-asset boundaries so hidden state never crosses assets.
5. Add code-quality artifacts: reconstruction error by feature/symbol, code utilization by epoch, transition churn, and candidate stability tables.
6. Refresh or exclude XAU before comparing learned-state information metrics across symbols.
7. Run seed-repeat evaluation before treating any state or candidate as stable.
8. Run first model-spec ablations in this order: `codes=64`, activation function, light dropout, then LayerNorm.
9. Test optional `torch.compile` behind an explicit `[learn.train]` flag only after the profiling baseline is stable.
10. Test AMP only as an explicit opt-in experiment and compare validation loss, active-code utilization, morphology, and candidate output drift against fp32.
11. Re-run the report after strict promotion support and time-split support are wired for learned-state patterns.
