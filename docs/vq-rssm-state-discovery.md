# VQ-RSSM State Discovery

Stage 2 learns discrete `behavior_state_id` labels from causal OHLCV shape windows. The labels are research diagnostics, not strategy identities.

## Data Flow

1. `LearnedStateConfig.prepare()` computes known-at-close relative OHLCV fields from current bar values and prior close/volume.
2. `LearnedStateConfig.prepare_many()` applies the same flow independently per asset, then merges windows into one shared-codebook dataset.
3. Optional causal volatility scaling divides relative OHLCV fields by a per-symbol EWM volatility estimate computed from current and past rows only.
4. `WindowConfig.assign_split()` applies chronological `train`, `valid`, and `test` labels per asset before windows are merged.
5. `WindowConfig.windows()` returns `(PreparedWindows, WindowProvenance)` so numeric model data and row/timestamp/symbol alignment are separated at construction time.
4. `PreparedWindows.to_dataset()` strips provenance and produces `WindowDataset` for `qooi.ai.vq_rssm`.
7. `vq_rssm.train()` updates model parameters only from train windows.
8. `vq_rssm.predict_codes()` emits `CodeSequence` over all windows.
9. `WindowProvenance.states_from_codes()` reconstructs `StateSequence` with `symbol`, `behavior_state_id`, `code_distance`, and optional `volatility_scale`.
10. `StateSequence.research_frame()` adapts labels into the existing `ResearchFrame` pipe with `state_source="vq_rssm"`.

## Features

The first implementation uses relative shape features:

```text
open_rel = open_t / close_{t-1} - 1
high_rel = high_t / close_{t-1} - 1
low_rel = low_t / close_{t-1} - 1
close_rel = close_t / close_{t-1} - 1
volume_log_rel = log((volume_t + eps) / (volume_{t-1} + eps))
```

Forward returns are never encoder inputs. They are used only later as evaluation labels.

When enabled, volatility scaling estimates a per-symbol EWM standard deviation from `close_rel`, clamps it to configured floor/cap bounds, and divides configured price-relative feature columns by that scale. `volume_log_rel` remains unscaled by default because volume has a separate relative scale. The scale is exported with state provenance for diagnostics and is not added as an encoder feature by default.

## Model Boundary

`qooi.ai.contracts` is import-safe without PyTorch. `qooi.ai.vq_rssm` imports PyTorch directly and belongs to the optional ML dependency group:

```bash
uv sync --group ml
```

The model is a fixed VQ-RSSM: linear input embedding, GRU deterministic state, posterior and prior Gaussian projections, vector quantization, straight-through codebook updates, decoder reconstruction, and GRU transition on the quantized latent. Hidden state starts at zero for every training/prediction window, so no memory crosses assets or unrelated time segments.

## Current Limitations

- Learned states are diagnostic labels until promotion gates and execution-aware backtests justify a strategy hypothesis.
- No smoothing or post-hoc state relabeling is applied in the first implementation.
- The shared codebook is intentionally asset-invariant; symbol IDs and asset embeddings are not encoder inputs.
- Robustness across symbols, sequence lengths, volatility-scaling settings, and codebook sizes remains empirical work.
