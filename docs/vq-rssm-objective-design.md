# VQ-RSSM Objective Design

This note documents the current learned-state VQ-RSSM objective before changing the design. The prerequisite code-bug triage tests pass for VQ gradient direction, shared objective equivalence, sequence hidden isolation, and feature/provenance alignment.

Stage 2 learned states remain diagnostic research labels. A `behavior_state_id` is not a strategy identity, promotion rule, or allocation signal unless later execution-aware backtests and promotion gates support a concrete hypothesis.

## Current Objective

`qooi.ai.vq_rssm` owns the torch model, training, checkpoint, and prediction lifecycle. Research feature preparation and provenance adaptation stay outside `qooi.ai`; `qooi.ai.contracts` remains torch-free.

At each step, the model builds a posterior latent `z_e` from current known-at-close features and recurrent hidden state, selects the nearest codebook vector `z_q`, and decodes same-step features through straight-through quantization:

```python
z_q_st = z_e + (z_q - z_e).detach()
```

This means reconstruction and recurrent-transition gradients flow to the encoder/posterior path through `z_e`, not directly to the codebook. The codebook is updated by the raw VQ codebook loss:

```text
codebook_loss = mean((stop_gradient(z_e) - z_q)^2)
commitment_loss = mean((z_e - stop_gradient(z_q))^2)
total = recon + codebook + commitment_cost * commitment + kl
```

The tests now document this behavior explicitly:

- Codebook loss moves selected code vectors toward fixed encoder outputs.
- Commitment loss moves encoder outputs toward fixed selected code vectors.
- Same-step reconstruction does not update the codebook through the straight-through path.
- Window and sequence losses use the same decomposed objective when hidden state starts at zero.
- KL weighting changes only the KL contribution and total loss.

## Observed Design Pressure

The sequence artifact showed reconstruction loss improving while VQ codebook and commitment losses rose. With the current straight-through estimator, that is mechanically plausible: reconstruction pressure can move `z_e` toward values that improve same-step decoding while the codebook only chases detached encoder outputs through the VQ loss.

This is not currently evidence of a gradient wiring bug. It is evidence that objective balance, codebook update dynamics, or latent scale control need controlled experiments before interpreting learned-state quality.

## Candidate Designs

| Design | What Changes | Why Test It | Main Risk |
|---|---|---|---|
| Current VQ-VAE-style loss with stronger diagnostics or commitment | Keep the current objective, add VQ distance/norm metrics, and test stronger `commitment_cost` values. | Minimal change; directly tests whether encoder-codebook anchoring fixes divergence. | Too much commitment may harm reconstruction or collapse useful state variation. |
| EMA codebook updates | Replace optimizer-driven codebook embedding updates with exponential moving average updates from assigned encoder outputs. | Can stabilize sparse or noisy codebook updates when Adam-updated embeddings lag encoder drift. | Adds a second update rule and extra state; should be gated by codebook movement diagnostics first. |
| Predictive RSSM variant | Decode or predict next-step/future features from recurrent state rather than treating same-step reconstruction as the main objective. | Better aligns with world-model interpretation and may reduce same-step autoencoder shortcut pressure. | Larger semantic change; learned labels must remain known-at-close diagnostics and still pass provenance/no-leakage tests. |

## Decision Gate

Do not rewrite the objective solely because final-epoch validation degraded. The next experiment should first record VQ diagnostics by epoch: `z_e` norm, selected `z_q` norm, selected distance mean/p95/max, codebook norm mean/max, active codes, and decomposed losses.

If VQ-only gradient tests continue to pass but reconstruction pressure increases selected VQ distance, treat the issue as objective-design imbalance. Test stronger commitment and codebook update diagnostics before EMA. Consider a predictive objective only after the same-step diagnostic encoder is understood and the Stage 2 diagnostic contract remains intact.
