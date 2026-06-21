# Primitive feature benchmark root cause and next plan

## Environment note

Use `uv` for Python execution.

Verified:

```bash
uv run python - <<'PY'
import sys, polars as pl
print(sys.executable)
print(pl.__version__)
PY
```

Output:

```text
python C:\Users\nostalgia\proj\dev\qooi\.venv\Scripts\python.exe
polars 1.40.1
```

## What happened

The broad primitive bar feature pass added seven columns:

```text
return_12bar
return_48bar
momentum_accel_4_24
realized_vol_ratio_24_168
volume_participation_24_168
range_position_720
range_compression_48_720
```

They reached the model metadata, so the benchmark was real:

```text
continuous_count: 28
missing_required: []
```

But the selection-efficiency surface worsened versus the normalized-bounded baseline:

```text
best hpo_score: 60.109815 -> 54.187079 (-9.85%)
best valid_tail_lift: 56.816060 -> 51.614937 (-9.15%)
best utility mean: 2.746921 -> 2.331626 (-15.12%)
best utility p90: 9.367261 -> 8.225128 (-12.19%)
feature_count: 26 -> 33 (+26.92%)
```

## Important nuance

The result did not worsen everywhere.

At proper comparison grain, average by score bucket:

```text
top_1pct:  hpo -1.95, lift -1.64, utility -0.27
top_2pct:  hpo +0.10, lift +0.15, utility -0.08
top_5pct:  hpo -0.12, lift -0.21, utility +0.06
top_10pct: hpo +0.14, lift +0.04, utility +0.04
```

So the broad feature pass mainly damaged the most concentrated bucket (`top_1pct`). It mildly helped broader buckets. This matters because scanner promotion relies on high-concentration opportunity selection, not broad weak separation.

## Why it behaved worse

### 1. The new features dominated the tree too much

Aggregate primitive-run LightGBM importance:

```text
return_48bar                rank 4   share 7.34%
range_compression_48_720    rank 5   share 6.40%
range_position_720          rank 10  share 3.79%
volume_participation_24_168 rank 12  share 2.85%
realized_vol_ratio_24_168   rank 13  share 2.51%
return_12bar                rank 14  share 2.34%
momentum_accel_4_24         rank 20  share 0.59%
```

All new features together consumed about:

```text
25.8% of total split gain
```

That is too much surface area for an unproven feature pack. The model did not merely ignore the new columns; it reorganized around them.

### 2. Long-window range features likely became lifecycle/history-depth proxies

The 720-bar features have a 30d warmup:

```text
range_position_720
range_compression_48_720
```

For old symbols this is fine. For young/recently listed altcoins, nullness and partial history become informative but ambiguous.

For 30% altcoin tails, lifecycle is real signal, but this implementation encoded it indirectly through missing long-window geometry rather than explicitly as a past-only lifecycle primitive.

That is a design implementation problem:

```text
wanted: current setup compression/location
got: setup mixed with history-depth/listing-age missingness
```

### 3. Feature grain did not match event-lift objective grain

`tail_event_lift` asks:

```text
which current states concentrate future >=30% events?
```

The broad pass mixed three grains at once:

```text
12h/48h velocity
24h/7d vol/volume regime
48h/30d range geometry
```

That may help broad ranking (`top_5pct`, `top_10pct`) but hurt top bucket concentration because LightGBM can split on several correlated, high-cardinality regime proxies before it isolates the very sharp event-lift pockets.

### 4. Some features duplicate existing state rather than complement it

Existing inputs already included:

```text
return_24bar
return_4bar
return_24bar_vol_scaled
close_to_range_high_ratio
categorical decision_transition
```

The new broad pass added:

```text
return_48bar
return_12bar
range_position_720
range_compression_48_720
```

These are theoretically valid, but not necessarily marginally useful. The model already had recent momentum and local range location. The added windows may have shifted split budget toward slower context and away from the sharper existing h24 signal.

### 5. `momentum_accel_4_24` was not the issue

Its aggregate importance was low:

```text
rank 20, share 0.59%
```

So acceleration-only is the least risky next probe. The damage mostly came from the long-window/context pack, not the acceleration idea.

## Is the theory grammar wrong?

No, not fundamentally.

The grammar says features should be:

```text
known-at-close
scale-normalized
multi-scale
primitive market states
```

The failed columns satisfy much of that. The mistake was treating a valid grammar as permission to add a full family pack at once.

The theory grammar was too broad for one benchmark pass; the implementation was too coarse.

Better classification:

```text
Theory principle: mostly sound
Implementation grain: wrong
Benchmark acceptance: correctly rejected it
```

## Is the design implementation astray?

Yes, in two specific ways.

### Implementation mistake A: broad bundle instead of isolated primitive family

We changed seven columns at once. That made the benchmark answer:

```text
does this whole pack help?
```

not:

```text
which primitive family helps?
```

Ponytail should have tested one family at a time.

### Implementation mistake B: lifecycle leaked in through null geometry

Long-window 720 features are not label leakage, but they may encode symbol age/history-depth implicitly.

If lifecycle matters, it should be explicit and past-only:

```text
history_depth_days_at_decision
```

not hidden inside:

```text
range_position_720 is null
```

## Next improvement proposal

Do not re-add the broad primitive pack.

Keep current proven baseline:

```text
event-lift objective
bounded Optuna h24/h48
normalized bar/source aliases
```

Then test one tiny family.

### Probe A: acceleration-only

Add only:

```text
momentum_accel_4_24
```

Reason:

```text
- low importance in failed broad run, so unlikely to hijack the model
- directly tests convexity beyond existing return_4bar/return_24bar
- no long warmup
- no lifecycle null proxy
```

Acceptance:

```text
keep only if top_1pct hpo/lift/utility does not worsen and at least one improves
```

### Probe B: source interaction only

If acceleration-only fails or is neutral, test one source interaction:

```text
price_oi_pressure_24 = return_24bar_vol_scaled * oi_delta_pct
```

Reason:

```text
- aligns forced-positioning theory with existing normalized columns
- no new long window
- directly combines price move + positioning expansion
```

Do not add funding stress yet; funding cadence/staleness is less clean.

### Probe C: lifecycle explicit, later

Only after A/B:

```text
history_depth_days_at_decision
```

Reason:

```text
- makes lifecycle explicit instead of hidden in 720-window nullness
- no future target leakage if derived from available past bar depth
```

Do not add symbol prior tail rate yet.

## Verification plan

For each probe:

```bash
uv run python -m ruff format src/qooi/scanner/state.py src/qooi/scanner/tailrun/core.py
uv run python -m ruff check src tests scripts/scanner_backfill.py scripts/scanner_potential.py
uv run python -m ty check
uv run python -m pytest tests/test_state.py -q
uv run python -u scripts/scanner_potential.py --config configs/potential-advanced-tailtree.toml
```

Compare against:

```text
data/output/potential/benchmarks/normalized-bounded-tailtree-selection-efficiency.csv
```

Use `tailtree-selection-efficiency.csv` only; no new artifact family.

## Ponytail decision

Current broad primitive pass stays reverted.

Next code should be only one feature:

```text
momentum_accel_4_24
```

If it fails, revert it immediately and record the result.
