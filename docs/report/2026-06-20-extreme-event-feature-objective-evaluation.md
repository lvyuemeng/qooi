# Extreme-event scanner feature/objective evaluation

## User thesis

```text
threshold should be >= 30% due to high volatility of altcoin.
the goal is to grasp the extreme event or behavior, which is the profit source.
```

This changes the scanner target from “moderate tail opportunity” to “rare explosive path event”. The model should no longer optimize mainly for 8%/15% path tails. It should identify known-at-close conditions that materially raise the probability and/or utility of a future ±30% move.

## Current implementation summary

### Current configured scale

Current local configs already reflect the new hypothesis:

```toml
[potential.bars]
timeframes = ["1H"]
days = 120 / 180

[potential.transition]
horizon = 24
mae_mfe_horizon = 24

[potential.evidence.tailtree]
threshold_pct = 30.0
outcome_horizon = [24]
```

Important: although the observation builder supports `4H` swing and `1D` background contexts, current configs only load `1H`. Therefore many intended multi-scale categorical fields become `market_context_missing` / null.

### Current categorical feature grain

Tailtree currently trains on these categorical fields:

```text
background_regime
swing_core
decision_core
decision_transition
decision_direction
```

Where they come from:

- `decision_*`: current 1H state.
- `swing_*`: as-of 4H context, but currently missing because `4H` is not loaded.
- `background_*`: as-of 1D context, but currently missing because `1D` is not loaded.

Current state construction in `src/qooi/scanner/state.py` uses:

| Feature family | Current logic | Time scale |
|---|---|---|
| ATR | true range rolling mean 14, percentile over 100 | 1H only in current config |
| Swing structure | 5-bar swing high/low, 24-bar HH/HL/LH/LL counts | 1H; 4H/1D only if loaded |
| Range | 48-bar high/low, range width in ATR units | 1H = 2 days |
| Liquidity event | 20-bar prior high/low sweep or breakout acceptance | 1H = 20 hours |
| State age | state/event run-length buckets | per timeframe |

This is useful for a generic state machine, but weak for 30% altcoin extremes because the strongest precursor often lives in multi-day compression, liquidity vacuum, post-listing drift, leverage/crowding, and flow acceleration rather than only a 20–48 hourly-bar state.

### Current continuous feature grain

Tailtree currently trains on:

```text
atr_percentile
range_width_atr
return_1bar
return_4bar
return_24bar
vol_anomaly
close_to_range_high_ratio
funding_rate
oi_delta
taker_buy_sell_ratio
long_short_ratio
funding_age_ms
oi_age_ms
taker_age_ms
lsr_age_ms
```

Source features are aligned known-at-close by symbol-safe backward as-of joins. Stale source values are nulled by family-specific max ages while age columns remain visible.

Current feature gaps for ≥30% events:

1. **No direct multi-horizon volatility scale**
   - ATR percentile exists, but there is no realized volatility at 6h/24h/72h, no Parkinson/Garman-Klass range volatility, no volatility compression/expansion slope.

2. **No distance-to-extreme / breakout geometry beyond 48 bars**
   - `close_to_range_high_ratio` uses a 48-bar range only.
   - 30% events often require context from 7d/14d/30d range, all-time-post-listing high/low, local supply vacuum, or post-consolidation breakout distance.

3. **Returns are too sparse/coarse**
   - Current returns: 1h, 4h, 24h.
   - Missing: 2h, 6h, 12h, 48h, 72h, 7d, acceleration terms, return/volatility normalized momentum, realized skew.

4. **Source features are mostly raw levels, not changes**
   - Funding level, OI delta, taker ratio, long/short ratio.
   - Missing: OI/funding/taker/LSR z-scores, slopes, percent changes, divergence vs price, acceleration, crowding reversal signals.

5. **Current-only books/trades are not historical training features**
   - Books/trades can confirm the current setup but cannot yet train historical extreme behavior without archival snapshots.
   - For a 30% event model, this means microstructure should be a review/current confirmation layer unless we build a historical archive.

6. **No symbol lifecycle / listing-age feature**
   - Extreme tails are highly concentrated in new/illiquid/high-beta symbols.
   - Current model sees symbol indirectly through state/source data, but not listing age, cache depth, age bucket, or historical volatility class.

7. **No market beta / cross-sectional context**
   - A 30% alt move often depends on BTC/ETH regime, sector/market breadth, or cross-sectional meme/alt impulse.
   - Current per-symbol features do not include market-wide alt breadth or BTC/ETH leading returns.

## Current objective and label behavior

### Current label

`label_tail_exceedances()` labels path tails as:

```text
up:   forward_max_return_pct > threshold_pct
down: forward_min_return_pct < -threshold_pct
```

For `threshold_pct = 30.0`, this means:

```text
up event   = future path reaches +30% within horizon
down event = future path reaches -30% within horizon
```

This is the correct primitive for “extreme event” because it uses path max/min, not close-to-close return.

### Current utility

For up tails:

```text
tail_utility_up =
  (forward_max_return_pct - threshold_pct)
  * close_retention_ratio
  * path_efficiency
  * speed_to_max
  - 0.1 * post_max_drawdown_pct
```

For down tails:

```text
tail_utility_down =
  (abs(forward_min_return_pct) - threshold_pct)
  * close_retention_ratio
  * path_efficiency
  * speed_to_min
  - 0.1 * post_min_rebound_pct
```

This is good directionally: it rewards exceedance, retention, efficiency, and speed while penalizing adverse path after the extreme.

But for 30% events, the current utility has issues:

1. **Utility is positive only after a hard 30% threshold**
   - Good for purity.
   - But model gets no graded signal from 20–29% near-misses that may share precursors.

2. **Current LightGBM trains only on tail rows**
   - `tailtree_training_frame()` sends only `tail_observations` to `TailTreeModel.train()`.
   - Denominators are used afterward for tail lift, but the model itself learns severity among known tails, not event probability over all states.
   - For rare 30% events, this is a key gap: the first question should be “which states increase event probability?”; severity/utility is the second question.

3. **Absolute threshold is not volatility-normalized**
   - User is right that altcoins require high absolute threshold.
   - But absolute 30% still mixes two phenomena:
     - ordinary volatility of very new/high-beta coins;
     - genuinely abnormal event behavior relative to a coin’s own volatility.
   - We should keep absolute `>=30%` as the profit-source definition, but add normalized event descriptors for model learning and diagnostics.

4. **Single horizon h24 is too narrow for 30% path events**
   - 30% in 24h is real but sparse.
   - Many extreme alt moves unfold over 48–72h.
   - A scanner trying to catch event behavior should train/evaluate `h24`, `h48`, and `h72`, then report whether a current candidate is early ignition (`h24`) or broader event setup (`h48/h72`).

## Empirical label support from current cache

I computed path-tail support directly from cached `1H` bars across 89 symbols and horizons 6/12/24/48/72.

| Horizon | Threshold | Rows | Up | Up % | Down | Down % | Either | Either % |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 6h | 8% | 647,487 | 27,971 | 4.320 | 21,520 | 3.324 | 48,020 | 7.416 |
| 6h | 15% | 647,487 | 6,956 | 1.074 | 3,596 | 0.555 | 10,332 | 1.596 |
| 6h | 30% | 647,487 | 1,260 | 0.195 | 532 | 0.082 | 1,779 | 0.275 |
| 12h | 30% | 647,487 | 3,475 | 0.537 | 1,290 | 0.199 | 4,726 | 0.730 |
| 24h | 30% | 647,487 | 8,810 | 1.361 | 3,110 | 0.480 | 11,782 | 1.820 |
| 48h | 30% | 647,487 | 20,259 | 3.129 | 7,478 | 1.155 | 27,327 | 4.220 |
| 72h | 30% | 647,487 | 32,154 | 4.966 | 13,450 | 2.077 | 44,682 | 6.901 |
| 24h | 50% | 647,487 | 2,826 | 0.436 | 943 | 0.146 | 3,767 | 0.582 |
| 72h | 50% | 647,487 | 11,780 | 1.819 | 3,539 | 0.547 | 15,291 | 2.362 |

Interpretation:

- `30% / h24` is sparse but trainable at universe level: 11,782 either-side events in cache.
- Up tails are much more common than down tails at 30% (`h24 up 1.36%`, `h24 down 0.48%`). Direction-specific support matters.
- `h6 30%` is very sparse; it should be an “ignition speed” metric, not the only training target.
- `h48/h72 30%` has much better support and is probably the right event-behavior horizon for the scanner.
- `50%` should be tracked as an appendix/severity tier, not the first training target unless universe/depth expands.

## Current model artifact sanity

The latest daily model artifact still reflects the earlier low-threshold run, not the local 30% config. It shows huge tail rates:

```text
up train exceedances:   109,465 / 535,260 ≈ 20.4%
down train exceedances:  86,680 / 535,260 ≈ 16.2%
```

That is consistent with the old 8% threshold, not the new 30% thesis.

Top feature importances in that old run:

```text
up:   decision_transition, long_short_ratio, return_24bar, funding_rate, close_to_range_high_ratio
 down: decision_transition, return_24bar, funding_rate, long_short_ratio, close_to_range_high_ratio, oi_delta
```

This is useful: the model is already using decision transition + market positioning + 24h momentum. But for 30% events, these need richer multi-scale/normalized forms.

## Gap analysis for >=30% extreme-event scanner

### Gap 1 — current config says 30%, but model design is still 8%-style

With 8% tails, event support is abundant; severity ranking among tail rows can work.

With 30% tails, event support is sparse. We need explicit rare-event design:

```text
all observations -> event probability / event lift
selected event rows -> severity / utility
```

Current tailtree trains only on the tail subset, so it does not directly learn the boundary between normal states and extreme-event states.

### Gap 2 — single horizon hides event formation

A 30% alt event has phases:

```text
ignition: 6h/12h
main path: 24h
continuation / full event: 48h/72h
```

Current h24-only config collapses this into one label. That misses whether the candidate is:

- early but not confirmed;
- already in expansion;
- late after most path is gone;
- a slower 48–72h setup.

### Gap 3 — current timeframes do not use designed multi-scale fields

`potential_observation_frame()` already expects:

```text
1H decision
4H swing
1D background
```

But current configs load only:

```toml
timeframes = ["1H"]
```

So `background_*` and `swing_*` are missing. For 30% events this is a major blind spot.

### Gap 4 — feature granularity is too low for altcoin volatility

The current continuous feature set is a compact scanner set, not an extreme-event set. It lacks:

- volatility compression/expansion slopes;
- multi-window returns and accelerations;
- range breakout distance across 2d/7d/14d/30d;
- volume/taker/OI/funding z-scores and deltas;
- listing age / new-coin regime;
- market beta / cross-sectional impulse.

### Gap 5 — selection gates remain tuned for frequent tails

Current gates:

```toml
min_selected_observation_count = 500
min_selected_tail_count = 20
min_valid_tail_lift = 3.0
```

For a 30% h24 event with base rate ~1.82% either-side, requiring 500 selected observations means at least ~9 baseline tails expected. A 3x lift means ~27 tails. That is reasonable for h24/h48/h72, but too strict for h6 ignition. Gates should be horizon-specific.

## Proposed target design

### 1. Outcome: absolute extreme + normalized abnormality

Keep the absolute event threshold as the core profit-source label:

```toml
threshold_pct = 30.0
```

Add event descriptor columns:

```text
tail_abs_up_30
 tail_abs_down_30
 tail_abs_up_50
 tail_abs_down_50
 forward_max_atr_multiple
 forward_min_atr_multiple
 forward_max_realized_vol_multiple
 forward_min_realized_vol_multiple
 event_speed_bucket          # <=6h, <=12h, <=24h, <=48h, <=72h
 event_efficiency_bucket     # terminal retention / path range
 event_reversal_after_extreme_bucket
```

Do not replace the absolute threshold with normalized threshold. The profit source is absolute move. But normalized features/labels tell us whether the move is abnormal for that coin.

### 2. Horizon: train multi-horizon 24/48/72, inspect 6/12

Recommended config direction:

```toml
[potential.bars]
timeframes = ["1H", "4H", "1D"]
days = 365  # if cache/API budget allows; otherwise 180 first

[potential.evidence.tailtree]
threshold_pct = 30.0
outcome_horizon = [24, 48, 72]
```

Use h6/h12 not as first training targets but as speed/ignition diagnostics:

```text
if h24/h48/h72 event selected and h6/h12 already shows extension -> candidate may be late
if h6/h12 quiet but h48/h72 setup strong -> early setup
```

### 3. Feature set: add extreme-event feature families

#### A. Multi-scale price/range features

Add windows:

```text
returns: 1h, 2h, 4h, 6h, 12h, 24h, 48h, 72h, 168h
range position: 48h, 7d, 14d, 30d
breakout distance: close vs prior high/low for same windows
realized volatility: 6h, 24h, 72h, 168h
volatility compression ratio: rv_24h / rv_168h
range compression ratio: range_48h / range_30d
```

Purpose: distinguish random volatility from coiled/extreme setup.

#### B. Acceleration / convexity features

```text
return_accel_1_4 = return_1h - return_4h/4
return_accel_4_24 = return_4h - return_24h/6
volume_accel_1_24
range_expansion_6_48
```

Purpose: catch ignition and behavior shift before the full 30% move completes.

#### C. Source/crowding dynamics

For OI, funding, taker volume, long/short ratio:

```text
level
1h/4h/24h delta
percent change
rolling z-score 24h/7d
price divergence term
age_ms
```

Examples:

```text
oi_change_4h_pct
oi_change_24h_pct
oi_price_divergence = sign(return_4h) * oi_change_4h_pct
funding_z_7d
taker_buy_sell_ratio_z_24h
long_short_ratio_delta_24h
crowding_stress = funding_z * opposite price return
```

Purpose: extreme events often come from forced positioning/crowding, not just candles.

#### D. Symbol lifecycle / volatility class

```text
listing_age_hours
history_depth_days
symbol_realized_vol_rank
symbol_tail_rate_30_h24_prior
symbol_tail_rate_30_h72_prior
```

Purpose: 30% base rates vary wildly by symbol. Without lifecycle/volatility class, the model can confuse “new coin normally moves 30%” with an actual abnormal setup.

#### E. Market/cross-sectional context

Add simple global frames from cached bars:

```text
btc_return_1h/4h/24h
eth_return_1h/4h/24h
alt_universe_breadth_1h/24h
alt_top_decile_return_24h
cross_sectional_rank_return_24h
cross_sectional_rank_volume_anomaly
```

Purpose: a 30% alt event is often conditional on broad alt impulse or market risk regime.

### 4. Objective: two-stage rare-event model

Current objective options:

```text
tail_severity_gpd
tail_utility_quantile
```

For >=30%, add a profile that separates event probability from event utility:

#### Stage 1 — event-lift classifier/ranker over all observations

Train on all observations:

```text
y = 1 if forward_max_return_pct >= 30 for up
 y = 1 if forward_min_return_pct <= -30 for down
```

Use rare-event weights:

```text
positive_weight ~= sqrt(n_negative / n_positive)
```

Output:

```text
event_probability_score
event_lift
selected_event_rate
selected_tail_count
```

This directly answers: “which current states make a 30% event more likely?”

#### Stage 2 — severity/utility model over event rows

For selected/event rows, train current quantile/GPD-style objective:

```text
utility = exceedance_beyond_30 * retention * efficiency * speed - adverse_path_penalty
```

Output:

```text
expected_event_utility_bucket
speed bucket
path quality bucket
```

#### Final promotion surface

Promotion should require both:

```text
event_lift_pass
utility_pass
support_pass
freshness/source_pass
```

Not just high utility among already-tail rows.

### 5. Selection metrics for 30% event scanner

Canonical metrics should be re-centered around rare-event utility:

```text
base_event_rate
selected_event_rate
event_lift
selected_event_count
selected_observation_count
event_per_1k_obs
median_time_to_extreme
utility_mean
utility_p90
horizon_confirmation_count
opposite_direction_conflict
```

For report rows, show:

```text
symbol | side | horizon | event_lift | base_rate | selected_rate | event_count | time_to_extreme | utility_p90 | blockers
```

### 6. Config recommendations

#### Extreme research config

```toml
[potential]
max_symbols = 160

[potential.bars]
timeframes = ["1H", "4H", "1D"]
days = 365
latest_staleness_hours = 2

[potential.transition]
horizon = 72
mae_mfe_horizon = 72
recent_window = 720
long_window = 4320

[potential.evidence.tailtree]
threshold_pct = 30.0
outcome_horizon = [24, 48, 72]

[potential.evidence.tailtree.selection]
min_selected_observation_count = 300
min_selected_tail_count = 15
min_valid_tail_lift = 4.0
```

Rationale:

- h24 is sparse but immediate.
- h48/h72 provide more support and full event path.
- 4H/1D context activates existing background/swing features.
- 365d increases rare-event count if provider/cache budget permits.

#### Daily config

For daily scanner, keep runtime bounded:

```toml
max_symbols = 80
threshold_pct = 30.0
outcome_horizon = [24, 48]
timeframes = ["1H", "4H", "1D"]
days = 180
```

Use fixed HPs after an advanced run selects profile settings.

### 7. Implementation phases

#### Phase 1 — measurement-only benchmark

No model changes yet.

Add/report:

```text
tail-support-by-threshold-horizon.csv
feature-null-rate.csv
symbol-tail-rate-30.csv
```

Verify:

```text
30% support by horizon/direction
which symbols dominate events
whether 4H/1D context exists or is missing
feature null rates after source age gating
```

#### Phase 2 — activate multi-scale context

Update configs to load:

```toml
timeframes = ["1H", "4H", "1D"]
```

Ensure kline history and observations actually contain non-null:

```text
background_regime
swing_core
market_alignment != market_context_missing
```

#### Phase 3 — add extreme-event feature family

Add a compact, typed feature block in `state.py`:

```text
multi-window returns
range positions
realized volatility
compression/expansion ratios
source deltas/z-scores
listing age / history depth
```

Keep Polars-native, no row loops.

#### Phase 4 — add rare-event objective/profile

Add new profile objective, e.g.:

```text
tail_event_lift
```

This trains on all observations with rare-event weighting, then keeps existing utility/GPD post-analysis for selected event rows.

Do not delete current `tail_utility_quantile`; compare both on shared selection-efficiency surface.

#### Phase 5 — report event scanner output

Report should show:

```text
Extreme Event Board
- side: up/down
- horizon: h24/h48/h72
- base 30% event rate
- selected event rate
- lift
- support
- median time-to-extreme
- utility p90
- blockers
```

## Recommended next concrete step

Do not jump directly to code-heavy objective changes.

First implement Phase 1 + Phase 2:

1. Add threshold/horizon support diagnostics for `[8, 15, 30, 50]` and `[6, 12, 24, 48, 72]`.
2. Change advanced config to include `1H`, `4H`, `1D` timeframes.
3. Run a cache/backfill-aware advanced smoke to prove background/swing features are populated.
4. Only then add new extreme-event features and a rare-event event-lift objective.

## Bottom line

The user’s `>=30%` threshold is directionally correct for altcoin extreme behavior. The current scanner has the right skeleton — known-at-close features, path max/min labels, direction-specific tails, source freshness, and utility scoring — but it is still shaped like a moderate-tail scanner.

For extreme-event profit-source research, the key redesign is:

```text
absolute 30% path-event label
+ multi-horizon 24/48/72 event formation
+ 1H/4H/1D multi-scale context
+ volatility/crowding/source dynamics
+ event-probability lift over all observations
+ severity/utility ranking over selected events
```

That will make the scanner ask the right question:

```text
Which current known-at-close states materially increase the chance of a future ±30% path event, and among those, which have the best speed/retention/utility profile?
```
