# Tailtree performance direction detail

> Superseded for implementation by
> `docs/report/2026-06-21-tailtree-explicit-label-api-implementation-plan.md`.
> Keep this file as theory/background only. Use the canonical plan for names,
> module ownership, and phase order.

## Purpose

The previous report named the direction but was still too abstract. This document makes the next model-development direction concrete in three layers:

1. theory base — what probability/random-process/information problem we are solving;
2. market behavior — what crypto perpetual behavior the model should distinguish;
3. model development — exact product shapes, training targets, HPO objective, and evaluation gates.

The goal is not “better ML” in general. The goal is:

```text
known-at-close state -> probability and utility of future directional extreme path
```

with explicit separation between:

```text
extreme volatility regime
clean up directional extreme
clean down directional extreme
both-tail / gray-zone regime
```

## Current failure mode stated precisely

Current `tail_event_lift` trains two one-vs-rest classifiers:

```text
up model:   Y_up   = 1{forward_max_return_pct > threshold}
down model: Y_down = 1{forward_min_return_pct < -threshold}
```

This answers:

```text
Which state buckets concentrate future up tails?
Which state buckets concentrate future down tails?
```

But the scanner asks a stronger question:

```text
Which side, if any, has clean directional dominance and usable path utility?
```

Those are not the same. A state can have high `P(down tail | X)` and still be a bad clean-down candidate if it also has high `P(up tail | X)` or if the path frequently whipsaws before the down move.

The current empirical artifact confirms this:

```text
down = higher lift, higher false-direction rate, negative margin
up   = lower lift, lower false-direction rate, positive margin
```

So the next direction is not “make down stronger” or “balance up/down mechanically.” It is:

```text
separate event intensity from directional skew and path utility
```

## Layer 1 — theory base

### 1.1 Joint distribution, not two isolated binary labels

The real outcome is joint:

```text
Y_up   ∈ {0,1}
Y_down ∈ {0,1}
```

Therefore there are four states:

```text
none:      Y_up=0, Y_down=0
up_only:   Y_up=1, Y_down=0
down_only: Y_up=0, Y_down=1
both_tail: Y_up=1, Y_down=1
```

The current one-vs-rest models estimate marginal probabilities:

```text
P(Y_up=1 | X)
P(Y_down=1 | X)
```

But decision quality depends on joint quantities:

```text
P(up_only | X)
P(down_only | X)
P(both_tail | X)
P(none | X)
```

A clean up candidate is not just high `P(Y_up=1 | X)`. It is high:

```text
P(up_only | X) or P(up utility dominates down utility | X)
```

A high-risk volatility watch is high:

```text
P(both_tail | X) or P(any_extreme | X) with low side dominance
```

This is why up/down should be clarified rather than “balanced.” Balancing class weights may help training, but it does not solve the missing joint-state definition.

### 1.2 Random-process decomposition

Model the future path over horizon `h` as a random process with three latent components:

```text
future path = direction/drift component + volatility component + jump component
```

Tail events can arise from different mechanisms:

1. directional drift/skew:

```text
price process has directional pressure; one side tail dominates
```

2. volatility expansion:

```text
large range expected; both up and down extremes possible
```

3. jump/liquidation regime:

```text
rare discontinuous move; direction may depend on positioning/crowding
```

Current `tail_event_lift` mainly captures:

```text
state -> tail event intensity
```

It does not explicitly say whether the state is:

```text
up-skewed jump risk
down-skewed jump risk
symmetric volatility expansion
```

The next model architecture should estimate these separately:

```text
extreme intensity: P(any_extreme | X)
directional skew:  P(up_only | X) - P(down_only | X)
gray risk:         P(both_tail | X)
path utility:      E[utility_up - utility_down | X]
```

### 1.3 Information theory view

Raw lift is:

```text
lift = P(tail | selected) / P(tail)
```

When base rate is small, lift can be huge even if selected probability is not practically strong. Down tails are rarer, so down lift is naturally easier to inflate.

A better evidence score should consider information gain and uncertainty:

```text
information = logit(P(tail | selected)) - logit(P(tail base))
```

with shrinkage for count:

```text
shrunk_information = information * sqrt(n_selected_tail) / sqrt(n_selected_tail + k)
```

or use a lower-confidence bound:

```text
score_probability = lower_bound(P(tail | selected)) - P(tail base)
```

This prevents tiny, rare-side buckets from dominating only because base rate is tiny.

### 1.4 Decision theory view

The scanner is not only estimating probabilities. It is selecting review candidates. The decision utility should be side-aware:

```text
U(select up)   = expected_up_utility - false_down_cost - gray_zone_cost
U(select down) = expected_down_utility - false_up_cost - gray_zone_cost
U(watch)       = high any_extreme but low side dominance
U(skip)        = low information or unstable evidence
```

Therefore HPO should optimize expected decision utility, not raw model logloss or raw lift.

## Layer 2 — market behavior interpretation

### 2.1 Why downside looks high-lift but dirty

In crypto perpetuals, downside extremes often occur in states with:

```text
high leverage/crowding
thin liquidity
funding/positioning imbalance
forced liquidation cascades
volatility expansion
```

These states are often not cleanly directional before the move. They are unstable energy states. The path can include:

```text
short squeeze then liquidation
liquidation wick then rebound
wide two-sided range
```

So a down model can find a very high-lift regime while still having high false-direction exposure.

Interpretation:

```text
high down lift + high false-direction + negative margin = crash-risk / volatility-risk watch
```

not automatically:

```text
clean short candidate
```

### 2.2 Why upside may be lower-lift but cleaner

Upside extremes may be less rare in altcoin samples and therefore have lower lift. But some upside states can be cleaner:

```text
positive taker pressure
short crowding stress
OI expansion with price up
compression breakout
funding not overheated yet
```

These states may produce lower lift but better directional dominance:

```text
positive score margin
low false-direction rate
higher utility per selected observation
```

Interpretation:

```text
moderate up lift + low false-direction + positive margin = cleaner up candidate
```

### 2.3 Both-tail / gray-zone is not just failure

Both-tail outcomes mean:

```text
forward_max > threshold and forward_min < -threshold inside same horizon
```

This can be useful, but it has a different semantic:

```text
high range / danger / optionality regime
```

It should become:

```text
watch/risk/regime candidate
```

not:

```text
promote clean up/down candidate
```

So the model should explicitly identify both-tail. Suppressing it inside marginal up/down labels loses useful market information.

### 2.4 Path order matters

A horizon can contain both favorable and adverse extremes. For a real path utility lens:

```text
clean up path:
  max up occurs early or before severe drawdown
  post-max drawdown is tolerable
  close retention is high

bad up path:
  huge drawdown first, then up wick
  max up not retained
  opposite tail appears first
```

Same for down.

Therefore the labels should include order/path shape, not only max/min existence:

```text
first_extreme_side
first_extreme_time
adverse_before_favorable_pct
retention_after_extreme
path_efficiency
```

The current utility already uses retention/efficiency/speed, which is good. The missing piece is explicit first-side and opposite-before-favorable diagnostics.

## Layer 3 — concrete model-development direction

## 3.1 Product 1: label state table

Add one label product, still inside `tailtree/model.py` or nearby model label code.

For every decision row/horizon:

```text
symbol
decision_bar_close_ms
outcome_horizon
up_event
down_event
any_extreme
both_tail
clean_up
clean_down
extreme_class
up_utility
down_utility
utility_margin_up_minus_down
first_extreme_side
first_extreme_time_bar
```

Definitions:

```text
up_event = forward_max_return_pct > threshold_pct
down_event = forward_min_return_pct < -threshold_pct
any_extreme = up_event | down_event
both_tail = up_event & down_event
clean_up = up_event & !down_event
clean_down = down_event & !up_event

extreme_class:
  none if !any_extreme
  up_only if clean_up
  down_only if clean_down
  both_tail if both_tail

utility_margin_up_minus_down = tail_utility_up - tail_utility_down
```

If path order columns are not sufficient yet, first implement joint class without first-side, then add first-side once outcome paths expose it cleanly.

Why this product matters:

```text
It turns vague “up/down imbalance” into measurable prevalence and confusion structure.
```

## 3.2 Product 2: feature state table

Add known-at-close feature groups in `state.py`, then whitelist them in tailtree.

### Group A — regime age / transition dynamics

Use known-at-close state/history fields:

```text
state_age_bars
event_age_bars
fresh_event_int
transition_kind categorical
compression_state categorical
expansion_state categorical
extreme_range categorical
extreme_vol categorical
```

Market meaning:

```text
new transitions and stale regimes have different hazard rates
compression/expansion states condition volatility and jump probability
```

### Group B — realized volatility and jump budget

Historical rolling features:

```text
realized_vol_6h_pct
realized_vol_24h_pct
realized_vol_72h_pct
realized_vol_ratio_6h_72h
abs_return_1h_pct
max_abs_return_24h_pct
range_pct_1h
range_pct_24h_mean
range_expansion_1h_vs_24h
volume_z_24h
```

Market meaning:

```text
volatility clusters; extremes require a volatility budget; jump-like bars persist in local regimes
```

### Group C — directional skew features

```text
upside_range_share_24h
downside_range_share_24h
close_position_24h
close_position_72h
signed_return_vol_ratio_24h
```

Market meaning:

```text
separate symmetric volatility from directional pressure
```

### Group D — derivative/source pressure and disagreement

```text
funding_z
funding_delta
funding_abs_z
oi_change_z
oi_acceleration_z
taker_pressure_delta
lsr_log_ratio_delta
source_bullish_count
source_bearish_count
source_conflict_count
price_flow_divergence_flag
```

Market meaning:

```text
crowding + price/flow disagreement often precedes squeezes, liquidations, and false-direction traps
```

### Group E — missing/freshness features

```text
funding_present_int
oi_present_int
taker_present_int
lsr_present_int
source_fresh_count
max_source_age_ms
```

Market meaning:

```text
source availability is not random; missingness can bias tail diagnostics
```

## 3.3 Product 3: train objectives

Do not jump directly to a complex model. Build in layers and compare.

### Objective A — baseline side binary, current behavior

Keep:

```text
tail_event_lift up/down binary models
```

This is the baseline.

Use it to answer:

```text
Does new feature input improve current objective without label redesign?
```

### Objective B — extreme intensity

Train:

```text
any_extreme = up_event | down_event
```

Output:

```text
p_any_extreme
```

This detects volatility/jump-prone states regardless of direction.

Expected behavior:

```text
high p_any_extreme + low side margin => watch/risk regime
```

### Objective C — joint extreme class

Train multiclass:

```text
none / up_only / down_only / both_tail
```

Output:

```text
p_none
p_up_only
p_down_only
p_both_tail
```

Decision features:

```text
p_clean_up = p_up_only
p_clean_down = p_down_only
p_gray = p_both_tail
side_margin = p_up_only - p_down_only
```

This directly solves the “up/down extreme/not should be clarified” issue.

### Objective D — side utility

Train side utility models:

```text
log1p(tail_utility_up)
log1p(tail_utility_down)
```

or train only on event rows:

```text
E[utility_up | up_event, X]
E[utility_down | down_event, X]
```

Output:

```text
u_up
u_down
utility_margin = u_up - u_down
```

This prevents probability-only optimization from selecting low-utility tails.

## 3.4 Scoring architecture

Final candidate score should be composed, not hidden inside the learner.

For up side:

```text
score_up =
    intensity_gate(p_any_extreme)
  + clean_prob_weight * p_up_only
  + utility_weight * u_up
  + margin_weight * (p_up_only - p_down_only)
  - gray_weight * p_both_tail
  - false_cost_weight * expected_down_utility_when_select_up
```

For down side:

```text
score_down =
    intensity_gate(p_any_extreme)
  + clean_prob_weight * p_down_only
  + utility_weight * u_down
  + margin_weight * (p_down_only - p_up_only)
  - gray_weight * p_both_tail
  - false_cost_weight * expected_up_utility_when_select_down
```

For watch:

```text
score_watch =
    high p_any_extreme
  + high p_both_tail
  + low absolute side margin
```

This gives three semantic outputs:

```text
promote_up
promote_down
watch_volatility_gray_zone
```

## 3.5 HPO objective

Current HPO should stop rewarding raw lift as the primary objective. New HPO should optimize validation decision quality.

Per fold and bucket:

```text
base_information = shrink(logit(selected_tail_rate) - logit(base_tail_rate), selected_tail_count)
selected_utility = selected_utility_mean
side_margin = calibrated_selected_side_prob - calibrated_opposite_side_prob
false_rate = P(opposite_tail & !selected_tail | selected bucket)
false_cost = E(opposite_utility | false_direction)
gray_rate = P(selected_tail & opposite_tail | selected bucket)
```

Objective:

```text
objective_score =
    information_weight * base_information
  + utility_weight * selected_utility
  + margin_weight * side_margin
  - false_rate_weight * false_rate
  - false_cost_weight * false_cost
  - gray_weight * gray_rate
```

Across folds:

```text
hpo_score = mean(objective_score_by_fold) - fold_std_weight * std(objective_score_by_fold)
```

or stricter:

```text
hpo_score = percentile_25(objective_score_by_fold)
```

Reason:

```text
tail models must be stable across time; best-fold performance is not enough
```

## 3.6 Calibration requirement

Raw LightGBM scores from separate up/down models are not comparable. The paired margin currently uses raw model score difference:

```text
score_up - score_down
```

That is only a rough diagnostic. For real side comparison, use calibrated probabilities or empirical bucket tail rates.

Minimal calibration:

```text
for each fold/direction/bucket:
  calibrated_tail_rate = observed_tail_count / selected_count
```

Then compare:

```text
calibrated_margin = calibrated_up_tail_rate - calibrated_down_tail_rate
```

Better later:

```text
Platt/isotonic calibration per fold
```

But start with bucket calibration to avoid dependency/complexity.

## 3.7 Training imbalance policy

Clarify imbalance, do not erase it.

Rules:

1. Natural prevalence must be reported:

```text
P(up_only), P(down_only), P(both_tail), P(none)
```

2. Training may use sample weights:

```text
weight_class = sqrt(total_count / class_count)
```

with cap:

```text
max_weight = 10
```

3. HPO/evaluation must use natural validation distribution, not balanced validation.

4. Direction comparison must be calibrated by validation buckets.

This gives the learner enough minority examples without lying about market base rates.

## Development roadmap

### Phase 1 — make the problem measurable

Add artifacts only, no behavior change:

```text
tailtree-label-distribution.csv
tailtree-joint-class-by-fold.csv
tailtree-side-confusion.csv
```

Metrics:

```text
none_count
up_only_count
down_only_count
both_tail_count
any_extreme_rate
both_tail_given_up_rate
both_tail_given_down_rate
false_direction_by_bucket
utility_margin_by_bucket
```

Expected output:

```text
We can say exactly whether down is high lift because down_only is strong or because both_tail/volatility is strong.
```

### Phase 2 — feature enrichment, same model

Add known-at-close features and run current `tail_event_lift` unchanged.

Success criteria:

```text
false_direction_rate decreases
objective_hpo_score improves
fold stability improves
raw lift does not need to increase
```

This isolates feature value from objective redesign.

### Phase 3 — joint label objective

Add `extreme_class` training profile.

Success criteria:

```text
p_both_tail explains gray-zone/watch cases
p_up_only/p_down_only improves directional margin
promotion candidates have lower false-direction rate
```

### Phase 4 — calibrated selection score

Replace raw score margin with calibrated bucket margin.

Success criteria:

```text
up/down bucket comparison is interpretable
side choice agrees with validation tail rates
fewer cases where raw down score dominates but opposite event rate is high
```

### Phase 5 — robust HPO

Use fold-stability-aware objective:

```text
mean - lambda * std
```

Success criteria:

```text
selected hyperparams are not best only on one fold
paired replay metrics improve on held-out folds
promotion/watch split is more stable
```

## Concrete target outputs after this redesign

Each candidate should be explainable as one of:

```text
clean_up:
  high p_up_only
  positive calibrated side margin
  acceptable false-down risk
  good up utility

clean_down:
  high p_down_only
  positive calibrated side margin for down
  acceptable false-up risk
  good down utility

volatility_watch:
  high p_any_extreme
  high p_both_tail or low side margin
  do not promote as directional

skip:
  low information, low utility, unstable fold evidence, or missing-data blocker
```

This is the concrete semantic endpoint.

## Bottom line

The next direction is:

```text
1. define the joint outcome state;
2. add known-at-close features that describe volatility, skew, state age, and source disagreement;
3. train separate products for intensity, clean side, both-tail risk, and utility;
4. calibrate up/down scores before comparing them;
5. optimize HPO for robust directional utility, not raw lift.
```

This converts the model from:

```text
rare tail bucket finder
```

to:

```text
probabilistic scanner for clean up/down extreme opportunity vs volatility watch regime
```
