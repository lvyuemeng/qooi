# Tailtree model-performance theory redesign

> Superseded for implementation by
> `docs/report/2026-06-21-tailtree-explicit-label-api-implementation-plan.md`.
> Keep this file as theory/background only. Use the canonical plan for names,
> module ownership, and phase order.

## Purpose

Focus the next tailtree work on model performance, not another API shuffle. The scanner goal is:

```text
known-at-close state -> information about future up/down extreme behavior and path utility
```

The current implementation now has a reduced enough runtime surface to reason about the model stack directly:

```text
features input -> labels -> train objective -> HPO objective -> training protocol -> selection artifacts
```

This report evaluates the problem and proposes theory-backed improvements without coding them yet.

## Current architecture snapshot

Durable scanner boundary from `docs/architecture/scanner.md`:

```text
state.py      -> known-at-close observations/features
outcome.py    -> future/path/source outcomes
 tailtree/model.py -> labels, training frame, LightGBM model
 tailtree/evidence.py -> prediction/evidence buckets
 tailrun/planning.py -> profiles/trials/folds/jobs
 tailrun/core.py -> train/load/score lifecycle
 tailrun/selection.py -> paired replay + selection/HPO metrics
```

Important current contracts:

- model inputs must be known-at-close only;
- future returns/path quantities are labels/outcomes only;
- direction and threshold are objective slices, not input features;
- scanner artifacts are research hypotheses, not trading/execution instructions;
- missing/stale/provider-bounded data must stay visible.

## Current configuration behavior

Current tailtree config axes:

```text
TailtreeConfig:
  lifecycle: train | load_predict
  threshold_pct: fixed tail threshold, currently 30.0 in active configs
  outcome_horizon: e.g. [24] or [24, 48]
  selection: top_k/top_pct/gates
  profiles:
    objective: tail_event_lift | tail_utility_quantile | tail_severity_gpd
    training: fixed | optuna
    evaluation: single_split | walkforward
```

Active research configs use:

```text
objective = tail_event_lift
threshold_pct = 30.0
outcome_horizon = [24] or [24, 48]
walkforward for advanced/paired replay; single_split for daily
```

`tail_event_lift` currently trains one binary classifier per horizon × direction:

```text
P(up tail over threshold | known-at-close features)
P(down tail over threshold | known-at-close features)
```

This is conceptually clean, but it does not by itself decide whether an event is a clean directional opportunity or a both-tail volatility regime.

## Current feature input

Whitelisted tailtree features in `tailrun/core.py` are known-at-close:

Categorical:

```text
background_regime
swing_core
decision_core
decision_transition
decision_direction
```

Continuous:

```text
atr_percentile
range_width_atr
bar_return_1h_pct
bar_return_4h_pct
bar_return_24h_pct
bar_return_4h_per_vol_7d
bar_return_24h_per_vol_7d
bar_volume_1h_to_ma_20h
bar_close_position_48h
funding_rate_raw
funding_rate_bps
oi_change_raw
oi_change_pct
taker_buy_sell_ratio_raw
taker_buy_pressure
lsr_ratio_raw
lsr_log_ratio
funding_age_ms
oi_age_ms
taker_age_ms
lsr_age_ms
```

These are mostly point-in-time state and short lookback features. They are useful, but they underrepresent three things that matter for extreme tail prediction:

1. **state persistence / transition age** — hazard changes when a regime is new vs stale;
2. **realized volatility / jump pressure** — extremes are volatility-clustered random-process events;
3. **cross-source disagreement** — tails often arise when price, OI, taker flow, funding, and L/S positioning disagree.

## Current labels

`label_tail_exceedances(...)` labels direction-specific events:

```text
tail_up   = forward_max_return_pct > threshold_pct
tail_down = forward_min_return_pct < -threshold_pct
```

and utility:

```text
up utility   = (forward_max_return_pct - threshold) * retention * efficiency * speed - drawdown penalty
down utility = (abs(forward_min_return_pct) - threshold) * retention * efficiency * speed - rebound penalty
```

This is direction-aware and path-aware. The key limitation is that it still treats up and down as separate binary facts, while the real outcome space is joint:

```text
not extreme
up-only extreme
down-only extreme
both-tail / gray-zone extreme
```

That distinction matters because current down buckets often find generic crash/volatility regimes, not necessarily clean down opportunities.

## Current empirical problem

Latest paired-replay smoke artifact:

```text
data/output/potential/paired-replay-test/tailtree-selection-efficiency.csv
shape = (16, 44)
```

Direction summary from current output:

```text
down:
  max_lift              = 55.749853
  max_base_hpo          = 59.190124
  max_objective_hpo     = 52.017283
  mean_false_rate       = 0.356409
  mean_false_cost       = 1.909621
  mean_margin           = -0.200262
  mean_gray             = 0.107080

up:
  max_lift              = 29.330425
  max_base_hpo          = 34.799956
  max_objective_hpo     = 36.172602
  mean_false_rate       = 0.079218
  mean_false_cost       = 2.285395
  mean_margin           = 0.320216
  mean_gray             = 0.107076
```

Interpretation:

- down has much higher lift because downside extreme events are rarer;
- down also has much worse false-direction rate and negative directional margin;
- up has lower lift but cleaner directional separation;
- gray-zone rate is almost equal between sides, so gray-zone alone is not the discriminating metric.

The current asymmetry is not only an implementation issue. It is a data-generating-process issue:

```text
down tail = often volatility/crash-risk regime with opposite-tail contamination
up tail   = less concentrated but more directionally separable in current data
```

## Theory evaluation

### 1. Probability view

The scanner should not only estimate:

```text
P(Y_up = 1 | X)
P(Y_down = 1 | X)
```

It needs the joint event distribution:

```text
P(Y = none | X)
P(Y = up_only | X)
P(Y = down_only | X)
P(Y = both_tail | X)
```

Current separate up/down models are one-vs-rest projections of this joint distribution. One-vs-rest is valid when classes are cleanly separable. It is weak when `both_tail` is material, because both classifiers can be correct and still produce a bad directional promotion.

### 2. Random-process view

Crypto tails behave like clustered jump processes with stochastic volatility:

```text
return process = drift/state component + volatility component + jump component
```

A high downside lift bucket may mean:

```text
high jump intensity / high volatility
```

not:

```text
clean negative directional drift
```

So the model should separate at least two latent factors:

```text
extreme intensity:   P(any extreme | X)
directional skew:    P(up | extreme, X) - P(down | extreme, X)
path utility:        E[utility | selected side, X]
```

Current binary up/down event classifiers mix intensity and skew. This explains why down lift can dominate while directional margin is negative.

### 3. Information view

Tail lift is a ratio:

```text
P(tail | selected bucket) / P(tail)
```

For rare events, the denominator is tiny. This inflates lift and over-rewards rare-side concentration. A better information criterion should use a calibrated evidence term such as:

```text
log-likelihood improvement / KL contribution / information gain over base rate
```

and include uncertainty/count shrinkage. In practice, use stable score terms:

```text
logit(p_selected_tail) - logit(p_base_tail)
credible_lower_bound(lift or probability delta)
```

instead of raw lift alone.

### 4. ML objective view

Current LightGBM objective for `tail_event_lift` is binary logloss per side. That is a good base estimator for event concentration, but the selection/HPO target is a utility decision problem:

```text
choose side/bucket that maximizes expected directional utility under false-direction risk
```

Therefore the model stack should be two-stage or multi-head in logic even if implemented with simple flat functions:

```text
stage A: estimate extreme intensity / side probabilities
stage B: score selection utility and directional dominance
```

Do not force all goals into a single binary target.

## Proposed model-performance redesign

### A. Feature input: add state dynamics, volatility, and disagreement features

Keep all features known-at-close. Add only features derivable from historical bars/source states at or before the decision bar.

Recommended feature groups:

#### A1. State age and transition features

Use already produced path/state concepts as model inputs where known at close:

```text
state_age_bars
event_age_bars
fresh_event
transition_kind
compression_state
expansion_state
extreme_range
extreme_vol
```

Reason:

- tail hazard is not stationary across the life of a regime;
- new compression breakouts and stale trends have different jump/skew distributions.

#### A2. Multi-scale realized-volatility and jump proxies

Add rolling historical-only features:

```text
realized_vol_6h_pct
realized_vol_24h_pct
realized_vol_72h_pct
realized_vol_ratio_6h_72h
abs_return_1h_pct
max_abs_return_24h_pct
range_expansion_1h_vs_24h
volume_z_24h
```

Theory:

- volatility clusters;
- jump intensity is state dependent;
- extremes require enough volatility budget.

#### A3. Directional skew / asymmetry features

Add historical-only asymmetry measures:

```text
upside_range_share_24h = rolling max positive move / rolling total range
downside_range_share_24h
close_position_24h / 72h
signed_return_vol_ratio
```

Purpose:

- separate “high volatility both-tail” from “directionally skewed volatility”.

#### A4. Cross-source disagreement features

Derive compact features from funding/OI/taker/LSR:

```text
flow_pressure_z
funding_extreme_z
oi_acceleration_z
taker_pressure_delta
lsr_crowding_delta
source_direction_agreement_count
source_direction_conflict_count
price_flow_divergence_flag
```

Theory:

- extremes often occur when crowding and realized price path disagree;
- false-direction risk is high when sources conflict.

#### A5. Missingness/freshness as information

Do not hide missing/stale sources. Expose stable missingness/freshness features:

```text
funding_present_int
oi_present_int
taker_present_int
lsr_present_int
source_fresh_count
max_source_age_ms
```

Reason:

- data absence is not random across symbols/sources;
- diagnostics still need to distinguish missing-data artifacts from real signal.

### B. Label redesign: joint extreme class plus path utility

Keep the current up/down labels, but add a joint outcome product:

```text
extreme_class:
  none
  up_only
  down_only
  both_tail
```

Definitions:

```text
up_event   = forward_max_return_pct > threshold_pct
down_event = forward_min_return_pct < -threshold_pct

none      = !up_event & !down_event
up_only   =  up_event & !down_event
down_only = !up_event &  down_event
both_tail =  up_event &  down_event
```

Add side-conditional utility labels:

```text
utility_up
utility_down
net_directional_utility = utility_selected - utility_opposite
```

Add path-order labels if available:

```text
first_extreme_side = up | down | none
first_extreme_time
adverse_before_favorable_ratio
```

Why path order matters:

- A path that hits +30% after first drawing down -35% is not the same as a clean up path;
- utility should reflect realizable path quality, not only max/min extrema.

### C. Train objective: split intensity, direction, and utility

Do not replace everything with one bigger target. Use a small stack of flat products.

Recommended training products:

#### C1. Extreme intensity model

Train:

```text
P(any_extreme | X), where any_extreme = up_event | down_event
```

Purpose:

- identify volatility/jump-prone states;
- should be direction-neutral.

Loss:

```text
binary logloss with class weights or positive downsampling guard
```

#### C2. Conditional side model

Train only on extreme rows or with sample weights emphasizing extremes:

```text
P(up_only | any_extreme, X)
P(down_only | any_extreme, X)
P(both_tail | any_extreme, X)
```

Implementation can start simple:

```text
multiclass LightGBM over {up_only, down_only, both_tail}
```

or two binary margins:

```text
side_margin = P(up_event | X) - P(down_event | X)
both_tail_probability = P(both_tail | X)
```

#### C3. Utility model

Train side-specific utility only after event/intensity is understood:

```text
E[utility_up | up_event, X]
E[utility_down | down_event, X]
```

Use robust target transform:

```text
log1p(max(utility, 0))
```

or quantile objective for high utility tails.

#### C4. Keep current `tail_event_lift` as baseline

Do not delete current binary side models immediately. Keep them as baseline and compare against the split objective stack.

### D. HPO objective: optimize decision utility, not raw lift

Current base:

```text
base_hpo_score = lift + utility_mean + sqrt(selected_tail_count) / 10
```

Problem:

- raw lift over-rewards rare downside base-rate effects;
- false-direction risk is underweighted;
- count uncertainty is weakly handled.

Recommended HPO objective:

```text
score =
  + information_gain_lower_bound
  + expected_selected_utility
  + directional_margin_weight * directional_margin
  - false_direction_weight * false_direction_rate
  - false_cost_weight * false_direction_cost
  - both_tail_weight * both_tail_rate
  - instability_penalty
```

Where:

```text
information_gain_lower_bound = shrink(logit(selected_tail_rate) - logit(base_tail_rate), count)
directional_margin = mean(score_selected - score_opposite)
false_direction_rate = P(opposite tail & !selected tail | selected bucket)
both_tail_rate = P(selected tail & opposite tail | selected bucket)
instability_penalty = fold-to-fold variance or negative min-fold score
```

Key change:

```text
optimize min/mean walkforward utility, not one pooled best bucket
```

Use:

```text
fold_score = mean(score_folds) - lambda * std(score_folds)
```

or:

```text
fold_score = percentile_25(score_folds)
```

Reason:

- robust performance matters more than best-fold concentration;
- tail events are sparse, so variance control is part of the objective.

### E. Training protocol: balance side support without hiding base rates

The user concern is correct: up/down extreme/not should be clarified rather than just left imbalanced.

Recommended protocol:

1. Keep natural base rates in diagnostics.
2. Use class weighting/sample weighting for training so the learner sees enough minority events.
3. Report calibrated natural probabilities after training.
4. Use fold-level prevalence and selected support gates in HPO.

For separate up/down binary models:

```text
positive_weight_direction = sqrt(n_negative / n_positive)
```

Use sqrt/log weighting, not full inverse prevalence, to avoid overfitting rare down tails.

For joint class:

```text
class_weight = sqrt(total_count / class_count)
```

Cap class weights.

### F. Calibration and comparability

Current score buckets compare raw LightGBM scores between up/down models. Raw scores are not necessarily calibrated across separately trained models.

Add a calibration step per fold/model:

```text
calibrated_probability = isotonic/logistic calibration on validation fold
```

If avoiding new dependencies initially, use simple empirical bucket calibration:

```text
score_bucket -> observed tail rate on validation fold
```

Then paired replay should compare calibrated probabilities or calibrated expected utility, not raw model scores.

This directly attacks the current “up/down bucket not equally identified” issue.

## Recommended implementation sequence

### Step 1 — diagnostics first, no model behavior change

Add/emit label distribution diagnostics:

```text
none_count
up_only_count
down_only_count
both_tail_count
up_base_rate
down_base_rate
both_tail_rate
horizon × fold × direction prevalence
```

Goal: prove the imbalance and gray-zone source before changing training.

### Step 2 — add known-at-close feature groups behind whitelist

Add state-age/volatility/disagreement features to `state.py`, then append selected names to the tailtree whitelist in `tailrun/core.py`.

Do not include future/path columns.

### Step 3 — add joint label product

Extend `label_tail_exceedances` with:

```text
extreme_class
any_extreme
both_tail
clean_tail_up
clean_tail_down
```

Keep existing columns for compatibility during evaluation.

### Step 4 — add candidate-calibrated replay metrics

Add validation bucket calibration columns:

```text
selected_calibrated_tail_rate
opposite_calibrated_tail_rate
calibrated_directional_margin
```

Use these in objective diagnostics first.

### Step 5 — make HPO robust and direction-aware

Replace HPO fold score with robust objective:

```text
mean(objective_score_by_fold) - 0.5 * std(objective_score_by_fold)
```

Use objective columns, not raw lift.

### Step 6 — evaluate split objective stack

Compare three profiles on same folds:

```text
A. current side binary `tail_event_lift`
B. joint extreme-class model
C. two-stage intensity + side/utility score
```

Promotion criterion:

- not max lift;
- lower false-direction rate;
- positive calibrated directional margin;
- stable fold score;
- sufficient selected tail count;
- utility per selected observation improves.

## Proposed minimal target API additions

Avoid manager classes. Use product functions and typed config only where needed.

Potential additions:

```text
model.py:
  label_joint_extreme_outcomes(...)
  joint_extreme_training_values(...)

selection.py:
  calibrated_candidate_replay_frame(...)
  robust_directional_objective_score(...)

config.py:
  TailtreeHpoObjectiveConfig
    false_rate_weight
    false_cost_weight
    margin_weight
    both_tail_weight
    fold_std_weight
    calibration: none | bucket
```

Config should keep default behavior stable until explicitly switched.

## What not to do

Do not:

- add direction or threshold as model input features;
- hide class imbalance by balanced resampling without reporting natural base rates;
- optimize raw lift as the final objective;
- treat both-tail/gray-zone as always bad — it may be useful as a watch/risk regime, but it should not be promoted as clean directional;
- add a new model framework before exhausting label/objective/calibration fixes;
- add managers/orchestrators around the reduced pipeline.

## Bottom-line recommendation

The next performance improvement should be:

```text
joint label clarity + known-at-close volatility/disagreement features + calibrated directional HPO
```

not simply more Optuna trials.

The current model is learning extreme concentration. The performance gap is that the scanner needs to separate:

```text
extreme intensity
from directional dominance
from path utility
```

Once those are separated, down buckets with high lift but high false-direction exposure become watch/risk regimes, while cleaner up/down buckets become promotion candidates.
