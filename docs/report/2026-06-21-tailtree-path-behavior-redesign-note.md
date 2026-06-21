# Tailtree Path Behavior Redesign Note

## Goal

Record the proposed redesign before coding: separate fixed-horizon path behavior, side utility, and final scanner action. The current tailtree learns side labels from future-window extrema; this is too vague for trading because one horizon can contain clean movement while a longer horizon can contain reversal or chop.

## Current problem

Current labels are derived from a fixed future horizon:

```text
forward_max_return_pct = max future return in H bars
forward_min_return_pct = min future return in H bars

tail_up   = forward_max_return_pct > threshold_pct
tail_down = forward_min_return_pct < -threshold_pct
```

This answers whether the future interval touched an up/down threshold, not whether the path was a good directional trade.

A row can be:

```text
clean in 6h
both-sided in 24h
choppy / reversal in 48h
```

So horizon is not just another parameter; horizon changes the behavior class.

## Redesign principle

Do not make `up`/`down` the first model target.

Use this causal order:

```text
1. Input features known at decision close
2. Future path measurement per horizon
3. Path behavior label per horizon
4. Side utility label per horizon
5. Final action policy across horizons
```

The model predicts future path behavior and side utility from known-at-close features. The final scanner action is a policy decision over calibrated multi-horizon predictions.

## Input features

Inputs must be known at the decision bar close. They should not include future path fields, future utility, or threshold-derived labels.

Candidate feature groups:

```text
symbol/state keys:
  symbol
  decision_bar_close_ms
  timeframe

market state:
  market_stage
  structure_trend_state
  liquidity_event_type
  atr_percentile
  range_width_atr

recent bar features:
  bar_return_1h_pct
  bar_return_4h_pct
  bar_return_24h_pct
  bar_return_4h_per_vol_7d
  bar_return_24h_per_vol_7d
  bar_volume_1h_to_ma_20h
  bar_close_position_48h

source features known at close:
  funding_rate_bps
  oi_change_pct
  taker_buy_sell_ratio
  book pressure / spread features if cached
```

Forbidden input features:

```text
forward_max_return_pct
forward_min_return_pct
time_to_max_bar
time_to_min_bar
post_max_drawdown_pct
post_min_rebound_pct
path_efficiency
close_retention_ratio
tail_up / tail_down / tail_state
utility_up / utility_down / utility_margin
```

Those are labels or label ingredients, not inputs.

## Output labels

### 1. Path behavior label per horizon

One row per:

```text
symbol × decision_bar_close_ms × horizon
```

Suggested `path_state` vocabulary:

```text
none
clean_up
clean_down
up_first_both
down_first_both
chop_both
late_up
late_down
```

Minimal first version can be:

```text
none
clean_up
clean_down
up_first_both
down_first_both
chop_both
```

Definitions:

```text
clean_up:
  up threshold touched, down threshold not touched, utility_up usable

clean_down:
  down threshold touched, up threshold not touched, utility_down usable

up_first_both:
  both thresholds touched and time_to_max_bar < time_to_min_bar

down_first_both:
  both thresholds touched and time_to_min_bar < time_to_max_bar

chop_both:
  both thresholds touched but ordering/utility dominance is weak, inefficient, or unstable

none:
  neither side produced actionable tail behavior
```

### 2. Side utility labels per horizon

Utility stays side-specific, but it is no longer the behavior label itself.

```text
utility_up
utility_down
utility_margin_up = utility_up - utility_down
utility_margin_down = utility_down - utility_up
```

Potential binary labels:

```text
tradable_up   = path_state is clean_up and utility_margin_up exceeds threshold
tradable_down = path_state is clean_down and utility_margin_down exceeds threshold
```

Potential regression targets:

```text
expected_utility_up
expected_utility_down
expected_utility_margin_up
expected_utility_margin_down
```

### 3. Final action label is policy output, not raw model label

Suggested action vocabulary:

```text
long_candidate
short_candidate
volatility_watch
reversal_watch
gray_zone
no_action
```

Final action should be derived from calibrated path predictions and utility predictions, not trained as one opaque label first.

## How prediction works

### Model inputs

For each decision row, provide only known-at-close feature vector:

```text
X(t) = market state + bar/source features known at close t
```

### Model outputs per horizon

For each horizon `h`, predict:

```text
P(path_state_h = clean_up)
P(path_state_h = clean_down)
P(path_state_h = up_first_both)
P(path_state_h = down_first_both)
P(path_state_h = chop_both)
E[utility_up_h]
E[utility_down_h]
E[utility_margin_up_h]
E[utility_margin_down_h]
```

The simplest implementation can train independent horizon heads first. Do not average raw tree scores across horizons.

### Horizon-aware policy

A short horizon can be clean while a long horizon is bad. That should not be treated as a contradiction; it is a different action profile.

Example:

```text
6h:  clean_up high probability
24h: up_first_both / chop_both high probability
48h: down_first_both high probability
```

Interpretation:

```text
short-lived long opportunity, not a swing long
longer horizon has reversal/chop risk
```

Final output should therefore carry a holding-horizon/actionability profile:

```text
action_side: up
actionability: scalp_candidate / short_horizon_only
valid_horizon: 6h
blocker_horizon: 24h, 48h
reason: clean short-horizon path but longer-horizon both/chop risk
```

## Multi-horizon selection policy

Do not require all horizons to agree. Use horizon roles.

Suggested roles:

```text
entry horizon:
  shortest horizon where clean path probability and utility are strong

confirmation horizon:
  medium horizon should not strongly contradict the side

risk horizon:
  longer horizon reports reversal/chop/blocker risk
```

Policy examples:

```text
scalp/short-horizon candidate:
  short horizon clean side probability high
  short horizon utility margin positive
  longer horizon chop/reversal risk present

swing candidate:
  short and medium horizons clean same side
  long horizon does not show strong opposite/both risk

watch/blocker:
  tail_any high but both/chop/reversal probability high
  or calibrated side margin <= 0
```

This avoids forcing a symbol into one global up/down label when behavior changes by horizon.

## Training recommendation

Start with diagnostic and label products before changing objective.

### Product 1: path behavior distribution

Artifact:

```text
tailtree-path-behavior-distribution.csv
```

Grain:

```text
horizon × path_state
```

Columns:

```text
row_count
class_rate
utility_up_mean
utility_down_mean
utility_margin_up_mean
utility_margin_down_mean
time_to_max_bar_mean
time_to_min_bar_mean
path_efficiency_mean
close_retention_ratio_mean
```

### Product 2: selected candidate behavior

Artifact:

```text
tailtree-selected-path-behavior.csv
```

Grain:

```text
horizon × selected_direction × score_bucket × path_state
```

This tells whether selected down candidates are truly `clean_down` or mostly `down_first_both` / `chop_both`.

### Product 3: horizon action panel

Artifact:

```text
tailtree-horizon-action-panel.csv
```

Grain:

```text
symbol × decision_bar_close_ms × direction
```

Columns:

```text
entry_horizon
clean_horizon_count
contradicting_horizon_count
chop_horizon_count
best_utility_margin
actionability
blocker_reason
```

## Preferred final scanner output

Do not output only:

```text
symbol, side, score
```

Prefer:

```text
symbol
entry_horizon
max_valid_horizon
action_side
actionability
path_state_profile
utility_margin
risk_state
blocker_reason
```

Examples:

```text
actionability = trade_candidate
action_side = up
entry_horizon = 6h
max_valid_horizon = 24h
path_state_profile = clean_up@6h, clean_up@24h, chop_both@48h
reason = short/medium clean up, long horizon chop risk
```

```text
actionability = watch
action_side = down
path_state_profile = clean_down@6h, down_first_both@24h, chop_both@48h
reason = down tail exists but longer horizon rebound/chop risk blocks short trade
```

## Open design questions

1. Which horizon set should tailtree own first: current `[24, 48]` or add a shorter entry horizon such as 6/12?
2. Should `clean_up/down` require positive close retention, or only threshold touch without opposite threshold?
3. Should `chop_both` be defined by both thresholds only, or require poor path efficiency / weak utility dominance?
4. Should actionability be rule-based first, then learned later?
5. Should down-first-both become `reversal_watch` rather than `short_candidate` by default?

## Recommended next step

Do not change model objective yet.

First implement diagnostic labels/artifacts only:

```text
PathOutcome → PathBehavior → PathUtility → behavior distribution artifacts
```

Then inspect whether current selected down candidates are mostly:

```text
clean_down
```

or:

```text
down_first_both / chop_both
```

Only after that choose training objectives and final policy gates.
