# Directional HPO replay penalty implementation

## Goal

Apply the corrected direction → threshold → utility design without changing model features or labels.

Keep:

```text
binary up/down threshold-event concentration training
```

Patch only:

```text
selection-efficiency / HPO replay score
```

## Why this is architecture-consistent

Direction and threshold remain in outcome labels:

```text
tail_up
tail_down
```

Known-at-close model features are unchanged.

Utility and false/opposite direction are used only in replay scoring.

## First ponytail rule

Use same-horizon opposite evidence only:

```text
h24 up vs h24 down
h48 up vs h48 down
```

Do not handle cross-horizon conflict in this pass.

## Score formula

For each evidence row:

```text
base_score = selected_side_lift + selected_side_utility + sqrt(selected_tail_count) / 10
opposite_quality = opposite_side_lift + opposite_side_utility
same_horizon_gray_zone = selected side material and opposite side material
hpo_score = base_score - opposite_quality - 5.0 * gray_zone_flag
```

Material means:

```text
selected_observations >= 500
valid_tail_lift >= 3.0
```

## Acceptance

Run fast scan and compare against conflict-abstain baseline.

Keep only if clean promotion quality improves or remains flat without worsening lift/utility materially.
