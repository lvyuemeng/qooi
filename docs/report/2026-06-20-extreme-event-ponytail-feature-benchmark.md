# Extreme-event ponytail feature benchmark

## Scope

Tested one compact, non-sin/cos feature block for the current local advanced target:

```toml
max_symbols = 160
threshold_pct = 30.0
outcome_horizon = [24]
max_trials = 5
```

The feature block added market-mechanism columns only:

```text
return_6bar
return_12bar
return_48bar
return_72bar
realized_vol_24bar
realized_vol_72bar
vol_compression_24_168
volume_anomaly_72bar
range_position_168bar
range_position_720bar
range_compression_48_720
oi_change_24h_pct
taker_buy_sell_ratio_delta_24h
long_short_ratio_delta_24h
```

No sin/cos time sampling was used.
No new diagnostics were added.
Comparison used the existing `tailtree-selection-efficiency.csv` surface.

## Runs

### Baseline

```text
current code + current local advanced config
runtime: 212s
artifact: data/output/potential/benchmarks/baseline-tailtree-selection-efficiency.csv
```

### Improved attempt

```text
same config + compact feature block
runtime: 210s
artifact: data/output/potential/benchmarks/improved-tailtree-selection-efficiency.csv
```

## Aggregate comparison

| Metric | Baseline | Feature block | Delta |
|---|---:|---:|---:|
| feature_count | 20 | 34 | +14 |
| passing_rows | 51 | 33 | -18 |
| best_hpo_score | 22.297248 | 16.991378 | -23.80% |
| best_valid_tail_lift | 18.613551 | 13.283620 | -28.64% |
| best_profit_proxy_per_selected_obs | 2.820659 | 2.272787 | -19.42% |
| max_selected_tail_count | 1650 | 1940 | +290 |

## Best-row comparison

| Metric | Baseline best | Feature-block best | Delta |
|---|---:|---:|---:|
| selected_observation_count | 2189 | 2189 | 0 |
| selected_tail_count | 165 | 510 | +345 |
| valid_tail_lift | 18.613551 | 12.662890 | -31.97% |
| selected_utility_mean | 2.395288 | 2.067958 | -13.67% |
| selected_utility_p90 | 10.949607 | 6.581315 | -39.89% |
| profit_proxy_per_selected_obs | 2.395288 | 2.067958 | -13.67% |
| hpo_score | 22.297248 | 16.991378 | -23.80% |

## Decision

Rejected and reverted.

Ponytail rule: if a feature block adds 14 training columns but worsens the existing selection-efficiency surface, do not keep it.

## Interpretation

The added features increased event support in the best selected bucket, but diluted concentration and utility:

```text
more selected tails
lower lift
lower utility p90
lower HPO score
fewer passing rows
```

So the issue is not “more market information is always better”. For this current `30% / h24` objective, the extra multi-window columns likely gave LightGBM more ways to split common volatility regimes, but not better extreme-event concentration.

## Next ponytail direction

Do not add another broad feature block.

Next useful improvement should target one structural gap, not feature count:

```text
objective: train event-lift over all observations, then utility over tails
```

or:

```text
horizon: compare h24 vs h48/h72 for 30% event formation
```

Those address the current failure more directly than adding more rolling columns.
