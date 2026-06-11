# Tailtree Module Graph

`qooi.scanner.tailtree` is the LightGBM + GPD extreme-value evidence path. It consumes scanner observation/outcome frames and emits tree models plus per-leaf numeric evidence.

Design doc: `docs/architecture/scanner.md`.

---

## Dependency contract

Optional dependencies only for this path:

```toml
[project.optional-dependencies]
tailtree = [
  "lightgbm>=4.0",
  "scipy>=1.11",
]
```

Rules:

- no pandas/pyarrow dependency is required for the tailtree path;
- use Polars → numpy matrices for LightGBM input;
- pydantic validates config/metadata serialization boundaries;
- model persistence is JSON: `booster.model_to_string()` + metadata JSON.

---

## Statistical role of GPD

GPD models threshold exceedance severity only.

```text
up tail:   forward_max_return_pct >= threshold_pct
up excess: forward_max_return_pct - threshold_pct

down tail:   forward_min_return_pct <= -threshold_pct
down excess: abs(forward_min_return_pct) - threshold_pct
```

Division of labor:

| Quantity | Population | Purpose |
|---|---|---|
| tree split training | tail rows only | partition extreme events by exceedance behavior |
| GPD fit | tail rows only | estimate tail shape/scale severity |
| `N_total` | all rows assigned to leaf | denominator for leaf frequency |
| `tail_rate` | all rows assigned to leaf | probability of entering tail |
| `tail_lift` | leaf rate vs global all-row rate | concentration of tail frequency |

GPD is not a full-return model and not an up/down/flat classifier.

---

## Public data products

### Labeled outcome frame

```text
qooi.scanner.tailtree.label_tail_exceedances(
    outcome_frame: pl.DataFrame,
    *,
    threshold_pct: float,
) -> pl.DataFrame
```

Adds:

```text
tail_up: bool
tail_down: bool
tail_exceedance_value_up: float
tail_exceedance_value_down: float
```

### Tailtree training frame

Public helper:

```text
qooi.scanner.tailtree.tailtree_training_frame(
    observations: pl.DataFrame,
    labeled_outcomes: pl.DataFrame,
    *,
    direction: Literal["up", "down"],
) -> TailtreeTrainingFrame
```

Contract:

```text
TailtreeTrainingFrame
├── direction
├── all_observations        # denominator population
├── tail_observations       # tree/GPD training population
├── exceedance_values       # positive threshold excesses
└── global_tail_rate        # len(tail_observations) / len(all_observations)
```

Training uses `tail_observations`; diagnostics project `all_observations` through the trained model.

---

## Model classes

### `qooi.scanner.tailtree.TrainConfig`

```text
num_leaves: int
min_data_in_leaf: int
learning_rate: float
num_iterations: int
early_stopping_rounds: int
validation_fraction: float
random_seed: int
```

### `qooi.scanner.tailtree.GPDParams`

```text
xi: float          # shape; bounded by validation
sigma: float       # scale; positive
tail_rate: float   # frequency diagnostic, not severity
```

### `qooi.scanner.tailtree.TreeMetadata`

```text
direction: "up" | "down"
num_leaves_actual: int
categorical_features: list[str]
continuous_features: list[str]
global_baseline: GPDParams
leaf_params: dict[int, GPDParams]
feature_importance: list[tuple[str, float]]
train_config: TrainConfig
train_timestamp: str
train_n_observations: int
train_n_exceedances: int
```

No `leaf_paths`: recursive path extraction is cosmetic and not part of the model/evidence contract.

### `qooi.scanner.tailtree.TailTreeModel`

```text
TailTreeModel.train(
    features: pl.DataFrame,              # tail rows for tree/GPD training
    exceedance_values: Sequence[float],  # positive exceedances, same length
    *,
    config: TrainConfig,
    categorical_features: list[str],
    continuous_features: list[str],
    direction: Literal["up", "down"],
    global_tail_rate: float,
    train_n_observations: int,
) -> TailTreeModel

TailTreeModel.predict_leaf(features: pl.DataFrame) -> pl.DataFrame
TailTreeModel.predict_leaf_params(features: pl.DataFrame) -> pl.DataFrame
TailTreeModel.to_json(path) -> None
TailTreeModel.from_json(path) -> TailTreeModel
```

`global_tail_rate` and `train_n_observations` come from the all-row training frame. The model is trained on tail rows, but its baseline frequency is not computed from tail rows alone.

---

## Training graph per direction

```text
labeled_outcomes = label_tail_exceedances(outcomes, threshold_pct)
training_frame   = tailtree_training_frame(observations, labeled_outcomes, direction)

all_n       = training_frame.train_n_observations
tail_train  = training_frame.tail_observations
excess      = training_frame.exceedance_values
global_rate = training_frame.global_tail_rate

model = TailTreeModel.train(
    tail_train,
    excess,
    config=TrainConfig(...),
    categorical_features=present_categoricals,
    continuous_features=present_continuous,
    direction=direction,
    global_tail_rate=global_rate,
    train_n_observations=all_n,
)
```

---

## Leaf diagnostics graph

```text
qooi.scanner.tailtree.leaf_evidence_frame(
    tree: TailTreeModel,
    observations: pl.DataFrame,
    outcomes: pl.DataFrame,
    *,
    recent_window_days: int = 30,
) -> pl.DataFrame
```

Implementation shape:

```text
1. with_leaf = tree.predict_leaf(observations)
2. collapse market/source duplicate outcomes by (symbol, decision_bar_close_ms), preserving any source tail labels
3. group by leaf_id over all rows:
     N_total
     N_tail_exceedances
     tail_rate = N_tail_exceedances / N_total
     tail_lift = tail_rate / tree.metadata.global_baseline.tail_rate
4. join per-leaf GPD params from tree metadata
5. add recent-window stability metrics
```

Leaf evidence and leaf context must use the same decision-key outcome aggregation so market baseline rows cannot erase source tail labels.

---

## Selection graph

```text
qooi.scanner.tailtree.select_tail_leaves(leaf_evidence) -> pl.DataFrame
```

Default hard gate:

```text
N_tail_exceedances >= 30
tail_lift >= 1.5
tail_lift_stability numeric threshold when available
```

Output semantics:

```text
selection_mode = "hard_gate"       # at least one leaf passed the hard gate
selection_mode = "best_available"  # no leaf passed; top leaves are written for inspection
selected_evidence_level = true      # promoted evidence row
selected_evidence_level = false     # fallback row, not promoted
```

Selection is a research filter, not a trading authorization.

---

## Persistence graph

```text
model.to_json("tail-tree-up.json")
  → {
      "lightgbm_model": booster.model_to_string(),
      "metadata": TreeMetadata.model_dump(mode="json"),
    }

TailTreeModel.from_json("tail-tree-up.json")
  → TreeMetadata.model_validate(...)
  → lgb.Booster(model_str=...)
  → TailTreeModel(...)
```

---

## Removed/stale surfaces

Do not reintroduce:

```text
leaf_path_to_text(...)
TreeMetadata.leaf_paths
_walk_node(...)
Booster.trees_to_dataframe()        # pandas surface
features.to_pandas()                # pyarrow/pandas dependency path
lgb.train(..., fobj=...)            # LightGBM 4 uses params["objective"]
```
