"""LightGBM + GPD tail-detection tree.

Optional dependency group: [tailtree] = lightgbm, scipy.
All imports are lazy; import this module directly only when evidence="tailtree".
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
import polars as pl
from pydantic import BaseModel, Field

# ── pydantic models ──────────────────────────────────────────────────────────


class TrainConfig(BaseModel):
    """Training hyperparameters. Validated on construction."""

    objective: Literal[
        "tail_severity_gpd",
        "tail_utility_quantile",
        "tail_event_lift",
        "tail_any_event",
        "tail_side_only",
    ] = "tail_severity_gpd"
    num_leaves: int = Field(default=64, ge=8, le=256)
    min_data_in_leaf: int = Field(default=30, ge=10, le=500)
    learning_rate: float = Field(default=0.05, gt=0, le=1.0)
    num_iterations: int = Field(default=200, ge=10, le=2000)
    early_stopping_rounds: int = Field(default=20, ge=5, le=100)
    validation_fraction: float = Field(default=0.2, ge=0.05, le=0.5)
    random_seed: int = Field(default=42, ge=0)


class GPDParams(BaseModel):
    """Fitted GPD parameters for one leaf or the global baseline."""

    xi: float = Field(ge=-0.2, le=0.6)
    sigma: float = Field(gt=0)
    tail_rate: float = Field(ge=0, le=1.0)


class TreeMetadata(BaseModel):
    """Serializable metadata stored alongside the LightGBM booster string."""

    direction: Literal["up", "down"]
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


class TailTreePayload(BaseModel):
    """JSON artifact payload for one serialized tailtree model."""

    lightgbm_model: str
    metadata: TreeMetadata


@dataclass(frozen=True)
class TailtreeTrainingFrame:
    """Direction-specific tailtree data product.

    `tail_observations` trains the LightGBM/GPD tree. `all_observations`
    supplies denominators for tail rate and lift diagnostics.
    """

    direction: Literal["up", "down"]
    all_observations: pl.DataFrame
    tail_observations: pl.DataFrame
    exceedance_values: np.ndarray
    utility_values: np.ndarray
    global_tail_rate: float

    @property
    def train_n_observations(self) -> int:
        return len(self.all_observations)

    @property
    def train_n_exceedances(self) -> int:
        return len(self.tail_observations)

    def has_min_exceedances(self, min_count: int) -> bool:
        return self.train_n_exceedances >= min_count


def _leaf_id_vector(pred_leaf: object) -> np.ndarray:
    """Return one leaf id per row from LightGBM pred_leaf output."""
    leaves = np.asarray(pred_leaf)
    if leaves.ndim == 2:
        leaves = leaves[:, -1]
    return leaves.astype("int32").ravel()


@dataclass
class TailTreeModel:
    """Trained tail-detection tree.

    Construct via TailTreeModel.train() or TailTreeModel.from_json().
    """

    booster: str
    metadata: TreeMetadata

    # ── static factory: train ────────────────────────────────────────────────

    @staticmethod
    def train(
        features: pl.DataFrame,
        exceedance_values: np.ndarray,
        *,
        config: TrainConfig | dict,
        categorical_features: list[str],
        continuous_features: list[str],
        direction: Literal["up", "down"],
        global_tail_rate: float | None = None,
        train_n_observations: int | None = None,
        utility_values: np.ndarray | None = None,
    ) -> TailTreeModel:
        """Train a LightGBM tree with GPD-based objective on tail exceedances.

        Args:
            features: Observation rows with categorical + continuous columns.
            exceedance_values: Positive floats > threshold_pct, shape (n,).
            config: TrainConfig or dict of hyperparameters.
            categorical_features: Column names for LightGBM categorical handling.
            continuous_features: Column names for continuous splits.
            direction: "up" (upper tail) or "down" (lower tail).
        """
        import lightgbm as lgb
        import scipy.stats

        if isinstance(config, dict):
            config = TrainConfig.model_validate(config)

        if len(exceedance_values) != len(features):
            raise ValueError(
                "exceedance_values length "
                f"{len(exceedance_values)} != features length {len(features)}"
            )
        if len(exceedance_values) < config.min_data_in_leaf:
            raise ValueError(
                f"Not enough exceedances ({len(exceedance_values)}) for "
                f"min_data_in_leaf={config.min_data_in_leaf}"
            )
        utility_values = (
            utility_values.astype(float) if utility_values is not None else exceedance_values
        )
        if len(utility_values) != len(features):
            raise ValueError(
                f"utility_values length {len(utility_values)} != features length {len(features)}"
            )

        is_binary_objective = config.objective in {
            "tail_event_lift",
            "tail_any_event",
            "tail_side_only",
        }
        event_count = (
            int(np.sum(exceedance_values > 0.0))
            if is_binary_objective
            else len(exceedance_values)
        )

        # 1. Global tail baseline. Event-lift uses binary labels, not a severity GPD.
        if is_binary_objective:
            xi_global, sigma_global = 0.0, 1.0
        else:
            xi_global, _, sigma_global = scipy.stats.genpareto.fit(exceedance_values, floc=0)
        train_n_observations = train_n_observations or len(features)
        global_tail_rate = (
            global_tail_rate if global_tail_rate is not None else event_count / len(features)
        )
        global_baseline = GPDParams(
            xi=float(np.clip(xi_global, -0.2, 0.6)),
            sigma=float(max(sigma_global, 1e-6)),
            tail_rate=float(global_tail_rate),
        )

        # 2. Build LightGBM datasets from numpy (no pandas dependency)
        all_cols = categorical_features + continuous_features
        present_cols = [c for c in all_cols if c in features.columns]
        x = features.select(present_cols).to_numpy()
        # Map categorical columns to integer codes for LightGBM
        cat_indices = []
        for col in categorical_features:
            if col in features.columns:
                cat_indices.append(present_cols.index(col))
                # Convert string categorical to integer codes
                col_data = features.get_column(col)
                if col_data.dtype == pl.String:
                    codes = col_data.cast(pl.Categorical).to_physical().to_numpy()
                    idx = present_cols.index(col)
                    x[:, idx] = codes.astype(np.float64)

        n_valid = max(1, int(len(x) * config.validation_fraction))
        x_train, x_valid = x[:-n_valid], x[-n_valid:]
        if is_binary_objective:
            target_values = (exceedance_values > 0.0).astype(float)
        elif config.objective == "tail_utility_quantile":
            target_values = np.log1p(np.maximum(utility_values, 0.0))
        else:
            target_values = exceedance_values
        y_train, y_valid = target_values[:-n_valid], target_values[-n_valid:]

        train_data = lgb.Dataset(
            x_train,
            label=y_train,
            categorical_feature=cat_indices if cat_indices else "auto",
        )
        valid_data = lgb.Dataset(
            x_valid,
            label=y_valid,
            categorical_feature=cat_indices if cat_indices else "auto",
            reference=train_data,
        )

        # 3. Train LightGBM
        is_gpd_objective = config.objective == "tail_severity_gpd"
        params = {
            "objective": "binary"
            if is_binary_objective
            else (_gpd_xi_objective if is_gpd_objective else "quantile"),
            "metric": "binary_logloss"
            if is_binary_objective
            else ("None" if is_gpd_objective else "quantile"),
            "alpha": 0.8,
            "num_leaves": config.num_leaves,
            "min_data_in_leaf": config.min_data_in_leaf,
            "learning_rate": config.learning_rate,
            "verbosity": -1,
            "num_threads": 4,
            "seed": config.random_seed,
            "deterministic": True,
        }
        booster = lgb.train(
            params,
            train_data,
            num_boost_round=config.num_iterations,
            valid_sets=[valid_data],
            valid_names=["valid"],
            feval=_gpd_nll_eval if is_gpd_objective else None,
            callbacks=[
                lgb.early_stopping(config.early_stopping_rounds),
                lgb.log_evaluation(0),
            ],
        )

        # 4. Per-leaf GPD fit
        leaf_ids = _leaf_id_vector(booster.predict(x_train, pred_leaf=True))
        leaf_params: dict[int, GPDParams] = {}
        for lid in np.unique(leaf_ids):
            mask = leaf_ids == lid
            leaf_y = y_train[mask]
            if is_binary_objective:
                xi_l, sigma_l = 0.0, 1.0
            elif len(leaf_y) >= 10:
                xi_l, _, sigma_l = scipy.stats.genpareto.fit(leaf_y, floc=0)
            else:
                xi_l, sigma_l = xi_global, sigma_global
            leaf_tail_contribution = len(leaf_y) / train_n_observations
            leaf_params[int(lid)] = GPDParams(
                xi=float(np.clip(xi_l, -0.2, 0.6)),
                sigma=float(max(sigma_l, 1e-6)),
                tail_rate=float(leaf_tail_contribution),
            )

        # 5. Feature importance
        importance = booster.feature_importance(importance_type="gain")
        feature_importance = sorted(
            [
                (present_cols[i], float(importance[i]))
                for i in range(min(len(present_cols), len(importance)))
            ],
            key=lambda x: x[1],
            reverse=True,
        )
        return TailTreeModel(
            booster=booster.model_to_string(),
            metadata=TreeMetadata(
                direction=direction,
                num_leaves_actual=booster.num_trees(),
                categorical_features=categorical_features,
                continuous_features=continuous_features,
                global_baseline=global_baseline,
                leaf_params=leaf_params,
                feature_importance=feature_importance,
                train_config=config,
                train_timestamp=datetime.now(UTC).isoformat(),
                train_n_observations=train_n_observations,
                train_n_exceedances=event_count,
            ),
        )

    # ── prediction ───────────────────────────────────────────────────────────

    @property
    def _booster(self) -> Any:
        """Reconstruct lightgbm Booster from stored model string."""
        import lightgbm as lgb

        return lgb.Booster(model_str=self.booster)

    def _feature_matrix(self, features: pl.DataFrame) -> np.ndarray:
        all_cols = self.metadata.categorical_features + self.metadata.continuous_features
        present = [c for c in all_cols if c in features.columns]
        x = features.select(present).to_numpy()
        for col in self.metadata.categorical_features:
            if col in features.columns and features[col].dtype == pl.String:
                codes = features.get_column(col).cast(pl.Categorical).to_physical().to_numpy()
                x[:, present.index(col)] = codes.astype(np.float64)
        return x

    def predict_leaf(self, features: pl.DataFrame) -> pl.DataFrame:
        """Assign each row to a leaf. Returns features + 'leaf_id' (Int32)."""
        leaf_ids = _leaf_id_vector(
            self._booster.predict(self._feature_matrix(features), pred_leaf=True)
        )
        return features.with_columns(pl.Series("leaf_id", leaf_ids))

    def predict_score(self, features: pl.DataFrame) -> pl.DataFrame:
        """Score each row with the full boosted model ensemble."""
        scores = (
            np.asarray(self._booster.predict(self._feature_matrix(features)))
            .astype(float)
            .ravel()
        )
        return features.with_columns(pl.Series("tailtree_score", scores))

    def predict_leaf_params(self, features: pl.DataFrame) -> pl.DataFrame:
        """Predict leaf and join per-leaf GPD params. Adds xi, sigma, tail_rate."""
        with_leaf = self.predict_leaf(features)
        rows = [
            {"leaf_id": lid, "gpd_xi": p.xi, "gpd_sigma": p.sigma, "leaf_tail_rate": p.tail_rate}
            for lid, p in self.metadata.leaf_params.items()
        ]
        leaf_df = pl.DataFrame(
            rows,
            schema={
                "leaf_id": pl.Int32,
                "gpd_xi": pl.Float64,
                "gpd_sigma": pl.Float64,
                "leaf_tail_rate": pl.Float64,
            },
        )
        return with_leaf.join(leaf_df, on="leaf_id", how="left")

    # ── persistence ──────────────────────────────────────────────────────────

    def to_json(self, path: str | Path) -> None:
        """Serialize booster + metadata as one JSON file."""
        payload = TailTreePayload(lightgbm_model=self.booster, metadata=self.metadata)
        Path(path).write_text(payload.model_dump_json(indent=2), encoding="utf-8")

    @classmethod
    def from_json(cls, path: str | Path) -> TailTreeModel:
        """Load from JSON. pydantic validates metadata on construction."""
        payload = TailTreePayload.model_validate_json(Path(path).read_text(encoding="utf-8"))
        return TailTreeModel(booster=payload.lightgbm_model, metadata=payload.metadata)


# ── LightGBM custom objective ────────────────────────────────────────────────


def _gpd_xi_objective(
    preds: np.ndarray,
    train_data: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Custom LightGBM objective: GPD NLL gradient w.r.t. xi.

    Scalar preds → ξ ∈ (−0.2, 0.6) via scaled sigmoid.
    σ held at global median during training.
    Per-leaf (ξ, σ) fit post-hoc via scipy MLE.
    """
    y = train_data.get_label()
    xi = 1.0 / (1.0 + np.exp(-preds)) * 0.8 - 0.2
    sigma = float(np.median(y))

    z = 1.0 + xi * y / sigma
    valid = (z > 0) & (sigma > 0)

    # d(NLL)/dξ
    d_nll_d_xi = np.where(
        valid,
        -y / (sigma * z) * (1.0 + 1.0 / np.maximum(np.abs(xi), 1e-8))
        + np.log(np.maximum(z, 1e-8)) / np.maximum(xi * xi, 1e-8),
        0.0,
    )
    # dξ/dpreds (chain rule through scaled sigmoid)
    xi_raw = (xi + 0.2) / 0.8
    d_xi_d_preds = xi_raw * (1.0 - xi_raw) * 0.8

    grad = d_nll_d_xi * d_xi_d_preds
    grad = np.where(valid, grad, 0.0)
    hess = np.abs(grad) + 1e-6

    return grad.astype(np.float64), hess.astype(np.float64)


def _gpd_nll_eval(
    preds: np.ndarray,
    train_data: Any,
) -> tuple[str, float, bool]:
    """LightGBM eval metric for early stopping."""
    y = train_data.get_label()
    xi = 1.0 / (1.0 + np.exp(-preds)) * 0.8 - 0.2
    sigma = float(np.median(y))

    z = 1.0 + xi * y / sigma
    valid = z > 0

    nll = np.where(
        valid,
        np.log(sigma) + (1.0 + 1.0 / np.maximum(np.abs(xi), 1e-8)) * np.log(np.maximum(z, 1e-8)),
        1e10,
    )
    return ("gpd_nll", float(np.mean(nll)), False)


# ── evidence functions ───────────────────────────────────────────────────────


def label_tail_paths(
    outcome_frame: pl.DataFrame,
    *,
    threshold_pct: float = 5.0,
    utility_floor: float = 0.0,
    margin_floor: float = 0.0,
    path_efficiency_floor: float = 0.0,
    late_bar_ratio: float | None = None,
) -> pl.DataFrame:
    """Label fixed-horizon path behavior and side utility."""
    has_max = "forward_max_return_pct" in outcome_frame.columns
    has_min = "forward_min_return_pct" in outcome_frame.columns

    retention = (
        pl.col("close_retention_ratio").cast(pl.Float64).clip(0.0, 1.0)
        if "close_retention_ratio" in outcome_frame.columns
        else pl.lit(1.0)
    )
    efficiency = (
        pl.col("path_efficiency").cast(pl.Float64).clip(0.0, 1.0)
        if "path_efficiency" in outcome_frame.columns
        else pl.lit(1.0)
    )
    max_speed = (
        1.0 / (1.0 + pl.col("time_to_max_bar").cast(pl.Float64).fill_null(0.0)).sqrt()
        if "time_to_max_bar" in outcome_frame.columns
        else pl.lit(1.0)
    )
    min_speed = (
        1.0 / (1.0 + pl.col("time_to_min_bar").cast(pl.Float64).fill_null(0.0)).sqrt()
        if "time_to_min_bar" in outcome_frame.columns
        else pl.lit(1.0)
    )
    max_drawdown_penalty = (
        pl.col("post_max_drawdown_pct").cast(pl.Float64).fill_null(0.0).clip(0.0, None)
        if "post_max_drawdown_pct" in outcome_frame.columns
        else pl.lit(0.0)
    )
    min_rebound_penalty = (
        pl.col("post_min_rebound_pct").cast(pl.Float64).fill_null(0.0).clip(0.0, None)
        if "post_min_rebound_pct" in outcome_frame.columns
        else pl.lit(0.0)
    )

    exprs = []
    if has_max:
        exprs.extend(
            [
                (pl.col("forward_max_return_pct").cast(pl.Float64) > threshold_pct).alias(
                    "tail_up"
                ),
                pl.when(pl.col("forward_max_return_pct").cast(pl.Float64) > threshold_pct)
                .then(pl.col("forward_max_return_pct").cast(pl.Float64) - threshold_pct)
                .otherwise(None)
                .alias("tail_exceedance_value_up"),
                pl.when(pl.col("forward_max_return_pct").cast(pl.Float64) > threshold_pct)
                .then(
                    (
                        (pl.col("forward_max_return_pct").cast(pl.Float64) - threshold_pct)
                        * retention
                        * efficiency
                        * max_speed
                        - 0.1 * max_drawdown_penalty
                    ).clip(0.0, None)
                )
                .otherwise(0.0)
                .alias("tail_utility_up"),
            ]
        )
    if has_min:
        exprs.extend(
            [
                (pl.col("forward_min_return_pct").cast(pl.Float64) < -threshold_pct).alias(
                    "tail_down"
                ),
                pl.when(pl.col("forward_min_return_pct").cast(pl.Float64) < -threshold_pct)
                .then(pl.col("forward_min_return_pct").cast(pl.Float64).abs() - threshold_pct)
                .otherwise(None)
                .alias("tail_exceedance_value_down"),
                pl.when(pl.col("forward_min_return_pct").cast(pl.Float64) < -threshold_pct)
                .then(
                    (
                        (pl.col("forward_min_return_pct").cast(pl.Float64).abs() - threshold_pct)
                        * retention
                        * efficiency
                        * min_speed
                        - 0.1 * min_rebound_penalty
                    ).clip(0.0, None)
                )
                .otherwise(0.0)
                .alias("tail_utility_down"),
            ]
        )

    if not exprs:
        return outcome_frame

    labeled = outcome_frame.with_columns(exprs)
    if "tail_up" not in labeled.columns or "tail_down" not in labeled.columns:
        return labeled

    up = pl.col("tail_up").fill_null(False).cast(pl.Boolean)
    down = pl.col("tail_down").fill_null(False).cast(pl.Boolean)
    up_utility = pl.col("tail_utility_up").fill_null(0.0).cast(pl.Float64)
    down_utility = pl.col("tail_utility_down").fill_null(0.0).cast(pl.Float64)
    up_margin = up_utility - down_utility
    down_margin = down_utility - up_utility
    if "time_to_max_bar" in labeled.columns:
        time_to_max = pl.col("time_to_max_bar").cast(pl.Float64).fill_null(0.0)
    else:
        time_to_max = pl.lit(0.0)
    if "time_to_min_bar" in labeled.columns:
        time_to_min = pl.col("time_to_min_bar").cast(pl.Float64).fill_null(0.0)
    else:
        time_to_min = pl.lit(0.0)
    if "path_efficiency" in labeled.columns:
        efficient_path = (
            pl.col("path_efficiency").cast(pl.Float64).fill_null(0.0) >= path_efficiency_floor
        )
    else:
        efficient_path = pl.lit(True)
    horizon = (
        pl.col("outcome_horizon").cast(pl.Float64).fill_null(0.0)
        if "outcome_horizon" in labeled.columns
        else pl.lit(0.0)
    )
    late_limit = pl.lit(None) if late_bar_ratio is None else horizon * float(late_bar_ratio)
    late_up = up & ~down & late_limit.is_not_null() & (time_to_max > late_limit)
    late_down = down & ~up & late_limit.is_not_null() & (time_to_min > late_limit)
    first_up = up & (~down | (time_to_max < time_to_min))
    first_down = down & (~up | (time_to_min < time_to_max))
    first_touch = (
        pl.when(first_up)
        .then(pl.lit("up"))
        .when(first_down)
        .then(pl.lit("down"))
        .when(up & down)
        .then(pl.lit("tie"))
        .otherwise(pl.lit("none"))
    )
    clean_up = up & ~down & efficient_path & (up_utility >= utility_floor)
    clean_down = down & ~up & efficient_path & (down_utility >= utility_floor)
    weak_both = up & down & (
        ~efficient_path | ((up_margin.abs() < margin_floor) & (down_margin.abs() < margin_floor))
    )
    path_state = (
        pl.when(late_up)
        .then(pl.lit("late_up"))
        .when(late_down)
        .then(pl.lit("late_down"))
        .when(clean_up)
        .then(pl.lit("clean_up"))
        .when(clean_down)
        .then(pl.lit("clean_down"))
        .when(weak_both)
        .then(pl.lit("chop_both"))
        .when(up & down & first_up)
        .then(pl.lit("up_first_both"))
        .when(up & down & first_down)
        .then(pl.lit("down_first_both"))
        .otherwise(pl.lit("none"))
    )
    path_actionability = (
        pl.when(clean_up & (up_margin >= margin_floor))
        .then(pl.lit("tradable_up"))
        .when(clean_down & (down_margin >= margin_floor))
        .then(pl.lit("tradable_down"))
        .when(up & down)
        .then(pl.lit("reversal_watch"))
        .when(up | down)
        .then(pl.lit("gray_zone"))
        .otherwise(pl.lit("no_action"))
    )
    return labeled.with_columns(
        up.alias("tail_any_up"),
        down.alias("tail_any_down"),
        up.alias("tail_touch_up"),
        down.alias("tail_touch_down"),
        (up | down).alias("tail_any"),
        (up | down).alias("tail_touch_any"),
        (up & down).alias("tail_both"),
        (up & down).alias("tail_touch_both"),
        pl.when(up & down)
        .then(pl.lit("both"))
        .when(up)
        .then(pl.lit("up"))
        .when(down)
        .then(pl.lit("down"))
        .otherwise(pl.lit("none"))
        .alias("tail_state"),
        first_touch.alias("first_touch_side"),
        path_state.alias("path_state"),
        path_actionability.alias("path_actionability"),
        pl.when(path_actionability == "no_action")
        .then(pl.lit("no_tail_touch"))
        .when(path_actionability == "gray_zone")
        .then(pl.lit("weak_path_utility"))
        .when(path_actionability == "reversal_watch")
        .then(pl.lit("both_or_mixed_path"))
        .otherwise(pl.lit(""))
        .alias("path_blocker"),
        up_utility.alias("path_utility_up"),
        down_utility.alias("path_utility_down"),
        up_margin.alias("tail_utility_margin_up"),
        down_margin.alias("tail_utility_margin_down"),
        up_margin.alias("path_utility_margin_up"),
        down_margin.alias("path_utility_margin_down"),
    )


def tailtree_label_distribution_frame(labeled_outcomes: pl.DataFrame) -> pl.DataFrame:
    """Summarize orthogonal tail-state prevalence by horizon."""
    if labeled_outcomes.is_empty() or "tail_state" not in labeled_outcomes.columns:
        return pl.DataFrame()
    total_by_horizon = labeled_outcomes.group_by("outcome_horizon").agg(
        pl.len().alias("horizon_row_count")
    )
    return (
        labeled_outcomes.group_by("outcome_horizon", "tail_state")
        .agg(
            pl.len().alias("row_count"),
            pl.col("tail_up").fill_null(False).sum().alias("tail_up_count"),
            pl.col("tail_down").fill_null(False).sum().alias("tail_down_count"),
            pl.col("tail_any").fill_null(False).sum().alias("tail_any_count"),
            pl.col("tail_both").fill_null(False).sum().alias("tail_both_count"),
            (pl.col("tail_state") == "up").sum().alias("tail_state_up_count"),
            (pl.col("tail_state") == "down").sum().alias("tail_state_down_count"),
            pl.col("tail_utility_up")
            .fill_null(0.0)
            .mean()
            .alias("tail_utility_up_mean"),
            pl.col("tail_utility_down")
            .fill_null(0.0)
            .mean()
            .alias("tail_utility_down_mean"),
            pl.col("tail_utility_margin_up")
            .fill_null(0.0)
            .mean()
            .alias("tail_utility_margin_up_mean"),
        )
        .join(total_by_horizon, on="outcome_horizon", how="left")
        .with_columns((pl.col("row_count") / pl.col("horizon_row_count")).alias("class_rate"))
        .drop("horizon_row_count")
        .sort("outcome_horizon", "tail_state")
    )


def tailtree_training_frame(
    observations: pl.DataFrame,
    labeled_outcomes: pl.DataFrame,
    *,
    direction: Literal["up", "down"],
) -> TailtreeTrainingFrame:
    """Build the direction-specific tailtree training data product.

    The returned frame keeps two populations separate:
    all aligned observations for denominator diagnostics, and tail-only rows for
    LightGBM/GPD training.
    """
    empty_obs = observations.head(0)
    exceed_col = f"tail_exceedance_value_{direction}"
    utility_col = f"tail_utility_{direction}"
    tail_col = f"tail_{direction}"
    if (
        observations.is_empty()
        or labeled_outcomes.is_empty()
        or exceed_col not in labeled_outcomes.columns
    ):
        return TailtreeTrainingFrame(
            direction=direction,
            all_observations=empty_obs,
            tail_observations=empty_obs,
            exceedance_values=np.array([], dtype=float),
            utility_values=np.array([], dtype=float),
            global_tail_rate=0.0,
        )

    all_keys = labeled_outcomes.select("symbol", "decision_bar_close_ms").unique()
    all_observations = observations.join(
        all_keys,
        on=["symbol", "decision_bar_close_ms"],
        how="inner",
    )
    if all_observations.is_empty() or tail_col not in labeled_outcomes.columns:
        return TailtreeTrainingFrame(
            direction=direction,
            all_observations=all_observations,
            tail_observations=empty_obs,
            exceedance_values=np.array([], dtype=float),
            utility_values=np.array([], dtype=float),
            global_tail_rate=0.0,
        )

    selected_cols = ["symbol", "decision_bar_close_ms", exceed_col]
    if utility_col in labeled_outcomes.columns:
        selected_cols.append(utility_col)
    tail_outcomes = (
        labeled_outcomes.filter(pl.col(tail_col).fill_null(False))
        .select(selected_cols)
        .unique(subset=["symbol", "decision_bar_close_ms"], keep="first")
    )
    tail_observations = observations.join(
        tail_outcomes,
        on=["symbol", "decision_bar_close_ms"],
        how="inner",
    ).filter(pl.col(exceed_col).is_not_null())
    exceedance_values = (
        tail_observations.get_column(exceed_col).to_numpy()
        if not tail_observations.is_empty()
        else np.array([], dtype=float)
    )
    utility_values = (
        tail_observations.get_column(utility_col).fill_null(0.0).to_numpy()
        if utility_col in tail_observations.columns and not tail_observations.is_empty()
        else exceedance_values
    )
    global_tail_rate = (
        len(tail_observations) / len(all_observations) if not all_observations.is_empty() else 0.0
    )
    return TailtreeTrainingFrame(
        direction=direction,
        all_observations=all_observations,
        tail_observations=tail_observations,
        exceedance_values=exceedance_values,
        utility_values=utility_values,
        global_tail_rate=global_tail_rate,
    )


TailtreeBinaryTarget = Literal["tail_event_lift", "tail_any_event", "tail_side_only"]


def tailtree_target_training_values(
    observations: pl.DataFrame,
    labeled_outcomes: pl.DataFrame,
    *,
    target: TailtreeBinaryTarget,
    direction: Literal["up", "down"],
) -> tuple[pl.DataFrame, np.ndarray, np.ndarray]:
    """Build target-specific binary labels through one training surface."""
    if observations.is_empty() or labeled_outcomes.is_empty():
        return observations.head(0), np.array([], dtype=float), np.array([], dtype=float)

    if target == "tail_event_lift":
        label_col = f"tail_{direction}"
        utility_col = f"tail_utility_{direction}"
        if label_col not in labeled_outcomes.columns:
            return observations.head(0), np.array([], dtype=float), np.array([], dtype=float)
        utility_expr = (
            pl.col(utility_col).fill_null(0.0).cast(pl.Float64).max().alias(utility_col)
            if utility_col in labeled_outcomes.columns
            else pl.lit(0.0).alias(utility_col)
        )
        outcome = labeled_outcomes.group_by("symbol", "decision_bar_close_ms").agg(
            pl.col(label_col).fill_null(False).cast(pl.Boolean).max().alias(label_col),
            utility_expr,
        )
    elif target == "tail_any_event":
        label_col = "tail_any"
        utility_col = "tail_utility_any"
        if label_col not in labeled_outcomes.columns:
            return observations.head(0), np.array([], dtype=float), np.array([], dtype=float)
        outcome = labeled_outcomes.group_by("symbol", "decision_bar_close_ms").agg(
            pl.col(label_col).fill_null(False).cast(pl.Boolean).max().alias(label_col),
            pl.max_horizontal(
                pl.col("tail_utility_up").fill_null(0.0),
                pl.col("tail_utility_down").fill_null(0.0),
            ).max().alias(utility_col),
        )
    else:
        label_col = f"tail_side_only_{direction}"
        utility_col = f"tail_utility_margin_{direction}"
        state_col = "path_state" if "path_state" in labeled_outcomes.columns else "tail_state"
        if state_col not in labeled_outcomes.columns or utility_col not in labeled_outcomes.columns:
            return observations.head(0), np.array([], dtype=float), np.array([], dtype=float)
        outcome = labeled_outcomes.group_by("symbol", "decision_bar_close_ms").agg(
            pl.col(state_col).is_in([direction, f"clean_{direction}"]).max().alias(label_col),
            pl.col(utility_col).fill_null(0.0).clip(0.0, None).max().alias(utility_col),
        )

    joined = observations.join(outcome, on=["symbol", "decision_bar_close_ms"], how="inner").sort(
        "decision_bar_close_ms"
    )
    labels = joined.get_column(label_col).fill_null(False).cast(pl.Int8).to_numpy().astype(float)
    utilities = joined.get_column(utility_col).fill_null(0.0).to_numpy().astype(float)
    return joined.drop(label_col, utility_col), labels, utilities


def _tailtree_outcome_by_decision(outcomes: pl.DataFrame) -> pl.DataFrame:
    """Collapse market/source duplicate outcomes to one decision-key row.

    Market baseline rows carry terminal context but no source forward return; source rows
    carry tail labels. Tailtree diagnostics need one all-row denominator with any source
    tail evidence preserved, not an arbitrary first row.
    """
    if outcomes.is_empty():
        return outcomes
    exprs: list[pl.Expr] = []
    if "outcome_bucket" in outcomes.columns:
        exprs.append(
            pl.when((pl.col("outcome_bucket") == "up").any())
            .then(pl.lit("up"))
            .when((pl.col("outcome_bucket") == "down").any())
            .then(pl.lit("down"))
            .otherwise(pl.lit("flat"))
            .alias("outcome_bucket")
        )
    for col in ("tail_up", "tail_down", "direction_changed", "returned_to_origin"):
        if col in outcomes.columns:
            exprs.append(pl.col(col).fill_null(False).cast(pl.Boolean).max().alias(col))
    for col in ("tail_utility_up", "tail_utility_down"):
        if col in outcomes.columns:
            exprs.append(pl.col(col).fill_null(0.0).cast(pl.Float64).max().alias(col))
    if not exprs:
        return outcomes.select("symbol", "decision_bar_close_ms").unique()
    return outcomes.group_by("symbol", "decision_bar_close_ms").agg(*exprs)
