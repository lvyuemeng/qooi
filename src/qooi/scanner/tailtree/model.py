"""LightGBM + GPD tail-detection tree.

Optional dependency group: [tailtree] = lightgbm, scipy.
All imports are lazy; import this module directly only when evidence="tailtree".
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Protocol

import numpy as np
import polars as pl
from pydantic import BaseModel, Field


class LightGbmBooster(Protocol):
    def predict(self, data: np.ndarray, *, pred_leaf: bool = False) -> np.ndarray: ...
    def feature_importance(self, *, importance_type: str) -> np.ndarray: ...
    def model_to_string(self) -> str: ...
    def num_trees(self) -> int: ...


class LightGbmDataset(Protocol):
    def get_label(self) -> np.ndarray: ...


# ── pydantic models ──────────────────────────────────────────────────────────


class TrainConfig(BaseModel):
    """Training hyperparameters. Validated on construction."""

    objective: Literal["tail_severity_gpd", "tail_utility_quantile"] = "tail_severity_gpd"
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
                "utility_values length " f"{len(utility_values)} != features length {len(features)}"
            )

        # 1. Global GPD fit over tail exceedance severity.
        xi_global, _, sigma_global = scipy.stats.genpareto.fit(exceedance_values, floc=0)
        train_n_observations = train_n_observations or len(features)
        global_tail_rate = (
            global_tail_rate
            if global_tail_rate is not None
            else len(exceedance_values) / len(features)
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
        target_values = (
            np.log1p(np.maximum(utility_values, 0.0))
            if config.objective == "tail_utility_quantile"
            else exceedance_values
        )
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
            "objective": _gpd_xi_objective if is_gpd_objective else "quantile",
            "metric": "None" if is_gpd_objective else "quantile",
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
            if len(leaf_y) >= 10:
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
                train_n_exceedances=len(exceedance_values),
            ),
        )

    # ── prediction ───────────────────────────────────────────────────────────

    @property
    def _booster(self) -> LightGbmBooster:
        """Reconstruct lightgbm Booster from stored model string."""
        import lightgbm as lgb

        return lgb.Booster(model_str=self.booster)

    def predict_leaf(self, features: pl.DataFrame) -> pl.DataFrame:
        """Assign each row to a leaf. Returns features + 'leaf_id' (Int32)."""
        all_cols = self.metadata.categorical_features + self.metadata.continuous_features
        present = [c for c in all_cols if c in features.columns]
        x = features.select(present).to_numpy()
        for col in self.metadata.categorical_features:
            if col in features.columns and features[col].dtype == pl.String:
                codes = features.get_column(col).cast(pl.Categorical).to_physical().to_numpy()
                idx = present.index(col)
                x[:, idx] = codes.astype(np.float64)
        leaf_ids = _leaf_id_vector(self._booster.predict(x, pred_leaf=True))
        return features.with_columns(pl.Series("leaf_id", leaf_ids))

    def predict_score(self, features: pl.DataFrame) -> pl.DataFrame:
        """Score each row with the full boosted model ensemble."""
        all_cols = self.metadata.categorical_features + self.metadata.continuous_features
        present = [c for c in all_cols if c in features.columns]
        x = features.select(present).to_numpy()
        for col in self.metadata.categorical_features:
            if col in features.columns and features[col].dtype == pl.String:
                codes = features.get_column(col).cast(pl.Categorical).to_physical().to_numpy()
                idx = present.index(col)
                x[:, idx] = codes.astype(np.float64)
        scores = np.asarray(self._booster.predict(x)).astype(float).ravel()
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
        data = {
            "lightgbm_model": self.booster,
            "metadata": self.metadata.model_dump(mode="json"),
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    @classmethod
    def from_json(cls, path: str | Path) -> TailTreeModel:
        """Load from JSON. pydantic validates metadata on construction."""
        with open(path) as f:
            data = json.load(f)
        metadata = TreeMetadata.model_validate(data["metadata"])
        return TailTreeModel(booster=data["lightgbm_model"], metadata=metadata)


# ── LightGBM custom objective ────────────────────────────────────────────────


def _gpd_xi_objective(
    preds: np.ndarray,
    train_data: LightGbmDataset,
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
    train_data: LightGbmDataset,
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


def label_tail_exceedances(
    outcome_frame: pl.DataFrame,
    *,
    threshold_pct: float = 5.0,
) -> pl.DataFrame:
    """Label tail exceedances in the outcome frame."""
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

    return outcome_frame.with_columns(exprs)


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
    if not exprs:
        return outcomes.select("symbol", "decision_bar_close_ms").unique()
    return outcomes.group_by("symbol", "decision_bar_close_ms").agg(*exprs)
