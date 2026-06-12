"""LightGBM + GPD tail-detection tree.

Optional dependency group: [tailtree] = lightgbm, scipy.
All imports are lazy; import this module directly only when evidence="tailtree".
"""

from __future__ import annotations

import json
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
    global_tail_rate: float

    @property
    def train_n_observations(self) -> int:
        return len(self.all_observations)

    @property
    def train_n_exceedances(self) -> int:
        return len(self.tail_observations)

    def has_min_exceedances(self, min_count: int) -> bool:
        return self.train_n_exceedances >= min_count


# ── TailTreeModel ────────────────────────────────────────────────────────────


@dataclass
class TailTreeModel:
    """Trained tail-detection tree.

    Construct via TailTreeModel.train() or TailTreeModel.from_json().
    """

    booster: Any  # lightgbm.Booster
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
        y_train, y_valid = exceedance_values[:-n_valid], exceedance_values[-n_valid:]

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
        params = {
            "objective": _gpd_xi_objective,
            "metric": "None",
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
            feval=_gpd_nll_eval,
            callbacks=[
                lgb.early_stopping(config.early_stopping_rounds),
                lgb.log_evaluation(0),
            ],
        )

        # 4. Per-leaf GPD fit
        leaf_ids = booster.predict(x_train, pred_leaf=True).astype(int).ravel()
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
    def _booster(self) -> Any:
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
        leaf_ids = self._booster.predict(x, pred_leaf=True).astype("int32").ravel()
        return features.with_columns(pl.Series("leaf_id", leaf_ids))

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
    train_data: Any,  # lgb.Dataset
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


def label_tail_exceedances(
    outcome_frame: pl.DataFrame,
    *,
    threshold_pct: float = 5.0,
) -> pl.DataFrame:
    """Label tail exceedances in the outcome frame."""
    has_max = "forward_max_return_pct" in outcome_frame.columns
    has_min = "forward_min_return_pct" in outcome_frame.columns

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
            global_tail_rate=0.0,
        )

    tail_outcomes = (
        labeled_outcomes.filter(pl.col(tail_col).fill_null(False))
        .select("symbol", "decision_bar_close_ms", exceed_col)
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
    global_tail_rate = (
        len(tail_observations) / len(all_observations) if not all_observations.is_empty() else 0.0
    )
    return TailtreeTrainingFrame(
        direction=direction,
        all_observations=all_observations,
        tail_observations=tail_observations,
        exceedance_values=exceedance_values,
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


def leaf_evidence_frame(
    tree: TailTreeModel,
    observations: pl.DataFrame,
    outcomes: pl.DataFrame,
    *,
    recent_window_days: int = 30,
) -> pl.DataFrame:
    """Per-leaf tail evidence: N_total, N_tail, xi, sigma, tail_lift, stability."""
    with_leaf = tree.predict_leaf(observations)

    global_tr = tree.metadata.global_baseline.tail_rate
    tail_col = "tail_up" if tree.metadata.direction == "up" else "tail_down"
    has_tail = tail_col in outcomes.columns

    if not has_tail:
        return pl.DataFrame(
            schema={
                "leaf_id": pl.Int32,
                "tree_direction": pl.String,
                "N_total": pl.UInt32,
                "N_tail_exceedances": pl.UInt32,
                "gpd_shape_xi": pl.Float64,
                "gpd_scale_sigma": pl.Float64,
                "tail_lift": pl.Float64,
                "tail_lift_stability": pl.Float64,
                "leaf_tail_rate": pl.Float64,
                "global_tail_rate": pl.Float64,
            }
        )

    outcome_by_decision = _tailtree_outcome_by_decision(outcomes)
    joined = with_leaf.join(
        outcome_by_decision.select("symbol", "decision_bar_close_ms", tail_col),
        on=["symbol", "decision_bar_close_ms"],
        how="left",
    )

    leaf_stats = joined.group_by("leaf_id").agg(
        pl.len().cast(pl.UInt32).alias("N_total"),
        pl.col(tail_col).cast(pl.UInt32).sum().alias("N_tail_exceedances"),
    )

    # Recent window
    max_ts = observations.get_column("decision_bar_close_ms").max()
    recent_cutoff = max_ts - recent_window_days * 24 * 60 * 60 * 1000
    recent = joined.filter(pl.col("decision_bar_close_ms") >= recent_cutoff)
    if not recent.is_empty():
        recent_stats = recent.group_by("leaf_id").agg(
            pl.len().cast(pl.UInt32).alias("N_recent"),
            pl.col(tail_col).cast(pl.UInt32).sum().alias("N_tail_recent"),
        )
        leaf_stats = leaf_stats.join(recent_stats, on="leaf_id", how="left")
    else:
        leaf_stats = leaf_stats.with_columns(
            pl.lit(0, dtype=pl.UInt32).alias("N_recent"),
            pl.lit(0, dtype=pl.UInt32).alias("N_tail_recent"),
        )

    leaf_params_df = pl.DataFrame(
        [
            {"leaf_id": lid, "gpd_shape_xi": p.xi, "gpd_scale_sigma": p.sigma}
            for lid, p in tree.metadata.leaf_params.items()
        ],
        schema={
            "leaf_id": pl.Int32,
            "gpd_shape_xi": pl.Float64,
            "gpd_scale_sigma": pl.Float64,
        },
    )

    result = (
        leaf_stats.join(leaf_params_df, on="leaf_id", how="left")
        .with_columns(
            pl.lit(tree.metadata.direction).alias("tree_direction"),
            (
                pl.col("N_tail_exceedances").cast(pl.Float64)
                / pl.when(pl.col("N_total") > 0).then(pl.col("N_total")).otherwise(None)
            )
            .fill_null(0.0)
            .alias("leaf_tail_rate"),
            pl.lit(global_tr).alias("global_tail_rate"),
        )
        .with_columns(
            (
                pl.col("leaf_tail_rate")
                / pl.when(pl.col("global_tail_rate") > 0)
                .then(pl.col("global_tail_rate"))
                .otherwise(None)
            )
            .fill_null(0.0)
            .alias("tail_lift"),
        )
        .with_columns(
            (
                (
                    pl.col("N_tail_recent").cast(pl.Float64)
                    / pl.when(pl.col("N_recent") > 0).then(pl.col("N_recent")).otherwise(None)
                )
                / pl.when(pl.col("leaf_tail_rate") > 0)
                .then(pl.col("leaf_tail_rate"))
                .otherwise(None)
            )
            .clip(0, 2)
            .fill_null(0.0)
            .alias("tail_lift_stability"),
        )
    )

    return result.select(
        "leaf_id",
        "tree_direction",
        "N_total",
        "N_tail_exceedances",
        "gpd_shape_xi",
        "gpd_scale_sigma",
        "tail_lift",
        "tail_lift_stability",
        "leaf_tail_rate",
        "global_tail_rate",
    )


def leaf_context_frame(
    tree: TailTreeModel,
    observations: pl.DataFrame,
    outcomes: pl.DataFrame,
    *,
    global_baseline: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Per-leaf context: directional probabilities, path diagnostics."""
    with_leaf = tree.predict_leaf(observations)

    if outcomes.is_empty():
        return pl.DataFrame(
            schema={
                "leaf_id": pl.Int32,
                "p_up": pl.Float64,
                "p_down": pl.Float64,
                "p_flat": pl.Float64,
                "conditioned_entropy_bits": pl.Float64,
                "information_gain_bits": pl.Float64,
                "tail_up_rate": pl.Float64,
                "tail_down_rate": pl.Float64,
                "path_skew": pl.Float64,
                "returned_to_origin_rate": pl.Float64,
                "statistical_direction": pl.String,
                "research_suggestion": pl.String,
            }
        )

    outcome_by_decision = _tailtree_outcome_by_decision(outcomes)
    joined = with_leaf.join(
        outcome_by_decision.select(
            [
                "symbol",
                "decision_bar_close_ms",
                "outcome_bucket",
                "tail_up",
                "tail_down",
                "direction_changed",
                "returned_to_origin",
            ]
        ),
        on=["symbol", "decision_bar_close_ms"],
        how="left",
    )

    leaf_agg = joined.group_by("leaf_id").agg(
        pl.len().cast(pl.UInt32).alias("N_leaf"),
        (pl.col("outcome_bucket") == "up").mean().alias("p_up"),
        (pl.col("outcome_bucket") == "down").mean().alias("p_down"),
        (pl.col("outcome_bucket") == "flat").mean().alias("p_flat"),
    )

    has_tails = "tail_up" in joined.columns and "tail_down" in joined.columns
    if has_tails:
        tail_agg = joined.group_by("leaf_id").agg(
            pl.col("tail_up").cast(pl.Float64).mean().alias("tail_up_rate"),
            pl.col("tail_down").cast(pl.Float64).mean().alias("tail_down_rate"),
        )
        leaf_agg = leaf_agg.join(tail_agg, on="leaf_id", how="left")

    path_agg = (
        joined.group_by("leaf_id").agg(
            (
                pl.col("tail_up").cast(pl.Float64).mean().fill_null(0.0)
                - pl.col("tail_down").cast(pl.Float64).mean().fill_null(0.0)
            ).alias("path_skew"),
            pl.col("returned_to_origin").cast(pl.Float64).mean().alias("returned_to_origin_rate"),
        )
        if "returned_to_origin" in joined.columns and has_tails
        else joined.group_by("leaf_id").agg(
            pl.lit(0.0).alias("path_skew"),
            pl.lit(0.0).alias("returned_to_origin_rate"),
        )
    )
    leaf_agg = leaf_agg.join(path_agg, on="leaf_id", how="left")

    # Entropy
    from qooi.scanner import entropy_expr

    leaf_agg = leaf_agg.with_columns(
        entropy_expr("p_up", "p_down", "p_flat").alias("conditioned_entropy_bits"),
        pl.lit(0.0).alias("information_gain_bits"),
    )

    # Statistical direction
    leaf_agg = leaf_agg.with_columns(
        pl.when(pl.col("p_up") > pl.max_horizontal("p_down", "p_flat"))
        .then(pl.lit("up"))
        .when(pl.col("p_down") > pl.max_horizontal("p_up", "p_flat"))
        .then(pl.lit("down"))
        .otherwise(pl.lit("flat"))
        .alias("statistical_direction"),
    )

    # Research suggestion
    leaf_agg = leaf_agg.with_columns(
        pl.when((pl.col("returned_to_origin_rate") >= 0.25) & (pl.col("path_skew").abs() <= 0.10))
        .then(pl.lit("chop_avoid"))
        .otherwise(pl.lit("insufficient_evidence"))
        .alias("research_suggestion"),
    )

    return leaf_agg.select(
        "leaf_id",
        "p_up",
        "p_down",
        "p_flat",
        "conditioned_entropy_bits",
        "information_gain_bits",
        "tail_up_rate",
        "tail_down_rate",
        "path_skew",
        "returned_to_origin_rate",
        "statistical_direction",
        "research_suggestion",
    )


def select_tail_leaves(
    leaf_evidence: pl.DataFrame,
    *,
    min_tail_exceedances: int = 30,
    min_tail_lift: float = 1.5,
    min_tail_lift_stability: float = 0.3,
    fallback_top_n: int = 10,
) -> pl.DataFrame:
    """Select tail leaves by hard gate, or top-ranked best available leaves.

    The fallback is deliberately labelled; it does not pretend weak leaves passed.
    It gives review/candidate ranking a quantitative surface when the strict gate
    selects zero leaves.
    """
    if leaf_evidence.is_empty():
        return leaf_evidence

    scored = leaf_evidence.with_columns(
        (pl.col("N_tail_exceedances") >= min_tail_exceedances).alias("passes_tail_count_gate"),
        (pl.col("tail_lift") >= min_tail_lift).alias("passes_tail_lift_gate"),
        (
            (pl.col("tail_lift_stability") >= min_tail_lift_stability) | (pl.col("N_total") < 200)
        ).alias("passes_stability_gate"),
    ).with_columns(
        (
            pl.col("passes_tail_count_gate")
            & pl.col("passes_tail_lift_gate")
            & pl.col("passes_stability_gate")
        ).alias("selected_evidence_level"),
        (
            pl.max_horizontal(pl.col("tail_lift").fill_null(0.0), pl.lit(0.0))
            * (pl.col("N_tail_exceedances").fill_null(0).cast(pl.Float64) + 1.0).log()
            * pl.max_horizontal(pl.col("tail_lift_stability").fill_null(0.0), pl.lit(0.05))
        ).alias("tail_evidence_score"),
    )

    hard = scored.filter(pl.col("selected_evidence_level"))
    if not hard.is_empty():
        return hard.with_columns(pl.lit("hard_gate").alias("selection_mode"))

    return (
        scored.sort(
            ["tail_evidence_score", "tail_lift", "N_tail_exceedances"],
            descending=[True, True, True],
        )
        .head(fallback_top_n)
        .with_columns(pl.lit("best_available").alias("selection_mode"))
    )
