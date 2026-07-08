"""LightGBM path-prototype model artifact and training helpers."""

from __future__ import annotations

from collections import Counter
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

    objective: Literal["path_prototype"] = "path_prototype"
    num_leaves: int = Field(default=64, ge=8, le=256)
    min_data_in_leaf: int = Field(default=30, ge=10, le=500)
    learning_rate: float = Field(default=0.05, gt=0, le=1.0)
    num_iterations: int = Field(default=200, ge=10, le=2000)
    early_stopping_rounds: int = Field(default=20, ge=5, le=100)
    validation_fraction: float = Field(default=0.2, ge=0.05, le=0.5)
    random_seed: int = Field(default=42, ge=0)


class FeatureContractError(ValueError):
    """Raised when a parsed model artifact does not match its feature contract."""


class FixedSearchProvenance(BaseModel):
    """Training used fixed parameters rather than an HPO/search result."""

    kind: Literal["fixed"] = "fixed"


class OptunaSearchProvenance(BaseModel):
    """Optuna search provenance embedded as metadata only, never predict behavior."""

    kind: Literal["optuna"] = "optuna"
    study_name: str = ""
    trial_number: int = Field(default=0, ge=0)
    score: float = 0.0
    seed: int = Field(default=42, ge=0)


SearchProvenance = FixedSearchProvenance | OptunaSearchProvenance


class GPDParams(BaseModel):
    """Fitted GPD parameters for one leaf or the global baseline."""

    xi: float = Field(ge=-0.2, le=0.6)
    sigma: float = Field(gt=0)
    tail_rate: float = Field(ge=0, le=1.0)


class TreeMetadata(BaseModel):
    """Serializable metadata stored alongside the LightGBM booster string."""

    direction: Literal["up", "down", "path"]
    artifact_version: int = 1
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
    num_class: int | None = None
    selected_columns: list[str] = Field(default_factory=list)
    feature_schema_hash: str = ""
    feature_manifest_id: str = ""
    feature_manifest_checksum: str = ""
    label_contract_id: str = ""
    weight_column: str = "sample_weight"
    model_num_data: int = 0
    search: SearchProvenance = Field(default_factory=FixedSearchProvenance)
    class_names: list[str] = Field(default_factory=list)
    class_counts: dict[str, int] = Field(default_factory=dict)
    valid_n_observations: int = 0


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

        event_count = len(exceedance_values)

        # 1. Global tail baseline.
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
        params = {
            "objective": "quantile",
            "metric": "quantile",
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
            np.asarray(self._booster.predict(self._feature_matrix(features))).astype(float).ravel()
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

    def assert_compatible_matrix(self, feature_matrix: pl.DataFrame) -> None:
        """Fail unless a score matrix contains exactly this model's selected feature space."""
        selected = list(self.metadata.selected_columns or self.metadata.continuous_features)
        missing = [column for column in selected if column not in feature_matrix.columns]
        if missing:
            raise FeatureContractError(
                f"feature matrix missing selected path model features: {', '.join(missing)}"
            )
        actual_order = [column for column in feature_matrix.columns if column in set(selected)]
        if actual_order != selected:
            raise FeatureContractError(
                f"selected feature order mismatch: model={selected!r} matrix={actual_order!r}"
            )

    def score_path(self, feature_matrix: pl.DataFrame) -> pl.DataFrame:
        """Score a path-prototype matrix after enforcing this artifact's feature contract."""
        if (
            self.metadata.direction != "path"
            or self.metadata.train_config.objective != "path_prototype"
        ):
            raise ValueError("score_path requires a path_prototype TailTreeModel")
        self.assert_compatible_matrix(feature_matrix)
        missing_keys = [column for column in _PATH_KEYS if column not in feature_matrix.columns]
        if missing_keys:
            raise ValueError(f"feature_matrix missing path score keys: {', '.join(missing_keys)}")
        selected_columns = list(self.metadata.selected_columns or self.metadata.continuous_features)
        probabilities = np.asarray(
            self._booster.predict(_path_matrix(feature_matrix, selected_columns))
        )
        if probabilities.ndim == 1:
            probabilities = probabilities.reshape((-1, 5))
        probabilities = probabilities.astype(float)
        pred_labels = probabilities.argmax(axis=1).astype(int)
        confidences = probabilities.max(axis=1).astype(float)
        result = feature_matrix.select(_PATH_KEYS)
        for index, column in enumerate(_PATH_PROB_COLUMNS):
            result = result.with_columns(pl.Series(column, probabilities[:, index]))
        return result.with_columns(
            pl.Series("path_pred_label", pred_labels),
            pl.Series(
                "path_pred_label_name",
                [_PATH_CLASS_NAMES[int(label)] for label in pred_labels],
            ),
            pl.Series("path_confidence", confidences),
        )

    @classmethod
    def train_path(
        cls,
        train_matrix: pl.DataFrame,
        valid_matrix: pl.DataFrame,
        *,
        config: TrainConfig | dict,
        selected_manifest: object,
        label_contract_id: str,
        feature_manifest_id: str | None = None,
    ) -> TailTreeModel:
        """Train one multiclass path-prototype model using the existing TailTreeModel artifact."""
        import lightgbm as lgb

        if isinstance(config, dict):
            config = TrainConfig.model_validate(config)
        if config.objective != "path_prototype":
            raise ValueError(
                "TailTreeModel.train_path requires TrainConfig(objective='path_prototype')"
            )
        selected_columns = _selected_columns_from_manifest(selected_manifest)
        x_train = _path_matrix(train_matrix, selected_columns)
        x_valid = _path_matrix(valid_matrix, selected_columns)
        y_train = _path_labels(train_matrix, name="train_matrix")
        y_valid = _path_labels(valid_matrix, name="valid_matrix")
        w_train = _path_weights(train_matrix, name="train_matrix")
        w_valid = _path_weights(valid_matrix, name="valid_matrix")
        if len(y_train) < config.min_data_in_leaf:
            raise ValueError(
                f"Not enough path rows ({len(y_train)}) "
                f"for min_data_in_leaf={config.min_data_in_leaf}"
            )
        train_data = lgb.Dataset(x_train, label=y_train, weight=w_train)
        valid_data = lgb.Dataset(x_valid, label=y_valid, weight=w_valid, reference=train_data)
        params = {
            "objective": "multiclass",
            "num_class": 5,
            "metric": "multi_logloss",
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
            callbacks=[
                lgb.early_stopping(config.early_stopping_rounds, verbose=False),
                lgb.log_evaluation(0),
            ],
        )
        importance = booster.feature_importance(importance_type="gain")
        feature_importance = sorted(
            [
                (selected_columns[index], float(importance[index]))
                for index in range(min(len(selected_columns), len(importance)))
            ],
            key=lambda item: item[1],
            reverse=True,
        )
        class_counts = {str(label): int(count) for label, count in Counter(y_train).items()}
        for label in range(5):
            class_counts.setdefault(str(label), 0)
        return cls(
            booster=booster.model_to_string(),
            metadata=TreeMetadata(
                direction="path",
                num_leaves_actual=booster.num_trees(),
                categorical_features=[],
                continuous_features=selected_columns,
                global_baseline=GPDParams(xi=0.0, sigma=1.0, tail_rate=1.0),
                leaf_params={},
                feature_importance=feature_importance,
                train_config=config,
                train_timestamp=datetime.now(UTC).isoformat(),
                train_n_observations=len(train_matrix),
                train_n_exceedances=len(train_matrix),
                num_class=5,
                selected_columns=selected_columns,
                feature_schema_hash=str(getattr(selected_manifest, "schema_hash", "")),
                feature_manifest_id=feature_manifest_id
                or str(getattr(selected_manifest, "artifact_id", "")),
                feature_manifest_checksum=selected_manifest.checksum(),
                label_contract_id=label_contract_id,
                weight_column=str(getattr(selected_manifest, "weight_column", "sample_weight")),
                model_num_data=len(train_matrix),
                search=FixedSearchProvenance(),
                class_names=list(_PATH_CLASS_NAMES),
                class_counts=dict(sorted(class_counts.items())),
                valid_n_observations=len(valid_matrix),
            ),
        )

    # ── persistence

    def to_json(self, path: str | Path) -> None:
        """Serialize booster + metadata as one JSON file."""
        payload = TailTreePayload(lightgbm_model=self.booster, metadata=self.metadata)
        Path(path).write_text(payload.model_dump_json(indent=2), encoding="utf-8")

    @classmethod
    def from_json(cls, path: str | Path) -> TailTreeModel:
        """Load from JSON. pydantic validates metadata on construction."""
        payload = TailTreePayload.model_validate_json(Path(path).read_text(encoding="utf-8"))
        return cls(booster=payload.lightgbm_model, metadata=payload.metadata)


_PATH_CLASS_NAMES = ["calm", "smooth_up", "smooth_down", "chop", "fake_breakout"]
_PATH_PROB_COLUMNS = [
    "path_prob_calm",
    "path_prob_smooth_up",
    "path_prob_smooth_down",
    "path_prob_chop",
    "path_prob_fake_breakout",
]
_PATH_KEYS = ["symbol", "decision_bar_close_ms", "horizon_hours"]


def _selected_columns_from_manifest(selected_manifest: object) -> list[str]:
    columns = [str(column) for column in getattr(selected_manifest, "selected_columns", ())]
    if not columns:
        raise ValueError("selected_manifest.selected_columns must not be empty")
    return columns


def _require_path_columns(frame: pl.DataFrame, *, columns: list[str], name: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{name} missing selected path model features: {', '.join(missing)}")


def _path_matrix(frame: pl.DataFrame, columns: list[str]) -> np.ndarray:
    _require_path_columns(frame, columns=columns, name="path feature matrix")
    return frame.select(columns).to_numpy().astype(np.float64)


def _path_labels(frame: pl.DataFrame, *, name: str) -> np.ndarray:
    if "path_label" not in frame.columns:
        raise ValueError(f"{name} missing path_label")
    labels = frame.get_column("path_label").cast(pl.Int64).to_numpy()
    invalid = [int(value) for value in labels if int(value) < 0 or int(value) >= 5]
    if invalid:
        raise ValueError("path_label values must be integers in [0, 4]")
    return labels.astype(np.int64)


def _path_weights(frame: pl.DataFrame, *, name: str) -> np.ndarray:
    if "sample_weight" not in frame.columns:
        raise ValueError(f"{name} missing sample_weight")
    weights = frame.get_column("sample_weight").cast(pl.Float64).to_numpy()
    if np.any(weights <= 0.0):
        raise ValueError("sample_weight values must be positive")
    return weights.astype(np.float64)


# ── legacy tail training helpers ─────────────────────────────────────────────


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


def _path_guard_target_frame(
    behavior_targets: pl.DataFrame,
    target: str,
    label_col: str,
    utility_col: str,
) -> pl.DataFrame:
    obvious_guard = (
        pl.col("behavior_false_direction").fill_null(False).cast(pl.Boolean)
        | pl.col("behavior_blocker")
        .fill_null("")
        .is_in(["opposite_clean_path", "opposite_tail_dominates"])
        | (pl.col("behavior_path_state").fill_null("none") == "clean_down")
    )
    blocker_guard = obvious_guard | (pl.col("behavior_blocker").fill_null("") != "")
    tradability_guard = obvious_guard | (
        pl.col("behavior_actionability").fill_null("none") != "tradable_up"
    )
    full_guard = (
        blocker_guard
        | tradability_guard
        | (pl.col("behavior_utility_margin").fill_null(0.0).cast(pl.Float64) <= 0.0)
    )
    guard = (
        full_guard
        if target == "path_guard_full"
        else tradability_guard
        if target == "path_guard_tradability"
        else blocker_guard
        if target == "path_guard_blocker"
        else obvious_guard
    )
    return (
        behavior_targets.with_columns(
            guard.alias(label_col),
            pl.when(guard)
            .then(1.0 + pl.col("behavior_utility_margin").fill_null(0.0).cast(pl.Float64).abs())
            .otherwise(1.0)
            .alias(utility_col),
        )
        .group_by("symbol", "decision_bar_close_ms")
        .agg(
            pl.col(label_col).fill_null(False).cast(pl.Boolean).max().alias(label_col),
            pl.col(utility_col).fill_null(1.0).cast(pl.Float64).max().alias(utility_col),
        )
    )


def tailtree_target_training_values(
    observations: pl.DataFrame,
    labeled_outcomes: pl.DataFrame,
    *,
    target: str,
    direction: Literal["up", "down"],
    behavior_targets: pl.DataFrame | None = None,
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
            )
            .max()
            .alias(utility_col),
        )
    elif target == "tail_side_only":
        label_col = f"tail_side_only_{direction}"
        utility_col = f"tail_utility_margin_{direction}"
        state_col = "path_state" if "path_state" in labeled_outcomes.columns else "tail_state"
        if state_col not in labeled_outcomes.columns or utility_col not in labeled_outcomes.columns:
            return observations.head(0), np.array([], dtype=float), np.array([], dtype=float)
        outcome = labeled_outcomes.group_by("symbol", "decision_bar_close_ms").agg(
            pl.col(state_col).is_in([direction, f"clean_{direction}"]).max().alias(label_col),
            pl.col(utility_col).fill_null(0.0).clip(0.0, None).max().alias(utility_col),
        )
    elif target.startswith("path_guard"):
        label_col = "behavior_guard"
        utility_col = "behavior_guard_weight"
        if direction != "up" or behavior_targets is None or behavior_targets.is_empty():
            return observations.head(0), np.array([], dtype=float), np.array([], dtype=float)
        outcome = _path_guard_target_frame(behavior_targets, target, label_col, utility_col)
    else:
        return observations.head(0), np.array([], dtype=float), np.array([], dtype=float)

    joined = observations.join(outcome, on=["symbol", "decision_bar_close_ms"], how="inner").sort(
        "decision_bar_close_ms"
    )
    labels = joined.get_column(label_col).fill_null(False).cast(pl.Int8).to_numpy().astype(float)
    utilities = joined.get_column(utility_col).fill_null(0.0).to_numpy().astype(float)
    return joined.drop(label_col, utility_col), labels, utilities
