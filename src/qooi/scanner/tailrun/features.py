"""Tailtree feature manifest, matrix, and candidate-artifact boundaries."""

from __future__ import annotations

import hashlib
import json
import warnings
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from pydantic import BaseModel, ConfigDict, ValidationError
from tsflex.features import FeatureCollection, FeatureDescriptor, FuncWrapper

_KEY_COLUMNS = ("symbol", "decision_bar_close_ms")
_TRAIN_LABEL_COLUMNS = (
    "path_label",
    "path_label_name",
    "sample_weight",
    "trend_cleanliness",
    "risk_adjusted_path_weight",
    "first_touch_hours",
    "final_return",
    "path_reason",
)


class FeatureSpec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    horizons: tuple[int, ...]
    windows_hours: tuple[int, ...] = ()
    tsfresh_value_columns: tuple[str, ...] = ()
    tsfresh_required_value_columns: tuple[str, ...] = ()
    tsfresh_optional_min_finite_rate: float = 0.95
    tsfresh_calculators: tuple[str, ...] = (
        "mean",
        "standard_deviation",
        "minimum",
        "maximum",
        "median",
        "absolute_maximum",
        "skewness",
        "kurtosis",
        "abs_energy",
    )
    base_prefix: str = "base__"
    context_prefix: str = "ctx__"
    tsfresh_prefix: str = "tsf__"
    cross_prefix: str = "cross__"
    selected_generated_columns: tuple[str, ...] = ()
    source_tsfresh_value_columns: tuple[str, ...] = ()
    source_tsfresh_calculators: tuple[str, ...] = (
        "last_minus_first",
        "minimum",
        "maximum",
        "median",
        "valid_ratio",
        "q90_q10_range",
        "sample_count",
    )
    source_tsfresh_prefix: str = "tsfsrc__"
    source_cross_prefix: str = "crosssrc__"

    def train_frame(
        self,
        observations: pl.DataFrame,
        histories: pl.DataFrame | None,
        labels: pl.DataFrame,
    ) -> pl.DataFrame:
        """Build one training feature candidate frame for scanner and research."""
        base = _feature_base_with_tsfresh(observations, histories, labels, spec=self)
        return train_candidates(labels, base, spec=self)

    def predict_frame(
        self,
        observations: pl.DataFrame,
        histories: pl.DataFrame | None,
    ) -> pl.DataFrame:
        """Build one prediction feature candidate frame without future labels."""
        samples = observations.select(*_KEY_COLUMNS)
        base = _feature_base_with_tsfresh(observations, histories, samples, spec=self)
        return predict_candidates(observations, base, horizons=self.horizons, spec=self)

    def generated_by_window(self) -> dict[int, frozenset[str]]:
        selected: dict[int, set[str]] = {}
        for column in self.selected_generated_columns:
            if not column.startswith((self.tsfresh_prefix, self.cross_prefix)):
                continue
            parts = column.split("__")
            if len(parts) != 4 or not parts[2].startswith("w") or not parts[2].endswith("h"):
                raise ValueError(f"invalid generated feature column: {column}")
            selected.setdefault(int(parts[2][1:-1]), set()).add(column)
        return {window: frozenset(columns) for window, columns in selected.items()}

    def source_generated_by_window(self) -> dict[int, frozenset[str]]:
        selected: dict[int, set[str]] = {}
        for column in self.selected_generated_columns:
            if not column.startswith((self.source_tsfresh_prefix, self.source_cross_prefix)):
                continue
            parts = column.split("__")
            if len(parts) != 4 or not parts[2].startswith("w") or not parts[2].endswith("h"):
                raise ValueError(f"invalid generated source feature column: {column}")
            selected.setdefault(int(parts[2][1:-1]), set()).add(column)
        return {window: frozenset(columns) for window, columns in selected.items()}

    def tsflex_output_name(self, column: str, window: int) -> str:
        series, calculator, *_rest = column.split("__")
        series = series.replace("|", "_")
        prefix = (
            self.cross_prefix if calculator == "corr" and "_" in series else self.tsfresh_prefix
        )
        return f"{prefix.rstrip('_')}__{series}__w{window}h__{calculator}"


class SelectSpec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    min_features: int = 20
    max_features: int = 50
    label_column: str = "path_label"
    weight_column: str = "sample_weight"
    feature_prefixes: tuple[str, ...] = (
        "ctx__",
        "base__",
        "tsf__",
        "cross__",
        "tsfsrc__",
        "crosssrc__",
    )


class AcceptanceRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    accepted_at: str
    accepted_from_checksum: str
    accepted_by: str = ""
    note: str = ""


class FeatureManifestBase(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    artifact_id: str
    artifact_kind: str
    spec: FeatureSpec
    selected_columns: tuple[str, ...]
    candidate_feature_columns: tuple[str, ...]
    fold_ids: tuple[int, ...]
    fit_row_count: int
    validation_row_count: int
    schema_hash: str
    label_column: str
    label_contract_id: str = "path_prototype"
    weight_column: str = "sample_weight"
    selection_metric: str = ""
    created_at: str = ""
    review_artifact_ids: tuple[str, ...] = ()

    @classmethod
    def parse_json(cls, text: str) -> FeatureManifest:
        """Parse manifest JSON into the requested manifest class."""
        try:
            data = json.loads(text)
            if cls is FeatureManifestBase:
                if "acceptance" in data:
                    return AcceptedFeatureManifest.model_validate(data)
                return ProposalFeatureManifest.model_validate(data)
            return cls.model_validate(data)
        except ValidationError as exc:
            if any(error.get("loc") == ("spec",) for error in exc.errors()):
                raise ValueError("FeatureManifest spec is required") from exc
            raise

    @classmethod
    def read(cls, path: Path) -> FeatureManifest:
        """Read manifest JSON from disk into the requested manifest class."""
        return cls.parse_json(path.read_text(encoding="utf-8"))

    def write(self, path: Path) -> None:
        """Persist this manifest as JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")

    def checksum(self) -> str:
        """Return a stable checksum for this manifest contract payload."""
        payload = json.dumps(
            self.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
        )
        return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def select_matrix(self, feature_candidates: pl.DataFrame) -> pl.DataFrame:
        """Project candidate features to keys, labels, weights, and selected columns."""
        required = [*_KEY_COLUMNS, "horizon_hours", *self.selected_columns]
        missing = sorted(set(required) - set(feature_candidates.columns))
        if missing:
            raise ValueError(f"missing selected feature columns: {', '.join(missing)}")
        label_columns = tuple(
            self.weight_column if column == "sample_weight" else column
            for column in _TRAIN_LABEL_COLUMNS
        )
        optional = [column for column in label_columns if column in feature_candidates.columns]
        context = [
            column
            for column in ("base__source_any_present", "base__source_min_age_ms")
            if column in feature_candidates.columns and column not in self.selected_columns
        ]
        return feature_candidates.select(
            *_KEY_COLUMNS,
            "horizon_hours",
            *optional,
            *context,
            *self.selected_columns,
        )

    def source_blended(
        self, matrix: pl.DataFrame, *, artifact_suffix: str = "source_blended_all"
    ) -> FeatureManifestBase:
        """Return this manifest plus all source context features present in matrix."""
        current = tuple(self.selected_columns)
        source = tuple(
            column
            for column in matrix.columns
            if (
                (
                    column.startswith(("base__", "ctx__"))
                    and any(token in column for token in SOURCE_CONTEXT_TOKENS)
                )
                or column.startswith(("tsfsrc__", "crosssrc__"))
            )
            and column not in current
        )
        columns = current + source
        digest = hashlib.sha256("\n".join(columns).encode("utf-8")).hexdigest()[:16]
        return self.model_copy(
            update={
                "artifact_id": f"{self.artifact_id}-{artifact_suffix}",
                "selected_columns": columns,
                "schema_hash": f"sha256:{digest}",
                "selection_metric": f"{self.selection_metric}+{artifact_suffix}",
            }
        )


class AcceptedFeatureManifest(FeatureManifestBase):
    artifact_kind: str = "feature_manifest.accepted"
    acceptance: AcceptanceRecord = AcceptanceRecord(
        accepted_at="",
        accepted_from_checksum="",
    )

    @classmethod
    def read(cls, path: Path) -> AcceptedFeatureManifest:
        data = json.loads(path.read_text(encoding="utf-8"))
        if "acceptance" not in data:
            raise ValueError("accepted feature manifest required")
        try:
            return cls.model_validate(data)
        except ValidationError as exc:
            if any(error.get("loc") == ("spec",) for error in exc.errors()):
                raise ValueError("FeatureManifest spec is required") from exc
            raise


class ProposalFeatureManifest(FeatureManifestBase):
    artifact_kind: str = "feature_manifest.proposal"

    @classmethod
    def read(cls, path: Path) -> ProposalFeatureManifest:
        data = json.loads(path.read_text(encoding="utf-8"))
        if "acceptance" in data:
            raise ValueError("proposal feature manifest required")
        try:
            return cls.model_validate(data)
        except ValidationError as exc:
            if any(error.get("loc") == ("spec",) for error in exc.errors()):
                raise ValueError("FeatureManifest spec is required") from exc
            raise

    def accepted(self, *, accepted_by: str = "", note: str = "") -> AcceptedFeatureManifest:
        """Parse this proposal into an accepted manifest with review provenance."""
        data = self.model_dump(mode="python")
        data["artifact_kind"] = "feature_manifest.accepted"
        data["acceptance"] = AcceptanceRecord(
            accepted_at=datetime.now(UTC).isoformat(),
            accepted_from_checksum=self.checksum(),
            accepted_by=accepted_by,
            note=note,
        ).model_dump(mode="python")
        return AcceptedFeatureManifest.model_validate(data)


FeatureManifest = ProposalFeatureManifest | AcceptedFeatureManifest

SOURCE_CONTEXT_TOKENS = ("source_", "funding", "oi_", "taker", "lsr", "market_")


def _require_columns(frame: pl.DataFrame, *, name: str, columns: tuple[str, ...]) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{name} missing required columns: {', '.join(missing)}")


def _ensure_unique_grain(frame: pl.DataFrame, *, name: str, columns: tuple[str, ...]) -> None:
    if frame.is_empty():
        return
    duplicate_count = frame.group_by(*columns).len().filter(pl.col("len") > 1).height
    if duplicate_count:
        raise ValueError(f"duplicate {name} grain: {' × '.join(columns)}")


def _schema_hash(frame: pl.DataFrame) -> str:
    payload = "\n".join(
        f"{name}:{dtype}" for name, dtype in zip(frame.columns, frame.dtypes, strict=True)
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def base_features(
    observations: pl.DataFrame,
    *,
    spec: FeatureSpec,
) -> pl.DataFrame:
    """Build namespaced causal base features at symbol × decision grain."""
    _require_columns(observations, name="observations", columns=_KEY_COLUMNS)
    _ensure_unique_grain(observations, name="observation", columns=_KEY_COLUMNS)
    numeric_dtypes = {
        pl.Int8,
        pl.Int16,
        pl.Int32,
        pl.Int64,
        pl.UInt8,
        pl.UInt16,
        pl.UInt32,
        pl.UInt64,
        pl.Float32,
        pl.Float64,
    }
    feature_exprs = []
    for column, dtype in zip(observations.columns, observations.dtypes, strict=True):
        if column in _KEY_COLUMNS or dtype not in numeric_dtypes:
            continue
        feature_exprs.append(pl.col(column).cast(pl.Float64).alias(f"{spec.base_prefix}{column}"))
    return observations.select(*_KEY_COLUMNS, *feature_exprs).sort(*_KEY_COLUMNS)


def tsfresh_long(
    histories: pl.DataFrame,
    samples: pl.DataFrame,
    *,
    spec: FeatureSpec,
) -> pl.DataFrame:
    """Build finite causal long-format rows for automatic feature extraction."""
    return _filter_tsfresh_long(_tsfresh_raw_long(histories, samples, spec=spec), spec=spec)


def tsfresh_input_review(
    histories: pl.DataFrame,
    samples: pl.DataFrame,
    *,
    spec: FeatureSpec,
) -> pl.DataFrame:
    """Review finite coverage before tsfresh receives long-format values."""
    return _tsfresh_input_review(_tsfresh_raw_long(histories, samples, spec=spec), spec=spec)


def _empty_tsfresh_long() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "symbol": pl.Utf8,
            "decision_bar_close_ms": pl.Int64,
            "window_hours": pl.Int64,
            "value_column": pl.Utf8,
            "tsfresh_id": pl.Utf8,
            "tsfresh_time": pl.Int64,
            "value": pl.Float64,
        }
    )


_DECISION_RELATIVE_PRICE_COLUMNS = {
    "open_rel_decision": "open",
    "high_rel_decision": "high",
    "low_rel_decision": "low",
    "close_rel_decision": "close",
}


def _tsfresh_source_columns(value_columns: tuple[str, ...]) -> tuple[str, ...]:
    source = [_DECISION_RELATIVE_PRICE_COLUMNS.get(column, column) for column in value_columns]
    return tuple(dict.fromkeys(source))


def _with_decision_relative_prices(
    frame: pl.DataFrame, value_columns: tuple[str, ...]
) -> pl.DataFrame:
    relative_columns = [
        (column, source)
        for column, source in _DECISION_RELATIVE_PRICE_COLUMNS.items()
        if column in value_columns
    ]
    if not relative_columns:
        return frame
    decision_close = pl.when(pl.col("decision_close") > 0.0).then(pl.col("decision_close"))
    return frame.with_columns(
        [
            (pl.col(source).cast(pl.Float64) / decision_close - 1.0).alias(column)
            for column, source in relative_columns
        ]
    )


def _tsfresh_raw_long(
    histories: pl.DataFrame,
    samples: pl.DataFrame,
    *,
    spec: FeatureSpec,
) -> pl.DataFrame:
    """Build raw causal long-format rows before finite coverage gating."""
    value_columns = tuple(spec.tsfresh_value_columns)
    source_columns = _tsfresh_source_columns(value_columns)
    _require_columns(
        histories,
        name="histories",
        columns=("symbol", "bar_close_ms", *source_columns),
    )
    _require_columns(samples, name="samples", columns=_KEY_COLUMNS)
    if not value_columns or not spec.windows_hours:
        return _empty_tsfresh_long()
    decisions = (
        samples.select(*_KEY_COLUMNS)
        .unique()
        .join(
            histories.select(
                "symbol",
                pl.col("bar_close_ms").alias("decision_bar_close_ms"),
                pl.col("close").cast(pl.Float64).alias("decision_close"),
            ),
            on=_KEY_COLUMNS,
            how="left",
        )
        .sort(*_KEY_COLUMNS)
    )
    windows = pl.DataFrame({"window_hours": [int(window) for window in spec.windows_hours]})
    sample_windows = decisions.join(windows, how="cross")
    hour_ms = 60 * 60 * 1000
    history = histories.select(
        pl.col("symbol").alias("history_symbol"), "bar_close_ms", *source_columns
    )
    window_rows = _with_decision_relative_prices(
        sample_windows.join_where(
            history,
            pl.col("symbol") == pl.col("history_symbol"),
            pl.col("bar_close_ms") <= pl.col("decision_bar_close_ms"),
            pl.col("bar_close_ms")
            > pl.col("decision_bar_close_ms") - (pl.col("window_hours") * hour_ms),
        ),
        value_columns,
    )
    return (
        window_rows.unpivot(
            index=["symbol", "decision_bar_close_ms", "window_hours", "bar_close_ms"],
            on=list(value_columns),
            variable_name="value_column",
            value_name="value",
        )
        .with_columns(
            pl.col("bar_close_ms").cast(pl.Int64).alias("tsfresh_time"),
            pl.col("value").cast(pl.Float64),
            pl.concat_str(
                [
                    pl.col("symbol"),
                    pl.col("decision_bar_close_ms").cast(pl.Utf8),
                    pl.col("window_hours").cast(pl.Utf8),
                ],
                separator="|",
            ).alias("tsfresh_id"),
        )
        .select(
            "symbol",
            "decision_bar_close_ms",
            "window_hours",
            "value_column",
            "tsfresh_id",
            "tsfresh_time",
            "value",
        )
        .sort("decision_bar_close_ms", "symbol", "window_hours", "value_column", "tsfresh_time")
    )


def _tsfresh_input_review(tsfresh_long: pl.DataFrame, *, spec: FeatureSpec) -> pl.DataFrame:
    required = set(spec.tsfresh_required_value_columns)
    schema = {
        "value_column": pl.Utf8,
        "window_hours": pl.Int64,
        "row_count": pl.Int64,
        "finite_count": pl.Int64,
        "null_count": pl.Int64,
        "nan_count": pl.Int64,
        "finite_rate": pl.Float64,
        "required": pl.Boolean,
        "status": pl.Utf8,
        "reason": pl.Utf8,
    }
    if tsfresh_long.is_empty():
        return pl.DataFrame(schema=schema)
    finite = pl.col("value").is_finite().fill_null(False)
    review = tsfresh_long.group_by("value_column", "window_hours").agg(
        pl.len().alias("row_count"),
        finite.cast(pl.Int64).sum().alias("finite_count"),
        pl.col("value").null_count().alias("null_count"),
        pl.col("value").is_nan().fill_null(False).cast(pl.Int64).sum().alias("nan_count"),
    )
    return (
        review.with_columns(
            (pl.col("finite_count") / pl.col("row_count")).alias("finite_rate"),
            pl.col("value_column").is_in(required).alias("required"),
        )
        .with_columns(
            pl.when(pl.col("required") & (pl.col("finite_rate") >= 1.0))
            .then(pl.lit("included_required"))
            .when(pl.col("required"))
            .then(pl.lit("rejected_required"))
            .when(pl.col("finite_rate") >= spec.tsfresh_optional_min_finite_rate)
            .then(pl.lit("included_optional"))
            .otherwise(pl.lit("rejected_optional"))
            .alias("status")
        )
        .with_columns(
            pl.when(pl.col("status") == "rejected_required")
            .then(pl.lit("required_non_finite"))
            .when(pl.col("status") == "rejected_optional")
            .then(pl.lit("optional_finite_rate_below_threshold"))
            .otherwise(pl.lit("finite_coverage_ok"))
            .alias("reason")
        )
        .select(*schema)
        .sort("value_column", "window_hours")
    )


def _filter_tsfresh_long(tsfresh_long: pl.DataFrame, *, spec: FeatureSpec) -> pl.DataFrame:
    if tsfresh_long.is_empty():
        return tsfresh_long
    review = _tsfresh_input_review(tsfresh_long, spec=spec)
    bad_required = review.filter(pl.col("status") == "rejected_required")
    if not bad_required.is_empty():
        pairs = ", ".join(
            f"{row['value_column']}:w{row['window_hours']}h={row['finite_rate']:.3f}"
            for row in bad_required.select("value_column", "window_hours", "finite_rate").to_dicts()
        )
        raise ValueError(f"required tsfresh inputs contain non-finite values: {pairs}")
    eligible = review.filter(pl.col("status").str.starts_with("included")).select(
        "value_column", "window_hours"
    )
    if eligible.is_empty():
        return _empty_tsfresh_long()
    return (
        tsfresh_long.join(eligible, on=["value_column", "window_hours"], how="semi")
        .filter(pl.col("value").is_finite().fill_null(False))
        .sort("decision_bar_close_ms", "symbol", "window_hours", "value_column", "tsfresh_time")
    )


def tsfresh_features(
    tsfresh_long: pl.DataFrame,
    *,
    spec: FeatureSpec,
) -> pl.DataFrame:
    """Extract `tsf__` columns through real tsflex feature descriptors."""
    _require_columns(
        tsfresh_long,
        name="tsfresh_long",
        columns=("symbol", "decision_bar_close_ms", "window_hours", "value_column", "value"),
    )
    if tsfresh_long.is_empty() or not spec.tsfresh_calculators:
        return pl.DataFrame(schema={"symbol": pl.Utf8, "decision_bar_close_ms": pl.Int64})
    tsfresh_long = tsfresh_long.unique(
        subset=("symbol", "decision_bar_close_ms", "window_hours", "value_column", "tsfresh_time")
    )
    rows: list[dict[str, object]] = []
    for grain in tsfresh_long.group_by("symbol", "decision_bar_close_ms", "window_hours"):
        keys, frame = grain
        symbol, decision_ms, window = keys
        segment = frame.pivot(
            index="tsfresh_time",
            on="value_column",
            values="value",
            aggregate_function="first",
        ).sort("tsfresh_time")
        rows.append(
            _tsflex_feature_row(
                str(symbol),
                int(decision_ms),
                int(window),
                segment,
                spec=spec,
            )
        )
    return _fill_tsfresh_nulls(_tsflex_rows_frame(rows)).sort("decision_bar_close_ms", "symbol")


def _tsflex_features_from_histories(
    histories: pl.DataFrame,
    samples: pl.DataFrame,
    *,
    spec: FeatureSpec,
) -> pl.DataFrame:
    if histories.is_empty() or samples.is_empty() or not spec.tsfresh_calculators:
        return pl.DataFrame(schema={"symbol": pl.Utf8, "decision_bar_close_ms": pl.Int64})
    _require_columns(histories, name="histories", columns=("symbol", "bar_close_ms"))
    rows: list[dict[str, object]] = []
    for symbol in samples.get_column("symbol").unique().sort().to_list():
        symbol_histories = histories.filter(pl.col("symbol") == symbol).sort("bar_close_ms")
        symbol_samples = samples.filter(pl.col("symbol") == symbol).select(*_KEY_COLUMNS).unique()
        if symbol_histories.is_empty() or symbol_samples.is_empty():
            continue
        data = _tsflex_symbol_frame(symbol_histories, spec=spec)
        if data.empty:
            continue
        selected_by_window = spec.generated_by_window()
        windows = tuple(selected_by_window) if selected_by_window else tuple(spec.windows_hours)
        for window in windows:
            decision_times = [
                int(value)
                for value in symbol_samples.get_column("decision_bar_close_ms").to_list()
                if value is not None
            ]
            starts = pd.to_datetime(
                [value - int(window) * 60 * 60 * 1000 for value in decision_times], unit="ms"
            )
            ends = pd.to_datetime(decision_times, unit="ms")
            if len(ends) == 0:
                continue
            collection = _tsflex_collection(
                int(window),
                tuple(spec.tsfresh_calculators),
                tuple(data.columns),
                spec=spec,
                selected_outputs=selected_by_window.get(int(window)),
            )
            if collection.get_nb_output_features() == 0:
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                extracted = collection.calculate(
                    data,
                    segment_start_idxs=starts,
                    segment_end_idxs=ends,
                    return_df=True,
                    include_final_window=True,
                    window_idx="end",
                    n_jobs=1,
                )
            for decision_ms, (_index, row) in zip(
                decision_times, extracted.iterrows(), strict=False
            ):
                out: dict[str, object] = {
                    "symbol": str(symbol),
                    "decision_bar_close_ms": decision_ms,
                }
                for column, value in row.items():
                    renamed = spec.tsflex_output_name(str(column), int(window))
                    out[renamed] = float(value) if pd.notna(value) else None
                rows.append(out)
    return (
        _fill_tsfresh_nulls(_tsflex_rows_frame(rows)).sort("decision_bar_close_ms", "symbol")
        if rows
        else pl.DataFrame(schema={"symbol": pl.Utf8, "decision_bar_close_ms": pl.Int64})
    )


def _source_name(column: str) -> str:
    return column.replace("base__", "base_").replace("__", "_").replace("|", "_")


def _source_cross_pairs(columns: tuple[str, ...]) -> tuple[tuple[str, str], ...]:
    semantic = (
        ("funding", "oi"),
        ("funding", "lsr"),
        ("funding", "taker"),
        ("oi", "taker"),
        ("oi", "lsr"),
        ("taker", "lsr"),
    )
    pairs: list[tuple[str, str]] = []
    for left_token, right_token in semantic:
        left = next((column for column in columns if left_token in column), None)
        right = next((column for column in columns if right_token in column), None)
        if left is not None and right is not None and left != right:
            pair = (left, right)
            if pair not in pairs and pair[::-1] not in pairs:
                pairs.append(pair)
    return tuple(pairs)


def _source_tsflex_output_name(raw_name: str, window: int, *, spec: FeatureSpec) -> str:
    raw = raw_name.replace("'", "").replace('"', "").replace("__w=manual", "")
    source_names = tuple(_source_name(column) for column in spec.source_tsfresh_value_columns)
    for left, right in _source_cross_pairs(source_names):
        if "lead_corr" in raw and left in raw and right in raw:
            return (
                f"{spec.source_cross_prefix.rstrip('_')}__"
                f"{_source_name(left)}_{_source_name(right)}__w{window}h__lead_corr"
            )
        if "corr" in raw and left in raw and right in raw:
            return (
                f"{spec.source_cross_prefix.rstrip('_')}__"
                f"{_source_name(left)}_{_source_name(right)}__w{window}h__corr"
            )
    calculator = next(
        (
            name
            for name in (*spec.source_tsfresh_calculators, "lead_corr", "corr")
            if raw.endswith(name)
        ),
        raw.rsplit("__", 1)[-1],
    )
    source = next((column for column in source_names if column in raw), raw)
    return (
        f"{spec.source_tsfresh_prefix.rstrip('_')}__"
        f"{_source_name(source)}__w{window}h__{calculator}"
    )


def source_tsflex_features(
    matrix: pl.DataFrame,
    samples: pl.DataFrame,
    *,
    spec: FeatureSpec,
) -> pl.DataFrame:
    """Extract source-series descriptors with real tsflex; never fill/interpolate source gaps."""
    columns = tuple(spec.source_tsfresh_value_columns)
    aliases = tuple(_source_name(column) for column in columns)
    if matrix.is_empty() or samples.is_empty() or not columns or not spec.windows_hours:
        return pl.DataFrame(schema={"symbol": pl.Utf8, "decision_bar_close_ms": pl.Int64})
    _require_columns(
        matrix, name="source matrix", columns=("symbol", "decision_bar_close_ms", *columns)
    )
    _require_columns(samples, name="samples", columns=_KEY_COLUMNS)
    rows: list[dict[str, object]] = []
    for symbol in samples.get_column("symbol").unique().sort().to_list():
        symbol_matrix = matrix.filter(pl.col("symbol") == symbol).sort("decision_bar_close_ms")
        symbol_samples = samples.filter(pl.col("symbol") == symbol).select(*_KEY_COLUMNS).unique()
        if symbol_matrix.is_empty() or symbol_samples.is_empty():
            continue
        data = _pandas_frame(
            symbol_matrix.select(
                "decision_bar_close_ms",
                *(
                    pl.col(column).alias(alias)
                    for column, alias in zip(columns, aliases, strict=True)
                ),
            )
        )
        data.index = pd.to_datetime(data.pop("decision_bar_close_ms"), unit="ms")
        data = data.astype(float)
        decisions = [
            int(value)
            for value in symbol_samples.get_column("decision_bar_close_ms").to_list()
            if value is not None
        ]
        if not decisions:
            continue
        selected_by_window = spec.source_generated_by_window()
        windows = tuple(selected_by_window) if selected_by_window else tuple(spec.windows_hours)
        for window in windows:
            starts = pd.to_datetime(
                [value - int(window) * 60 * 60 * 1000 for value in decisions], unit="ms"
            )
            ends = pd.to_datetime(decisions, unit="ms")
            collection = _source_tsflex_collection(
                int(window),
                tuple(spec.source_tsfresh_calculators),
                aliases,
                spec=spec,
                selected_outputs=selected_by_window.get(int(window)),
            )
            if collection.get_nb_output_features() == 0:
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                extracted = collection.calculate(
                    data,
                    segment_start_idxs=starts,
                    segment_end_idxs=ends,
                    return_df=True,
                    include_final_window=True,
                    window_idx="end",
                    n_jobs=1,
                )
            for decision_ms, (_index, row) in zip(decisions, extracted.iterrows(), strict=False):
                out: dict[str, object] = {
                    "symbol": str(symbol),
                    "decision_bar_close_ms": decision_ms,
                }
                for column, value in row.items():
                    out[_source_tsflex_output_name(str(column), int(window), spec=spec)] = (
                        float(value) if pd.notna(value) else None
                    )
                rows.append(out)
    return (
        _tsflex_rows_frame(rows).sort("decision_bar_close_ms", "symbol")
        if rows
        else pl.DataFrame(schema={"symbol": pl.Utf8, "decision_bar_close_ms": pl.Int64})
    )


def _source_tsflex_collection(
    window: int,
    calculators: tuple[str, ...],
    columns: tuple[str, ...],
    *,
    spec: FeatureSpec,
    selected_outputs: frozenset[str] | None = None,
) -> FeatureCollection:
    unknown = sorted(set(calculators) - set(_SOURCE_TSFLEX_FUNCTIONS))
    if unknown:
        raise ValueError(f"unsupported source tsflex calculators: {', '.join(unknown)}")
    descriptors = []
    for column in columns:
        for calculator in calculators:
            output_name = (
                f"{spec.source_tsfresh_prefix.rstrip('_')}__"
                f"{_source_name(column)}__w{window}h__{calculator}"
            )
            if selected_outputs is not None and output_name not in selected_outputs:
                continue
            descriptors.append(
                FeatureDescriptor(
                    FuncWrapper(_SOURCE_TSFLEX_FUNCTIONS[calculator], calculator),
                    series_name=column,
                )
            )
    if window >= 4:
        for left, right in _source_cross_pairs(columns):
            for name, func in (("corr", _source_corr), ("lead_corr", _source_lead_corr)):
                output_name = (
                    f"{spec.source_cross_prefix.rstrip('_')}__"
                    f"{_source_name(left)}_{_source_name(right)}__w{window}h__{name}"
                )
                if selected_outputs is not None and output_name not in selected_outputs:
                    continue
                descriptors.append(
                    FeatureDescriptor(
                        FuncWrapper(func, name),
                        series_name=(left, right),
                    )
                )
    return FeatureCollection(descriptors)


def _tsflex_rows_frame(rows: list[dict[str, object]]) -> pl.DataFrame:
    frame = pl.from_dicts(rows, infer_schema_length=None)
    value_columns = [column for column in frame.columns if column not in _KEY_COLUMNS]
    return frame.group_by(*_KEY_COLUMNS).agg(
        *(pl.col(column).drop_nulls().first().alias(column) for column in value_columns)
    )


def _pandas_frame(frame: pl.DataFrame) -> pd.DataFrame:
    """Convert Polars to pandas without requiring pyarrow."""
    return pd.DataFrame({column: frame.get_column(column).to_list() for column in frame.columns})


def _tsflex_symbol_frame(histories: pl.DataFrame, *, spec: FeatureSpec) -> pd.DataFrame:
    source_columns = _tsfresh_source_columns(tuple(spec.tsfresh_value_columns))
    missing = sorted(set(source_columns) - set(histories.columns))
    if missing:
        raise ValueError(f"histories missing required columns: {', '.join(missing)}")
    frame = histories.select("bar_close_ms", *source_columns).sort("bar_close_ms")
    pdf = _pandas_frame(frame)
    pdf.index = pd.to_datetime(pdf.pop("bar_close_ms"), unit="ms")
    if {"high", "low", "close"} <= set(pdf.columns):
        pdf["range"] = (pdf["high"] - pdf["low"]) / pdf["close"].replace(0.0, np.nan)
    return pdf.astype(float)


def _tsflex_feature_row(
    symbol: str,
    decision_ms: int,
    window: int,
    segment: pl.DataFrame,
    *,
    spec: FeatureSpec,
) -> dict[str, object]:
    pdf = _pandas_frame(segment)
    if "tsfresh_time" in pdf:
        pdf.index = pd.to_datetime(pdf.pop("tsfresh_time"), unit="ms")
    collection = _tsflex_collection(
        window, tuple(spec.tsfresh_calculators), tuple(pdf.columns), spec=spec
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        extracted = collection.calculate(
            pdf,
            segment_start_idxs=[pdf.index.min() - pd.Timedelta(milliseconds=1)],
            segment_end_idxs=[pdf.index.max()],
            return_df=True,
            include_final_window=True,
            window_idx="end",
            n_jobs=1,
        )
    row: dict[str, object] = {"symbol": symbol, "decision_bar_close_ms": decision_ms}
    for column, value in extracted.iloc[0].items():
        row[spec.tsflex_output_name(str(column), window)] = (
            float(value) if pd.notna(value) else None
        )
    return row


def _tsflex_collection(
    window: int,
    calculators: tuple[str, ...],
    columns: tuple[str, ...],
    *,
    spec: FeatureSpec,
    selected_outputs: frozenset[str] | None = None,
) -> FeatureCollection:
    unknown = sorted(set(calculators) - set(_TSFLEX_FUNCTIONS))
    if unknown:
        raise ValueError(f"unsupported tsflex calculators: {', '.join(unknown)}")
    descriptors = []
    for column in columns:
        for calculator in calculators:
            if calculator in _SPECTRAL_CALCULATORS and (
                window < 24 or column not in {"close", "volume"}
            ):
                continue
            output_name = f"{spec.tsfresh_prefix.rstrip('_')}__{column}__w{window}h__{calculator}"
            if selected_outputs is not None and output_name not in selected_outputs:
                continue
            descriptors.append(
                FeatureDescriptor(
                    FuncWrapper(_TSFLEX_FUNCTIONS[calculator], calculator),
                    series_name=column,
                )
            )
    if window >= 24:
        for left, right in (("close", "volume"), ("range", "volume")):
            if {left, right} <= set(columns):
                output_name = f"{spec.cross_prefix.rstrip('_')}__{left}_{right}__w{window}h__corr"
                if selected_outputs is not None and output_name not in selected_outputs:
                    continue
                descriptors.append(
                    FeatureDescriptor(
                        FuncWrapper(_corr, "corr"),
                        series_name=(left, right),
                    )
                )
    return FeatureCollection(descriptors)


def _finite(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return values[np.isfinite(values)]


def _relative(values: np.ndarray) -> np.ndarray:
    finite = _finite(values)
    if finite.size == 0 or finite[-1] == 0.0:
        return finite
    return finite / finite[-1] - 1.0


def _series(values: np.ndarray) -> np.ndarray:
    return _relative(values)


def _volume(values: np.ndarray) -> np.ndarray:
    return _finite(values)


def _mean(values: np.ndarray) -> float:
    values = _series(values)
    return float(np.mean(values)) if values.size else np.nan


def _std(values: np.ndarray) -> float:
    values = _series(values)
    return float(np.std(values, ddof=1)) if values.size >= 2 else np.nan


def _min(values: np.ndarray) -> float:
    values = _series(values)
    return float(np.min(values)) if values.size else np.nan


def _max(values: np.ndarray) -> float:
    values = _series(values)
    return float(np.max(values)) if values.size else np.nan


def _median(values: np.ndarray) -> float:
    values = _series(values)
    return float(np.median(values)) if values.size else np.nan


def _abs_max(values: np.ndarray) -> float:
    values = _series(values)
    return float(np.max(np.abs(values))) if values.size else np.nan


def _skew(values: np.ndarray) -> float:
    values = _series(values)
    if values.size < 3:
        return np.nan
    std = np.std(values)
    return float(np.mean(((values - np.mean(values)) / std) ** 3)) if std else np.nan


def _kurtosis(values: np.ndarray) -> float:
    values = _series(values)
    if values.size < 4:
        return np.nan
    std = np.std(values)
    return float(np.mean(((values - np.mean(values)) / std) ** 4) - 3.0) if std else np.nan


def _abs_energy(values: np.ndarray) -> float:
    values = _series(values)
    return float(np.sum(values * values)) if values.size else np.nan


def _q90_q10_range(values: np.ndarray) -> float:
    values = _series(values)
    return float(np.quantile(values, 0.9) - np.quantile(values, 0.1)) if values.size else np.nan


def _mean_abs_change(values: np.ndarray) -> float:
    values = _series(values)
    return float(np.mean(np.abs(np.diff(values)))) if values.size >= 2 else np.nan


def _last_minus_mean(values: np.ndarray) -> float:
    values = _series(values)
    return float(values[-1] - np.mean(values)) if values.size else np.nan


def _slope(values: np.ndarray) -> float:
    values = _series(values)
    if values.size < 2:
        return np.nan
    return float(np.polyfit(np.arange(values.size, dtype=float), values, 1)[0])


def _sample_count(values: np.ndarray) -> float:
    return float(_finite(values).size)


def _source_values(values: np.ndarray) -> np.ndarray:
    return _finite(values)


def _source_min(values: np.ndarray) -> float:
    values = _source_values(values)
    return float(np.min(values)) if values.size else np.nan


def _source_max(values: np.ndarray) -> float:
    values = _source_values(values)
    return float(np.max(values)) if values.size else np.nan


def _source_median(values: np.ndarray) -> float:
    values = _source_values(values)
    return float(np.median(values)) if values.size else np.nan


def _source_first(values: np.ndarray) -> float:
    values = _source_values(values)
    return float(values[0]) if values.size else np.nan


def _source_last(values: np.ndarray) -> float:
    values = _source_values(values)
    return float(values[-1]) if values.size else np.nan


def _source_last_minus_first(values: np.ndarray) -> float:
    values = _source_values(values)
    return float(values[-1] - values[0]) if values.size >= 2 else np.nan


def _source_skew(values: np.ndarray) -> float:
    values = _source_values(values)
    if values.size < 3:
        return np.nan
    std = np.std(values)
    return float(np.mean(((values - np.mean(values)) / std) ** 3)) if std else np.nan


def _source_kurtosis(values: np.ndarray) -> float:
    values = _source_values(values)
    if values.size < 4:
        return np.nan
    std = np.std(values)
    return float(np.mean(((values - np.mean(values)) / std) ** 4) - 3.0) if std else np.nan


def _source_abs_energy(values: np.ndarray) -> float:
    values = _source_values(values)
    return float(np.sum(values * values)) if values.size else np.nan


def _source_mean_abs_change(values: np.ndarray) -> float:
    values = _source_values(values)
    return float(np.mean(np.abs(np.diff(values)))) if values.size >= 2 else np.nan


def _source_mean_change(values: np.ndarray) -> float:
    values = _source_values(values)
    return float(np.mean(np.diff(values))) if values.size >= 2 else np.nan


def _source_positive_change_rate(values: np.ndarray) -> float:
    values = _source_values(values)
    changes = np.diff(values)
    return float(np.mean(changes > 0.0)) if changes.size else np.nan


def _source_negative_change_rate(values: np.ndarray) -> float:
    values = _source_values(values)
    changes = np.diff(values)
    return float(np.mean(changes < 0.0)) if changes.size else np.nan


def _source_valid_ratio(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    return float(np.isfinite(values).mean()) if values.size else np.nan


def _source_max_valid_gap(values: np.ndarray) -> float:
    valid = np.flatnonzero(np.isfinite(np.asarray(values, dtype=float)))
    return float(np.max(np.diff(valid))) if valid.size >= 2 else np.nan


def _source_slope(values: np.ndarray) -> float:
    values = _source_values(values)
    if values.size < 2:
        return np.nan
    return float(np.polyfit(np.arange(values.size, dtype=float), values, 1)[0])


def _source_trend_r2(values: np.ndarray) -> float:
    values = _source_values(values)
    if values.size < 3:
        return np.nan
    x = np.arange(values.size, dtype=float)
    slope, intercept = np.polyfit(x, values, 1)
    fitted = slope * x + intercept
    total = float(np.sum((values - np.mean(values)) ** 2))
    residual = float(np.sum((values - fitted) ** 2))
    return float(1.0 - residual / total) if total else np.nan


def _source_q90_q10_range(values: np.ndarray) -> float:
    values = _source_values(values)
    return float(np.quantile(values, 0.9) - np.quantile(values, 0.1)) if values.size else np.nan


def _source_corr(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    valid = np.isfinite(left) & np.isfinite(right)
    if valid.sum() < 3:
        return np.nan
    return float(np.corrcoef(left[valid], right[valid])[0, 1])


def _source_lead_corr(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if min(left.size, right.size) < 4:
        return np.nan
    lead = left[:-1]
    lagged = right[1:]
    valid = np.isfinite(lead) & np.isfinite(lagged)
    if valid.sum() < 3:
        return np.nan
    return float(np.corrcoef(lead[valid], lagged[valid])[0, 1])


def _fft_power(values: np.ndarray) -> np.ndarray:
    values = _series(values)
    if values.size < 24:
        return np.array([], dtype=float)
    demeaned = (values - np.mean(values)) * np.hanning(values.size)
    power = np.abs(np.fft.rfft(demeaned))[1:] ** 2
    total = power.sum()
    return power / total if total > 0.0 else np.array([], dtype=float)


def _fft_low_power_share(values: np.ndarray) -> float:
    power = _fft_power(values)
    return float(power[: max(1, power.size // 3)].sum()) if power.size else np.nan


def _fft_high_power_share(values: np.ndarray) -> float:
    power = _fft_power(values)
    return float(power[-max(1, power.size // 3) :].sum()) if power.size else np.nan


def _spectral_entropy(values: np.ndarray) -> float:
    power = _fft_power(values)
    return (
        float(-(power * np.log2(power + 1e-12)).sum() / np.log2(power.size))
        if power.size > 1
        else np.nan
    )


def _dominant_period_hours(values: np.ndarray) -> float:
    power = _fft_power(values)
    if power.size == 0:
        return np.nan
    return float(_series(values).size / (int(np.argmax(power)) + 1))


def _corr(left: np.ndarray, right: np.ndarray) -> float:
    left_values = _relative(left)
    right_values = _volume(right)
    size = min(left_values.size, right_values.size)
    if size < 3:
        return np.nan
    return float(np.corrcoef(left_values[-size:], right_values[-size:])[0, 1])


_SPECTRAL_CALCULATORS = {
    "fft_low_power_share",
    "fft_high_power_share",
    "spectral_entropy",
    "dominant_period_hours",
}
_SOURCE_TSFLEX_FUNCTIONS = {
    "first_value": _source_first,
    "last_value": _source_last,
    "last_minus_first": _source_last_minus_first,
    "minimum": _source_min,
    "maximum": _source_max,
    "median": _source_median,
    "skewness": _source_skew,
    "kurtosis": _source_kurtosis,
    "abs_energy": _source_abs_energy,
    "mean_change": _source_mean_change,
    "mean_abs_change": _source_mean_abs_change,
    "positive_change_rate": _source_positive_change_rate,
    "negative_change_rate": _source_negative_change_rate,
    "valid_ratio": _source_valid_ratio,
    "max_valid_gap": _source_max_valid_gap,
    "trend_slope": _source_slope,
    "trend_r2": _source_trend_r2,
    "q90_q10_range": _source_q90_q10_range,
    "sample_count": _sample_count,
}


_TSFLEX_FUNCTIONS = {
    "mean": _mean,
    "standard_deviation": _std,
    "minimum": _min,
    "maximum": _max,
    "median": _median,
    "absolute_maximum": _abs_max,
    "skewness": _skew,
    "kurtosis": _kurtosis,
    "abs_energy": _abs_energy,
    "q90_q10_range": _q90_q10_range,
    "mean_abs_change": _mean_abs_change,
    "last_minus_mean": _last_minus_mean,
    "slope": _slope,
    "sample_count": _sample_count,
    "fft_low_power_share": _fft_low_power_share,
    "fft_high_power_share": _fft_high_power_share,
    "spectral_entropy": _spectral_entropy,
    "dominant_period_hours": _dominant_period_hours,
}


def _fill_tsfresh_nulls(frame: pl.DataFrame) -> pl.DataFrame:
    if frame.is_empty():
        return frame
    fill_exprs = []
    for column in frame.columns:
        if not column.startswith(("tsf__", "cross__")):
            continue
        median = frame.select(
            pl.col(column).cast(pl.Float64).filter(pl.col(column).is_finite()).median()
        ).item()
        if median is not None:
            fill_exprs.append(
                pl.when(pl.col(column).is_finite().fill_null(False))
                .then(pl.col(column))
                .otherwise(float(median))
                .alias(column)
            )
    return frame.with_columns(fill_exprs) if fill_exprs else frame


def _join_base_features(samples: pl.DataFrame, base_features: pl.DataFrame) -> pl.DataFrame:
    _require_columns(samples, name="samples", columns=(*_KEY_COLUMNS, "horizon_hours"))
    _require_columns(base_features, name="base_features", columns=_KEY_COLUMNS)
    _ensure_unique_grain(samples, name="sample", columns=(*_KEY_COLUMNS, "horizon_hours"))
    _ensure_unique_grain(base_features, name="base feature", columns=_KEY_COLUMNS)
    marker = "__base_feature_present"
    joined = samples.join(
        base_features.with_columns(pl.lit(True).alias(marker)),
        on=list(_KEY_COLUMNS),
        how="left",
    )
    if joined.filter(pl.col(marker).is_null()).height:
        raise ValueError("missing base feature rows for sample grain")
    return joined.drop(marker)


def predict_candidates(
    observations: pl.DataFrame,
    base_features: pl.DataFrame,
    *,
    horizons: tuple[int, ...],
    spec: FeatureSpec,
) -> pl.DataFrame:
    """Expand current observations across horizons without future label columns."""
    _require_columns(observations, name="observations", columns=_KEY_COLUMNS)
    decisions = observations.select(*_KEY_COLUMNS).unique().sort(*_KEY_COLUMNS)
    horizon_values = [int(horizon) for horizon in (horizons or spec.horizons)]
    if not horizon_values or decisions.is_empty():
        return pl.DataFrame(schema={"symbol": pl.Utf8, "decision_bar_close_ms": pl.Int64})
    samples = decisions.join(pl.DataFrame({"horizon_hours": horizon_values}), how="cross")
    return (
        _join_base_features(samples, base_features)
        .with_columns(
            pl.col("horizon_hours").cast(pl.Int64).alias(f"{spec.context_prefix}horizon_hours")
        )
        .sort("decision_bar_close_ms", "symbol", "horizon_hours")
    )


def train_candidates(
    labels: pl.DataFrame,
    base_features: pl.DataFrame,
    *,
    spec: FeatureSpec,
) -> pl.DataFrame:
    """Join labeled samples to known-at-close base features."""
    _require_columns(labels, name="labels", columns=(*_KEY_COLUMNS, "horizon_hours"))
    return (
        _join_base_features(labels, base_features)
        .with_columns(
            pl.col("horizon_hours").cast(pl.Int64).alias(f"{spec.context_prefix}horizon_hours")
        )
        .sort("decision_bar_close_ms", "symbol", "horizon_hours")
    )


def _feature_base_with_tsfresh(
    observations: pl.DataFrame,
    histories: pl.DataFrame | None,
    samples: pl.DataFrame,
    *,
    spec: FeatureSpec,
) -> pl.DataFrame:
    base = base_features(observations, spec=spec)
    if spec.source_tsfresh_value_columns and spec.windows_hours:
        source_generated = source_tsflex_features(base, samples, spec=spec)
        if not source_generated.is_empty():
            base = base.join(source_generated, on=list(_KEY_COLUMNS), how="left")
    if (
        histories is None
        or histories.is_empty()
        or not spec.tsfresh_value_columns
        or not spec.windows_hours
    ):
        return base
    generated = _tsflex_features_from_histories(histories, samples, spec=spec)
    if generated.is_empty():
        return base
    return base.join(generated, on=list(_KEY_COLUMNS), how="left")


def financial_history_features(
    histories: pl.DataFrame,
    samples: pl.DataFrame,
    *,
    spec: FeatureSpec,
) -> pl.DataFrame:
    """Build causal financial base features from OHLCV history at decision grain."""
    if histories.is_empty():
        return pl.DataFrame(schema={"symbol": pl.Utf8, "decision_bar_close_ms": pl.Int64})
    _require_columns(histories, name="histories", columns=("symbol", "bar_close_ms", "close"))
    _require_columns(samples, name="samples", columns=_KEY_COLUMNS)
    decisions = samples.select(*_KEY_COLUMNS).unique().sort(*_KEY_COLUMNS)
    hour_ms = 60 * 60 * 1000
    windows = (3, 6, 12, 24)
    rows: list[dict[str, object]] = []
    for sample in decisions.to_dicts():
        symbol = str(sample["symbol"])
        decision_ms = int(sample["decision_bar_close_ms"])
        history = histories.filter(
            (pl.col("symbol") == symbol)
            & (pl.col("bar_close_ms") <= decision_ms)
            & (pl.col("bar_close_ms") > decision_ms - (max(windows) * hour_ms))
        ).sort("bar_close_ms")
        row: dict[str, object] = {"symbol": symbol, "decision_bar_close_ms": decision_ms}
        closes = (
            history.get_column("close").cast(pl.Float64).to_list()
            if "close" in history.columns
            else []
        )
        volumes = (
            history.get_column("volume").cast(pl.Float64).to_list()
            if "volume" in history.columns
            else []
        )
        highs = (
            history.get_column("high").cast(pl.Float64).to_list()
            if "high" in history.columns
            else closes
        )
        lows = (
            history.get_column("low").cast(pl.Float64).to_list()
            if "low" in history.columns
            else closes
        )
        if closes:
            current = float(closes[-1])
            for window in windows:
                count = min(window, len(closes))
                segment = [float(value) for value in closes[-count:]]
                if len(segment) >= 2 and segment[0] != 0.0:
                    ret = (segment[-1] / segment[0] - 1.0) * 100.0
                    row[f"{spec.base_prefix}ret_{window}h"] = ret
                    returns = [
                        (segment[index] / segment[index - 1] - 1.0) * 100.0
                        for index in range(1, len(segment))
                        if segment[index - 1] != 0.0
                    ]
                    vol = _std(returns)
                    row[f"{spec.base_prefix}realized_vol_{window}h"] = vol
                    path = sum(abs(value) for value in returns)
                    row[f"{spec.base_prefix}trend_efficiency_{window}h"] = (
                        abs(ret) / path if path else None
                    )
                    row[f"{spec.base_prefix}vol_adjusted_momentum_{window}h"] = (
                        ret / vol if vol else None
                    )
                high_segment = [float(value) for value in highs[-count:]]
                low_segment = [float(value) for value in lows[-count:]]
                high = max(high_segment) if high_segment else None
                low = min(low_segment) if low_segment else None
                if high is not None and low is not None and high > low:
                    row[f"{spec.base_prefix}close_position_{window}h"] = (current - low) / (
                        high - low
                    )
                    row[f"{spec.base_prefix}distance_to_{window}h_high_pct"] = (
                        (current / high - 1.0) * 100.0 if high else None
                    )
                    row[f"{spec.base_prefix}distance_to_{window}h_low_pct"] = (
                        (current / low - 1.0) * 100.0 if low else None
                    )
            if volumes:
                for window in windows:
                    segment = [float(value) for value in volumes[-min(window, len(volumes)) :]]
                    mean = sum(segment) / len(segment) if segment else None
                    std = _std(segment)
                    latest = float(segment[-1]) if segment else None
                    row[f"{spec.base_prefix}volume_z_{window}h"] = (
                        (latest - mean) / std
                        if latest is not None and mean is not None and std
                        else None
                    )
                    row[f"{spec.base_prefix}volume_ratio_{window}h"] = (
                        latest / mean if latest is not None and mean else None
                    )
                row[f"{spec.base_prefix}dollar_volume_proxy"] = current * float(volumes[-1])
        rows.append(row)
    return pl.DataFrame(rows).sort(*_KEY_COLUMNS)


def _std(values: list[float]) -> float | None:
    finite = [float(value) for value in values if value is not None]
    if len(finite) < 2:
        return None
    mean = sum(finite) / len(finite)
    return (sum((value - mean) ** 2 for value in finite) / (len(finite) - 1)) ** 0.5


def _feature_candidate_columns(
    frame: pl.DataFrame,
    *,
    spec: SelectSpec,
) -> tuple[str, ...]:
    numeric_dtypes = {
        pl.Int8,
        pl.Int16,
        pl.Int32,
        pl.Int64,
        pl.UInt8,
        pl.UInt16,
        pl.UInt32,
        pl.UInt64,
        pl.Float32,
        pl.Float64,
    }
    return tuple(
        column
        for column, dtype in zip(frame.columns, frame.dtypes, strict=True)
        if column.startswith(spec.feature_prefixes) and dtype in numeric_dtypes
    )


def _fold_masks(
    frame: pl.DataFrame,
    folds: tuple[object, ...],
) -> tuple[pl.Expr, pl.Expr, tuple[int, ...]]:
    train_expr: pl.Expr | None = None
    valid_expr: pl.Expr | None = None
    fold_ids: list[int] = []
    for fold in folds:
        fold_ids.append(int(getattr(fold, "fold_id", 0)))
        train_window = getattr(fold, "train_window", None)
        valid_window = getattr(fold, "valid_window", None)
        if train_window is None or valid_window is None:
            train_part = pl.lit(True)
            valid_part = pl.lit(False)
        else:
            train_part = (pl.col("decision_bar_close_ms") >= int(train_window.start_ms)) & (
                pl.col("decision_bar_close_ms") < int(train_window.end_ms)
            )
            valid_part = (pl.col("decision_bar_close_ms") >= int(valid_window.start_ms)) & (
                pl.col("decision_bar_close_ms") < int(valid_window.end_ms)
            )
        train_expr = train_part if train_expr is None else train_expr | train_part
        valid_expr = valid_part if valid_expr is None else valid_expr | valid_part
    if train_expr is None:
        train_expr = pl.lit(True)
    if valid_expr is None:
        valid_expr = pl.lit(False)
    return train_expr, valid_expr, tuple(fold_ids)


def select_manifest(
    feature_candidates: pl.DataFrame,
    folds: tuple[object, ...],
    *,
    spec: SelectSpec,
    artifact_id: str,
    feature_spec: FeatureSpec | None = None,
    label_contract_id: str = "path_prototype",
) -> ProposalFeatureManifest:
    """Select candidate feature columns using train-fold rows only."""
    _require_columns(
        feature_candidates,
        name="feature_candidates",
        columns=(*_KEY_COLUMNS, "horizon_hours"),
    )
    if spec.max_features < spec.min_features:
        raise ValueError("max_features must be >= min_features")
    candidate_columns = _feature_candidate_columns(feature_candidates, spec=spec)
    if not candidate_columns:
        raise ValueError("feature_candidates contain no selectable feature columns")
    train_expr, valid_expr, fold_ids = _fold_masks(feature_candidates, folds)
    train_frame = feature_candidates.filter(train_expr)
    valid_frame = feature_candidates.filter(valid_expr)
    if train_frame.is_empty():
        raise ValueError("feature selection requires non-empty train-fold rows")
    scores: list[tuple[str, float, int]] = []
    for index, column in enumerate(candidate_columns):
        value = train_frame.select(pl.col(column).cast(pl.Float64).var()).item()
        score = 0.0 if value is None else float(value)
        scores.append((column, score, index))
    ranked = sorted(scores, key=lambda item: (-item[1], item[2]))
    positive = [column for column, score, _index in ranked if score > 0.0]
    if len(positive) < spec.min_features:
        selected = [column for column, _score, _index in ranked[: spec.min_features]]
    else:
        selected = positive[: spec.max_features]
    selected = selected[: spec.max_features]
    selected_frame = feature_candidates.select(*_KEY_COLUMNS, "horizon_hours", *selected)
    manifest_spec = feature_spec or FeatureSpec(
        horizons=tuple(
            int(item)
            for item in feature_candidates.get_column("horizon_hours").unique().sort().to_list()
        )
    )
    return ProposalFeatureManifest(
        artifact_id=artifact_id,
        spec=manifest_spec,
        selected_columns=tuple(selected),
        candidate_feature_columns=candidate_columns,
        fold_ids=fold_ids,
        fit_row_count=train_frame.height,
        validation_row_count=valid_frame.height,
        schema_hash=_schema_hash(selected_frame),
        label_column=spec.label_column,
        label_contract_id=label_contract_id,
        weight_column=spec.weight_column,
        selection_metric="train_variance",
        created_at=datetime.now(UTC).isoformat(),
    )


__all__ = [
    "AcceptanceRecord",
    "AcceptedFeatureManifest",
    "FeatureManifestBase",
    "FeatureSpec",
    "ProposalFeatureManifest",
    "SelectSpec",
    "FeatureManifest",
    "base_features",
    "financial_history_features",
    "predict_candidates",
    "train_candidates",
    "tsfresh_features",
    "tsfresh_input_review",
    "tsfresh_long",
    "source_tsflex_features",
    "select_manifest",
]
