"""Import-safe AI data contracts.

These contracts intentionally avoid market, research, strategy, and PyTorch concepts.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

SplitName = Literal["train", "valid", "test"]
SPLIT_NAMES = ("train", "valid", "test")

WindowTensor = tuple[tuple[tuple[float, ...], ...], ...]


@dataclass(frozen=True)
class WindowDataset:
    features: WindowTensor
    feature_columns: tuple[str, ...]
    splits: tuple[SplitName, ...]
    seq_len: int
    stride: int

    def __post_init__(self) -> None:
        _validate_windows(
            self.features,
            feature_columns=self.feature_columns,
            splits=self.splits,
            seq_len=self.seq_len,
            stride=self.stride,
        )


@dataclass(frozen=True)
class CodeSequence:
    codes: tuple[int, ...]
    distances: tuple[float, ...]
    row_index: tuple[int, ...]
    splits: tuple[SplitName, ...]

    def __post_init__(self) -> None:
        lengths = {len(self.codes), len(self.distances), len(self.row_index), len(self.splits)}
        if len(lengths) != 1:
            raise ValueError("codes, distances, row_index, and splits must have equal lengths")
        invalid_splits = [split for split in self.splits if split not in SPLIT_NAMES]
        if invalid_splits:
            raise ValueError("splits must be one of: train, valid, test")
        if any(code < 0 for code in self.codes):
            raise ValueError("codes must be non-negative integers")
        if any(not math.isfinite(distance) or distance < 0.0 for distance in self.distances):
            raise ValueError("distances must be finite non-negative floats")


def _validate_windows(
    features: WindowTensor,
    *,
    feature_columns: tuple[str, ...],
    splits: tuple[SplitName, ...],
    seq_len: int,
    stride: int,
) -> None:
    if seq_len <= 0:
        raise ValueError("seq_len must be positive")
    if stride <= 0:
        raise ValueError("stride must be positive")
    if not feature_columns:
        raise ValueError("feature_columns must be non-empty")
    if not features:
        raise ValueError("features must contain at least one window")
    if len(features) != len(splits):
        raise ValueError("splits length must match feature window count")
    invalid_splits = [split for split in splits if split not in SPLIT_NAMES]
    if invalid_splits:
        raise ValueError("splits must be one of: train, valid, test")
    feature_width = len(feature_columns)
    for window in features:
        if len(window) != seq_len:
            raise ValueError("each feature window must match seq_len")
        for row in window:
            if len(row) != feature_width:
                raise ValueError("each feature row must match feature_columns width")
            if any(not math.isfinite(value) for value in row):
                raise ValueError("feature values must be finite floats")
