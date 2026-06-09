from __future__ import annotations

import importlib

import pytest

from qooi.dynamic.contracts import CodeSequence, WindowDataset


def test_dynamic_contracts_import_without_torch() -> None:
    importlib.import_module("qooi.dynamic.contracts")


def test_window_dataset_rejects_provenance_and_invalid_shape() -> None:
    with pytest.raises(TypeError):
        WindowDataset(  # type: ignore[call-arg]
            features=(((0.1, 0.2),),),
            row_index=(10,),
            feature_columns=("a", "b"),
            splits=("train",),
            seq_len=1,
            stride=1,
        )

    with pytest.raises(ValueError, match="seq_len"):
        WindowDataset(
            features=(((0.1, 0.2),),),
            feature_columns=("a", "b"),
            splits=("train",),
            seq_len=2,
            stride=1,
        )


def test_code_sequence_validates_lengths_and_values() -> None:
    with pytest.raises(ValueError, match="equal lengths"):
        CodeSequence(codes=(1,), distances=(0.1, 0.2), row_index=(0,), splits=("train",))

    with pytest.raises(ValueError, match="non-negative"):
        CodeSequence(codes=(-1,), distances=(0.1,), row_index=(0,), splits=("train",))


