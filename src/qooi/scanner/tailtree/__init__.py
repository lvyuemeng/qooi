"""Public tailtree API."""

from qooi.scanner.tailtree.evidence import (
    leaf_context_frame,
    leaf_evidence_frame,
    score_bucket_evidence_frame,
    select_tail_leaves,
)
from qooi.scanner.tailtree.model import (
    GPDParams,
    TailTreeModel,
    TailtreeTrainingFrame,
    TrainConfig,
    TreeMetadata,
    _gpd_nll_eval,
    _gpd_xi_objective,
    _leaf_id_vector,
    _tailtree_outcome_by_decision,
    label_tail_exceedances,
    tailtree_training_frame,
)

__all__ = [
    "GPDParams",
    "TailTreeModel",
    "TailtreeTrainingFrame",
    "TrainConfig",
    "TreeMetadata",
    "_gpd_nll_eval",
    "_gpd_xi_objective",
    "_leaf_id_vector",
    "_tailtree_outcome_by_decision",
    "label_tail_exceedances",
    "tailtree_training_frame",
    "leaf_context_frame",
    "leaf_evidence_frame",
    "score_bucket_evidence_frame",
    "select_tail_leaves",
]
