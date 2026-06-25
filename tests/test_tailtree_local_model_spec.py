from qooi.scanner.tailrun.core import LocalModelSpec


def test_local_model_spec_dumps_train_and_score_contract() -> None:
    spec = LocalModelSpec(
        role="promoter",
        label_column="promoter_label",
        weight_column="promoter_weight",
        score_column="promotion_score",
        objective="tail_event_lift",
    )

    assert spec.model_dump() == {
        "role": "promoter",
        "label_column": "promoter_label",
        "weight_column": "promoter_weight",
        "score_column": "promotion_score",
        "objective": "tail_event_lift",
    }
