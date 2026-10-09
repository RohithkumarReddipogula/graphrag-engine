import json

import pytest

from graphrag.eval.judge import build_items, item_id, validate


def row(qid, pred, unknown=False):
    return {"id": qid, "type": "comparison", "question": "q?", "answer": "Paris", "prediction": pred, "unknown": unknown}


def test_items_are_unique_per_question_and_normalised_answer_and_skip_unknown():
    rows = {"a": [row("q1", "Paris"), row("q2", "unknown", unknown=True)],
            "b": [row("q1", "paris."), row("q1", "Lyon")]}
    items = build_items(rows, {"q1": ["Paris"], "q2": ["X"]})
    assert len(items) == 2                                   # "Paris" and "paris." are one item; unknown skipped
    assert item_id("q1", "Paris") == item_id("q1", "The Paris") == item_id("q1", "paris.")
    assert all("system" not in it for it in items.values())  # the judge never sees the system


def test_validate_requires_every_item_and_a_known_verdict():
    ok = {"verdicts": [{"item_id": "j1", "verdict": "correct", "reason": "r"}]}
    assert set(validate(json.dumps(ok), ["j1"])) == {"j1"}
    with pytest.raises(ValueError):
        validate(json.dumps({"verdicts": []}), ["j1"])
    with pytest.raises(ValueError):
        validate(json.dumps({"verdicts": [{"item_id": "j1", "verdict": "maybe", "reason": "r"}]}), ["j1"])
