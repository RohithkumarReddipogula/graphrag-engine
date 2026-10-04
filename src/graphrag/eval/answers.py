"""Answer scoring as in the official 2WikiMultihopQA evaluation script v1.1
(github.com/Alab-NII/2wikimultihop, 2wikimultihop_evaluate_v1.1.py, itself adapted from HotpotQA).

normalize_answer, f1_score and exact_match_score are copied from that script with only formatting
changes. Copyright the 2WikiMultihopQA authors, licensed under the Apache License, Version 2.0
(http://www.apache.org/licenses/LICENSE-2.0). The max over accepted answers (gold + Wikidata aliases and
demonyms) follows its update_answer.
"""

import re
import string
from collections import Counter


def normalize_answer(s: str) -> str:
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction: str, ground_truth: str) -> tuple[float, float, float]:
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    zero = (0.0, 0.0, 0.0)
    if normalized_prediction in ["yes", "no", "noanswer"] and normalized_prediction != normalized_ground_truth:
        return zero
    if normalized_ground_truth in ["yes", "no", "noanswer"] and normalized_prediction != normalized_ground_truth:
        return zero

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return zero
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall


def exact_match_score(prediction: str, ground_truth: str) -> bool:
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def score(prediction: str, accepted: list[str]) -> dict[str, float]:
    """EM and F1, each the max over all accepted answers (as update_answer in v1.1)."""
    em = max(float(exact_match_score(prediction, g)) for g in accepted)
    f1 = max(f1_score(prediction, g)[0] for g in accepted)
    return {"em": em, "f1": f1}
