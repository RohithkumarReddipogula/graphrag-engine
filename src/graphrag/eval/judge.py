"""LLM judge for test answers (docs/PLAN.md, M5.5, approved 2026-10-09).

meta-llama/llama-3.3-70b-instruct on the pinned OpenRouter endpoint parasail/fp8 (a different model
family than the gpt-oss generator), temperature 0, strict JSON output, ITEMS_PER_CALL items per call.
The judge sees the question, the gold answer with its accepted aliases, and the answer; never the system.
"unknown" answers are incorrect without a judge call. Each unique (question, normalised answer) is judged
once. Invalid output is retried once (separately cached); if it fails again, the items are "unsure".
"""

import hashlib
import json
import time

from pydantic import BaseModel, ValidationError

from graphrag.eval.answers import normalize_answer
from graphrag.extraction.extract import QuotaExhausted, _is_daily_quota, _is_retryable
from graphrag.llm.client import CachedLLM, LLMResult

ITEMS_PER_CALL = 25
VERDICTS = ("correct", "incorrect", "unsure")

SYSTEM = """You grade answers to questions against a gold answer.

For each item you get a question, the gold answer with its accepted aliases, and an answer to grade.
- "correct": the answer names the same entity or value as the gold answer. Other wording, spellings,
  aliases, abbreviations and date formats are fine.
- "incorrect": the answer names a different entity or value, or is less specific than the question asks
  for (for example only a year when the question asks for a full date, or a country when it asks for a
  city).
- "unsure": you cannot tell.
Judge only against the gold answer; do not use your own knowledge of the facts. Give a reason of at most
15 words. Return exactly one verdict for every item_id you were given."""


class Verdict(BaseModel):
    item_id: str
    verdict: str
    reason: str


class Verdicts(BaseModel):
    verdicts: list[Verdict]


def response_options() -> dict:
    schema = {
        "type": "object",
        "properties": {"verdicts": {"type": "array", "items": {
            "type": "object",
            "properties": {"item_id": {"type": "string"}, "verdict": {"type": "string", "enum": list(VERDICTS)},
                           "reason": {"type": "string"}},
            "required": ["item_id", "verdict", "reason"], "additionalProperties": False}}},
        "required": ["verdicts"], "additionalProperties": False,
    }
    return {"response_format": {"type": "json_schema", "json_schema": {"name": "answer_verdicts", "strict": True, "schema": schema}}}


def item_id(question_id: str, prediction: str) -> str:
    return "j" + hashlib.sha1(f"{question_id}|{normalize_answer(prediction)}".encode("utf-8")).hexdigest()[:10]


def build_items(rows_by_system: dict[str, list[dict]], accepted: dict[str, list[str]]) -> dict[str, dict]:
    """Unique (question, normalised answer) items over all systems; "unknown" answers are not items."""
    items: dict[str, dict] = {}
    for system in sorted(rows_by_system):
        for r in rows_by_system[system]:
            if r["unknown"]:
                continue
            iid = item_id(r["id"], r["prediction"])
            items.setdefault(iid, {"item_id": iid, "question_id": r["id"], "type": r["type"],
                                   "question": r["question"], "gold": r["answer"],
                                   "accepted": accepted[r["id"]], "prediction": r["prediction"]})
    return dict(sorted(items.items()))


def batch_prompt(batch: list[dict]) -> str:
    blocks = [f"item_id: {it['item_id']}\nquestion: {it['question']}\n"
              f"gold answer and accepted aliases: {'; '.join(it['accepted'])}\nanswer to grade: {it['prediction']}"
              for it in batch]
    return "Items:\n\n" + "\n\n---\n\n".join(blocks)


def validate(raw: str, ids: list[str]) -> dict[str, Verdict]:
    parsed = Verdicts.model_validate(json.loads(raw))
    got = {v.item_id: v for v in parsed.verdicts}
    if sorted(got) != sorted(ids) or any(v.verdict not in VERDICTS for v in got.values()):
        raise ValueError(f"item ids or verdicts differ: missing {sorted(set(ids) - set(got))}")
    return got


def judge_batch(llm: CachedLLM, batch: list[dict], max_waits: int = 5) -> tuple[dict[str, Verdict], list[LLMResult], bool]:
    prompt = batch_prompt(batch)
    ids = [it["item_id"] for it in batch]
    calls: list[LLMResult] = []
    for attempt in (1, 2):
        salt = None if attempt == 1 else "retry-1"
        for wait in range(max_waits + 1):
            try:
                out = llm.complete(prompt, system=SYSTEM, options=response_options(), salt=salt)
                break
            except Exception as exc:  # noqa: BLE001 - classify provider errors
                if not _is_retryable(exc):
                    raise
                if _is_daily_quota(exc) or wait == max_waits:
                    raise QuotaExhausted(str(exc)[:300]) from exc
                time.sleep(60)
        calls.append(out)
        try:
            return validate(out.text, ids), calls, True
        except (ValidationError, ValueError, json.JSONDecodeError):
            continue
    return {i: Verdict(item_id=i, verdict="unsure", reason="invalid output twice") for i in ids}, calls, False
