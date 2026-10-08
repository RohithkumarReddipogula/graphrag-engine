"""LLM check for borderline pairs (docs/PLAN.md, M3, step 3). gpt-oss-120b on the pinned endpoint,
strict JSON output, PAIRS_PER_CALL pairs per call. Each pair shows both mentions with name, type,
description and their source paragraph. Verdicts: same, different, unsure. Unsure means don't merge.
Invalid output is retried once (separately cached); a batch that fails twice gets "unsure" for every pair.
"""

import json
import time

from pydantic import BaseModel, ValidationError

from graphrag.extraction.extract import QuotaExhausted, _is_daily_quota, _is_retryable
from graphrag.llm.client import CachedLLM, LLMResult
from graphrag.resolution.mentions import Mention

PAIRS_PER_CALL = 10
PARAGRAPH_WORDS = 120
VERDICTS = ("same", "different", "unsure")

SYSTEM = """You decide whether two entity mentions from Wikipedia paragraphs refer to the same real-world
entity (the same person, film, place, organisation or work).

For each pair you get, for both mentions: the name, a type, a short description and the paragraph it
comes from. Use the paragraphs: dates, roles, relatives, countries and titles tell different people or
works with the same name apart. Two different films or people can share a name.

Answer "same" only if the paragraphs make it clear they are the same entity. Answer "different" if they
are clearly different. Otherwise answer "unsure". Give a reason of at most 15 words.
Return exactly one decision for every pair_id you were given."""


class Decision(BaseModel):
    pair_id: str
    verdict: str
    reason: str


class Decisions(BaseModel):
    decisions: list[Decision]


def response_options() -> dict:
    schema = {
        "type": "object",
        "properties": {"decisions": {"type": "array", "items": {
            "type": "object",
            "properties": {"pair_id": {"type": "string"}, "verdict": {"type": "string", "enum": list(VERDICTS)},
                           "reason": {"type": "string"}},
            "required": ["pair_id", "verdict", "reason"], "additionalProperties": False}}},
        "required": ["decisions"], "additionalProperties": False,
    }
    return {"response_format": {"type": "json_schema", "json_schema": {"name": "pair_decisions", "strict": True, "schema": schema}}}


def _para(m: Mention, corpus: dict[str, dict]) -> str:
    c = corpus[m.chunk_id]
    words = c["text"].split()
    text = " ".join(words[:PARAGRAPH_WORDS]) + (" ..." if len(words) > PARAGRAPH_WORDS else "")
    return f"[{c['title']}] {text}"


def batch_prompt(pairs: list[tuple[str, Mention, Mention]], corpus: dict[str, dict]) -> str:
    blocks = []
    for pid, a, b in pairs:
        blocks.append(
            f"pair_id: {pid}\n"
            f"A: {a.name} ({a.type}) - {a.description}\n   paragraph: {_para(a, corpus)}\n"
            f"B: {b.name} ({b.type}) - {b.description}\n   paragraph: {_para(b, corpus)}"
        )
    return "Pairs:\n\n" + "\n\n---\n\n".join(blocks)


def validate(raw: str, pair_ids: list[str]) -> dict[str, Decision]:
    parsed = Decisions.model_validate(json.loads(raw))
    got = {d.pair_id: d for d in parsed.decisions}
    if sorted(got) != sorted(pair_ids) or any(d.verdict not in VERDICTS for d in got.values()):
        raise ValueError(f"pair ids or verdicts differ: missing {sorted(set(pair_ids) - set(got))}")
    return got


def judge_batch(llm: CachedLLM, pairs: list[tuple[str, Mention, Mention]], corpus: dict[str, dict],
                max_waits: int = 5) -> tuple[dict[str, Decision], list[LLMResult], bool]:
    """Returns (decisions, call results, ok). On two invalid answers every pair gets "unsure"."""
    prompt = batch_prompt(pairs, corpus)
    ids = [pid for pid, _, _ in pairs]
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
    return {pid: Decision(pair_id=pid, verdict="unsure", reason="invalid output twice") for pid in ids}, calls, False
