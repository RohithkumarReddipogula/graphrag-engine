"""Batched, cached entity and relation extraction with strict structured output (docs/PLAN.md, M2).

The extractor is openai/gpt-oss-120b on the pinned OpenRouter endpoint (no fallbacks), the same
model and endpoint as the generator.

- 8 paragraphs per call, batches built from chunk ids in sorted order, so the batches are stable.
- The response must match BatchExtraction and return exactly the chunk ids that were sent.
- Invalid output is retried once (with a cache salt, so the retry is a new call); if it fails again, the
  batch is logged as failed and skipped.
- Rate limits and temporary overload (429, 500, 503) are retried after a minute, up to 5 times. A daily
  quota error, or an outage that outlasts the retries, stops the run; everything done so far is cached,
  and a re-run resumes from the cache.
"""

import json
import time
from dataclasses import dataclass, field

from pydantic import ValidationError

from graphrag.extraction.schema import ENTITY_TYPES, RELATIONS, BatchExtraction, strict_json_schema
from graphrag.llm.client import CachedLLM

BATCH_SIZE = 8

SYSTEM = f"""You extract a knowledge graph from Wikipedia paragraphs.

For each paragraph, independently of the others:
1. List the named entities it mentions: the paragraph's own subject first, then every person, film,
   place, organisation or work it names. Use the name as written in the paragraph. Type is one of:
   {", ".join(ENTITY_TYPES)}. Description: at most 12 words, taken from the paragraph.
2. List the facts it states as relations (subject, relation, object), using only these relations:
{chr(10).join(f"   - {name}: {desc}" for name, desc in RELATIONS.items())}
   Use OTHER with a short other_label for an important fact that fits none of them.
   Follow the direction given above (subject -> object).
   For date relations, the object is the date as written in the paragraph.
   Subject and object must be entity names from step 1, except date objects.
Only state what the paragraph says. Do not use outside knowledge. Return every chunk_id you were given."""


def make_batches(chunks: list[dict], size: int = BATCH_SIZE) -> list[list[dict]]:
    ordered = sorted(chunks, key=lambda c: c["id"])
    return [ordered[i:i + size] for i in range(0, len(ordered), size)]


def batch_prompt(batch: list[dict]) -> str:
    blocks = [f"chunk_id: {c['id']}\ntitle: {c['title']}\ntext: {c['text']}" for c in batch]
    return "Paragraphs:\n\n" + "\n\n---\n\n".join(blocks)


def response_options() -> dict:
    """OpenRouter structured output. With require_parameters on, the pinned endpoint must honour it."""
    return {
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "batch_extraction", "strict": True, "schema": strict_json_schema()},
        }
    }


def validate(raw: str, batch: list[dict]) -> BatchExtraction:
    parsed = BatchExtraction.model_validate(json.loads(raw))
    sent = sorted(c["id"] for c in batch)
    got = sorted(p.chunk_id for p in parsed.paragraphs)
    if got != sent:
        raise ValueError(f"chunk ids differ: missing {sorted(set(sent) - set(got))}, extra {sorted(set(got) - set(sent))}")
    return parsed


class QuotaExhausted(RuntimeError):
    """Daily quota used up, or the provider stayed unavailable: stop now and resume from the cache later."""


def _is_retryable(exc: Exception) -> bool:
    """Rate limits (429) and temporary server overload (500, 503) are worth waiting for."""
    text = str(exc)
    return any(code in text for code in ("429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE", "500 INTERNAL"))


def _is_daily_quota(exc: Exception) -> bool:
    text = str(exc)
    return "PerDay" in text or "per day" in text.lower()


@dataclass
class RunStats:
    batches: int = 0
    ok: int = 0
    retried: int = 0
    failed: list[dict] = field(default_factory=list)
    live_calls: int = 0
    input_tokens: int = 0       # over every call made for these batches, cached or live
    output_tokens: int = 0
    cost_usd: float = 0.0       # cost of the original live calls


def extract_batch(llm: CachedLLM, batch: list[dict], stats: RunStats, max_quota_waits: int = 5) -> BatchExtraction | None:
    prompt = batch_prompt(batch)
    for attempt in (1, 2):
        salt = None if attempt == 1 else "retry-1"
        for wait in range(max_quota_waits + 1):
            try:
                out = llm.complete(prompt, system=SYSTEM, options=response_options(), salt=salt)
                break
            except Exception as exc:  # noqa: BLE001 - classify provider errors
                if not _is_retryable(exc):
                    raise
                if _is_daily_quota(exc) or wait == max_quota_waits:
                    raise QuotaExhausted(str(exc)[:300]) from exc
                time.sleep(60)  # per-minute limit or temporary overload: wait and try again
        stats.live_calls += not out.cached
        stats.input_tokens += out.input_tokens
        stats.output_tokens += out.output_tokens
        stats.cost_usd += out.cost_usd
        try:
            parsed = validate(out.text, batch)
            stats.ok += 1
            stats.retried += attempt == 2
            return parsed
        except (ValidationError, ValueError, json.JSONDecodeError) as exc:
            last_error = f"{type(exc).__name__}: {str(exc)[:300]}"
    stats.failed.append({"chunk_ids": [c["id"] for c in batch], "error": last_error})
    return None
