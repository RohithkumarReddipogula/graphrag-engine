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
import re
import time
from dataclasses import dataclass, field
from typing import Callable

from pydantic import ValidationError

from graphrag.extraction.schema import ENTITY_TYPES, RELATIONS, BatchExtraction, strict_json_schema
from graphrag.llm.client import CachedLLM

BATCH_SIZE = 8
# v1 (commit 06f195b): no OTHER rule, no adjective rule; results/m2/pilot_v1.json.
# v2 (commit bc8eb06): adds the OTHER and adjective rules; results/m2/pilot_v2.json.
# v3: adds subject-by-title naming, family facts from the paragraph's person, performer for music only.
PROMPT_VERSION = "v3"

SYSTEM = f"""You extract a knowledge graph from Wikipedia paragraphs.

For each paragraph, independently of the others:
1. List the named entities it mentions: the paragraph's own subject first, then every person, film,
   place, organisation or work it names. Name the paragraph's own subject by the paragraph title,
   without any bracketed part (title "Heat (1995 film)" gives "Heat"), and use exactly that name for
   it in every relation. Name other entities as written in the paragraph. Type is one of:
   {", ".join(ENTITY_TYPES)}. Description: at most 12 words, taken from the paragraph.
2. List the facts it states as relations (subject, relation, object), using only these relations:
{chr(10).join(f"   - {name}: {desc}" for name, desc in RELATIONS.items())}
   If a fact fits one of these relations, you must use that relation. Use OTHER, with a short
   other_label, only for an important fact that fits none of them. Never use OTHER for a fact that a
   listed relation covers, and never put a relation name into the object or the other_label.
   Follow the direction given above (subject -> object). State family facts from the person the
   paragraph is about: "son of X" gives (subject, father, X) or (subject, mother, X); you may add the
   reverse child relation as well.
   performer is only for music and recordings (a singer or band performing a song or album); an actor
   appearing in a film or show is OTHER with other_label "cast member".
   Facts stated through an adjective or a description count as stated. A nationality adjective
   ("a French village", "a Polish-born British actor") states the country or the country of
   citizenship; give the country (France; Poland and United Kingdom).
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


ID_REPAIR_MIN_RATIO = 90


def repair_chunk_ids(parsed: BatchExtraction, sent: list[str]) -> BatchExtraction:
    """Fix one mistyped chunk id: when exactly one sent id is missing, exactly one unknown id came back,
    and the two are near-identical (one is a prefix of the other, or rapidfuzz ratio >= 90), the unknown
    id is mapped back to the sent one. Anything else is left for validate() to reject."""
    from rapidfuzz import fuzz

    got = [p.chunk_id for p in parsed.paragraphs]
    missing = sorted(set(sent) - set(got))
    extra = sorted(set(got) - set(sent))
    if len(missing) == 1 and len(extra) == 1 and got.count(extra[0]) == 1:
        m, e = missing[0], extra[0]
        if m.startswith(e) or e.startswith(m) or fuzz.ratio(m, e) >= ID_REPAIR_MIN_RATIO:
            for p in parsed.paragraphs:
                if p.chunk_id == e:
                    p.chunk_id = m
    return parsed


def validate(raw: str, batch: list[dict]) -> BatchExtraction:
    sent = sorted(c["id"] for c in batch)
    parsed = repair_chunk_ids(BatchExtraction.model_validate(json.loads(raw)), sent)
    got = sorted(p.chunk_id for p in parsed.paragraphs)
    if got != sent:
        raise ValueError(f"chunk ids differ: missing {sorted(set(sent) - set(got))}, extra {sorted(set(got) - set(sent))}")
    return parsed


class QuotaExhausted(RuntimeError):
    """Daily quota used up, or the provider stayed unavailable: stop now and resume from the cache later."""


_RETRYABLE_STATUS = re.compile(r"\b(408|429|5\d\d)\b")


def _is_retryable(exc: Exception) -> bool:
    """Request timeout (408), rate limits (429), any server or gateway error (5xx, including 504 and 524)
    and stalled calls are worth retrying."""
    text = str(exc)
    return bool(_RETRYABLE_STATUS.search(text)) or any(
        marker in text for marker in ("RESOURCE_EXHAUSTED", "UNAVAILABLE", "TIMEOUT", "timed out")
    )


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


def merge_stats(parts: list[RunStats]) -> RunStats:
    total = RunStats()
    for p in parts:
        total.batches += p.batches
        total.ok += p.ok
        total.retried += p.retried
        total.failed += p.failed
        total.live_calls += p.live_calls
        total.input_tokens += p.input_tokens
        total.output_tokens += p.output_tokens
        total.cost_usd += p.cost_usd
    return total


def extract_many(
    llm: CachedLLM,
    batches: dict[int, list[dict]],
    workers: int = 4,
    on_done: "Callable[[int, int, RunStats], None] | None" = None,
) -> tuple[dict[int, BatchExtraction | None], RunStats]:
    """Extract several batches in parallel threads. Each batch is an independent cached call, so the
    results do not depend on the number of workers. Each task keeps its own RunStats (merged at the end).
    on_done(done, total, stats_so_far) is called after every finished batch, for progress output.
    A QuotaExhausted is re-raised after the other running tasks finish; their work is cached and kept."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def one(i: int):
        st = RunStats(batches=1)
        return i, extract_batch(llm, batches[i], st), st

    results: dict[int, BatchExtraction | None] = {}
    parts: list[RunStats] = []
    stop: QuotaExhausted | None = None
    with ThreadPoolExecutor(workers) as pool:
        futures = [pool.submit(one, i) for i in sorted(batches)]
        for fut in as_completed(futures):
            try:
                i, parsed, st = fut.result()
            except QuotaExhausted as exc:
                stop = stop or exc
                for f in futures:
                    f.cancel()
                continue
            results[i] = parsed
            parts.append(st)
            if on_done:
                on_done(len(results), len(batches), merge_stats(parts))
    if stop:
        raise QuotaExhausted(f"{stop} (finished batches: {len(results)} of {len(batches)}, all cached)")
    return results, merge_stats(parts)
