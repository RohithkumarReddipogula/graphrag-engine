"""Answer generation. One prompt and one context budget for every system (docs/PLAN.md section 6.4)."""

import re
from dataclasses import dataclass

import tiktoken

from graphrag.retrieval.base import Passage

SYSTEM = (
    "Answer the question using only the context. Reply with the shortest possible answer: "
    "an entity name, a date, or yes/no. If the context does not contain the answer, reply unknown. "
    "No explanation."
)
# Closed-book uses the same instructions without context.
SYSTEM_CLOSED_BOOK = (
    "Answer the question. Reply with the shortest possible answer: an entity name, a date, or yes/no. "
    "If you do not know, reply unknown. No explanation."
)

# o200k_base is the tokenizer family of gpt-oss; used only to enforce the context budget.
_ENC = tiktoken.get_encoding("o200k_base")


def n_tokens(text: str) -> int:
    return len(_ENC.encode(text))


@dataclass(frozen=True)
class PackedContext:
    text: str
    chunk_ids: list[str]
    tokens: int


def pack_context(passages: list[Passage], budget_tokens: int) -> PackedContext:
    """Fill the budget in rank order. A passage that does not fit is skipped and the next one is tried,
    so one long paragraph cannot end the context early."""
    parts, ids, used = [], [], 0
    for p in passages:
        block = f"[{p.title}] {p.text}"
        cost = n_tokens(block) + 2          # the blank line between blocks
        if used + cost > budget_tokens:
            continue
        parts.append(block)
        ids.append(p.chunk_id)
        used += cost
    return PackedContext(text="\n\n".join(parts), chunk_ids=ids, tokens=used)


def build_prompt(question: str, context: PackedContext | None) -> tuple[str, str]:
    """(system, user prompt). context=None means closed-book."""
    if context is None:
        return SYSTEM_CLOSED_BOOK, f"Question: {question}\nAnswer:"
    return SYSTEM, f"Context:\n{context.text}\n\nQuestion: {question}\nAnswer:"


def parse_answer(raw: str) -> str:
    """First non-empty line, without an "Answer:" prefix, surrounding quotes or a trailing period."""
    line = next((ln.strip() for ln in raw.strip().splitlines() if ln.strip()), "")
    line = re.sub(r"^(final\s+)?answer\s*:\s*", "", line, flags=re.IGNORECASE)
    return line.strip().strip('"').strip("*").rstrip(".").strip()
