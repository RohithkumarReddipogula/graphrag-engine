"""Entity mentions for M3 (docs/PLAN.md, M3).

A mention is one extracted entity in one paragraph (deduplicated within the paragraph by normalised name
and type). The page entity of a paragraph is its subject: the mention whose normalised name equals the
normalised paragraph title, or else the first extracted entity. Page entities are identified by
chunk_id and never merge with each other.
"""

import hashlib
import re
from dataclasses import asdict, dataclass

TUNE_SHARE = 30          # percent of paragraphs in the tune split; the rest is the report split
SPLIT_SEED = "42:m3"


def norm_name(s: str) -> str:
    s = re.sub(r"\s*\([^)]*\)\s*$", "", s.lower())
    s = re.sub(r"[^\w\s]", " ", s)
    return " ".join(s.split())


def split_of(chunk_id: str) -> str:
    h = int(hashlib.sha1(f"{SPLIT_SEED}:{chunk_id}".encode("utf-8")).hexdigest(), 16)
    return "tune" if h % 100 < TUNE_SHARE else "report"


@dataclass(frozen=True)
class Mention:
    id: str             # "<chunk_id>::<index in that paragraph's entity list>"
    chunk_id: str
    name: str
    norm: str
    type: str
    description: str
    is_page: bool       # the paragraph's own subject
    split: str          # "tune" or "report", from the paragraph

    def to_dict(self) -> dict:
        return asdict(self)


def build_mentions(extraction_rows: list[dict], corpus: dict[str, dict]) -> list[Mention]:
    out: list[Mention] = []
    for r in extraction_rows:
        cid = r["chunk_id"]
        title_norm = norm_name(corpus[cid]["title"])
        ents = r["entities"]
        page_idx = next((k for k, e in enumerate(ents) if norm_name(e["name"]) == title_norm), 0 if ents else None)
        seen: set[tuple[str, str]] = set()
        for k, e in enumerate(ents):
            key = (norm_name(e["name"]), e["type"])
            if not key[0] or (key in seen and k != page_idx):
                continue
            seen.add(key)
            out.append(Mention(
                id=f"{cid}::{k}", chunk_id=cid, name=e["name"], norm=key[0], type=e["type"],
                description=e.get("description", ""), is_page=(k == page_idx), split=split_of(cid),
            ))
    return out


def pair_split(a: Mention, b: Mention) -> str:
    """A pair is tune only if every non-page mention in it comes from a tune paragraph; else report."""
    side = [x for x in (a, b) if not x.is_page]
    return "tune" if side and all(x.split == "tune" for x in side) else "report"
