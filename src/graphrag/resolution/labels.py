"""Labels for M3 from dev question gold triples only (docs/PLAN.md, M3). Test questions are never read.

A bridge link: a dev gold triple (s, relation, o) whose object o has its own paragraph (a gold paragraph
of the same question titled o), different from the subject's paragraph. The mention of o inside the
subject's paragraph should be resolved to o's page entity. Its split is the split of the subject's
paragraph (where the mention is). Negatives: the same mention paired with another page whose title has
the same normalised name (a same-name page that is not o).
"""

from dataclasses import dataclass

from rapidfuzz import fuzz

from graphrag.eval.extraction import gold_triples
from graphrag.extraction.schema import DATE_RELATIONS
from graphrag.resolution.mentions import Mention, norm_name, split_of

MENTION_MATCH_MIN = 85


@dataclass(frozen=True)
class BridgeLink:
    relation: str
    subject_chunk: str
    object_name: str
    object_chunk: str
    mention_id: str | None     # None when extraction has no mention of o in the subject's paragraph
    page_id: str | None        # page mention of the object's paragraph
    split: str


def _find_mention(cands: list[Mention], names: list[str]) -> Mention | None:
    norms = [norm_name(n) for n in names if norm_name(n)]
    best, best_score = None, 0.0
    for m in cands:
        if m.is_page:
            continue
        score = max(fuzz.ratio(m.norm, n) for n in norms)
        if score > best_score:
            best, best_score = m, score
    return best if best_score >= MENTION_MATCH_MIN else None


def bridge_links(dev_questions: list[dict], corpus: dict[str, dict], mentions: list[Mention],
                 evidence_aliases: dict) -> list[BridgeLink]:
    """dev_questions: dev multi-hop + dev single-hop records only."""
    by_chunk: dict[str, list[Mention]] = {}
    for m in mentions:
        by_chunk.setdefault(m.chunk_id, []).append(m)
    page_of = {m.chunk_id: m for m in mentions if m.is_page}
    triples, _ = gold_triples(dev_questions, corpus, evidence_aliases)
    gold_chunks = {cid for q in dev_questions for cid in q["gold_chunk_ids"]}

    out, seen = [], set()
    for t in triples:
        if t["relation"] in DATE_RELATIONS:
            continue
        obj_norms = {norm_name(a) for a in t["object_aliases"]}
        obj_chunks = [cid for cid in gold_chunks if norm_name(corpus[cid]["title"]) in obj_norms and cid != t["chunk_id"]]
        if len(obj_chunks) != 1:
            continue
        key = (t["chunk_id"], obj_chunks[0])
        if key in seen:
            continue
        seen.add(key)
        m = _find_mention(by_chunk.get(t["chunk_id"], []), t["object_aliases"])
        page = page_of.get(obj_chunks[0])
        out.append(BridgeLink(
            relation=t["relation"], subject_chunk=t["chunk_id"], object_name=t["object"],
            object_chunk=obj_chunks[0], mention_id=m.id if m else None, page_id=page.id if page else None,
            split=split_of(t["chunk_id"]),
        ))
    return out


def same_name_negatives(links: list[BridgeLink], mentions: list[Mention]) -> list[tuple[str, str]]:
    """(mention, page) pairs where the page has the same normalised name as the right page but is another
    paragraph: the mention must not be resolved to it."""
    pages_by_norm: dict[str, list[Mention]] = {}
    for m in mentions:
        if m.is_page:
            pages_by_norm.setdefault(m.norm, []).append(m)
    by_id = {m.id: m for m in mentions}
    out = []
    for link in links:
        if not link.mention_id or not link.page_id:
            continue
        right = by_id[link.page_id]
        for other in pages_by_norm.get(right.norm, []):
            if other.id != right.id:
                out.append((link.mention_id, other.id))
    return out
