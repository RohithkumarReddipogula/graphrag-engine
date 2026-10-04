"""Score extracted triples against 2Wiki gold evidence triples (docs/PLAN.md, M2).

2Wiki gold triples are only the facts each question needs, not every fact in a paragraph, so plain
precision against them would count correct extra facts as errors. Reported instead:
- recall: share of gold triples that some extracted triple matches;
- slot precision: among extracted triples whose subject and relation match a gold triple, the share
  whose object matches too.
A match needs the same relation and both entities to match: names by rapidfuzz ratio on normalised text
(threshold given), dates by year, month and day where both sides state them. As in the official 2Wiki
v1.1 evidence scoring, a gold subject or object also matches any Wikidata alias or demonym of its entity
(data/evidence_aliases.json), where the official ids line up with the evidence list.
"""

import re
from collections import defaultdict

from rapidfuzz import fuzz

from graphrag.extraction.schema import DATE_RELATIONS

MONTHS = {m: i for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
     "november", "december"], start=1)}


def norm_name(s: str) -> str:
    s = re.sub(r"\s*\([^)]*\)\s*$", "", s.lower())          # trailing disambiguation, e.g. "(1995 film)"
    s = re.sub(r"[^\w\s]", " ", s)
    return " ".join(s.split())


def parse_date(s: str) -> tuple[int | None, int | None, int | None] | None:
    """(year, month, day) from forms like "28 August 1921", "March 27, 1884", "1921", "August 1921"."""
    t = s.lower().replace(",", " ")
    year = re.search(r"\b(\d{3,4})\b", t)
    if not year:
        return None
    month = next((n for name, n in MONTHS.items() if re.search(rf"\b{name}\b", t)), None)
    day = re.search(r"\b(\d{1,2})\b", re.sub(r"\b\d{3,4}\b", " ", t))
    return int(year.group(1)), month, int(day.group(1)) if day and month else None


def dates_match(a: str, b: str) -> bool:
    pa, pb = parse_date(a), parse_date(b)
    if not pa or not pb:
        return False
    return all(x is None or y is None or x == y for x, y in zip(pa, pb)) and pa[0] == pb[0]


def names_match(a: str, b: str, threshold: float) -> bool:
    na, nb = norm_name(a), norm_name(b)
    return bool(na) and bool(nb) and fuzz.ratio(na, nb) >= threshold


def objects_match(relation: str, a: str, b: str, threshold: float) -> bool:
    return dates_match(a, b) if relation in DATE_RELATIONS else names_match(a, b, threshold)


def _subject_ok(t: dict, g: dict, threshold: float) -> bool:
    return any(names_match(t["subject"], alias, threshold) for alias in g.get("subject_aliases") or [g["subject"]])


def _object_ok(t: dict, g: dict, threshold: float) -> bool:
    if g["relation"] in DATE_RELATIONS:
        return dates_match(t["object"], g["object"])
    return any(names_match(t["object"], alias, threshold) for alias in g.get("object_aliases") or [g["object"]])


def score(extracted: list[dict], gold: list[dict], threshold: float) -> dict:
    """extracted / gold: dicts with subject, relation, object (gold also has chunk_id)."""
    by_rel: dict[str, list[dict]] = defaultdict(list)
    for t in extracted:
        by_rel[t["relation"]].append(t)

    per_rel: dict[str, dict[str, int]] = defaultdict(lambda: {"gold": 0, "recalled": 0, "slot_pred": 0, "slot_correct": 0})
    for g in gold:
        cands = [t for t in by_rel[g["relation"]] if _subject_ok(t, g, threshold)]
        r = per_rel[g["relation"]]
        r["gold"] += 1
        r["recalled"] += any(_object_ok(t, g, threshold) for t in cands)

    # Slot precision: each extracted triple counted once, against the gold slots it fills.
    for rel, triples in by_rel.items():
        slots = [g for g in gold if g["relation"] == rel]
        for t in triples:
            hits = [g for g in slots if _subject_ok(t, g, threshold)]
            if hits:
                per_rel[rel]["slot_pred"] += 1
                per_rel[rel]["slot_correct"] += any(_object_ok(t, g, threshold) for g in hits)

    def rates(d):
        return {
            **d,
            "recall": round(d["recalled"] / d["gold"], 4) if d["gold"] else None,
            "slot_precision": round(d["slot_correct"] / d["slot_pred"], 4) if d["slot_pred"] else None,
        }

    total = {k: sum(r[k] for r in per_rel.values()) for k in ("gold", "recalled", "slot_pred", "slot_correct")}
    return {"overall": rates(total), "per_relation": {k: rates(v) for k, v in sorted(per_rel.items()) if v["gold"]}}


def gold_triples(questions: list[dict], corpus: dict[str, dict], evidence_aliases: dict | None = None) -> tuple[list[dict], int]:
    """Gold evidence triples with the chunk that states them: the question's gold paragraph whose title
    matches the subject, plus alias sets for subject and object when available (single-hop questions use
    their source question's entry). Returns (triples, number whose paragraph could not be identified)."""
    alias_index: dict[tuple, dict] = {}
    for qid, entries in (evidence_aliases or {}).items():
        for e in entries:
            alias_index[(qid, tuple(e["evidence"]))] = e
    seen, out, unmapped = set(), [], 0
    for q in questions:
        source = q.get("source_question_id", q.get("id"))
        titles = {cid: corpus[cid]["title"] for cid in q["gold_chunk_ids"]}
        for s, rel, o in q["evidences"]:
            if (s, rel, o) in seen:
                continue
            seen.add((s, rel, o))
            match = [cid for cid, t in titles.items() if t.lower() == s.lower() or norm_name(t) == norm_name(s)]
            if len(match) != 1:
                unmapped += 1
                continue
            a = alias_index.get((source, (s, rel, o)), {})
            out.append({"subject": s, "relation": rel, "object": o, "chunk_id": match[0],
                        "subject_aliases": a.get("subject_aliases", [s]), "object_aliases": a.get("object_aliases", [o])})
    return out, unmapped
