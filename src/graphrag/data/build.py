"""Sampling, corpus and single-hop question construction for M1 (docs/PLAN.md section 6.1).

Everything is deterministic for a fixed seed. Random choices that belong to one question use an RNG seeded
with (seed, question id), so they do not depend on processing order.
"""

import hashlib
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

TYPES = ["compositional", "comparison", "bridge_comparison", "inference"]

# Single-hop templates for relations with one value per subject. Multi-valued relations (award received,
# child, sibling, educated at, employer, occupation, has part, ...) are excluded: a question like "Which
# award did X receive?" has several correct answers and EM would punish a correct one.
SINGLE_HOP_TEMPLATES = {
    "director": "Who directed {s}?",
    "date of birth": "When was {s} born?",
    "father": "Who is the father of {s}?",
    "date of death": "When did {s} die?",
    "publication date": "When was {s} released?",
    "country of citizenship": "What is the country of citizenship of {s}?",
    "place of birth": "Where was {s} born?",
    "mother": "Who is the mother of {s}?",
    "place of death": "Where did {s} die?",
    "country of origin": "What is the country of origin of {s}?",
    "country": "Which country is {s} in?",
    "place of burial": "Where was {s} buried?",
    "inception": "When was {s} founded?",
    "cause of death": "What was the cause of death of {s}?",
    "founded by": "Who founded {s}?",
    "publisher": "Who published {s}?",
    "manufacturer": "Who manufactured {s}?",
}


@dataclass(frozen=True)
class Question:
    id: str
    question: str
    answer: str
    type: str
    evidences: list[list[str]]
    gold_titles: list[str]
    context: list[tuple[str, str]] = field(repr=False)   # (title, text) in original order


def paragraph_text(sentences: Any) -> str:
    return " ".join(s.strip() for s in sentences).strip()


def from_row(row: Any) -> Question:
    titles = list(row.context["title"])
    texts = [paragraph_text(s) for s in row.context["sentences"]]
    return Question(
        id=row.id,
        question=row.question,
        answer=row.answer,
        type=row.type,
        evidences=[list(e) for e in row.evidences],
        gold_titles=list(dict.fromkeys(row.supporting_facts["title"])),
        context=list(zip(titles, texts)),
    )


def stratified_split(questions: list[Question], n_dev: int, n_test: int, seed: int) -> tuple[list[Question], list[Question]]:
    """Per type: sort by id, shuffle with the seed, take n_dev for dev and the next n_test for test."""
    dev, test = [], []
    for t in TYPES:
        pool = sorted((q for q in questions if q.type == t), key=lambda q: q.id)
        random.Random(f"{seed}:{t}").shuffle(pool)
        dev += pool[:n_dev]
        test += pool[n_dev:n_dev + n_test]
    return dev, test


def selected_paragraphs(q: Question, n_distractors: int, seed: int) -> list[tuple[str, str]]:
    """Gold paragraphs of the question plus n seeded-random distractors from its own context."""
    gold = set(q.gold_titles)
    distractors = [p for p in q.context if p[0] not in gold]
    random.Random(f"{seed}:{q.id}").shuffle(distractors)
    return [p for p in q.context if p[0] in gold] + distractors[:n_distractors]


def spacing_key(text: str) -> str:
    """Comparison key that ignores spacing around punctuation. 2Wiki contains some paragraphs in two
    tokenisations ("Silverstein (born" vs "Silverstein( born"); they must compare equal."""
    t = re.sub(r"\s+", " ", text)
    return re.sub(r"\s*([^\w\s])\s*", r"\1", t).strip()


def chunk_id(title: str, key: str, kept_apart: bool) -> str:
    if not kept_apart:
        return title
    return f"{title}#{hashlib.sha1(key.encode('utf-8')).hexdigest()[:8]}"


@dataclass
class Corpus:
    chunks: dict[str, dict[str, str]]                 # id -> {"id", "title", "text"}
    ids: dict[tuple[str, str], str]                   # (title, raw text) -> id, for every raw variant
    collisions: list[dict[str, Any]]
    multi_variant_titles: set[str] = field(default_factory=set)   # titles seen with more than one raw text


def build_corpus(questions: list[Question], n_distractors: int, seed: int) -> Corpus:
    """Pool paragraphs.

    - Raw texts of one title that differ only in spacing around punctuation are merged into one chunk.
      The stored text is the variant used by most questions (ties: lexicographically smallest), and
      every raw variant maps to that chunk id, so gold ids point to the merged paragraph.
    - Genuinely different texts of one title are all kept, each with an id carrying a short hash of its
      comparison key, so no version can be dropped.
    """
    uses: Counter[tuple[str, str]] = Counter()
    for q in questions:
        for p in selected_paragraphs(q, n_distractors, seed):
            uses[p] += 1

    variants: dict[tuple[str, str], list[str]] = defaultdict(list)   # (title, key) -> raw texts
    for title, text in uses:
        variants[(title, spacing_key(text))].append(text)
    keys_by_title: dict[str, list[str]] = defaultdict(list)
    for title, key in variants:
        keys_by_title[title].append(key)

    ids, chunks, collisions = {}, {}, []
    for (title, key), texts in variants.items():
        kept_apart = len(keys_by_title[title]) > 1
        cid = chunk_id(title, key, kept_apart)
        stored = min(texts, key=lambda t: (-uses[(title, t)], t))
        chunks[cid] = {"id": cid, "title": title, "text": stored}
        for t in texts:
            ids[(title, t)] = cid
        if len(texts) > 1:
            collisions.append({"title": title, "kind": "merged_spacing_variants", "id": cid, "variants": len(texts)})
    for title, keys in keys_by_title.items():
        if len(keys) > 1:
            collisions.append({"title": title, "kind": "kept_apart", "ids": [ids[(title, variants[(title, k)][0])] for k in keys]})

    multi = {t for t, keys in keys_by_title.items() if sum(len(variants[(t, k)]) for k in keys) > 1}
    return Corpus(chunks=chunks, ids=ids, collisions=collisions, multi_variant_titles=multi)


def gold_chunk_ids(q: Question, corpus: Corpus) -> list[str]:
    """Resolve gold titles through the question's own context, so gold never points to a same-title twin."""
    gold = set(q.gold_titles)
    return [corpus.ids[p] for p in q.context if p[0] in gold]


def question_record(q: Question, corpus: Corpus) -> dict[str, Any]:
    return {
        "id": q.id,
        "question": q.question,
        "answer": q.answer,
        "type": q.type,
        "evidences": q.evidences,
        "gold_titles": q.gold_titles,
        "gold_chunk_ids": gold_chunk_ids(q, corpus),
    }


def _strip_disambiguator(title: str) -> str:
    return re.sub(r"\s*\([^)]*\)$", "", title).strip().lower()


def single_hop_candidates(questions: list[Question], corpus: Corpus) -> list[dict[str, Any]]:
    """One candidate per (subject, relation), kept only if it is unambiguous and checkable:
    single-valued relation, exactly one gold paragraph of the source question matches the subject, the
    object appears verbatim in that paragraph, and the title was seen with only one raw text."""
    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    objects: dict[tuple[str, str], set[str]] = defaultdict(set)
    for q in questions:
        gold_paras = [p for p in q.context if p[0] in set(q.gold_titles)]
        for s, rel, o in q.evidences:
            objects[(s, rel)].add(o)
            if rel not in SINGLE_HOP_TEMPLATES or (s, rel) in by_key:
                continue
            matches = [p for p in gold_paras if p[0].lower() == s.lower() or _strip_disambiguator(p[0]) == s.lower()]
            if len(matches) != 1:
                continue
            title, text = matches[0]
            cid = corpus.ids[(title, text)]
            # Titles seen with more than one raw text are skipped (this also keeps the sample identical
            # to the one built before spacing variants were merged).
            if title in corpus.multi_variant_titles or o.lower() not in text.lower():
                continue
            by_key[(s, rel)] = {
                "id": "sh_" + hashlib.sha1(f"{s}|{rel}".encode("utf-8")).hexdigest()[:10],
                "question": SINGLE_HOP_TEMPLATES[rel].format(s=s),
                "answer": o,
                "type": "single_hop",
                "relation": rel,
                "evidences": [[s, rel, o]],
                "gold_titles": [title],
                "gold_chunk_ids": [cid],
                "source_question_id": q.id,
            }
    # Drop subjects that have conflicting objects for the same relation anywhere in the sample.
    return [c for k, c in by_key.items() if len(objects[k]) == 1]


def sample_single_hop(dev_q: list[Question], test_q: list[Question], corpus: Corpus, n_dev: int, n_test: int, seed: int):
    dev_pool = sorted(single_hop_candidates(dev_q, corpus), key=lambda c: c["id"])
    dev_keys = {(c["evidences"][0][0], c["relation"]) for c in dev_pool}
    # A fact used in dev never appears in test.
    test_pool = sorted(
        (c for c in single_hop_candidates(test_q, corpus) if (c["evidences"][0][0], c["relation"]) not in dev_keys),
        key=lambda c: c["id"],
    )
    random.Random(f"{seed}:single_hop:dev").shuffle(dev_pool)
    random.Random(f"{seed}:single_hop:test").shuffle(test_pool)
    return dev_pool[:n_dev], test_pool[:n_test], {"dev_pool": len(dev_pool), "test_pool": len(test_pool)}


def relation_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    return dict(Counter(r["relation"] for r in records).most_common())
