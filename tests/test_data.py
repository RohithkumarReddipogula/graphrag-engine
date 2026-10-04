import json

import pandas as pd
import pytest

from graphrag.config import get_settings
from graphrag.data import build
from graphrag.data.download import SHA256, sha256_of

DATA = get_settings().data_dir


def q(qid, type_, gold, context, evidences=()):
    return build.Question(
        id=qid, question=f"q {qid}", answer="a", type=type_,
        evidences=[list(e) for e in evidences], gold_titles=gold, context=context,
    )


# ---------- unit tests on small synthetic inputs ----------

def test_split_is_deterministic_and_stratified():
    qs = [q(f"{t}{i}", t, [], []) for t in build.TYPES for i in range(10)]
    a = build.stratified_split(qs, 2, 3, seed=42)
    b = build.stratified_split(list(reversed(qs)), 2, 3, seed=42)
    assert [x.id for x in a[0]] == [x.id for x in b[0]] and [x.id for x in a[1]] == [x.id for x in b[1]]
    assert len(a[0]) == 8 and len(a[1]) == 12
    assert not {x.id for x in a[0]} & {x.id for x in a[1]}


def test_title_collision_keeps_both_texts_with_distinct_ids():
    q1 = q("1", "comparison", ["Heat"], [("Heat", "Heat is a 1995 film."), ("X", "x")])
    q2 = q("2", "comparison", ["Heat"], [("Heat", "Heat is a 1972 film."), ("Y", "y")])
    corpus = build.build_corpus([q1, q2], n_distractors=4, seed=42)

    heat = [c for c in corpus.chunks.values() if c["title"] == "Heat"]
    assert {c["text"] for c in heat} == {"Heat is a 1995 film.", "Heat is a 1972 film."}
    assert len({c["id"] for c in heat}) == 2 and all("#" in c["id"] for c in heat)
    assert len(corpus.collisions) == 1 and corpus.collisions[0]["title"] == "Heat"
    assert set(corpus.collisions[0]["ids"]) == {c["id"] for c in heat}
    # Each question's gold resolves to its own version, not the twin.
    assert corpus.chunks[build.gold_chunk_ids(q1, corpus)[0]]["text"] == "Heat is a 1995 film."
    assert corpus.chunks[build.gold_chunk_ids(q2, corpus)[0]]["text"] == "Heat is a 1972 film."


def test_identical_title_and_text_is_stored_once_without_hash():
    q1 = q("1", "comparison", ["A"], [("A", "same text")])
    q2 = q("2", "comparison", ["A"], [("A", "same text")])
    corpus = build.build_corpus([q1, q2], 4, 42)
    assert list(corpus.chunks) == ["A"] and corpus.collisions == []


def test_distractor_count_and_gold_always_included():
    ctx = [("G", "gold")] + [(f"D{i}", f"d{i}") for i in range(8)]
    paras = build.selected_paragraphs(q("1", "inference", ["G"], ctx), 4, 42)
    assert ("G", "gold") in paras and len(paras) == 5


def test_single_hop_filters():
    ctx = [
        ("Film A (1990 film)", "Film A is directed by Jane Doe."),
        ("Jane Doe", "Jane Doe won the Oscar and the BAFTA."),
        ("Film B", "Film B is a film."),
    ]
    q1 = q("1", "compositional", [t for t, _ in ctx], ctx, evidences=[
        ("Film A", "director", "Jane Doe"),            # ok: disambiguator stripped, object in text
        ("Jane Doe", "award received", "Oscar"),       # multi-valued relation: excluded
        ("Film B", "director", "John Roe"),            # object not in paragraph: excluded
    ])
    corpus = build.build_corpus([q1], 4, 42)
    cands = build.single_hop_candidates([q1], corpus)
    assert [(c["question"], c["answer"], c["gold_chunk_ids"]) for c in cands] == [
        ("Who directed Film A?", "Jane Doe", ["Film A (1990 film)"])
    ]


def test_single_hop_dev_facts_never_in_test():
    ctx = [("Film A", "Film A is directed by Jane Doe.")]
    ev = [("Film A", "director", "Jane Doe")]
    dev_q = [q("1", "compositional", ["Film A"], ctx, ev)]
    test_q = [q("2", "compositional", ["Film A"], ctx, ev)]
    corpus = build.build_corpus(dev_q + test_q, 4, 42)
    sh_dev, sh_test, _ = build.sample_single_hop(dev_q, test_q, corpus, 5, 5, 42)
    assert len(sh_dev) == 1 and sh_test == []


# ---------- checks on the committed data files ----------

def read_jsonl(name):
    return [json.loads(line) for line in (DATA / name).read_text(encoding="utf-8").splitlines()]


needs_data = pytest.mark.skipif(not (DATA / "corpus.jsonl").exists(), reason="run scripts/build_data.py first")


@needs_data
def test_committed_splits_are_disjoint_and_gold_ids_exist():
    corpus = {c["id"] for c in read_jsonl("corpus.jsonl")}
    dev, test = read_jsonl("questions_dev.jsonl"), read_jsonl("questions_test.jsonl")
    assert not {r["id"] for r in dev} & {r["id"] for r in test}
    for r in dev + test + read_jsonl("single_hop_dev.jsonl") + read_jsonl("single_hop_test.jsonl"):
        assert r["gold_chunk_ids"], r["id"]
        assert set(r["gold_chunk_ids"]) <= corpus, r["id"]


@needs_data
def test_collisions_are_logged_and_every_version_is_in_the_corpus():
    corpus = {c["id"]: c for c in read_jsonl("corpus.jsonl")}
    for col in read_jsonl("collisions.jsonl"):
        assert len(col["ids"]) >= 2
        assert all(cid in corpus and corpus[cid]["title"] == col["title"] for cid in col["ids"])
        assert len({corpus[cid]["text"] for cid in col["ids"]}) == len(col["ids"])


RAW = DATA / "raw" / "2wiki_validation.parquet"


@pytest.mark.skipif(not RAW.exists(), reason="raw parquet not downloaded")
def test_every_gold_paragraph_is_in_the_corpus_with_its_exact_text():
    assert sha256_of(RAW) == SHA256
    corpus = {c["id"]: c for c in read_jsonl("corpus.jsonl")}
    sampled = {r["id"]: r for r in read_jsonl("questions_dev.jsonl") + read_jsonl("questions_test.jsonl")}
    raw = pd.read_parquet(RAW)
    checked = 0
    for row in raw[raw.id.isin(sampled)].itertuples(index=False):
        rec = sampled[row.id]
        gold = set(row.supporting_facts["title"])
        raw_gold = [
            (t, build.paragraph_text(s))
            for t, s in zip(row.context["title"], row.context["sentences"]) if t in gold
        ]
        assert len(raw_gold) == len(rec["gold_chunk_ids"]), row.id
        for (title, text), cid in zip(raw_gold, rec["gold_chunk_ids"]):
            assert corpus[cid]["title"] == title and corpus[cid]["text"] == text, (row.id, cid)
            checked += 1
    assert len(sampled) == 400 and checked > 0
