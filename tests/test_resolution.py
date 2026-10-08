import json

import numpy as np
import pytest

from graphrag.resolution.candidates import candidate_pairs, types_compatible
from graphrag.resolution.judge import validate
from graphrag.resolution.mentions import Mention, build_mentions, norm_name, split_of


def m(mid, chunk, name, type_="PERSON", page=False, desc=""):
    return Mention(id=mid, chunk_id=chunk, name=name, norm=norm_name(name), type=type_, description=desc,
                   is_page=page, split="tune")


def unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


def test_page_entity_is_the_title_match_else_the_first_entity():
    corpus = {"Heat (1995 film)": {"title": "Heat (1995 film)"}, "X": {"title": "X"}}
    rows = [
        {"chunk_id": "Heat (1995 film)", "entities": [{"name": "Michael Mann", "type": "PERSON"}, {"name": "Heat", "type": "FILM"}]},
        {"chunk_id": "X", "entities": [{"name": "Something", "type": "OTHER"}, {"name": "Y", "type": "PERSON"}]},
    ]
    pages = {x.chunk_id: x.name for x in build_mentions(rows, corpus) if x.is_page}
    assert pages == {"Heat (1995 film)": "Heat", "X": "Something"}


def test_two_page_entities_are_never_candidates_even_with_the_same_name():
    ms = [m("a::0", "Adam's Rib", "Adam's Rib", "FILM", page=True),
          m("b::0", "Adam's Rib (1923 film)", "Adam's Rib", "FILM", page=True),
          m("c::1", "Z", "Adam's Rib", "FILM")]
    emb = np.stack([unit([1, 0]), unit([1, 0]), unit([1, 0.01])])
    pairs = {(p.a, p.b) for p in candidate_pairs(ms, emb, t_low=0.5)}
    assert ("a::0", "b::0") not in pairs
    assert ("a::0", "c::1") in pairs and ("b::0", "c::1") in pairs


def test_type_compatibility():
    assert types_compatible("FILM", "WORK") and types_compatible("PERSON", "OTHER")
    assert not types_compatible("PERSON", "PLACE") and not types_compatible("ORG", "FILM")


def test_embedding_candidates_need_a_shared_name_part():
    ms = [m("a::1", "A", "Mongkut"), m("b::1", "B", "King Mongkut"), m("c::1", "C", "Jane Doe")]
    emb = np.stack([unit([1, 0]), unit([1, 0.05]), unit([1, 0.02])])   # all very similar descriptions
    pairs = {(p.a, p.b): p.sources for p in candidate_pairs(ms, emb, t_low=0.5)}
    assert ("a::1", "b::1") in pairs
    assert ("a::1", "c::1") not in pairs and ("b::1", "c::1") not in pairs


def test_split_is_deterministic_and_about_30_percent_tune():
    ids = [f"chunk {i}" for i in range(2000)]
    share = sum(split_of(c) == "tune" for c in ids) / len(ids)
    assert split_of("chunk 7") == split_of("chunk 7") and 0.25 < share < 0.35


def test_judge_validation_requires_every_pair_id_and_a_known_verdict():
    ok = {"decisions": [{"pair_id": "p1", "verdict": "same", "reason": "r"}, {"pair_id": "p2", "verdict": "unsure", "reason": "r"}]}
    assert set(validate(json.dumps(ok), ["p1", "p2"])) == {"p1", "p2"}
    with pytest.raises(ValueError):
        validate(json.dumps({"decisions": ok["decisions"][:1]}), ["p1", "p2"])
    bad = {"decisions": [{"pair_id": "p1", "verdict": "maybe", "reason": "r"}]}
    with pytest.raises(ValueError):
        validate(json.dumps(bad), ["p1"])


def test_chains_never_join_two_pages():
    from graphrag.resolution.cluster import cluster

    ms = [m("p1::0", "P1", "Louis", page=True), m("p2::0", "P2", "Louis", page=True),
          m("x::1", "X", "Louis"), m("y::1", "Y", "Louis")]
    # x=p1 (strongest), x=y, y=p2: the last edge would join P1 and P2 through x and y
    res = cluster(ms, [("p1::0", "x::1", 0.99), ("x::1", "y::1", 0.95), ("p2::0", "y::1", 0.90)])
    assert res.cluster_of["x::1"] == res.cluster_of["y::1"] == "P1"
    assert res.cluster_of["p2::0"] == "P2"
    assert res.rejected == [("p2::0", "y::1", "two_pages")]


def test_cluster_ids_are_stable_and_singletons_stay_alone():
    from graphrag.resolution.cluster import cluster

    ms = [m("b::1", "B", "Jane"), m("a::1", "A", "Jane"), m("c::1", "C", "Other")]
    res = cluster(ms, [("a::1", "b::1", 0.9)])
    assert res.cluster_of["a::1"] == res.cluster_of["b::1"] == "m:a::1"
    assert res.members["m:c::1"] == ["c::1"]


def test_cannot_link_blocks_a_chain_that_contradicts_a_different_verdict():
    from graphrag.resolution.cluster import cluster

    ms = [m("us::1", "A", "United States", "PLACE"), m("x::1", "B", "US", "PLACE"),
          m("uk::1", "C", "United Kingdom", "PLACE")]
    # x=us and x=uk were both judged "same" (wrongly), but us vs uk was judged "different"
    res = cluster(ms, [("us::1", "x::1", 0.95), ("uk::1", "x::1", 0.90)], [("uk::1", "us::1")])
    assert res.cluster_of["us::1"] == res.cluster_of["x::1"] != res.cluster_of["uk::1"]
    assert res.rejected == [("uk::1", "x::1", "cannot_link")]


def test_merge_order_is_most_confident_first_then_ids():
    from graphrag.resolution.cluster import merge_order

    edges = [("b", "c", 0.80), ("a", "d", 0.95), ("a", "c", 0.80)]
    assert merge_order(edges) == [("a", "d", 0.95), ("a", "c", 0.80), ("b", "c", 0.80)]


def test_judge_prompt_states_each_mentions_role():
    from graphrag.resolution.judge import JUDGE_VERSION, batch_prompt

    corpus = {"MacKenzie Scott": {"title": "MacKenzie Scott", "text": "She was married to Jeff Bezos."},
              "Jeff Bezos": {"title": "Jeff Bezos", "text": "Jeff Bezos founded Amazon."}}
    a = m("MacKenzie Scott::2", "MacKenzie Scott", "Jeff Bezos")
    b = m("Jeff Bezos::0", "Jeff Bezos", "Jeff Bezos", page=True)
    text = batch_prompt([("p1", a, b)], corpus)
    assert 'ROLE: mentioned (the paragraph is about "MacKenzie Scott")' in text
    assert "ROLE: subject (the paragraph is about this entity)" in text
    assert JUDGE_VERSION == "v2"
