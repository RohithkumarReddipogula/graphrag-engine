import numpy as np

from graphrag.graph.build import build_graph
from graphrag.graph.retrieve import HUB_DEGREE, GraphIndex, graph_context, score_paths
from graphrag.resolution.mentions import Mention, norm_name


def m(mid, chunk, name, type_="PERSON", page=False):
    return Mention(id=mid, chunk_id=chunk, name=name, norm=norm_name(name), type=type_, description="",
                   is_page=page, split="tune")


def unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


def small_graph(extra_rels=()):
    mentions = [
        m("Heat::0", "Heat", "Heat", "FILM", page=True), m("Heat::1", "Heat", "Michael Mann"),
        m("Michael Mann::0", "Michael Mann", "Michael Mann", page=True),
        m("Michael Mann::1", "Michael Mann", "Chicago", "PLACE"),
        m("Other::0", "Other", "Other film", "FILM", page=True), m("Other::1", "Other", "M. Mann"),
    ]
    cluster_of = {"Heat::0": "Heat", "Heat::1": "c:mann-mention", "Michael Mann::0": "Michael Mann",
                  "Michael Mann::1": "c:chicago", "Other::0": "Other", "Other::1": "c:mmann"}
    rows = [
        {"chunk_id": "Heat", "relations": [
            {"subject": "Heat", "relation": "director", "object": "Michael Mann", "other_label": None},
            {"subject": "Heat", "relation": "publication date", "object": "1995", "other_label": None},
            {"subject": "Heat", "relation": "occupation", "object": "not an entity", "other_label": None}]},
        {"chunk_id": "Michael Mann", "relations": [
            {"subject": "Michael Mann", "relation": "place of birth", "object": "Chicago", "other_label": None},
            {"subject": "Michael Mann", "relation": "date of birth", "object": "5 February 1943", "other_label": None},
            *extra_rels]},
        {"chunk_id": "Other", "relations": []},
    ]
    return mentions, cluster_of, rows


def test_build_resolves_endpoints_dates_and_exact_titles():
    mentions, cluster_of, rows = small_graph()
    g = build_graph(mentions, cluster_of, rows)
    assert [(r.source, r.type, r.target) for r in g.rels] == [
        ("Heat", "director", "c:mann-mention"), ("Michael Mann", "place of birth", "c:chicago")]
    assert g.entities["Heat"].dates == {"publication_date": [("1995", "Heat")]}
    assert g.stats["dropped_relations"] == {"object is not an extracted entity": 1}
    # "Michael Mann" mentioned in Heat's paragraph is a separate M3 cluster from his page: an exact-title edge links them
    assert [(x.source, x.target) for x in g.exact_titles] == [("c:mann-mention", "Michael Mann")]


def index_for(g, use_exact_title=True):
    emb = {e: unit([1.0, i * 0.01]) for i, e in enumerate(sorted(g.entities))}
    return GraphIndex(g, emb, use_exact_title=use_exact_title)


def test_exact_title_is_an_identity_link_not_a_hop():
    mentions, cluster_of, rows = small_graph()
    g = build_graph(mentions, cluster_of, rows)
    idx = index_for(g)
    texts = [p.text for p in idx.paths(["Heat"])]
    # Heat -> (Michael Mann mention == Michael Mann page) -> Chicago is 2 REL hops; the exact-title jump is free
    assert any("Heat -director-> Michael Mann; Michael Mann -place of birth-> Chicago" in t for t in texts)
    no_exact = index_for(g, use_exact_title=False)
    assert not any("Chicago" in p.text for p in no_exact.paths(["Heat"]))


def test_seeds_verbatim_longest_first_and_pages_first():
    mentions, cluster_of, rows = small_graph()
    g = build_graph(mentions, cluster_of, rows)
    idx = index_for(g)
    seeds = idx.seeds("Where was the director of Heat born, Michael Mann?", unit([0.0, 1.0]))
    ids = [s for s, how in seeds]
    # pages first; among pages the longer match first ("Michael Mann" before "Heat"); then the non-page
    # entity that shares the alias "Michael Mann"
    assert ids[:3] == ["Michael Mann", "Heat", "c:mann-mention"]
    assert all(how == "verbatim" for _, how in seeds[:3])


def test_hubs_end_paths_but_are_never_passed_through():
    extra = [{"subject": "Chicago", "relation": "country", "object": "Michael Mann", "other_label": None}]
    mentions, cluster_of, rows = small_graph()
    g = build_graph(mentions, cluster_of, rows)
    idx = index_for(g)
    idx.degree["c:chicago"] = HUB_DEGREE + 1
    idx.degree["c:mann-mention"] = HUB_DEGREE + 1
    texts = [p.text for p in idx.paths(["Heat"])]
    assert any(t.startswith("Heat -director-> Michael Mann") for t in texts)      # the hub ends a path
    assert not any("Chicago" in t for t in texts)                                  # but is not passed through


def test_graph_context_puts_facts_first_then_page_paragraphs():
    mentions, cluster_of, rows = small_graph()
    g = build_graph(mentions, cluster_of, rows)
    idx = index_for(g)
    paths = idx.paths(["Heat"])
    top = score_paths(paths, unit([1.0, 0.0]), np.stack([unit([1.0, 0.0])] * len(paths)), idx)
    corpus = {c: {"title": c, "text": f"Text of {c}."} for c in ("Heat", "Michael Mann", "Other")}
    ctx = graph_context(top, idx, corpus, budget_tokens=1500)
    assert ctx.text.startswith("Fact: ")
    assert "Heat" in ctx.chunk_ids and "Michael Mann" in ctx.chunk_ids
    assert ctx.text.index("Fact:") < ctx.text.index("[Heat] Text of Heat.")
