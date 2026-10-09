import json
from collections import Counter

from graphrag.config import ROOT, get_settings

DATA = json.loads((ROOT / "docs" / "graph_view" / "subgraph.json").read_text(encoding="utf-8"))


def test_links_and_questions_reference_existing_nodes_and_links():
    node_ids = {n["id"] for n in DATA["nodes"]}
    link_ids = {l["id"] for l in DATA["links"]}
    assert all(l["source"] in node_ids and l["target"] in node_ids for l in DATA["links"])
    for q in DATA["questions"]:
        assert set(q["path_node_ids"]) <= node_ids and set(q["path_link_ids"]) <= link_ids


def test_twenty_dev_questions_five_per_type_and_no_test_questions():
    assert Counter(q["type"] for q in DATA["questions"]) == {
        "comparison": 5, "inference": 5, "compositional": 5, "bridge_comparison": 5}
    d = get_settings().data_dir
    test_ids = {json.loads(l)["id"] for f in ("questions_test.jsonl", "single_hop_test.jsonl")
                for l in (d / f).read_text().splitlines()}
    assert not test_ids & {q["id"] for q in DATA["questions"]}


def test_answers_match_the_committed_dev_run():
    rows = {json.loads(l)["id"]: json.loads(l) for l in
            (ROOT / "results/m4/generation_dev_graph_plus_chunks_g0.5.jsonl").read_text().splitlines()}
    for q in DATA["questions"]:
        assert q["answer"] == rows[q["id"]]["prediction"] and q["exact_match"] == (rows[q["id"]]["em"] == 1.0)


def test_no_paragraph_text_is_exported():
    allowed = {"description", "source", "licence", "nodes", "links", "questions"}
    assert set(DATA) == allowed
    assert all(set(n) == {"id", "label", "type", "is_page"} for n in DATA["nodes"])
