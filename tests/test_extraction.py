import json

import pytest

from graphrag.eval.extraction import dates_match, gold_triples, names_match, norm_name, score
from graphrag.extraction.extract import make_batches, validate
from graphrag.extraction.schema import RELATIONS, strict_json_schema


def test_strict_schema_is_self_contained_and_closed():
    schema = strict_json_schema()
    text = json.dumps(schema)
    assert "$ref" not in text and "$defs" not in text

    def objects(node):
        if isinstance(node, dict):
            if node.get("type") == "object":
                yield node
            for v in node.values():
                yield from objects(v)
        elif isinstance(node, list):
            for v in node:
                yield from objects(v)

    for obj in objects(schema):
        assert obj["additionalProperties"] is False
        assert set(obj["required"]) == set(obj["properties"])
    rel_enum = schema["properties"]["paragraphs"]["items"]["properties"]["relations"]["items"]["properties"]["relation"]["enum"]
    assert set(rel_enum) == set(RELATIONS) | {"OTHER"}


def test_validate_requires_exactly_the_sent_chunk_ids():
    batch = [{"id": "A", "title": "A", "text": "a"}, {"id": "B", "title": "B", "text": "b"}]
    ok = {"paragraphs": [{"chunk_id": c, "entities": [], "relations": []} for c in ("B", "A")]}
    assert len(validate(json.dumps(ok), batch).paragraphs) == 2
    with pytest.raises(ValueError, match="missing"):
        validate(json.dumps({"paragraphs": ok["paragraphs"][:1]}), batch)


def test_batches_are_stable_and_sorted():
    chunks = [{"id": str(i)} for i in (5, 1, 3, 2, 4)]
    assert [[c["id"] for c in b] for b in make_batches(chunks, size=2)] == [["1", "2"], ["3", "4"], ["5"]]


def test_name_and_date_matching():
    assert norm_name("Polish-Russian War (film)") == "polish russian war"
    assert names_match("Xawery Zulawski", "Xawery Zulawski", 90)
    assert not names_match("John Adams", "John Quincy Adams", 90)
    assert dates_match("28 August 1921", "August 28, 1921")
    assert dates_match("1921", "28 August 1921")
    assert not dates_match("28 August 1921", "29 August 1921")
    assert not dates_match("1921", "1922")


def test_recall_and_slot_precision():
    gold = [
        {"subject": "Film A", "relation": "director", "object": "Jane Doe", "chunk_id": "c1"},
        {"subject": "Jane Doe", "relation": "date of birth", "object": "1 May 1950", "chunk_id": "c2"},
    ]
    extracted = [
        {"subject": "Film A", "relation": "director", "object": "Jane Doe"},          # correct
        {"subject": "Jane Doe", "relation": "date of birth", "object": "2 May 1950"},  # wrong object
        {"subject": "Jane Doe", "relation": "spouse", "object": "Joe"},               # no gold slot: not counted
    ]
    s = score(extracted, gold, 90)["overall"]
    assert (s["gold"], s["recalled"], s["slot_pred"], s["slot_correct"]) == (2, 1, 2, 1)


def test_gold_triples_map_to_the_subject_paragraph():
    corpus = {"Film A (1990 film)": {"title": "Film A (1990 film)"}, "Jane Doe": {"title": "Jane Doe"}}
    q = {"evidences": [["Film A", "director", "Jane Doe"], ["Nobody", "father", "X"]],
         "gold_chunk_ids": ["Film A (1990 film)", "Jane Doe"]}
    triples, unmapped = gold_triples([q, q], corpus)
    assert [t["chunk_id"] for t in triples] == ["Film A (1990 film)"] and unmapped == 1


def test_aliases_let_a_country_match_its_demonym():
    gold = [{"subject": "Michael X", "relation": "country of citizenship", "object": "Hungarian", "chunk_id": "c",
             "subject_aliases": ["Michael X"], "object_aliases": ["Hungarian", "Hungary"]}]
    extracted = [{"subject": "Michael X", "relation": "country of citizenship", "object": "Hungary"}]
    assert score(extracted, gold, 90)["overall"]["recalled"] == 1
    no_alias = [{**gold[0], "object_aliases": ["Hungarian"]}]
    assert score(extracted, no_alias, 90)["overall"]["recalled"] == 0
