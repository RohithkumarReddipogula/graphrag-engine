from graphrag.extraction.postprocess import clean_relation, clean_rows


def rel(subject, relation, obj, label=None):
    return {"subject": subject, "relation": relation, "object": obj, "other_label": label}


def test_other_with_a_listed_label_becomes_that_relation_and_loses_the_prefix():
    t, applied = clean_relation(rel("Richard Wingfield", "OTHER", "occupation: politician and peer", "occupation"))
    assert (t["relation"], t["object"], t["other_label"]) == ("occupation", "politician and peer", None)
    assert applied == ["other_to_listed", "strip_label_prefix"]


def test_label_prefix_is_stripped_but_other_label_kept_when_not_a_listed_relation():
    t, applied = clean_relation(rel("Why Did I Get Married Too?", "OTHER", "cast member: Richard T. Jones", "cast member"))
    assert (t["relation"], t["object"], t["other_label"]) == ("OTHER", "Richard T. Jones", "cast member")
    assert applied == ["strip_label_prefix"]


def test_relation_name_prefix_on_a_listed_relation_is_stripped():
    t, _ = clean_relation(rel("Heat", "director", "Director: Michael Mann"))
    assert t["object"] == "Michael Mann"


def test_colons_that_are_not_the_label_are_left_alone():
    t, applied = clean_relation(rel("X", "OTHER", "Star Wars: A New Hope", "sequel"))
    assert t["object"] == "Star Wars: A New Hope" and applied == [] and "postprocessed" not in t
    t, applied = clean_relation(rel("X", "director", "Jane Doe"))
    assert applied == []


def test_counts_per_rule():
    rows = [{"chunk_id": "c", "relations": [
        rel("A", "OTHER", "occupation: actor", "occupation"),
        rel("B", "OTHER", "cast member: C", "cast member"),
        rel("D", "director", "E"),
    ]}]
    cleaned, counts = clean_rows(rows)
    assert counts == {"other_to_listed": 1, "strip_label_prefix": 2, "relations_total": 3, "relations_changed": 2}
    assert [t["relation"] for t in cleaned[0]["relations"]] == ["occupation", "OTHER", "director"]
