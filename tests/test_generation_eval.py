from graphrag.eval.answers import normalize_answer, score
from graphrag.eval.bootstrap import mean_ci, paired_diff_ci
from graphrag.generation import build_prompt, n_tokens, pack_context, parse_answer
from graphrag.retrieval.base import Passage


def p(cid, text):
    return Passage(chunk_id=cid, title=cid, text=text, score=0.0)


def test_pack_context_respects_budget_and_skips_long_passages():
    short, long_ = "word " * 20, "word " * 2000
    ctx = pack_context([p("a", short), p("b", long_), p("c", short)], budget_tokens=200)
    assert ctx.chunk_ids == ["a", "c"]
    assert ctx.tokens <= 200 and n_tokens(ctx.text) <= ctx.tokens


def test_closed_book_prompt_has_no_context():
    system, prompt = build_prompt("Who?", None)
    assert "Context" not in prompt and "context" not in system


def test_parse_answer():
    assert parse_answer("Answer: Steven Spielberg.\n") == "Steven Spielberg"
    assert parse_answer('\n"Paris"\nbecause ...') == "Paris"
    assert parse_answer("**yes**") == "yes"


def test_official_normalisation_and_alias_max():
    assert normalize_answer("The Beatles!") == "beatles"
    assert score("Bruxellois", ["Brussels", "Bruxellois"]) == {"em": 1.0, "f1": 1.0}
    assert score("Brussels, Belgium", ["Brussels"])["em"] == 0.0
    assert score("Brussels, Belgium", ["Brussels"])["f1"] > 0.5


def test_yes_no_never_gets_partial_credit():
    assert score("no", ["yes"]) == {"em": 0.0, "f1": 0.0}
    assert score("yes it is", ["yes"])["f1"] == 0.0


def test_bootstrap_ci_contains_mean_and_is_deterministic():
    vals = [0.0, 1.0] * 50
    a, b = mean_ci(vals), mean_ci(vals)
    assert a == b and a["ci_low"] <= a["mean"] <= a["ci_high"] and a["n"] == 100
    d = paired_diff_ci([1.0] * 10, [0.0] * 10)
    assert d["mean"] == d["ci_low"] == d["ci_high"] == 1.0
