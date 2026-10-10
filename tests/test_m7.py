"""M7 adapters (docs/PLAN.md M7.2, M7.6). Skipped when the optional extra `m7` is not installed."""

from pathlib import Path

import pytest

pytest.importorskip("neo4j_graphrag")

from graphrag.extraction.schema import DATE_RELATIONS, ENTITY_TYPES, RELATIONS  # noqa: E402
from graphrag.llm.cache import DiskCache  # noqa: E402
from graphrag.llm.client import CachedLLM, ProviderResponse  # noqa: E402
from graphrag.m7 import CachedLibraryLLM, CostCapReached, library_schema  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def _llm(tmp_path, calls, cost=0.01):
    def fake(model, prompt, system, temperature, options):
        calls.append(prompt)
        return ProviderResponse("{}", "v", 10, 5, cost_usd=cost)

    return CachedLLM("fake-model", fake, DiskCache(tmp_path))


def test_library_llm_uses_the_cache_and_counts_the_build_cost_once_per_call(tmp_path):
    calls = []
    llm = CachedLibraryLLM(_llm(tmp_path, calls))
    assert llm.invoke("extract this").content == "{}"
    assert llm.invoke("extract this").content == "{}"
    assert calls == ["extract this"]                      # the second call is a cache hit
    s = llm.summary()
    assert (s["calls"], s["live_calls"]) == (2, 1)
    assert s["paid_now_usd"] == 0.01


def test_library_llm_refuses_calls_at_the_cost_cap(tmp_path):
    calls = []
    llm = CachedLibraryLLM(_llm(tmp_path, calls, cost=0.6), cost_cap_usd=1.0)
    llm.invoke("a")
    llm.invoke("b")
    with pytest.raises(CostCapReached):
        llm.invoke("c")
    assert calls == ["a", "b"]


def test_library_schema_has_our_types_and_relations():
    schema = library_schema()
    assert [n["label"] for n in schema["node_types"]] == list(ENTITY_TYPES)
    assert len(schema["relationship_types"]) == len(RELATIONS) - len(DATE_RELATIONS) == 30
    assert {p["name"] for p in schema["node_types"][0]["properties"]} == {"name"} | {r.replace(" ", "_") for r in DATE_RELATIONS}


def test_m7_files_never_name_the_test_questions():
    files = [ROOT / "src" / "graphrag" / "m7.py", *sorted((ROOT / "scripts").glob("*m7*.py"))]
    assert files
    for f in files:
        text = f.read_text(encoding="utf-8")
        assert "questions_test" not in text and "results/m5" not in text and '"m5"' not in text, f.name
