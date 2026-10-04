from graphrag.llm.cache import DiskCache, cache_key
from graphrag.llm.client import CachedLLM, ProviderResponse


def test_cache_key_is_stable_and_order_independent():
    a = cache_key("m", {"prompt": "p", "temperature": 0.0})
    b = cache_key("m", {"temperature": 0.0, "prompt": "p"})
    assert a == b
    assert a != cache_key("other-model", {"prompt": "p", "temperature": 0.0})


def test_disk_cache_roundtrip_handles_slash_in_model(tmp_path):
    cache = DiskCache(tmp_path)
    cache.put("openai/gpt-oss-20b", "ab12", {"x": 1})
    assert cache.get("openai/gpt-oss-20b", "ab12") == {"x": 1}
    assert cache.get("openai/gpt-oss-20b", "missing") is None


def test_cached_llm_calls_provider_once(tmp_path):
    calls = []

    def fake(model, prompt, system, temperature, options):
        calls.append((prompt, options))
        return ProviderResponse("ready", "fake-v1", input_tokens=7, output_tokens=3)

    llm = CachedLLM("fake-model", fake, DiskCache(tmp_path))
    first = llm.complete("hello")
    second = llm.complete("hello")
    assert calls == [("hello", {})]
    assert (first.cached, second.cached) == (False, True)
    assert second.text == "ready" and second.model_version == "fake-v1"
    assert (second.input_tokens, second.output_tokens) == (7, 3)


def test_options_are_part_of_the_cache_key(tmp_path):
    calls = []

    def fake(model, prompt, system, temperature, options):
        calls.append(options)
        return ProviderResponse("x", "v", 1, 1)

    llm = CachedLLM("fake-model", fake, DiskCache(tmp_path))
    llm.complete("q", options={"reasoning_effort": "low"})
    llm.complete("q", options={"reasoning_effort": "high"})
    llm.complete("q", options={"reasoning_effort": "low"})
    assert calls == [{"reasoning_effort": "low"}, {"reasoning_effort": "high"}]


def test_salt_makes_a_separately_cached_call(tmp_path):
    calls = []

    def fake(model, prompt, system, temperature, options):
        calls.append(prompt)
        return ProviderResponse("x", "v", 1, 1)

    llm = CachedLLM("m", fake, DiskCache(tmp_path))
    llm.complete("q")
    llm.complete("q", salt="retry-1")
    llm.complete("q", salt="retry-1")
    assert calls == ["q", "q"]
