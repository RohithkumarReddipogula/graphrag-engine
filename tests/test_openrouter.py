import json

import httpx
import pytest

from graphrag.config import Settings
from graphrag.llm import client as llm_client
from graphrag.llm.cache import DiskCache
from graphrag.llm.client import CachedLLM, ProviderResponse


def _settings(tmp_path) -> Settings:
    return Settings(
        _env_file=None,
        openrouter_api_key="test-key",
        generator_provider="deepinfra/bf16",
        cache_dir=tmp_path / "cache",
        results_dir=tmp_path / "results",
    )


def _fake_post(provider_name: str, captured: list):
    def post(url, headers, body, deadline_s=600):
        captured.append(body)
        return 200, {
            "model": body["model"],
            "provider": provider_name,
            "choices": [{"message": {"content": "ready"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "cost": 0.0001},
        }
    return post


def test_request_pins_provider_and_disables_fallbacks(tmp_path, monkeypatch):
    captured = []
    monkeypatch.setattr(llm_client, "_post_json", _fake_post("DeepInfra", captured))
    llm = llm_client.make_llm("openai/gpt-oss-120b", _settings(tmp_path))

    result = llm.complete("hi")

    body = captured[0]
    assert body["provider"] == {"order": ["deepinfra/bf16"], "allow_fallbacks": False, "require_parameters": True}
    assert body["temperature"] == 0.0
    assert body["reasoning"] == {"effort": "medium"}
    assert result.provider == "DeepInfra" and result.cost_usd == 0.0001


def test_every_live_call_is_logged_and_cache_hits_are_not(tmp_path, monkeypatch):
    monkeypatch.setattr(llm_client, "_post_json", _fake_post("DeepInfra", []))
    settings = _settings(tmp_path)
    llm = llm_client.make_llm("openai/gpt-oss-120b", settings)

    llm.complete("hi")
    llm.complete("hi")

    ledger = settings.results_dir / "spend" / "openrouter_calls.jsonl"
    rows = [json.loads(line) for line in ledger.read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["provider"] == "DeepInfra" and rows[0]["cost_usd"] == 0.0001


def test_wrong_provider_is_billed_but_not_cached(tmp_path, monkeypatch):
    monkeypatch.setattr(llm_client, "_post_json", _fake_post("Together", []))
    settings = _settings(tmp_path)
    llm = llm_client.make_llm("openai/gpt-oss-120b", settings)

    with pytest.raises(RuntimeError, match="expected pinned provider"):
        llm.complete("hi")

    assert (settings.results_dir / "spend" / "openrouter_calls.jsonl").exists()
    assert not any((settings.cache_dir / "llm").rglob("*.json"))


def test_backend_is_part_of_the_cache_key(tmp_path):
    calls = []

    def fake(model, prompt, system, temperature, options):
        calls.append(model)
        return ProviderResponse("x", "v", 1, 1)

    cache = DiskCache(tmp_path)
    CachedLLM("m", fake, cache, backend="openrouter:deepinfra/bf16").complete("q")
    CachedLLM("m", fake, cache, backend="openrouter:groq").complete("q")
    assert len(calls) == 2


class _SlowStream:
    """Stands in for httpx.stream: keeps sending padding, like OpenRouter does for slow requests."""
    status_code = 200
    request = None

    def __init__(self, *a, **k):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def iter_bytes(self):
        while True:
            yield b" "


def test_call_deadline_stops_a_stream_that_never_ends(monkeypatch):
    monkeypatch.setattr(llm_client.httpx, "stream", _SlowStream)
    with pytest.raises(llm_client.CallTimeout, match="TIMEOUT"):
        llm_client._post_json("https://x", {}, {}, deadline_s=0.05)


def test_timeouts_and_overload_are_retryable():
    from graphrag.extraction.extract import _is_retryable

    assert _is_retryable(llm_client.CallTimeout("TIMEOUT: OpenRouter call exceeded 600 s"))
    assert _is_retryable(httpx.ReadTimeout("The read operation timed out"))
    assert _is_retryable(RuntimeError("503 Service Unavailable"))
    assert not _is_retryable(RuntimeError("401 Unauthorized"))
