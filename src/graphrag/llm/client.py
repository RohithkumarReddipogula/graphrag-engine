"""Thin, cached LLM clients. Gemini serves the extractor and judge, OpenRouter serves the generator.

There is deliberately no cross-provider fallback: if a quota runs out or the pinned OpenRouter provider is
down, the call raises and the run is resumed later from the cache.
"""

import json
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import httpx

from graphrag.config import Settings
from graphrag.llm.cache import DiskCache, cache_key

OPENROUTER_URL = "https://openrouter.ai/api/v1"
_LEDGER_LOCK = threading.Lock()   # calls may run in parallel threads


@dataclass(frozen=True)
class ProviderResponse:
    text: str
    model_version: str    # version string the provider reports back
    input_tokens: int
    output_tokens: int    # includes hidden reasoning tokens, which are billed as output
    provider: str = ""    # upstream provider that actually served the call
    cost_usd: float = 0.0


@dataclass(frozen=True)
class LLMResult:
    text: str
    model: str            # model id we asked for
    model_version: str
    input_tokens: int
    output_tokens: int
    provider: str
    cost_usd: float       # cost of the original live call; a cache hit costs nothing now
    cached: bool


# (model, prompt, system, temperature, provider-specific options) -> response
ProviderCall = Callable[[str, str, str | None, float, dict[str, Any]], ProviderResponse]


def _gemini_call(settings: Settings) -> ProviderCall:
    from google import genai
    from google.genai import types

    if settings.gemini_api_key is None:
        raise RuntimeError("GEMINI_API_KEY is not set in .env")
    client = genai.Client(api_key=settings.gemini_api_key.get_secret_value())

    def call(model: str, prompt: str, system: str | None, temperature: float, options: dict[str, Any]) -> ProviderResponse:
        resp = client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system,
                temperature=temperature,
                # We never pass tools; disabling AFC also silences the SDK's per-call warning.
                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
                **options,
            ),
        )
        usage = resp.usage_metadata
        return ProviderResponse(
            text=resp.text or "",
            model_version=resp.model_version or model,
            input_tokens=(usage and usage.prompt_token_count) or 0,
            output_tokens=((usage and usage.candidates_token_count) or 0) + ((usage and usage.thoughts_token_count) or 0),
            provider="google-ai-studio",
        )

    return call


def _openrouter_call(settings: Settings, ledger: Path) -> ProviderCall:
    if settings.openrouter_api_key is None:
        raise RuntimeError("OPENROUTER_API_KEY is not set in .env")
    headers = {"Authorization": f"Bearer {settings.openrouter_api_key.get_secret_value()}"}
    pinned = settings.generator_provider            # e.g. "deepinfra/bf16"
    pinned_name = pinned.split("/")[0].lower()      # OpenRouter reports the provider name, e.g. "DeepInfra"

    def call(model: str, prompt: str, system: str | None, temperature: float, options: dict[str, Any]) -> ProviderResponse:
        messages = [{"role": "system", "content": system}] if system else []
        messages.append({"role": "user", "content": prompt})
        body = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            # One upstream endpoint, no silent fallback to another provider or quantization.
            "provider": {"order": [pinned], "allow_fallbacks": False, "require_parameters": True},
            "usage": {"include": True},
            **options,
        }
        resp = httpx.post(f"{OPENROUTER_URL}/chat/completions", headers=headers, json=body, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        if "error" in data:
            raise RuntimeError(f"OpenRouter error: {data['error']}")

        served_by = data.get("provider", "")
        usage = data.get("usage") or {}
        result = ProviderResponse(
            text=data["choices"][0]["message"].get("content") or "",
            model_version=data.get("model", model),
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
            provider=served_by,
            cost_usd=float(usage.get("cost") or 0.0),
        )
        # Log the spend before checking the provider: a wrong-provider call is still billed.
        append_ledger(ledger, model, result)
        if served_by.lower().replace(" ", "") != pinned_name:
            raise RuntimeError(f"served by {served_by!r}, expected pinned provider {pinned!r}; not caching")
        return result

    return call


def append_ledger(ledger: Path, model: str, r: ProviderResponse) -> None:
    ledger.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": model,
        "provider": r.provider,
        "input_tokens": r.input_tokens,
        "output_tokens": r.output_tokens,
        "cost_usd": r.cost_usd,
    }
    with _LEDGER_LOCK, ledger.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


class CachedLLM:
    def __init__(
        self,
        model: str,
        call: ProviderCall,
        cache: DiskCache,
        backend: str = "",
        defaults: dict[str, Any] | None = None,
    ):
        self.model = model
        self._call = call
        self._cache = cache
        self._backend = backend               # e.g. "openrouter:deepinfra/bf16"; part of the cache key
        self._defaults = defaults or {}       # options applied to every call (e.g. reasoning effort)

    def complete(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.0,
        options: dict[str, Any] | None = None,
    ) -> LLMResult:
        """`options` are provider-specific request fields. They, the backend and the defaults are all
        part of the cache key, so changing any of them never returns a stale answer."""
        options = {**self._defaults, **(options or {})}
        request: dict[str, Any] = {"prompt": prompt, "system": system, "temperature": temperature}
        if options:
            request["options"] = options
        if self._backend:
            request["backend"] = self._backend
        key = cache_key(self.model, request)
        hit = self._cache.get(self.model, key)
        if hit is not None:
            return LLMResult(**{**hit["result"], "cached": True})

        resp = self._call(self.model, prompt, system, temperature, options)
        result = LLMResult(
            text=resp.text,
            model=self.model,
            model_version=resp.model_version,
            input_tokens=resp.input_tokens,
            output_tokens=resp.output_tokens,
            provider=resp.provider,
            cost_usd=resp.cost_usd,
            cached=False,
        )
        self._cache.put(self.model, key, {"request": request, "result": asdict(result), "cached_at": time.time()})
        return result


def make_llm(model: str, settings: Settings) -> CachedLLM:
    cache = DiskCache(settings.cache_dir / "llm")
    if model.startswith("gemini"):
        return CachedLLM(model, _gemini_call(settings), cache, backend="gemini")
    ledger = settings.results_dir / "spend" / "openrouter_calls.jsonl"
    return CachedLLM(
        model,
        _openrouter_call(settings, ledger),
        cache,
        backend=f"openrouter:{settings.generator_provider}",
        defaults={"reasoning": {"effort": settings.generator_reasoning_effort}},
    )
