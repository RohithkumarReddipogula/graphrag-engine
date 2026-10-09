"""Thin, cached LLM clients. Every LLM call goes through OpenRouter with prepaid credit: the generator and
extractor (gpt-oss-120b, pinned deepinfra/bf16) and the M5 judge (llama-3.3-70b, pinned parasail/fp8).

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
# Hard wall-clock limit per call. httpx timeouts only limit the gap between received bytes, and
# OpenRouter keeps slow requests alive by sending padding, so a stalled upstream would never time out.
CALL_DEADLINE_S = 600
READ_GAP_S = 120


class CallTimeout(RuntimeError):
    """A call exceeded CALL_DEADLINE_S; message contains TIMEOUT so callers can retry it."""


def _post_json(url: str, headers: dict, body: dict, deadline_s: float = CALL_DEADLINE_S) -> tuple[int, dict]:
    """POST and read the JSON body as a stream, aborting once the whole call takes longer than deadline_s."""
    start = time.monotonic()
    timeout = httpx.Timeout(READ_GAP_S, connect=30.0)
    with httpx.stream("POST", url, headers=headers, json=body, timeout=timeout) as resp:
        chunks = []
        for chunk in resp.iter_bytes():
            chunks.append(chunk)
            if time.monotonic() - start > deadline_s:
                raise CallTimeout(f"TIMEOUT: OpenRouter call exceeded {deadline_s:.0f} s")
        if resp.status_code >= 400:
            raise httpx.HTTPStatusError(
                f"{resp.status_code} from OpenRouter: {b''.join(chunks)[:300]!r}", request=resp.request, response=resp
            )
        return resp.status_code, json.loads(b"".join(chunks))
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


def _openrouter_call(settings: Settings, ledger: Path, pinned: str) -> ProviderCall:
    """pinned: the OpenRouter endpoint tag, e.g. "deepinfra/bf16" (generator) or "parasail/fp8" (judge)."""
    if settings.openrouter_api_key is None:
        raise RuntimeError("OPENROUTER_API_KEY is not set in .env")
    headers = {"Authorization": f"Bearer {settings.openrouter_api_key.get_secret_value()}"}
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
        _, data = _post_json(f"{OPENROUTER_URL}/chat/completions", headers, body)
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
        salt: str | None = None,
    ) -> LLMResult:
        """`options` are provider-specific request fields. They, the backend and the defaults are all
        part of the cache key, so changing any of them never returns a stale answer. `salt` only changes
        the cache key (it is not sent), so a retry of the same request is a new, separately cached call."""
        options = {**self._defaults, **(options or {})}
        request: dict[str, Any] = {"prompt": prompt, "system": system, "temperature": temperature}
        if options:
            request["options"] = options
        if salt:
            request["salt"] = salt
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
    ledger = settings.results_dir / "spend" / "openrouter_calls.jsonl"
    if model == settings.judge_model:
        # The M5 judge: its own pinned endpoint, no reasoning parameter (the model has none, and with
        # require_parameters on, sending one would make the call fail).
        return CachedLLM(model, _openrouter_call(settings, ledger, settings.judge_provider), cache,
                         backend=f"openrouter:{settings.judge_provider}")
    return CachedLLM(
        model,
        _openrouter_call(settings, ledger, settings.generator_provider),
        cache,
        backend=f"openrouter:{settings.generator_provider}",
        defaults={"reasoning": {"effort": settings.generator_reasoning_effort}},
    )
