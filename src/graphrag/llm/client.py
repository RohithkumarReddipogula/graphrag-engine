"""Thin, cached LLM clients. Gemini serves the extractor and judge, Groq serves the generator.

There is deliberately no cross-provider fallback: if a quota runs out, the call raises and the run is
resumed later from the cache.
"""

from dataclasses import asdict, dataclass
from typing import Any, Callable

from graphrag.config import Settings
from graphrag.llm.cache import DiskCache, cache_key


@dataclass(frozen=True)
class ProviderResponse:
    text: str
    model_version: str    # version string the provider reports back
    input_tokens: int
    output_tokens: int    # includes hidden reasoning tokens, which are billed as output


@dataclass(frozen=True)
class LLMResult:
    text: str
    model: str            # model id we asked for
    model_version: str
    input_tokens: int
    output_tokens: int
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
        )

    return call


def _groq_call(settings: Settings) -> ProviderCall:
    from groq import Groq

    if settings.groq_api_key is None:
        raise RuntimeError("GROQ_API_KEY is not set in .env")
    client = Groq(api_key=settings.groq_api_key.get_secret_value())

    def call(model: str, prompt: str, system: str | None, temperature: float, options: dict[str, Any]) -> ProviderResponse:
        messages = [{"role": "system", "content": system}] if system else []
        messages.append({"role": "user", "content": prompt})
        resp = client.chat.completions.create(model=model, messages=messages, temperature=temperature, **options)
        return ProviderResponse(
            text=resp.choices[0].message.content or "",
            model_version=resp.model or model,
            input_tokens=resp.usage.prompt_tokens if resp.usage else 0,
            output_tokens=resp.usage.completion_tokens if resp.usage else 0,
        )

    return call


class CachedLLM:
    def __init__(self, model: str, call: ProviderCall, cache: DiskCache):
        self.model = model
        self._call = call
        self._cache = cache

    def complete(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.0,
        options: dict[str, Any] | None = None,
    ) -> LLMResult:
        """`options` are provider-specific request fields (for example Groq `reasoning_effort`).
        They are part of the cache key, so changing them never returns a stale answer."""
        options = options or {}
        request: dict[str, Any] = {"prompt": prompt, "system": system, "temperature": temperature}
        if options:
            request["options"] = options
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
            cached=False,
        )
        self._cache.put(self.model, key, {"request": request, "result": asdict(result)})
        return result


def make_llm(model: str, settings: Settings) -> CachedLLM:
    call = _gemini_call(settings) if model.startswith("gemini") else _groq_call(settings)
    return CachedLLM(model, call, DiskCache(settings.cache_dir / "llm"))
