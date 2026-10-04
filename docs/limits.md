# Model choices and rate limits

| Role | Model id | Provider and access | Limits | Source |
|---|---|---|---|---|
| Extractor, judge | gemini-3.8-flash | Google AI Studio, free tier | RPM TBD, RPD TBD, TPM TBD | aistudio.google.com/rate-limit |
| Generator | openai/gpt-oss-120b | OpenRouter, pinned endpoint `deepinfra/bf16`, fallbacks disabled | Prepaid credit, 7.60 USD bought | openrouter.ai/settings/credits |

Date the Gemini limits were read: TBD

## Generator history

1. Planned: `llama-3.3-70b-versatile` on Groq. Not available on my Groq account: the API returns 404
   `model_not_found`, and the account's model list contains no Llama chat models.
2. Planned next: `openai/gpt-oss-120b` on Groq's paid Developer tier. Developer upgrades are unavailable.
3. Current (2026-10-04): `openai/gpt-oss-120b` through OpenRouter.

## Generator reproducibility settings

- Endpoint pinned to `deepinfra/bf16` (DeepInfra, bf16 weights). Chosen for a low price, high uptime and a
  declared weight format; the tag also excludes DeepInfra's separate `turbo` endpoint.
- `allow_fallbacks: false` and `require_parameters: true`: OpenRouter may not route to another provider or
  to an endpoint that ignores a parameter.
- Temperature 0 and `reasoning.effort` = `medium` for every system.
- Every live call records the serving provider, token counts and cost in
  `results/spend/openrouter_calls.jsonl`. A call served by any other provider raises and is not cached.
- The pinned endpoint and the effort are part of the LLM cache key.

## Cost and credit

- Price of the pinned endpoint is read live from OpenRouter's endpoint list each time the estimate runs.
- Cost estimate: `results/m0/generator_cost.json` (`scripts/estimate_generator_cost.py`). Measured on the
  pinned endpoint (6 validation questions, all served by DeepInfra, max 271 output tokens including
  reasoning), at USD 0.037 per million input tokens and USD 0.17 per million output tokens. Each call is
  assumed to use the full 1,500-token context plus 200 prompt tokens and the max measured output:

  | Scenario | Dev iterations | Calls | Cost (USD) | Fits 7.60 USD credit |
  |---|---|---|---|---|
  | Expected | 10 | 8,500 | 0.93 | yes |
  | Pessimistic | 20 | 14,750 | 1.61 | yes |

- Remaining credit: `results/spend/openrouter_balance.json` (`scripts/credit_status.py`; also refreshed by
  `scripts/check_env.py`).
