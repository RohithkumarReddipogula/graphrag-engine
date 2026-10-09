# Model choices and rate limits

| Role | Model id | Provider and access | Limits | Source |
|---|---|---|---|---|
| Judge | gemini-3.8-flash | Google AI Studio, free tier | not published; to be read in AI Studio (see below) | aistudio.google.com/rate-limit |
| Extractor | openai/gpt-oss-120b | OpenRouter, same pinned endpoint as the generator | Prepaid credit (shared with the generator) | openrouter.ai/settings/credits |
| Generator | openai/gpt-oss-120b | OpenRouter, pinned endpoint `deepinfra/bf16`, fallbacks disabled | Prepaid credit, 7.60 USD bought | openrouter.ai/settings/credits |

Date the Gemini limits were read: TBD (must be read in AI Studio before the M5 judge runs). On
2026-10-09 the values were sent as blank placeholders ("RPM ___, TPM ___, RPD ___"), so nothing was
recorded; the real values are still needed before the judge runs.

## Gemini free-tier limits: what was checked (2026-10-09)

- Google's rate-limit page (ai.google.dev/gemini-api/docs/rate-limits) gives no free-tier RPM, TPM or RPD
  for gemini-3.8-flash. It says limits "can be viewed in Google AI Studio", that they are applied per
  project (not per API key), and that daily quotas reset at midnight Pacific time. Its Batch API tables
  start at Tier 1, so the batch mode is not available on the free tier.
- Third-party pages conflict: one claims about 20 requests per day for gemini-3.8-flash; another says
  the free-tier values are unpublished. Neither is verified.
- The M5 judge plan (`docs/PLAN.md`, M5.5) is designed to finish even at 20 requests per day.

## Generator history

1. Planned: `llama-3.3-70b-versatile` on Groq. Not available on my Groq account: the API returns 404
   `model_not_found`, and the account's model list contains no Llama chat models.
2. Planned next: `openai/gpt-oss-120b` on Groq's paid Developer tier. Developer upgrades are unavailable.
3. Current (2026-10-04): `openai/gpt-oss-120b` through OpenRouter.

## Extractor change (2026-10-04)

The plan first named `gemini-3.8-flash` (free tier) as the extractor. The first pilot call returned
503 "This model is currently experiencing high demand", and the free tier's daily limits are unknown, so
the extraction could stall for days. The extractor is now `openai/gpt-oss-120b` on the same pinned
OpenRouter endpoint as the generator, with strict JSON-schema output. The judge stays `gemini-3.8-flash`,
so it differs from both. Pilot tokens, cost and the full-corpus projection: `results/m2/pilot.json`.

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
