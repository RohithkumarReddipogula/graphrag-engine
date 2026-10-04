# Model choices and rate limits

Read the limits from the provider consoles (they are not published in the docs) and fill in the table.
Date of reading: TBD

| Role | Model id | Provider and tier | RPM | RPD | TPM | Source |
|---|---|---|---|---|---|---|
| Extractor, judge | gemini-3.8-flash | Google AI Studio, free | TBD | TBD | TBD | aistudio.google.com/rate-limit |
| Generator | openai/gpt-oss-120b | Groq, paid Developer tier | TBD | TBD | TBD | console.groq.com/settings/limits |

## Generator change (2026-10-04)

The plan first named `llama-3.3-70b-versatile` on Groq. It is not available on my Groq account: the API
returns 404 `model_not_found`, and the account's model list contains no Llama chat models. The generator
is now `openai/gpt-oss-120b`, a Groq production model, on the paid Developer tier with a spending limit set
in the Groq console.

Cost estimate: `results/m0/generator_cost.json`, produced by `scripts/estimate_generator_cost.py`. It
measures real token usage on 6 validation questions and projects the planned call volume with Groq's
published price (USD 0.15 per million input tokens, USD 0.60 per million output tokens) and the ECB
reference rate of 2026-10-02 (1 EUR = 1.1225 USD):

| Scenario | Dev iterations | Calls | Cost (EUR) |
|---|---|---|---|
| Expected | 10 | 8,500 | 3.14 |
| Pessimistic | 20 | 14,750 | 5.46 |

Both scenarios assume a full 1,500-token context plus 200 tokens of prompt per call, and the largest
measured output (267 tokens, which includes hidden reasoning tokens billed as output).

Spending limit set in the Groq console: TBD
