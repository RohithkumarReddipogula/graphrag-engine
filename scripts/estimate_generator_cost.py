"""Estimate the project's generator cost from measured token usage.

Runs 6 real 2Wiki validation questions through the configured generator (pinned OpenRouter endpoint,
configured reasoning effort; cached, so a re-run is free), reads the pinned endpoint's current price from
OpenRouter, and projects the cost for the planned call volume against the available credit.
Writes results/m0/generator_cost.json.
"""

import json
import random
import statistics
import sys

import pandas as pd

from graphrag.config import ROOT, get_settings
from graphrag.llm.client import make_llm
from graphrag.llm.spend import pinned_endpoint_price, write_balance

USD_PER_EUR = 1.1225                                # ECB reference rate, 2026-10-02
CONTEXT_BUDGET_TOKENS = 1500
PROMPT_OVERHEAD_TOKENS = 200                        # system prompt + question + formatting, rounded up

# Planned generator volume (docs/PLAN.md): dev = 125 questions, test = 375 questions.
DEV_Q, TEST_Q = 125, 375
SCENARIOS = {
    "expected": {"dev_iterations": 10, "systems": 5, "test_systems": 6},
    "pessimistic": {"dev_iterations": 20, "systems": 5, "test_systems": 6},
}

SYSTEM = ("Answer the question using only the context. Reply with the shortest possible answer: "
          "an entity name, a date, or yes/no. If the context does not contain the answer, reply unknown. "
          "No explanation.")


def sample_rows(path: str):
    df = pd.read_parquet(path)
    rng = random.Random(7)
    types = ["compositional", "comparison", "bridge_comparison", "inference"]
    picks = types + types[:2]
    return [df[df.type == t].sample(1, random_state=rng.randint(0, 10**6)).iloc[0] for t in picks]


def context(row, budget_words: int = 1100) -> str:
    out, n = [], 0
    for title, sents in zip(row.context["title"], row.context["sentences"]):
        text = " ".join(sents)
        w = len(text.split())
        if n + w > budget_words:
            continue
        out.append(f"[{title}] {text}")
        n += w
    return "\n\n".join(out)


def main(parquet_path: str) -> None:
    settings = get_settings()
    llm = make_llm(settings.generator_model, settings)
    price = pinned_endpoint_price(settings)
    calls = []
    for row in sample_rows(parquet_path):
        prompt = f"Context:\n{context(row)}\n\nQuestion: {row.question}\nAnswer:"
        out = llm.complete(prompt, system=SYSTEM)
        calls.append({
            "type": row.type,
            "provider": out.provider,
            "input_tokens": out.input_tokens,
            "output_tokens": out.output_tokens,
            "cost_usd": out.cost_usd,
        })
    outs = [c["output_tokens"] for c in calls]
    measured = {
        "calls": calls,
        "mean_output_tokens": round(statistics.mean(outs), 1),
        "max_output_tokens": max(outs),
    }

    # Conservative per-call figures: a full context budget, and the max measured output.
    in_per_call = CONTEXT_BUDGET_TOKENS + PROMPT_OVERHEAD_TOKENS
    out_per_call = measured["max_output_tokens"]
    balance = write_balance(settings)

    projections = {}
    for name, s in SCENARIOS.items():
        n_calls = s["dev_iterations"] * s["systems"] * DEV_Q + s["test_systems"] * TEST_Q
        usd = n_calls * (in_per_call * price["input"] + out_per_call * price["output"]) / 1e6
        projections[name] = {
            **s,
            "calls": n_calls,
            "usd": round(usd, 2),
            "eur": round(usd / USD_PER_EUR, 2),
            "fits_remaining_credit": usd <= balance["remaining_usd"],
        }

    report = {
        "model": settings.generator_model,
        "pinned_endpoint": settings.generator_provider,
        "reasoning_effort": settings.generator_reasoning_effort,
        "price_usd_per_million_tokens": price,
        "price_source": f"openrouter.ai/api/v1/models/{settings.generator_model}/endpoints",
        "usd_per_eur": USD_PER_EUR,
        "remaining_credit_usd": balance["remaining_usd"],
        "assumed_input_tokens_per_call": in_per_call,
        "assumed_output_tokens_per_call": out_per_call,
        "measured": measured,
        "projections": projections,
    }
    out_path = ROOT / "results" / "m0" / "generator_cost.json"
    out_path.write_text(json.dumps(report, indent=2) + "\n")
    keys = ["price_usd_per_million_tokens", "remaining_credit_usd", "assumed_output_tokens_per_call", "projections"]
    print(json.dumps({k: report[k] for k in keys}, indent=2))


if __name__ == "__main__":
    main(sys.argv[1])
