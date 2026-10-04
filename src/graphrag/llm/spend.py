"""OpenRouter credit tracking. Writes results/spend/openrouter_balance.json so the remaining credit is
always visible in the repo, next to the per-call ledger in results/spend/openrouter_calls.jsonl."""

import json
from datetime import datetime, timezone
from typing import Any

import httpx

from graphrag.config import Settings
from graphrag.llm.client import OPENROUTER_URL


def ledger_totals(settings: Settings) -> dict[str, Any]:
    ledger = settings.results_dir / "spend" / "openrouter_calls.jsonl"
    rows = [json.loads(line) for line in ledger.read_text().splitlines()] if ledger.exists() else []
    return {"calls": len(rows), "cost_usd": round(sum(r["cost_usd"] for r in rows), 6)}


def pinned_endpoint_price(settings: Settings) -> dict[str, float]:
    """USD per million tokens for the pinned endpoint, read live from OpenRouter's public endpoint list."""
    resp = httpx.get(f"{OPENROUTER_URL}/models/{settings.generator_model}/endpoints", timeout=30)
    resp.raise_for_status()
    for e in resp.json()["data"]["endpoints"]:
        if e.get("tag") == settings.generator_provider:
            p = e["pricing"]
            return {"input": round(float(p["prompt"]) * 1e6, 6), "output": round(float(p["completion"]) * 1e6, 6)}
    raise RuntimeError(f"pinned endpoint {settings.generator_provider!r} not listed for {settings.generator_model}")


def write_balance(settings: Settings) -> dict[str, Any]:
    if settings.openrouter_api_key is None:
        raise RuntimeError("OPENROUTER_API_KEY is not set in .env")
    headers = {"Authorization": f"Bearer {settings.openrouter_api_key.get_secret_value()}"}
    resp = httpx.get(f"{OPENROUTER_URL}/credits", headers=headers, timeout=30)
    resp.raise_for_status()
    data = resp.json()["data"]
    total, used = float(data["total_credits"]), float(data["total_usage"])
    balance = {
        "checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "total_credits_usd": round(total, 6),
        "total_usage_usd": round(used, 6),
        "remaining_usd": round(total - used, 6),
        "project_ledger": ledger_totals(settings),
    }
    out = settings.results_dir / "spend" / "openrouter_balance.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(balance, indent=2) + "\n")
    return balance
