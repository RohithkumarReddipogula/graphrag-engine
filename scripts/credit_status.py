"""Print and record the OpenRouter credit balance (results/spend/openrouter_balance.json)."""

from graphrag.config import get_settings
from graphrag.llm.spend import write_balance

if __name__ == "__main__":
    b = write_balance(get_settings())
    print(f"OpenRouter credit: {b['remaining_usd']:.4f} USD left of {b['total_credits_usd']:.2f} "
          f"(project ledger: {b['project_ledger']['calls']} calls, {b['project_ledger']['cost_usd']:.4f} USD)")
