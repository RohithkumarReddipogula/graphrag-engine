"""M1 end-to-end on the DEV split: closed-book, the chosen hybrid baseline, and the MSc thesis setting.

Retrieval runs first (sequential), then generation in parallel threads through the cached LLM client.
Scoring: official 2Wiki v1.1 EM/F1 against the gold answer plus Wikidata aliases (data/answer_aliases.json).
Writes results/m1/generation_dev_<system>.jsonl (per question) and results/m1/generation_dev.json (summary).
"""

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor

from graphrag.config import get_settings
from graphrag.embed import E5Encoder
from graphrag.eval.answers import score
from graphrag.eval.bootstrap import mean_ci, paired_diff_ci
from graphrag.generation import build_prompt, pack_context, parse_answer
from graphrag.llm.client import make_llm
from graphrag.llm.spend import ledger_totals, write_balance
from graphrag.retrieval.hybrid import CrossEncoderReranker, HybridConfig, HybridRetriever
from graphrag.store.corpus_store import fetch_chunks, vector_search
from graphrag.store.neo4j_client import connect

logging.getLogger("neo4j").setLevel(logging.ERROR)
BUDGET_TOKENS = 1500
WORKERS = 8
GROUPS = ["comparison", "inference", "compositional", "bridge_comparison", "multi_hop", "single_hop", "all"]


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def load_systems() -> dict[str, HybridConfig | None]:
    s = get_settings()
    chosen = json.loads((s.results_dir / "m1" / "retrieval_dev.json").read_text())["chosen"]
    return {
        "closed_book": None,
        "hybrid": HybridConfig(with_title=chosen["with_title"], fusion=chosen["fusion"], alpha=chosen["alpha"]),
        "hybrid_thesis": HybridConfig(with_title=True, fusion="weighted", alpha=0.7),
    }


def in_group(row: dict, group: str) -> bool:
    if group == "all":
        return True
    if group == "multi_hop":
        return row["type"] != "single_hop"
    return row["type"] == group


def summarize(rows: list[dict]) -> dict:
    out = {}
    for g in GROUPS:
        sub = [r for r in rows if in_group(r, g)]
        if sub:
            out[g] = {"em": mean_ci([r["em"] for r in sub]), "f1": mean_ci([r["f1"] for r in sub])}
    return out


def main() -> None:
    s = get_settings()
    questions = read_jsonl(s.data_dir / "questions_dev.jsonl") + read_jsonl(s.data_dir / "single_hop_dev.jsonl")
    accepted = json.loads((s.data_dir / "answer_aliases.json").read_text(encoding="utf-8"))
    systems = load_systems()

    # 1. Retrieval and context packing (sequential: the encoder and reranker share the GPU).
    contexts: dict[str, list] = {}
    encoder = E5Encoder(s.embedding_model)
    reranker = CrossEncoderReranker(s.reranker_model)
    with connect(s) as driver:
        chunks = fetch_chunks(driver)
        dense = lambda q, k, with_title: vector_search(driver, encoder.encode_queries([q])[0].tolist(), k, with_title)
        for name, cfg in systems.items():
            if cfg is None:
                contexts[name] = [None] * len(questions)
                continue
            retriever = HybridRetriever(chunks, dense, reranker, cfg)
            contexts[name] = [pack_context(retriever.retrieve(q["question"], cfg.rerank_top), BUDGET_TOKENS) for q in questions]

    # 2. Generation (parallel, cached, provider-pinned).
    llm = make_llm(s.generator_model, s)
    spent_before = ledger_totals(s)["cost_usd"]
    t0 = time.time()
    per_system: dict[str, list[dict]] = {}
    for name in systems:
        def run(i: int) -> dict:
            q, ctx = questions[i], contexts[name][i]
            system_prompt, prompt = build_prompt(q["question"], ctx)
            out = llm.complete(prompt, system=system_prompt)
            pred = parse_answer(out.text)
            gold = set(q["gold_chunk_ids"])
            return {
                "id": q["id"], "type": q["type"], "question": q["question"], "answer": q["answer"],
                "prediction": pred, "raw": out.text,
                **score(pred, accepted[q["id"]]),
                "unknown": pred.strip().lower() == "unknown",
                "context_chunk_ids": ctx.chunk_ids if ctx else [],
                "context_tokens": ctx.tokens if ctx else 0,
                "all_gold_in_context": float(gold <= set(ctx.chunk_ids)) if ctx else None,
                "provider": out.provider, "input_tokens": out.input_tokens, "output_tokens": out.output_tokens,
                "cached": out.cached,
            }
        with ThreadPoolExecutor(WORKERS) as pool:
            rows = list(pool.map(run, range(len(questions))))
        per_system[name] = rows
        out_path = s.results_dir / "m1" / f"generation_dev_{name}.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{name}: {len(rows)} answers, {sum(not r['cached'] for r in rows)} live calls", flush=True)

    # 3. Summary.
    cb = {r["id"]: r for r in per_system["closed_book"]}
    cb_wrong = {i for i, r in cb.items() if r["em"] == 0.0}
    summary = {
        "split": "dev",
        "generator": {"model": s.generator_model, "endpoint": s.generator_provider,
                      "reasoning_effort": s.generator_reasoning_effort, "temperature": 0.0},
        "context_budget_tokens": BUDGET_TOKENS,
        "scoring": "official 2Wiki v1.1 EM/F1, max over gold answer + Wikidata aliases and demonyms",
        "ci": "95% percentile bootstrap, 10,000 resamples, seed 0",
        "systems": {},
        "closed_book_wrong_subset": {"definition": "dev questions with closed-book EM = 0", "n": len(cb_wrong)},
    }
    for name, rows in per_system.items():
        entry = {
            "all_questions": summarize(rows),
            "closed_book_wrong": summarize([r for r in rows if r["id"] in cb_wrong]),
            "unknown_rate": round(sum(r["unknown"] for r in rows) / len(rows), 4),
            "providers": sorted({r["provider"] for r in rows}),
        }
        if name != "closed_book":
            ctx_rows = [r for r in rows if r["type"] != "single_hop"]
            entry["multi_hop_all_gold_in_context"] = round(sum(r["all_gold_in_context"] for r in ctx_rows) / len(ctx_rows), 4)
            entry["mean_context_tokens"] = round(sum(r["context_tokens"] for r in rows) / len(rows), 1)
            for metric in ("em", "f1"):
                entry[f"paired_diff_vs_closed_book_{metric}"] = paired_diff_ci(
                    [r[metric] for r in rows], [cb[r["id"]][metric] for r in rows]
                )
        summary["systems"][name] = entry
    h, t = per_system["hybrid"], {r["id"]: r for r in per_system["hybrid_thesis"]}
    summary["paired_diff_hybrid_vs_thesis_em"] = paired_diff_ci([r["em"] for r in h], [t[r["id"]]["em"] for r in h])
    balance = write_balance(s)
    summary["spend"] = {
        "this_run_usd": round(ledger_totals(s)["cost_usd"] - spent_before, 6),
        "openrouter_remaining_usd": balance["remaining_usd"],
        "generation_seconds": round(time.time() - t0),
    }
    (s.results_dir / "m1" / "generation_dev.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary["spend"]))


if __name__ == "__main__":
    main()
