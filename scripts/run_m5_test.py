"""M5: the single run on the test split, generation only (docs/PLAN.md, M5, approved 2026-10-09).

Before any test question is read, the script checks:
- nothing outside results/ differs from the commit tagged `m5-frozen`, and HEAD is that commit;
- the Neo4j graph matches results/m4/graph_stats.json;
- the run-once lock (results/m5/TEST_RUN.lock) can be acquired: refused if the run already finished;
  an interrupted run resumes only with the same commit and settings hash (all LLM calls are cached).

Systems: closed_book, hybrid, graph_only, graph_plus_chunks_g0.5, graph_plus_chunks_g0.5_no_exact_title,
with the frozen settings listed in docs/PLAN.md (M5.2) and recorded in results/m5/frozen_settings.json.
The judge is a separate step (M5.5) and does not run here.

Writes results/m5/generation_test_<system>.jsonl, results/m5/frozen_settings.json and
results/m5/test_summary.json.
"""

import hashlib
import json
import logging
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from graphrag import generation
from graphrag.config import ROOT, get_settings
from graphrag.embed import E5Encoder
from graphrag.eval.answers import score
from graphrag.eval.bootstrap import N_RESAMPLES, paired_diff_ci
from graphrag.eval.report import build_summary
from graphrag.eval.run_lock import acquire, finish
from graphrag.generation import PackedContext, build_prompt, pack_context, parse_answer
from graphrag.graph import retrieve as graph_retrieve
from graphrag.graph.retrieve import GraphIndex, graph_context, score_paths
from graphrag.graph.store import fetch_graph
from graphrag.llm.client import make_llm
from graphrag.llm.spend import ledger_totals, write_balance
from graphrag.resolution.embed_mentions import embed_mentions
from graphrag.resolution.mentions import build_mentions
from graphrag.retrieval.hybrid import CrossEncoderReranker, HybridConfig, HybridRetriever
from graphrag.store.corpus_store import fetch_chunks, vector_search
from graphrag.store.neo4j_client import connect
from graphrag.systems import bridge_chunks, combine

logging.getLogger("neo4j").setLevel(logging.ERROR)
TAG = "m5-frozen"
BUDGET = 1500
G = 0.5
WORKERS = 8
PRIMARY = ("graph_plus_chunks_g0.5", "hybrid")
SECONDARY_PAIRS = [("graph_only", "hybrid"), ("graph_plus_chunks_g0.5", "graph_only"),
                   ("graph_plus_chunks_g0.5_no_exact_title", "graph_plus_chunks_g0.5")]
FROZEN_FILES = ["data/corpus.jsonl", "data/questions_test.jsonl", "data/single_hop_test.jsonl",
                "data/answer_aliases.json", "results/m1/retrieval_dev.json", "results/m2/extractions.jsonl",
                "results/m3/clusters.jsonl", "results/m4/graph_stats.json"]


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def git(*args) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def sha256(path) -> str:
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def frozen_settings(s) -> dict:
    chosen = json.loads((ROOT / "results/m1/retrieval_dev.json").read_text())["chosen"]
    return {
        "hybrid": {**chosen, "bm25_k1": 1.5, "bm25_b": 0.75, "candidates": HybridConfig.candidates,
                   "rerank_top": HybridConfig.rerank_top, "embedding_model": s.embedding_model,
                   "reranker_model": s.reranker_model, "dense": "exact cosine in Neo4j"},
        "generation": {"model": s.generator_model, "endpoint": s.generator_provider, "temperature": 0.0,
                       "reasoning_effort": s.generator_reasoning_effort, "budget_tokens": BUDGET,
                       "tokenizer": "o200k_base", "system_prompt": generation.SYSTEM,
                       "system_prompt_closed_book": generation.SYSTEM_CLOSED_BOOK},
        "graph": {"g": G, **{k: getattr(graph_retrieve, k) for k in (
            "MAX_SEEDS", "SEED_MIN_COS", "MIN_ALIAS_CHARS", "MAX_NGRAM", "MAX_NEIGHBOURS", "HUB_DEGREE", "TOP_PATHS")}},
        "scoring": {"answers": "official 2Wiki v1.1 EM/F1 with Wikidata aliases",
                    "bootstrap": {"level": 0.95, "resamples": N_RESAMPLES, "seed": 0}},
        "files_sha256": {f: sha256(f) for f in FROZEN_FILES},
    }


def check_preconditions() -> str:
    # Code, data and docs must be exactly the frozen commit; run outputs under results/ may change
    # (the spend ledger, and the outputs of an interrupted run that is being resumed).
    if git("status", "--porcelain", "--", ".", ":(exclude)results"):
        raise SystemExit("refusing: uncommitted changes outside results/")
    head, frozen = git("rev-parse", "HEAD"), git("rev-parse", f"{TAG}^{{commit}}")
    if head != frozen:
        raise SystemExit(f"refusing: HEAD {head[:7]} is not the {TAG} commit {frozen[:7]}")
    return head


def main() -> None:
    commit = check_preconditions()
    s = get_settings()
    out = s.results_dir / "m5"
    out.mkdir(parents=True, exist_ok=True)
    settings = frozen_settings(s)
    settings_hash = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()

    expected = json.loads((ROOT / "results/m4/graph_stats.json").read_text())["neo4j_counts"]
    with connect(s) as driver:
        counts = driver.execute_query(
            "MATCH (e:Entity) WITH count(e) AS entities OPTIONAL MATCH ()-[r:REL]->() WITH entities, count(r) AS rels "
            "OPTIONAL MATCH ()-[x:EXACT_TITLE]->() WITH entities, rels, count(x) AS exact "
            "OPTIONAL MATCH ()-[m:MENTIONED_IN]->() RETURN entities, rels, exact, count(m) AS mentioned_in").records[0]
    if dict(counts) != expected:
        raise SystemExit(f"refusing: Neo4j graph {dict(counts)} differs from results/m4/graph_stats.json {expected}")

    lock = acquire(out / "TEST_RUN.lock", commit, settings_hash)
    (out / "frozen_settings.json").write_text(json.dumps({"commit": commit, "tag": TAG, "settings_hash": settings_hash,
                                                          **settings}, indent=1, ensure_ascii=False) + "\n")
    print(f"lock {lock['status']} (started {lock['started_at']}), commit {commit[:7]}, settings {settings_hash[:12]}", flush=True)

    corpus = {c["id"]: c for c in read_jsonl(s.data_dir / "corpus.jsonl")}
    questions = read_jsonl(s.data_dir / "questions_test.jsonl") + read_jsonl(s.data_dir / "single_hop_test.jsonl")
    dev_ids = {q["id"] for f in ("questions_dev.jsonl", "single_hop_dev.jsonl") for q in read_jsonl(s.data_dir / f)}
    assert not dev_ids & {q["id"] for q in questions} and len(questions) == 375
    accepted = json.loads((s.data_dir / "answer_aliases.json").read_text(encoding="utf-8"))

    mentions = build_mentions(read_jsonl(s.results_dir / "m2" / "extractions.jsonl"), corpus)
    m_emb = embed_mentions(mentions, s.embedding_model, s.cache_dir)
    m_idx = {m.id: i for i, m in enumerate(mentions)}
    by_mention = {m.id: m for m in mentions}
    encoder = E5Encoder(s.embedding_model)
    reranker = CrossEncoderReranker(s.reranker_model)
    hc = settings["hybrid"]

    t0 = time.time()
    with connect(s) as driver:
        g = fetch_graph(driver)

        def rep(e):
            ids = sorted(e.mention_ids)
            pages = [i for i in ids if by_mention[i].is_page]
            return m_emb[m_idx[(pages or ids)[0]]]

        entity_emb = {eid: rep(e) for eid, e in g.entities.items()}
        indexes = {True: GraphIndex(g, entity_emb, use_exact_title=True),
                   False: GraphIndex(g, entity_emb, use_exact_title=False)}
        chunks = fetch_chunks(driver)
        dense = lambda q, k, wt: vector_search(driver, encoder.encode_queries([q])[0].tolist(), k, wt)
        hybrid = HybridRetriever(chunks, dense, reranker,
                                 HybridConfig(with_title=hc["with_title"], fusion=hc["fusion"], alpha=hc["alpha"]))
        path_cache: dict[str, np.ndarray] = {}
        retrieved = []
        for q in questions:
            q_emb = encoder.encode_queries([q["question"]])[0]
            per_index = {}
            for exact, idx in indexes.items():
                seeds = idx.seeds(q["question"], q_emb)
                paths = idx.paths([e for e, _ in seeds])
                new = [p.text for p in paths if p.text not in path_cache]
                if new:
                    for text, e in zip(new, encoder.encode_passages(new)):
                        path_cache[text] = e
                top = score_paths(paths, q_emb, np.stack([path_cache[p.text] for p in paths]), idx) if paths else []
                per_index[exact] = {"seeds": seeds, "top": top}
            retrieved.append({"graph": per_index, "hybrid": hybrid.retrieve(q["question"], hybrid.config.rerank_top)})
    retrieval_seconds = round(time.time() - t0)

    def contexts(system: str) -> list[PackedContext | None]:
        out_ctx = []
        for r in retrieved:
            if system == "closed_book":
                out_ctx.append(None)
            elif system == "hybrid":
                out_ctx.append(pack_context(r["hybrid"], BUDGET))
            elif system == "graph_only":
                out_ctx.append(graph_context(r["graph"][True]["top"], indexes[True], corpus, BUDGET))
            else:
                exact = not system.endswith("_no_exact_title")
                gpart = graph_context(r["graph"][exact]["top"], indexes[exact], corpus, int(G * BUDGET))
                out_ctx.append(combine(gpart, r["hybrid"], BUDGET))
        return out_ctx

    llm = make_llm(s.generator_model, s)
    spent_before = ledger_totals(s)["cost_usd"]

    def run_system(name: str) -> list[dict]:
        ctxs = contexts(name)

        def one(i):
            q, ctx = questions[i], ctxs[i]
            system_prompt, prompt = build_prompt(q["question"], ctx)
            res = llm.complete(prompt, system=system_prompt)
            pred = parse_answer(res.text)
            gold, bridge = set(q["gold_chunk_ids"]), bridge_chunks(q, corpus)
            ids = ctx.chunk_ids if ctx else []
            return {"id": q["id"], "type": q["type"], "question": q["question"], "answer": q["answer"],
                    "prediction": pred, "raw": res.text, **score(pred, accepted[q["id"]]),
                    "unknown": pred.strip().lower() == "unknown", "context_chunk_ids": ids,
                    "context_tokens": ctx.tokens if ctx else 0,
                    "all_gold_in_context": float(gold <= set(ids)) if ctx else None,
                    "bridge_chunks": sorted(bridge),
                    "bridge_in_context": (float(bridge <= set(ids)) if (bridge and ctx) else None),
                    "provider": res.provider, "input_tokens": res.input_tokens, "output_tokens": res.output_tokens,
                    "cached": res.cached}

        with ThreadPoolExecutor(WORKERS) as pool:
            rows = list(pool.map(one, range(len(questions))))
        write_jsonl(out / f"generation_test_{name}.jsonl", rows)
        print(f"{name}: {len(rows)} answers, {sum(not r['cached'] for r in rows)} live calls", flush=True)
        return rows

    names = ["closed_book", "hybrid", "graph_only", "graph_plus_chunks_g0.5", "graph_plus_chunks_g0.5_no_exact_title"]
    per_system = {n: run_system(n) for n in names}

    header = {"split": "test", "commit": commit, "settings_hash": settings_hash,
              "generator": settings["generation"]["model"], "endpoint": settings["generation"]["endpoint"],
              "context_budget_tokens": BUDGET, "scoring": settings["scoring"]["answers"]}
    summary = build_summary(per_system, header)
    by_id = {n: {r["id"]: r for r in rows} for n, rows in per_system.items()}

    def paired(a, b, ids=None):
        rows = [r for r in per_system[a] if ids is None or r["id"] in ids]
        return paired_diff_ci([r["em"] for r in rows], [by_id[b][r["id"]]["em"] for r in rows])

    primary = paired(*PRIMARY)
    cb_wrong = {r["id"] for r in per_system["closed_book"] if r["em"] == 0.0}
    summary["primary_result"] = {
        "comparison": f"{PRIMARY[0]} vs {PRIMARY[1]}", "questions": len(questions),
        "paired_em_difference": primary,
        "pre_registered_expectation": "graph_plus_chunks_g0.5 beats hybrid on EM, 95% CI of the paired difference above 0",
        "expectation_met": primary["ci_low"] > 0,
    }
    summary["secondary_results"] = {
        "closed_book_wrong_subset": {"n": len(cb_wrong), "paired_em_difference": paired(*PRIMARY, cb_wrong),
                                     "note": "pre-specified secondary result, not a headline"},
        "exploratory": {f"{a} vs {b}": paired(a, b) for a, b in SECONDARY_PAIRS},
    }

    def frac(rows, key):
        vals = [r[key] for r in rows if r["type"] != "single_hop" and r.get(key) is not None]
        return round(sum(vals) / len(vals), 4) if vals else None

    summary["retrieval"] = {n: {"multi_hop_all_gold_in_context": frac(rows, "all_gold_in_context"),
                                "bridge_entity_recall": frac(rows, "bridge_in_context")}
                            for n, rows in per_system.items() if n != "closed_book"}
    balance = write_balance(s)
    summary["spend"] = {"this_run_usd": round(ledger_totals(s)["cost_usd"] - spent_before, 6),
                        "openrouter_remaining_usd": balance["remaining_usd"], "retrieval_seconds": retrieval_seconds}
    (out / "test_summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False) + "\n")
    finish(out / "TEST_RUN.lock", {"answers": sum(len(r) for r in per_system.values()),
                                   "summary_sha256": hashlib.sha256((out / "test_summary.json").read_bytes()).hexdigest()})
    print(json.dumps({"primary_result": summary["primary_result"], "secondary_results": summary["secondary_results"],
                      "spend": summary["spend"]}, indent=1))


if __name__ == "__main__":
    main()
