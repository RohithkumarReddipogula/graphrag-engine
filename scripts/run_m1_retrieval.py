"""M1 retrieval sweep on the DEV split only: recall@k of the baseline, before and after the reranker.

Grid: with_title in {true, false} x fusion in {weighted alpha 0.0..1.0 step 0.1, rrf}.
Selection rule (fixed before the run): highest post-rerank all_gold@10 on multi-hop dev, ties broken by
post-rerank recall@5 on multi-hop dev, then by the order of the grid.
Writes results/m1/retrieval_dev.json and results/m1/retrieval_dev_questions.jsonl (chosen config).
"""

import json
import logging
import time

from graphrag.config import get_settings
from graphrag.embed import E5Encoder
from graphrag.eval.retrieval import aggregate, per_question
from graphrag.retrieval.hybrid import BM25Index, CrossEncoderReranker, fuse_rrf, fuse_weighted
from graphrag.retrieval.base import passage_text
from graphrag.store.corpus_store import fetch_chunks, vector_search
from graphrag.store.neo4j_client import connect

logging.getLogger("neo4j").setLevel(logging.ERROR)
CANDIDATES, RERANK_TOP = 50, 30
ALPHAS = [round(a * 0.1, 1) for a in range(11)]


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def main() -> None:
    s = get_settings()
    questions = read_jsonl(s.data_dir / "questions_dev.jsonl") + read_jsonl(s.data_dir / "single_hop_dev.jsonl")
    encoder = E5Encoder(s.embedding_model)
    reranker = CrossEncoderReranker(s.reranker_model)
    qvecs = encoder.encode_queries([q["question"] for q in questions])

    rerank_cache: dict[tuple[str, str], float] = {}
    configs = []
    t0 = time.time()
    with connect(s) as driver:
        chunks = fetch_chunks(driver)
        by_id = {c["id"]: c for c in chunks}
        for with_title in (True, False):
            bm25 = BM25Index(chunks, with_title)
            cands = [
                (bm25.search(q["question"], CANDIDATES), vector_search(driver, v.tolist(), CANDIDATES, with_title))
                for q, v in zip(questions, qvecs)
            ]
            fusions = [("weighted", a) for a in ALPHAS] + [("rrf", None)]
            for fusion, alpha in fusions:
                pre_rows, post_rows, per_q = [], [], []
                for q, (b, d) in zip(questions, cands):
                    fused = fuse_weighted(b, d, alpha) if fusion == "weighted" else fuse_rrf(b, d)
                    fused_ids = [cid for cid, _ in fused]
                    top = fused_ids[:RERANK_TOP]
                    missing = [cid for cid in top if (q["id"], cid) not in rerank_cache]
                    if missing:
                        texts = [passage_text(by_id[c]["title"], by_id[c]["text"], True) for c in missing]
                        for cid, sc in zip(missing, reranker(q["question"], texts)):
                            rerank_cache[(q["id"], cid)] = sc
                    reranked = sorted(top, key=lambda cid: (-rerank_cache[(q["id"], cid)], cid))
                    pre_rows.append({"type": q["type"], **per_question(fused_ids, q["gold_chunk_ids"])})
                    post_rows.append({"type": q["type"], **per_question(reranked, q["gold_chunk_ids"])})
                    per_q.append({
                        "id": q["id"], "type": q["type"], "gold": q["gold_chunk_ids"],
                        "gold_rank_pre": [fused_ids.index(g) + 1 if g in fused_ids else None for g in q["gold_chunk_ids"]],
                        "gold_rank_post": [reranked.index(g) + 1 if g in reranked else None for g in q["gold_chunk_ids"]],
                    })
                configs.append({
                    "with_title": with_title, "fusion": fusion, "alpha": alpha,
                    "pre_rerank": aggregate(pre_rows), "post_rerank": aggregate(post_rows), "_per_q": per_q,
                })
                print(f"title={with_title!s:5} {fusion:8} alpha={alpha!s:4} "
                      f"post all_gold@10 multi={configs[-1]['post_rerank']['multi_hop']['all_gold@10']:.2f} "
                      f"pre={configs[-1]['pre_rerank']['multi_hop']['all_gold@10']:.2f}", flush=True)

    def key(c):
        m = c["post_rerank"]["multi_hop"]
        return (m["all_gold@10"], m["recall@5"])
    best_i = max(range(len(configs)), key=lambda i: (key(configs[i]), -i))
    best = configs[best_i]

    out = s.results_dir / "m1"
    out.mkdir(parents=True, exist_ok=True)
    with (out / "retrieval_dev_questions.jsonl").open("w") as f:
        for r in best["_per_q"]:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    for c in configs:
        c.pop("_per_q")
    report = {
        "split": "dev",
        "questions": {"multi_hop": sum(q["type"] != "single_hop" for q in questions), "single_hop": sum(q["type"] == "single_hop" for q in questions)},
        "settings": {"candidates_per_retriever": CANDIDATES, "rerank_top": RERANK_TOP, "embedding": s.embedding_model,
                     "reranker": s.reranker_model, "bm25": {"k1": 1.5, "b": 0.75}, "dense": "exact cosine in Neo4j"},
        "selection_rule": "max post-rerank all_gold@10 on multi-hop dev, then recall@5, then grid order",
        "chosen": {k: best[k] for k in ("with_title", "fusion", "alpha")},
        # Reference row: the MSc thesis setting, reported next to the chosen config.
        "thesis_setting": next(
            {k: c[k] for k in ("with_title", "fusion", "alpha", "pre_rerank", "post_rerank")}
            for c in configs if c["with_title"] and c["fusion"] == "weighted" and c["alpha"] == 0.7
        ),
        "spread_note": "all configs, post-rerank multi-hop all_gold@10: min {lo}, max {hi} (n = 100; one question = 0.01)",
        "configs": configs,
        "runtime_s": round(time.time() - t0),
    }
    vals = [c["post_rerank"]["multi_hop"]["all_gold@10"] for c in configs]
    report["spread_note"] = report["spread_note"].format(lo=min(vals), hi=max(vals))
    (out / "retrieval_dev.json").write_text(json.dumps(report, indent=1) + "\n")
    print("chosen:", report["chosen"])
    print(json.dumps({"pre_rerank": best["pre_rerank"], "post_rerank": best["post_rerank"]}, indent=1))


if __name__ == "__main__":
    main()
