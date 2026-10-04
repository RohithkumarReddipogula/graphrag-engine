"""M2 pilot: extract a small sample, measure tokens and cost, project the full-corpus cost against the
remaining OpenRouter credit, and score the sample against dev gold triples.

Sample: the 6 batches (8 paragraphs each) of the fixed full-run batching that contain the most dev gold
triples, so scoring has material and the full run reuses these calls from the cache.
Writes results/m2/pilot.json and results/m2/pilot_extractions.jsonl.
"""

import json

from graphrag.config import get_settings
from graphrag.eval.extraction import gold_triples, score
from graphrag.extraction.extract import RunStats, extract_batch, make_batches
from graphrag.extraction.schema import RELATIONS
from graphrag.llm.client import make_llm
from graphrag.llm.spend import write_balance

N_BATCHES = 6
THRESHOLDS = [80, 85, 90, 95, 100]
PRIMARY_THRESHOLD = 90      # fixed before looking at results; the others are a sensitivity check


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def main() -> None:
    s = get_settings()
    chunks = read_jsonl(s.data_dir / "corpus.jsonl")
    corpus = {c["id"]: c for c in chunks}
    dev = read_jsonl(s.data_dir / "questions_dev.jsonl") + read_jsonl(s.data_dir / "single_hop_dev.jsonl")
    gold_all, unmapped = gold_triples(dev, corpus)

    batches = make_batches(chunks)
    gold_per_chunk = {}
    for g in gold_all:
        gold_per_chunk[g["chunk_id"]] = gold_per_chunk.get(g["chunk_id"], 0) + 1
    ranked = sorted(range(len(batches)), key=lambda i: (-sum(gold_per_chunk.get(c["id"], 0) for c in batches[i]), i))
    picked = sorted(ranked[:N_BATCHES])

    llm = make_llm(s.extractor_model, s)
    stats = RunStats()
    rows = []
    for i in picked:
        stats.batches += 1
        parsed = extract_batch(llm, batches[i], stats)
        if parsed:
            rows += [{"chunk_id": p.chunk_id, "batch": i, **p.model_dump(exclude={"chunk_id"})} for p in parsed.paragraphs]

    sample_ids = {c["id"] for i in picked for c in batches[i]}
    extracted = [{**t, "chunk_id": r["chunk_id"]} for r in rows for t in r["relations"] if t["relation"] != "OTHER"]
    gold = [g for g in gold_all if g["chunk_id"] in sample_ids]
    scores = {str(t): score(extracted, gold, t) for t in THRESHOLDS}

    n_par = len(sample_ids)
    balance = write_balance(s)
    per_par = stats.cost_usd / n_par if n_par else 0.0
    projected = per_par * len(chunks)
    report = {
        "extractor": {"model": s.extractor_model, "endpoint": s.generator_provider,
                      "reasoning_effort": s.generator_reasoning_effort, "structured_output": "strict json_schema"},
        "sample": {"batches": picked, "paragraphs": n_par, "dev_gold_triples_in_sample": len(gold),
                   "selection": f"the {N_BATCHES} batches with the most dev gold triples"},
        "runs": {"batches_ok": stats.ok, "ok_after_retry": stats.retried, "failed": stats.failed,
                 "live_calls": stats.live_calls},
        "tokens": {"input": stats.input_tokens, "output": stats.output_tokens,
                   "input_per_paragraph": round(stats.input_tokens / n_par, 1),
                   "output_per_paragraph": round(stats.output_tokens / n_par, 1)},
        "cost": {"sample_usd": round(stats.cost_usd, 6), "per_paragraph_usd": round(per_par, 8),
                 "projected_full_corpus_usd": round(projected, 4), "full_corpus_paragraphs": len(chunks),
                 "remaining_credit_usd": balance["remaining_usd"],
                 "fits_remaining_credit": projected <= balance["remaining_usd"]},
        "extracted": {"entities": sum(len(r["entities"]) for r in rows), "relations": sum(len(r["relations"]) for r in rows),
                      "other_relations": sum(t["relation"] == "OTHER" for r in rows for t in r["relations"]),
                      "relations_in_fixed_list": len(RELATIONS)},
        "gold_triples_without_identified_paragraph": unmapped,
        "primary_threshold": PRIMARY_THRESHOLD,
        "scores_by_threshold": scores,
    }
    out = s.results_dir / "m2"
    out.mkdir(parents=True, exist_ok=True)
    with (out / "pilot_extractions.jsonl").open("w", encoding="utf-8") as f:
        for r in sorted(rows, key=lambda r: r["chunk_id"]):
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (out / "pilot.json").write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps({k: report[k] for k in ("sample", "runs", "tokens", "cost", "extracted")}, indent=1))
    print("overall by threshold:", {t: v["overall"] for t, v in scores.items()})


if __name__ == "__main__":
    main()
