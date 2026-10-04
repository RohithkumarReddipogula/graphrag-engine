"""M2: extract entities and relations for the whole corpus with prompt v3 (cached, resumable, parallel).

Every extracted relation goes through the deterministic post-processing in
graphrag.extraction.postprocess. Batches already extracted in the pilots come from the cache.
Writes results/m2/extractions.jsonl (one line per chunk, post-processed), results/m2/extraction_failures.jsonl
and results/m2/extraction_run.json. If the provider stays unavailable the run stops; re-running resumes.
"""

import argparse
import json
import time

from graphrag.config import get_settings
from graphrag.extraction.extract import PROMPT_VERSION, QuotaExhausted, extract_many, make_batches
from graphrag.extraction.postprocess import clean_rows
from graphrag.llm.client import make_llm
from graphrag.llm.spend import write_balance


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    s = get_settings()
    chunks = [json.loads(line) for line in (s.data_dir / "corpus.jsonl").read_text(encoding="utf-8").splitlines()]
    batches = make_batches(chunks)
    llm = make_llm(s.extractor_model, s)
    t0 = time.time()

    def progress(done: int, total: int, st) -> None:
        if done % 10 == 0 or done == total:
            print(f"[{time.time() - t0:6.0f}s] {done}/{total} batches  live calls {st.live_calls}  "
                  f"retries {st.retried}  failed {len(st.failed)}  cost so far {st.cost_usd:.4f} USD", flush=True)

    stopped = None
    try:
        parsed, stats = extract_many(llm, dict(enumerate(batches)), workers=args.workers, on_done=progress)
    except QuotaExhausted as exc:
        stopped = str(exc)[:500]
        print("STOPPED:", stopped, flush=True)
        raise SystemExit(1)

    rows = [
        {"chunk_id": p.chunk_id, "batch": i, **p.model_dump(exclude={"chunk_id"})}
        for i, res in sorted(parsed.items()) if res for p in res.paragraphs
    ]
    rows, post_counts = clean_rows(rows)
    rows.sort(key=lambda r: r["chunk_id"])

    out = s.results_dir / "m2"
    out.mkdir(parents=True, exist_ok=True)
    with (out / "extractions.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with (out / "extraction_failures.jsonl").open("w", encoding="utf-8") as f:
        for r in stats.failed:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    balance = write_balance(s)
    run = {
        "prompt_version": PROMPT_VERSION,
        "model": s.extractor_model,
        "endpoint": s.generator_provider,
        "batch_size": len(batches[0]),
        "workers": args.workers,
        "batches_total": len(batches),
        "batches_ok": stats.ok,
        "batches_ok_after_retry": stats.retried,
        "batches_failed": len(stats.failed),
        "chunks_extracted": len(rows),
        "chunks_total": len(chunks),
        "entities": sum(len(r["entities"]) for r in rows),
        "relations": sum(len(r["relations"]) for r in rows),
        "postprocessing": post_counts,
        "live_calls_this_run": stats.live_calls,
        "tokens": {"input": stats.input_tokens, "output": stats.output_tokens},
        "cost_usd_all_batches": round(stats.cost_usd, 6),
        "remaining_credit_usd": balance["remaining_usd"],
        "complete": len(rows) == len(chunks),
        "seconds": round(time.time() - t0),
    }
    (out / "extraction_run.json").write_text(json.dumps(run, indent=1) + "\n")
    print(json.dumps(run, indent=1))


if __name__ == "__main__":
    main()
