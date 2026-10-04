"""M2 pilot v3: extract fresh batches with the current prompt, score them, check the quality bar, and
write the hand-check sheet.

Batches come from the fixed full-run batching (so the full run reuses these calls from the cache):
- fresh: the next 9 batches by number of dev gold triples, excluding every batch extracted in pilots
  v1 and v2. The quality bar and the decision rule (docs/PLAN.md, M2) are judged on these only.
- v2_new: the 9 new batches of pilot v2, re-extracted with v3, for a like-for-like v2 vs v3 comparison.
Pilot v2's extractions on those batches are re-scored with the same scorer.
Writes results/m2/pilot_v3.json, results/m2/pilot_v3_extractions.jsonl and
results/m2/extraction_handcheck.md (30 seeded-random triples from the fresh batches; never overwritten).
Pilot v2 was run by this script at commit bc8eb06.
"""

import json
import random

from graphrag.config import get_settings
from graphrag.eval.extraction import gold_triples, score
from graphrag.extraction.extract import PROMPT_VERSION, extract_many, make_batches
from graphrag.extraction.schema import RELATIONS
from graphrag.llm.client import make_llm
from graphrag.llm.spend import write_balance

N_NEW = 9
WORKERS = 4
THRESHOLDS = [80, 85, 90, 95, 100]
PRIMARY_THRESHOLD = 90
BAR = {"recall": 0.75, "slot_precision": 0.90, "handcheck_correct": 27, "handcheck_n": 30}
SEED = 42


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def relation_triples(rows: list[dict]) -> list[dict]:
    return [{**t, "chunk_id": r["chunk_id"]} for r in rows for t in r["relations"] if t["relation"] != "OTHER"]


def other_misuse(rows: list[dict]) -> dict:
    other = [t for r in rows for t in r["relations"] if t["relation"] == "OTHER"]
    names = set(RELATIONS)
    misuse = [t for t in other if t["object"].lower() in names or (t.get("other_label") or "").lower() in names]
    return {"relations": sum(len(r["relations"]) for r in rows), "other": len(other), "other_using_a_list_relation_name": len(misuse)}


def group_report(rows: list[dict], gold: list[dict], chunk_ids: set[str]) -> dict:
    g = [x for x in gold if x["chunk_id"] in chunk_ids]
    return {
        "paragraphs": len(chunk_ids),
        "dev_gold_triples": len(g),
        "scores_by_threshold": {str(t): score(relation_triples(rows), g, t) for t in THRESHOLDS},
        "extracted": {"entities": sum(len(r["entities"]) for r in rows), **other_misuse(rows)},
    }


def write_handcheck(path, rows: list[dict], corpus: dict) -> list[dict]:
    triples = sorted(
        ({**t, "chunk_id": r["chunk_id"]} for r in rows for t in r["relations"]),
        key=lambda t: (t["chunk_id"], t["subject"], t["relation"], t["object"]),
    )
    random.Random(f"{SEED}:m2-handcheck").shuffle(triples)
    pick = triples[: BAR["handcheck_n"]]
    if path.exists():
        print(f"kept existing {path.name} (it may hold hand-check marks)")
        return pick
    lines = [
        "# Extraction hand check (pilot v3, fresh batches)",
        "",
        f"{len(pick)} seeded-random extracted triples from the fresh pilot v3 batches, all relation types",
        "including OTHER. For each: does the paragraph state this fact, with the right relation and",
        "direction (subject -> object)? For OTHER, does the label describe the fact? Mark ok or write the",
        f"problem. A verdict starting with \"ok\" counts as correct. Bar: at least {BAR['handcheck_correct']} of {len(pick)}.",
        "",
    ]
    for i, t in enumerate(pick, 1):
        rel = t["relation"] if t["relation"] != "OTHER" else f"OTHER ({t.get('other_label') or ''})"
        lines += [
            f"## {i}. ({t['subject']}) -[{rel}]-> ({t['object']})",
            "",
            f"- Paragraph `{t['chunk_id']}`: {corpus[t['chunk_id']]['text']}",
            "- Verdict: ",
            "",
        ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return pick


def main() -> None:
    s = get_settings()
    out = s.results_dir / "m2"
    chunks = read_jsonl(s.data_dir / "corpus.jsonl")
    corpus = {c["id"]: c for c in chunks}
    dev = read_jsonl(s.data_dir / "questions_dev.jsonl") + read_jsonl(s.data_dir / "single_hop_dev.jsonl")
    ev_aliases = json.loads((s.data_dir / "evidence_aliases.json").read_text(encoding="utf-8"))
    gold, unmapped = gold_triples(dev, corpus, ev_aliases)

    batches = make_batches(chunks)
    old = json.loads((out / "pilot_v1.json").read_text())["sample"]["batches"]
    v2_new = json.loads((out / "pilot_v2.json").read_text())["batches"]["new"]
    density = {}
    for g in gold:
        density[g["chunk_id"]] = density.get(g["chunk_id"], 0) + 1
    ranked = sorted(range(len(batches)), key=lambda i: (-sum(density.get(c["id"], 0) for c in batches[i]), i))
    seen = set(old) | set(v2_new)
    fresh = sorted([i for i in ranked if i not in seen][:N_NEW])

    llm = make_llm(s.extractor_model, s)
    parsed_by_batch, stats = extract_many(llm, {i: batches[i] for i in fresh + v2_new}, workers=WORKERS)
    rows_by_batch: dict[int, list[dict]] = {
        i: [{"chunk_id": p.chunk_id, "batch": i, **p.model_dump(exclude={"chunk_id"})} for p in parsed.paragraphs]
        if parsed else []
        for i, parsed in parsed_by_batch.items()
    }

    ids = lambda bs: {c["id"] for i in bs for c in batches[i]}
    rows = lambda bs: [r for i in bs for r in rows_by_batch[i]]
    v2_rows = [r for r in read_jsonl(out / "pilot_v2_extractions.jsonl") if r["batch"] in v2_new]

    fresh_report = group_report(rows(fresh), gold, ids(fresh))
    primary = fresh_report["scores_by_threshold"][str(PRIMARY_THRESHOLD)]["overall"]
    n_par = len(ids(fresh + v2_new))
    balance = write_balance(s)
    per_par = stats.cost_usd / n_par
    pick = write_handcheck(out / "extraction_handcheck.md", rows(fresh), corpus)

    report = {
        "prompt_version": PROMPT_VERSION,
        "extractor": {"model": s.extractor_model, "endpoint": s.generator_provider,
                      "reasoning_effort": s.generator_reasoning_effort, "structured_output": "strict json_schema"},
        "scoring": "recall and slot precision vs dev gold triples, names by rapidfuzz ratio, gold entities "
                   "also match their Wikidata aliases and demonyms (data/evidence_aliases.json), dates by parts",
        "batches": {"fresh": fresh, "v2_new": v2_new,
                    "fresh_selection": f"next {N_NEW} batches by dev gold triples, excluding all batches of pilots v1 and v2"},
        "runs": {"batches_ok": stats.ok, "ok_after_retry": stats.retried, "failed": stats.failed, "live_calls": stats.live_calls},
        "tokens": {"input": stats.input_tokens, "output": stats.output_tokens,
                   "input_per_paragraph": round(stats.input_tokens / n_par, 1),
                   "output_per_paragraph": round(stats.output_tokens / n_par, 1)},
        "cost": {"pilot_usd": round(stats.cost_usd, 6), "per_paragraph_usd": round(per_par, 8),
                 "projected_full_corpus_usd": round(per_par * len(chunks), 4),
                 "remaining_credit_usd": balance["remaining_usd"],
                 "fits_remaining_credit": per_par * len(chunks) <= balance["remaining_usd"]},
        "fresh_batches_v3": fresh_report,
        "v2_new_batches_v3": group_report(rows(v2_new), gold, ids(v2_new)),
        "v2_new_batches_v2": group_report(v2_rows, gold, ids(v2_new)),
        "gold_triples_without_identified_paragraph": unmapped,
        "primary_threshold": PRIMARY_THRESHOLD,
        "quality_bar": {
            **BAR,
            "judged_on": "fresh batches only",
            "recall": primary["recall"], "recall_pass": primary["recall"] is not None and primary["recall"] >= BAR["recall"],
            "slot_precision": primary["slot_precision"],
            "slot_precision_pass": primary["slot_precision"] is not None and primary["slot_precision"] >= BAR["slot_precision"],
            "handcheck": "pending: results/m2/extraction_handcheck.md",
        },
        "handcheck_sample": len(pick),
    }
    with (out / "pilot_v3_extractions.jsonl").open("w", encoding="utf-8") as f:
        for r in sorted(rows(fresh + v2_new), key=lambda r: r["chunk_id"]):
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (out / "pilot_v3.json").write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n")
    t = str(PRIMARY_THRESHOLD)
    for name in ("fresh_batches_v3", "v2_new_batches_v3", "v2_new_batches_v2"):
        g = report[name]
        print(f"{name:24} gold={g['dev_gold_triples']:3} overall={g['scores_by_threshold'][t]['overall']} extracted={g['extracted']}")
    print(json.dumps({k: report[k] for k in ("runs", "tokens", "cost", "quality_bar")}, indent=1))


if __name__ == "__main__":
    main()
