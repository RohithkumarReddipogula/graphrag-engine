"""M2 closing score: post-processed full-corpus extraction against all dev gold triples (strict scorer,
reported as it is). Also records the hand-check result from the marked sheet and links the manual
classification of the v3 pilot failures. Writes results/m2/extraction_scores.json."""

import json
import re

from graphrag.config import get_settings
from graphrag.eval.extraction import gold_triples, score

THRESHOLDS = [80, 85, 90, 95, 100]
PRIMARY_THRESHOLD = 90


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def hand_check(path) -> dict:
    verdicts = [v.strip() for v in re.findall(r"(?m)^- Verdict: *(.*)$", path.read_text(encoding="utf-8"))]
    correct = sum(v.lower().startswith("ok") for v in verdicts if v)
    return {"sheet": "results/m2/extraction_handcheck.md", "checked": sum(bool(v) for v in verdicts),
            "correct": correct, "bar": 27, "pass": correct >= 27}


def main() -> None:
    s = get_settings()
    out = s.results_dir / "m2"
    corpus = {c["id"]: c for c in read_jsonl(s.data_dir / "corpus.jsonl")}
    dev = read_jsonl(s.data_dir / "questions_dev.jsonl") + read_jsonl(s.data_dir / "single_hop_dev.jsonl")
    ev_aliases = json.loads((s.data_dir / "evidence_aliases.json").read_text(encoding="utf-8"))
    gold, unmapped = gold_triples(dev, corpus, ev_aliases)
    rows = read_jsonl(out / "extractions.jsonl")
    extracted_ids = {r["chunk_id"] for r in rows}
    triples = [{**t, "chunk_id": r["chunk_id"]} for r in rows for t in r["relations"] if t["relation"] != "OTHER"]
    gold_covered = [g for g in gold if g["chunk_id"] in extracted_ids]
    run = json.loads((out / "extraction_run.json").read_text())

    report = {
        "split": "dev",
        "extraction": "results/m2/extractions.jsonl (prompt v3, post-processed)",
        "scoring": "strict: recall and slot precision vs dev gold triples, names by rapidfuzz ratio, gold "
                   "entities also match Wikidata aliases and demonyms, dates by parts",
        "dev_gold_triples": len(gold),
        "dev_gold_triples_in_extracted_chunks": len(gold_covered),
        "dev_gold_triples_without_identified_paragraph": unmapped,
        "primary_threshold": PRIMARY_THRESHOLD,
        "scores_by_threshold": {str(t): score(triples, gold_covered, t) for t in THRESHOLDS},
        "hand_check": hand_check(out / "extraction_handcheck.md"),
        "postprocessing": run["postprocessing"],
        "notes": [
            "The strict score counts correct facts under other names, second valid objects and facts the "
            "paragraph does not state as errors. On the v3 pilot's fresh batches, all 10 failures were "
            "classified by hand: results/m2/pilot_v3_classification.json.",
            "Dev only. Test-split gold triples are not used in M2.",
        ],
    }
    (out / "extraction_scores.json").write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n")
    p = report["scores_by_threshold"][str(PRIMARY_THRESHOLD)]["overall"]
    print(json.dumps({"gold": len(gold_covered), "recall": p["recall"], "slot_precision": p["slot_precision"],
                      "hand_check": report["hand_check"]}, indent=1))


if __name__ == "__main__":
    main()
