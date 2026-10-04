"""M1 data step: pinned download, stratified dev/test sample, pooled corpus, single-hop questions.

Writes (all committed): data/questions_dev.jsonl, data/questions_test.jsonl, data/single_hop_dev.jsonl,
data/single_hop_test.jsonl, data/corpus.jsonl, data/collisions.jsonl, data/stats.json,
data/single_hop_handcheck.md. The raw parquet stays in data/raw/ (gitignored).
"""

import json
import random
import re
import statistics
from collections import Counter
from pathlib import Path

import pandas as pd

from graphrag.config import get_settings
from graphrag.data import build
from graphrag.data.download import REVISION, SHA256, fetch_validation

SEED = 42
N_DEV, N_TEST = 25, 75              # per multi-hop type
N_DISTRACTORS = 4
N_SH_DEV, N_SH_TEST = 25, 75
N_HANDCHECK = 30


def hand_check_summary(sheet: Path) -> dict | None:
    """Error rate from the verdicts in the hand-check sheet. A verdict starting with "ok" is correct."""
    if not sheet.exists():
        return None
    verdicts = re.findall(r"(?m)^- Verdict: *(.*)$", sheet.read_text(encoding="utf-8"))
    marked = [v.strip() for v in verdicts if v.strip()]
    if not marked:
        return {"sheet": f"data/{sheet.name}", "status": "not checked yet"}
    errors = [i + 1 for i, v in enumerate(verdicts) if v.strip() and not v.strip().lower().startswith("ok")]
    return {
        "sheet": f"data/{sheet.name}",
        "checked_set": "v1 single-hop sample (before the hand-check rules)",
        "checked": len(marked),
        "errors": len(errors),
        "error_rate": round(len(errors) / len(marked), 4),
        "error_items": errors,
    }


def write_jsonl(path: Path, rows) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    data_dir = get_settings().data_dir
    raw = fetch_validation(data_dir / "raw")
    questions = [build.from_row(r) for r in pd.read_parquet(raw).itertuples(index=False)]

    dev_q, test_q = build.stratified_split(questions, N_DEV, N_TEST, SEED)
    corpus = build.build_corpus(dev_q + test_q, N_DISTRACTORS, SEED)
    dev = [build.question_record(q, corpus) for q in dev_q]
    test = [build.question_record(q, corpus) for q in test_q]
    sh_dev, sh_test, sh_pools = build.sample_single_hop(dev_q, test_q, corpus, N_SH_DEV, N_SH_TEST, SEED)

    write_jsonl(data_dir / "questions_dev.jsonl", dev)
    write_jsonl(data_dir / "questions_test.jsonl", test)
    write_jsonl(data_dir / "single_hop_dev.jsonl", sh_dev)
    write_jsonl(data_dir / "single_hop_test.jsonl", sh_test)
    write_jsonl(data_dir / "corpus.jsonl", sorted(corpus.chunks.values(), key=lambda c: c["id"]))
    write_jsonl(data_dir / "collisions.jsonl", corpus.collisions)

    words = [len(c["text"].split()) for c in corpus.chunks.values()]
    stats = {
        "source": {"repo": "framolfese/2WikiMultihopQA", "revision": REVISION, "file_sha256": SHA256},
        "params": {"seed": SEED, "dev_per_type": N_DEV, "test_per_type": N_TEST, "distractors_per_question": N_DISTRACTORS},
        "validation_questions": len(questions),
        "multi_hop": {
            "dev": dict(Counter(r["type"] for r in dev)),
            "test": dict(Counter(r["type"] for r in test)),
        },
        "single_hop": {
            "rules": build.SINGLE_HOP_RULES,
            "excluded_after_hand_check": [
                {"subject": k[0], "relation": k[1], "reason": v} for k, v in build.EXCLUDED_SINGLE_HOP.items()
            ],
            "hand_check": hand_check_summary(data_dir / "single_hop_handcheck.md"),
            "candidate_pools": sh_pools,
            "dev": len(sh_dev),
            "test": len(sh_test),
            "dev_relations": build.relation_counts(sh_dev),
            "test_relations": build.relation_counts(sh_test),
        },
        "corpus": {
            "chunks": len(corpus.chunks),
            "words_total": sum(words),
            "words_median": statistics.median(words),
            "words_max": max(words),
            "merged_spacing_variants": sum(1 for c in corpus.collisions if c["kind"] == "merged_spacing_variants"),
            "titles_kept_apart": sum(1 for c in corpus.collisions if c["kind"] == "kept_apart"),
        },
    }
    (data_dir / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n")

    # Hand-check sheet: a seeded sample across both splits, with the paragraph to judge against.
    pick = (sh_dev + sh_test)[:]
    random.Random(f"{SEED}:handcheck").shuffle(pick)
    lines = [
        "# Single-hop hand check",
        "",
        f"{N_HANDCHECK} seeded-random single-hop questions. For each: is the question unambiguous, and is the",
        "answer correct and fully supported by the paragraph? Mark ok or write the problem.",
        "",
    ]
    for i, c in enumerate(pick[:N_HANDCHECK], 1):
        text = corpus.chunks[c["gold_chunk_ids"][0]]["text"]
        lines += [
            f"## {i}. {c['question']}",
            "",
            f"- Answer: {c['answer']}",
            f"- Relation: {c['relation']}",
            f"- Paragraph `{c['gold_chunk_ids'][0]}`: {text}",
            "- Verdict: ",
            "",
        ]
    sheet = data_dir / "single_hop_handcheck.md"
    if sheet.exists():
        print(f"kept existing {sheet.name} (it may hold hand-check marks)")
    else:
        sheet.write_text("\n".join(lines))
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
