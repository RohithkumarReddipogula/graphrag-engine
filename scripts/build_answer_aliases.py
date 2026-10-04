"""Accepted answers per question, as in the official 2Wiki evaluation script v1.1: the gold answer plus
all aliases and demonyms of its Wikidata entity. Single-hop questions use the entity id of the evidence
object. Writes data/answer_aliases.json ({question id: [accepted answers]}) and data/evidence_aliases.json
(alias sets for the subject and object of each evidence triple, where the official ids line up)."""

import json

from graphrag.config import get_settings
from graphrag.data.download import fetch_ids


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def main() -> None:
    data = get_settings().data_dir
    ids_dir = fetch_ids(data / "raw")
    official = {d["_id"]: d for d in json.loads((ids_dir / "dev.json").read_text(encoding="utf-8"))}
    aliases = {}
    with (ids_dir / "id_aliases.json").open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            aliases[row["Q_id"]] = set(row["aliases"] + row["demonyms"])

    out, stats = {}, {"multi_hop": 0, "single_hop": 0, "with_aliases": 0, "single_hop_without_ids": 0}
    for name in ("questions_dev.jsonl", "questions_test.jsonl"):
        for q in read_jsonl(data / name):
            d = official[q["id"]]
            assert d["answer"] == q["answer"], q["id"]
            golds = {q["answer"]} | aliases.get(d["answer_id"], set())
            out[q["id"]] = sorted(golds)
            stats["multi_hop"] += 1
            stats["with_aliases"] += len(golds) > 1
    for name in ("single_hop_dev.jsonl", "single_hop_test.jsonl"):
        for q in read_jsonl(data / name):
            src = official[q["source_question_id"]]
            s, rel, o = q["evidences"][0]
            idx = [i for i, e in enumerate(src["evidences"]) if e == [s, rel, o]]
            assert idx, q["id"]
            # evidences_id is not always parallel to evidences; only trust it when the lengths match.
            if len(src["evidences_id"]) == len(src["evidences"]):
                golds = {q["answer"], o} | aliases.get(src["evidences_id"][idx[0]][2], set())
            else:
                golds = {q["answer"], o}
                stats["single_hop_without_ids"] += 1
            out[q["id"]] = sorted(golds)
            stats["single_hop"] += 1
            stats["with_aliases"] += len(golds) > 1
    (data / "answer_aliases.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")

    # Evidence aliases, as the official v1.1 script uses for evidence scoring: for each multi-hop
    # question whose evidences_id lines up with its evidences, the alias sets of every subject and object.
    ev_out, ev_stats = {}, {"questions": 0, "aligned": 0}
    for name in ("questions_dev.jsonl", "questions_test.jsonl"):
        for q in read_jsonl(data / name):
            d = official[q["id"]]
            ev_stats["questions"] += 1
            if len(d["evidences_id"]) != len(d["evidences"]):
                continue
            ev_stats["aligned"] += 1
            ev_out[q["id"]] = [
                {"evidence": list(e),
                 "subject_aliases": sorted({e[0]} | aliases.get(ids[0], set())),
                 "object_aliases": sorted({e[2]} | aliases.get(ids[2], set()))}
                for e, ids in zip(d["evidences"], d["evidences_id"])
            ]
    (data / "evidence_aliases.json").write_text(json.dumps(ev_out, indent=1, ensure_ascii=False) + "\n")
    print(stats, ev_stats)


if __name__ == "__main__":
    main()
