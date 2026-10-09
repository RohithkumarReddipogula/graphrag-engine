"""Build the portfolio's knowledge-graph data from docs/graph_view/subgraph.json (M6.3 follow-up).

- Answer status per question, honest and recorded:
  "correct" (exact match against the gold answer and its aliases), "no answer" (the system said
  "unknown"), and for the other answers the M5 judge (llama-3.3-70b on parasail/fp8, same prompt) decides
  between "correct, different wording" and "wrong". Verdicts: results/m6/graph_view_judge.json.
- Same-name entities kept apart: only when both are page entities (two different Wikipedia pages are two
  different entities). Same-name pairs involving a mention cluster can be missed merges, so they are not
  presented as entity resolution working.
- Layout precomputed (no physics in the browser): a seeded Fruchterman-Reingold layout per connected
  component, components shelf-packed into one canvas.
- Opening question, by rule: a correctly answered bridge_comparison question whose two top-ranked facts are
  both film -director-> person chains with the director's life date and whose path has no two nodes with
  the same name; among those, the fewest OTHER facts (ties: question id).
Writes docs/graph_view/portfolio_graph.json (compact: integer node ids). No paragraph text.
"""

import hashlib
import json
import math
import re
from collections import defaultdict

import numpy as np

from graphrag.config import ROOT, get_settings
from graphrag.eval.judge import judge_batch
from graphrag.llm.client import make_llm

SRC = ROOT / "docs" / "graph_view" / "subgraph.json"
OUT = ROOT / "docs" / "graph_view" / "portfolio_graph.json"
CANVAS_WIDTH = 1600
PAD = 70


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", s.lower().replace("’", "'")).strip()


def fruchterman_reingold(n: int, edges: list[tuple[int, int]], seed: int, iters: int = 400) -> np.ndarray:
    if n == 1:
        return np.zeros((1, 2))
    rng = np.random.default_rng(seed)
    pos = rng.uniform(-1, 1, (n, 2))
    k = 1.0 / math.sqrt(n)
    t = 0.2
    for _ in range(iters):
        delta = pos[:, None, :] - pos[None, :, :]
        dist = np.linalg.norm(delta, axis=2) + 1e-9
        disp = ((k * k / dist ** 2)[:, :, None] * delta).sum(axis=1)          # repulsion
        for a, b in edges:                                                    # attraction
            d = pos[a] - pos[b]
            dl = np.linalg.norm(d) + 1e-9
            f = d / dl * (dl * dl / k)
            disp[a] -= f
            disp[b] += f
        length = np.linalg.norm(disp, axis=1) + 1e-9
        pos += disp / length[:, None] * np.minimum(length, t)[:, None]
        t *= 0.985
    return pos - pos.mean(axis=0)


def layout(nodes: list[dict], links: list[dict]) -> dict[str, tuple[int, int]]:
    ids = [n["id"] for n in nodes]
    adj = defaultdict(set)
    for l in links:
        adj[l["source"]].add(l["target"])
        adj[l["target"]].add(l["source"])
    seen, comps = set(), []
    for n in sorted(ids):
        if n in seen:
            continue
        stack, comp = [n], set()
        while stack:
            x = stack.pop()
            if x not in comp:
                comp.add(x)
                stack += adj[x]
        seen |= comp
        comps.append(sorted(comp))
    comps.sort(key=lambda c: (-len(c), c[0]))

    placed, x, y, row_h = {}, PAD, PAD, 0
    for comp in comps:
        idx = {n: i for i, n in enumerate(comp)}
        edges = sorted({tuple(sorted((idx[l["source"]], idx[l["target"]]))) for l in links
                        if l["source"] in idx and l["target"] in idx})
        seed = int(hashlib.sha1(comp[0].encode()).hexdigest()[:8], 16)
        pos = fruchterman_reingold(len(comp), edges, seed)
        span = np.ptp(pos, axis=0) if len(comp) > 1 else np.array([0.0, 0.0])
        target = 90 * math.sqrt(len(comp))                    # larger components get more room
        scale = target / max(span.max(), 1e-9) if len(comp) > 1 else 0
        pos = (pos - pos.min(axis=0)) * scale
        w, h = (pos.max(axis=0) if len(comp) > 1 else np.array([0.0, 0.0]))
        if x + w > CANVAS_WIDTH - PAD:
            x, y, row_h = PAD, y + row_h + 2 * PAD, 0
        for n, (px, py) in zip(comp, pos):
            placed[n] = (int(round(x + px)), int(round(y + py)))
        x += w + 2 * PAD
        row_h = max(row_h, h)
    return placed


def main() -> None:
    s = get_settings()
    d = json.loads(SRC.read_text(encoding="utf-8"))
    accepted = json.loads((s.data_dir / "answer_aliases.json").read_text(encoding="utf-8"))

    # Judge the answers that are neither exact matches nor "unknown" (same judge and prompt as M5).
    to_judge = [{"item_id": f"dev:{q['id']}", "question_id": q["id"], "type": q["type"], "question": q["question"],
                 "gold": q["gold"], "accepted": accepted[q["id"]], "prediction": q["answer"]}
                for q in d["questions"] if not q["exact_match"] and q["answer"].strip().lower() != "unknown"]
    verdicts, calls = {}, []
    if to_judge:
        dec, calls, _ = judge_batch(make_llm(s.judge_model, s), to_judge)
        verdicts = {i["question_id"]: dec[i["item_id"]] for i in to_judge}
    out_dir = s.results_dir / "m6"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "graph_view_judge.json").write_text(json.dumps({
        "judge": {"model": s.judge_model, "endpoint": s.judge_provider, "prompt": "graphrag.eval.judge.SYSTEM (as M5)"},
        "items": [{**i, "verdict": verdicts[i["question_id"]].verdict, "reason": verdicts[i["question_id"]].reason}
                  for i in to_judge],
        "cost_usd": round(sum(c.cost_usd for c in calls if not c.cached), 6)}, indent=1, ensure_ascii=False) + "\n")

    pos = layout(d["nodes"], d["links"])
    node_ids = [n["id"] for n in d["nodes"]]
    ix = {n: i for i, n in enumerate(node_ids)}
    pages_by_name = defaultdict(list)
    for n in d["nodes"]:
        if n["is_page"]:
            pages_by_name[norm(n["label"])].append(n["id"])

    questions = []
    for q in d["questions"]:
        if q["exact_match"]:
            status = "correct"
        elif q["answer"].strip().lower() == "unknown":
            status = "no answer"
        else:
            status = "correct, different wording" if verdicts[q["id"]].verdict == "correct" else "wrong"
        twins = []
        for nid in q["path_node_ids"]:
            node = next(n for n in d["nodes"] if n["id"] == nid)
            others = [o for o in pages_by_name.get(norm(node["label"]), []) if o != nid] if node["is_page"] else []
            for o in others:
                pair = sorted([ix[nid], ix[o]])
                if pair not in [t["nodes"] for t in twins]:
                    twins.append({"name": node["label"], "nodes": pair})
        questions.append({
            "type": q["type"], "question": q["question"], "gold": q["gold"], "answer": q["answer"], "status": status,
            "judge_reason": verdicts[q["id"]].reason if q["id"] in verdicts else None,
            "facts": q["facts"], "nodes": [ix[n] for n in q["path_node_ids"]],
            "links": [i for i, l in enumerate(d["links"]) if l["id"] in set(q["path_link_ids"])],
            "same_name_kept_apart": twins,
        })
    def clear_path(q):
        """Both top-ranked facts are film -director-> person chains with the director's life date, and no two
        nodes on the path share a name (so no duplicate cluster is on show)."""
        top2 = q["facts"][:2]
        names = [norm(data_nodes[n]["label"]) for n in q["nodes"]]
        return (len(top2) == 2 and all("-director->" in f and ("date of birth" in f or "date of death" in f) for f in top2)
                and len(names) == len(set(names)))

    data_nodes = d["nodes"]
    candidates = [i for i, q in enumerate(questions)
                  if q["type"] == "bridge_comparison" and q["status"] == "correct" and clear_path(q)]
    default = min(candidates, key=lambda i: (sum("-OTHER" in f for f in questions[i]["facts"]), d["questions"][i]["id"]))

    data = {
        "source": d["source"], "licence": d["licence"],
        "note": "Dev questions of 2WikiMultihopQA; graph paths the GraphRAG system used as facts; answers from the "
                "committed dev run. Status: exact match, or the M5 judge for other wordings.",
        "default_question": default,
        "nodes": [{"label": n["label"], "type": n["type"], "page": n["is_page"], "x": pos[n["id"]][0], "y": pos[n["id"]][1]}
                  for n in d["nodes"]],
        "links": [{"s": ix[l["source"]], "t": ix[l["target"]], "label": l["label"]} for l in d["links"]],
        "questions": questions,
    }
    OUT.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KB); default question {default}: {questions[default]['question']}")
    print({q["status"] for q in questions}, [(q["status"], q["judge_reason"]) for q in questions if q["judge_reason"]])
    print("same-name notes:", [(i, t["name"]) for i, q in enumerate(questions) for t in q["same_name_kept_apart"]])


if __name__ == "__main__":
    main()
