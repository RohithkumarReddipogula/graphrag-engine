"""Deterministic clean-up applied to every extracted relation (not a prompt change; docs/PLAN.md, M2).

Rule 1 (other_to_listed): relation OTHER whose other_label is a listed relation name becomes that
        relation, e.g. OTHER (occupation) -> occupation.
Rule 2 (strip_label_prefix): an object that starts with "<label>:" loses that prefix, where <label> is
        the triple's own other_label or relation name, e.g. "cast member: Richard T. Jones" ->
        "Richard T. Jones". Only the triple's own label is stripped, so names that contain a colon
        ("Star Wars: A New Hope") are left alone.
Each changed triple records the rules applied in "postprocessed".
"""

import re
from collections import Counter

from graphrag.extraction.schema import RELATIONS

RULES = ("other_to_listed", "strip_label_prefix")


def clean_relation(t: dict) -> tuple[dict, list[str]]:
    t = dict(t)
    applied = []
    label = (t.get("other_label") or "").strip()
    labels = {label.lower(), t["relation"].lower()} - {""}

    if t["relation"] == "OTHER" and label.lower() in RELATIONS:
        t["relation"] = label.lower()
        t["other_label"] = None
        applied.append("other_to_listed")

    for lab in sorted(labels, key=len, reverse=True):
        m = re.match(rf"\s*{re.escape(lab)}\s*:\s*(.+)$", t["object"], flags=re.IGNORECASE | re.S)
        if m and m.group(1).strip():
            t["object"] = m.group(1).strip()
            applied.append("strip_label_prefix")
            break

    if applied:
        t["postprocessed"] = applied
    return t, applied


def clean_rows(rows: list[dict]) -> tuple[list[dict], dict[str, int]]:
    """Apply the rules to every relation of every extracted paragraph; count changes per rule."""
    counts: Counter[str] = Counter({r: 0 for r in RULES})
    out = []
    for r in rows:
        rels = []
        for t in r["relations"]:
            cleaned, applied = clean_relation(t)
            counts.update(applied)
            rels.append(cleaned)
        out.append({**r, "relations": rels})
    counts["relations_total"] = sum(len(r["relations"]) for r in rows)
    counts["relations_changed"] = sum(1 for r in out for t in r["relations"] if t.get("postprocessed"))
    return out, dict(counts)
