# Next steps

Last updated: 2026-10-04. Plan and settled decisions: `docs/PLAN.md`. Rules: `CLAUDE.md`.

## 1. Where we are

- M0, M1 and M2 are done.
- M2: the whole corpus is extracted with prompt v3 and post-processed (`results/m2/extractions.jsonl`,
  run summary in `results/m2/extraction_run.json`, strict dev scores in
  `results/m2/extraction_scores.json`). Hand check 28 of 30 correct.

## 2. Next step: M3 entity resolution

1. Same-name entities ("Albert II", the two "Adam's Rib" films) must not be merged because the name
   matches; use the source paragraph (`chunk_id`) and the description.
2. Plan in `docs/PLAN.md`, M3: normalise, candidate pairs by embedding of name + description within the
   same type, LLM check only for borderline pairs, merge only when sure; measure the duplicate rate on
   about 100 hand-labelled entities and the precision of about 50 merges.
3. Decide the judge-limit and single-hop re-check questions in section 6 before M5.

## 3. History of the M2 pilots

- v1: 6 batches; `OTHER` misused for listed relations, adjective facts missed (`results/m2/pilot_v1.json`).
- v2: 6 old + 9 new batches; misuse fixed; failed the bar on the new batches, mostly name differences
  (`results/m2/pilot_v2.json`).
- v3: 9 fresh batches + v2's new batches; see section 1 (`results/m2/pilot_v3.json`).

## 4. Quality bar for the full run (decided)

Quality bar, fixed on 2026-10-04 before seeing any new-batch numbers. Measured on the new (unseen)
pilot batches only, at the primary name threshold 90:
  - recall of at least 0.75 against dev gold triples;
  - slot precision of at least 0.90;
  - at least 90% of the 30 hand-checked triples correct (at least 27 of 30).
If all three pass, the full corpus is extracted. If not, one more prompt fix on dev, then decide.

## 5. Decision rule after pilot v2 (decided)

Decision rule after pilot v2, fixed on 2026-10-04 before seeing any v3 numbers:
  - Prompt v3 is the last prompt fix. It is judged on 9 fresh batches (never extracted before) with the
    same scorer and the same quality bar.
  - If v3 passes the bar, the full corpus is extracted.
  - If v3 fails, every miss is classified as before (name or granularity difference that is actually
    a correct fact; relation stated in reverse; fact attributed to another subject; real miss; wrong
    fact; fact not stated in the paragraph):
    - if the remaining failures are mostly name or granularity differences that are actually correct
      facts, the full corpus is extracted anyway, the strict score is reported honestly, and M3 entity
      resolution handles names;
    - if there are many real missing or wrong facts, stop and rethink the extractor.

## 6. Open decisions

- Gemini judge limits: `docs/limits.md` still has TBD for the gemini-3.8-flash free-tier limits. The
  judge runs once, at M5, on about 2,250 answers, so the limits matter then.
- Single-hop set v2: regenerated after the hand check and not checked a second time. Decide whether to
  hand-check it again before M5.
- Reasoning effort for extraction: medium (shared setting with the generator). Most output tokens are
  reasoning. Lowering it would cut cost but may lower recall; only worth testing if cost becomes a
  problem, and it would apply to extraction only.
