# Next steps

Last updated: 2026-10-08. Plan and settled decisions: `docs/PLAN.md`. Rules: `CLAUDE.md`.

## 1. Where we are

- M0, M1 and M2 are done.
- M2: the whole corpus is extracted with prompt v3 and post-processed (`results/m2/extractions.jsonl`,
  run summary in `results/m2/extraction_run.json`, strict dev scores in
  `results/m2/extraction_scores.json`). Hand check 28 of 30 correct.
- M3: plan and quality bar approved (`docs/PLAN.md`, M3); auto-merge is off (every candidate pair is
  judged by the LLM). **M3 run 1 failed** (section 2). Steps 1 to 4 of section 3 are done (judge prompt
  v2, cannot-link, fixed merge order, two-direction pilot); the pilot passed both gates
  (`results/m3/pilot_v2.json`). Step 5, the full re-run, waits for approval. Nothing is running now.

## 2. M3 run 1: what went wrong and why

Run 1 (2026-10-08) judged all 17,309 candidate pairs with the LLM, for 0.59 USD. All outputs are kept in
`results/m3/run1_flawed/` (numbers below from `results/m3/run1_flawed/resolution_run.json`, explanation
in `results/m3/run1_flawed/README.md`). Its hand-check sheets were deleted before anyone labelled them.

- Verdicts: 3,628 same, 13,558 different, 123 unsure; 12 batches failed twice (counted as unsure).
- Clusters: 11,392 in total, 1,312 with 2 or more mentions; 2,558 merges applied, 34 rejected for joining
  two pages; 730 non-page mentions linked to a page.
- Bridge link recall on the report split fell from 0.83 before M3 (exact name matching, 57 of 69) to 0.51
  after M3 (35 of 69); on the tune split from 0.88 to 0.52.
- Largest clusters included false merges: United States with United Kingdom (117 mentions), Cannes,
  Berlin and Moscow film festivals (51), Mexico with Mexico City (18).

Two causes:

1. Judge prompt confused the mention with its paragraph's subject. Each mention was shown with the
   paragraph it came from, and the LLM judged the paragraph's subject instead of the mentioned entity.
   Examples from `results/m3/run1_flawed/decisions.jsonl`: "Jeff Bezos" mentioned in MacKenzie Scott's
   paragraph vs the Jeff Bezos page, reason "husband and wife, separate persons"; "Albert Capellani"
   mentioned in a film's paragraph vs his page, "one is a person, the other a film". All 24 lost bridge
   links were judged "different" (23) or "unsure" (1). The United States / United Kingdom merge came
   from the same confusion ("identical description refers to same producer").
2. No cannot-link check. Union-find only blocked merges between two page entities. 2,439 pairs judged
   "different" still ended up in one cluster through chains of "same" verdicts.

The pilot did not catch this: its "different" cases were genuinely different entities, and it did not
test mention-vs-page pairs that should be "same".

## 3. Approved next steps for M3 (steps 1 to 4 done on 2026-10-08; step 5 waits for approval)

1. Judge prompt v2 with explicit roles. Each side states its role: "A is the subject of its paragraph"
   or "A is mentioned in a paragraph about <title>". The prompt says to judge only the named entity:
   relations in the paragraph (father, spouse, director) describe the paragraph's subject, not
   necessarily the mention. Verdicts stay same / different / unsure; unsure means don't merge.
2. Cannot-link rule. Union-find refuses a merge when any pair across the two clusters was judged
   "different", in addition to the existing rule that two page entities never merge.
3. Merges applied in a fixed, deterministic order, most confident first. The judge gives no confidence
   score, so "most confident" is the pair's E5 cosine, highest first, ties broken by pair ids.
4. Pilot that tests both directions, before the full re-run, on the tune split and run 1's known errors:
   - the 24 tune-split bridge links (mention -> its correct page, from dev gold triples only) must be
     judged "same": gate at least 22 of 24;
   - known different pairs from run 1's bad clusters must be judged "different": United States vs
     United Kingdom, Cannes vs Berlin vs Moscow film festivals, Mexico vs Mexico City.
   Show the pilot results (both directions, cost) before the full re-run. Do not start the full re-run
   without approval.
5. Full re-run of all candidate pairs (expected about 0.59 USD and about 2 hours, judged from run 1;
   run 1's cached answers cannot be reused because the prompt changes). Then cluster stats, bridge link
   recall before and after on the report split, and the two hand-check sheets (50 cluster pairs, 60
   mentions with up to 5 candidates), as in `docs/PLAN.md`, M3.

Unchanged from the approved plan: `t_low` = 0.85 from dev gold, auto-merge off, the 30/70 tune/report
split, never-merge rule for page entities, quality bar (at least 48 of 50 merges correct; duplicate rate
after at most half of before), test questions never used.

Before M5, also decide the open questions in section 7.

## 4. History of the M2 pilots

- v1: 6 batches; `OTHER` misused for listed relations, adjective facts missed (`results/m2/pilot_v1.json`).
- v2: 6 old + 9 new batches; misuse fixed; failed the bar on the new batches, mostly name differences
  (`results/m2/pilot_v2.json`).
- v3: 9 fresh batches + v2's new batches; see section 1 (`results/m2/pilot_v3.json`).

## 5. M2 quality bar (decided)

Quality bar, fixed on 2026-10-04 before seeing any new-batch numbers. Measured on the new (unseen)
pilot batches only, at the primary name threshold 90:
  - recall of at least 0.75 against dev gold triples;
  - slot precision of at least 0.90;
  - at least 90% of the 30 hand-checked triples correct (at least 27 of 30).
If all three pass, the full corpus is extracted. If not, one more prompt fix on dev, then decide.

## 6. M2 decision rule after pilot v2 (decided)

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

## 7. Open decisions

- Gemini judge limits: `docs/limits.md` still has TBD for the gemini-3.8-flash free-tier limits. The
  judge runs once, at M5, on about 2,250 answers, so the limits matter then.
- Single-hop set v2: regenerated after the hand check and not checked a second time. Decide whether to
  hand-check it again before M5.
- Reasoning effort for extraction: medium (shared setting with the generator). Most output tokens are
  reasoning. Lowering it would cut cost but may lower recall; only worth testing if cost becomes a
  problem, and it would apply to extraction only.
