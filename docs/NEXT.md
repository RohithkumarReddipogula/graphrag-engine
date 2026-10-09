# Next steps

Last updated: 2026-10-09. Plan and settled decisions: `docs/PLAN.md`. Rules: `CLAUDE.md`.

## 1. Where we are

- M0, M1 and M2 are done.
- M2: the whole corpus is extracted with prompt v3 and post-processed (`results/m2/extractions.jsonl`,
  run summary in `results/m2/extraction_run.json`, strict dev scores in
  `results/m2/extraction_scores.json`). Hand check 28 of 30 correct.
- M3 is done and passed its quality bar (`results/m3/resolution_report.md`, scores in
  `results/m3/handcheck_scores.json`): 49 of 50 hand-checked merges correct (bar 48); duplicate rate on
  60 hand-labelled mentions 0.40 before, 0.10 after (bar: at most half). Bridge link recall on the report
  split: 0.83 before (exact names), 0.80 after M3. Hand checks first drafted by Claude (a different model
  from the gpt-oss-120b judge), reviewed and corrected by Rohith Kumar Reddipogula.
- M4 is done (`results/m4/graph_report.md`, numbers from `results/m4/generation_dev.json`): graph built
  and loaded into Neo4j (`results/m4/graph_stats.json`); `graph_plus_chunks` at g = 0.5 passed all three
  criteria of the M4 quality bar on dev and goes to M5 as the GraphRAG system.
- M5 is done. Test split run once at the `m5-frozen` tag (`results/m5/TEST_RUN.lock`: finished). Primary
  result (`results/m5/test_summary.json`): graph_plus_chunks_g0.5 vs hybrid, paired EM difference +0.221
  [+0.173, +0.269] over 375 test questions; the pre-registered expectation (CI above 0) is met. Judge
  (llama-3.3-70b on parasail/fp8) hand check: 29 of 30 agree (`results/m5/judge_handcheck_scores.json`).
  Full report: `BENCHMARK.md`, generated from the results files.
- Next: M6 packaging (README with the dev and test results, read-only API, simple graph view), see
  `docs/PLAN.md`. Nothing is running now.

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

## 3. Approved next steps for M3 (all done on 2026-10-08)

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

## 3a. M3 re-run results and open points

Numbers from `results/m3/resolution_run.json` and `results/m3/bridge_link_changes.json`:
- 17,309 pairs judged (8,671 same, 8,398 different, 240 unsure); 22 batches failed twice (unsure); 0
  error batches; live cost 0.43 USD (the first 600 batches came from the cache of the stopped attempt).
- 3,120 merges applied; 1,673 refused by cannot-link, 16 for joining two pages. 10,830 clusters, 1,613
  with 2 or more mentions; 742 non-page mentions linked to a page. Run 1's false merges are gone
  (United States and United Kingdom are separate clusters).
- Bridge link recall on the report split: 0.83 before (exact name matching, 57 of 69), 0.80 after M3
  (55 of 69). M3 lost 5 exact-name links and gained 3 that exact matching cannot find (name variants
  such as "Kaneto Shindō" / "Kaneto Shindo"). Of the 5 lost, 4 are the old role confusion, now rarer
  ("Abel Ferry" mentioned in a film's paragraph vs his page: "one is a person, the other a film"), and
  1 was judged same but blocked by cannot-link.
- The first re-run attempt stalled after 250 batches on an uncaught error in one batch; fixed (any 5xx,
  408 and 429 are retried; a batch error marks only that batch unsure and is logged in
  `results/m3/judge_errors.jsonl`) and restarted from the cache.

Closed on 2026-10-09:
1. Hand checks done; M3 passed (see section 1 and `results/m3/resolution_report.md`).
2. Bridge links for M4, decided after seeing the report-split numbers: keep the M3 clusters as they are
   and, in M4, add exact-title links (mention name equals exactly one page title) as a separate,
   labelled edge type. No M3 threshold, prompt or merge rule changes.

## 3b. Next step: M4 graph build and graph retrieval

See `docs/PLAN.md`, M4. Notes already recorded there:
- exact-title links as a separate, labelled edge type next to the M3 clusters;
- hub nodes (United States, France, Italy, frequent concepts such as "suicide") must be capped or
  down-ranked in graph retrieval so they do not flood the context;
- the deterministic post-processing of M2 relations is already applied in `results/m2/extractions.jsonl`.
Before writing M4 code: propose the graph schema, the retrieval method and the M4 dev evaluation, and get
them approved, as for M2 and M3.

## 3c. After M4: points to decide before M5

From `results/m4/graph_report.md` (dev only, optimistic because g was chosen on dev):
- Quality bar for `graph_plus_chunks_g0.5`: multi-hop all-gold-in-context +0.15 over hybrid (bar +0.10);
  paired EM difference over all 125 dev questions 0.18 [0.10, 0.26] (bar: CI low at least -0.05); EM on
  bridge_comparison + compositional 0.60 vs hybrid 0.12 (bar: above hybrid). All pass.
- `graph_only` scored higher than `graph_plus_chunks_g0.5` on dev (multi-hop EM 0.71 vs 0.67,
  multi-hop all-gold-in-context 0.88 vs 0.68, bridge entity recall 0.91 vs 0.58), but lower on
  comparison questions (0.84 vs 0.96). The approved plan did not make `graph_only` eligible as the
  GraphRAG system; changing that now would be a choice made after seeing dev results. M5 reports every
  system anyway, `graph_only` included.
- The `EXACT_TITLE` ablation changed the context of only a few questions and made no meaningful
  difference on dev.
- `graph_plus_chunks` at g = 0.25 had lower all-gold-in-context than hybrid (0.50 vs 0.53): a small graph
  share pushes out baseline chunks without adding enough graph paragraphs.

Open before M5 (together with section 7):
1. Confirm the M5 setup: test split once, every system (closed_book, hybrid, graph_only,
   graph_plus_chunks_g0.5, the no-exact-title ablation), frozen config, judge `gemini-3.8-flash` on test
   answers, 30-item hand check, `BENCHMARK.md`.
2. The Gemini judge limits (section 7) must be known before the judge runs.

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
