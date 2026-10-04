# Next steps

Last updated: 2026-10-04. Plan and settled decisions: `docs/PLAN.md`. Rules: `CLAUDE.md`.

## 1. Where we are

- M0 (environment) and M1 (data, hybrid baseline, closed-book baseline, dev results) are done and
  documented in `README.md`.
- M2 (extraction) pilot is done. Extractor: `openai/gpt-oss-120b` on the pinned OpenRouter endpoint
  `deepinfra/bf16`, strict JSON-schema output (reason for the switch from Gemini: `docs/limits.md`).
- Pilot results, all from `results/m2/pilot.json`:
  - sample: 6 batches (48 paragraphs), the batches with the most dev gold triples; 0 schema failures;
  - cost: USD 0.0085 for the sample, projected USD 0.36 for the full corpus of 2,049 paragraphs;
  - recall 17 of 22 dev gold triples, slot precision 17 of 17, the same at every name threshold
    from 80 to 100;
  - 311 relations extracted, 76 of them `OTHER`.
- OpenRouter credit left: USD 7.56 (`results/spend/openrouter_balance.json`).
- The full-corpus extraction has not been run.

## 2. Findings from the pilot that drive the next step

- 9 of the 76 `OTHER` triples are really fixed-list relations (6 occupation, 2 award received,
  1 founded by): the model wrote the relation name into the object or label instead of using it as
  the relation.
- Facts stated only through adjectives were missed: "a Syrian village" (country Syria),
  "Hungarian-born American" (country of citizenship).
- One miss is a scoring artefact, not an extraction error: "Akbank T.A.S." extracted for gold subject
  "Akbank" does not fuzzy-match. Scoring stays strict; entity resolution (M3) is where this belongs.
- 22 gold triples is too few to judge quality.

## 3. Next step (in this order)

1. Fix the extraction prompt (`src/graphrag/extraction/extract.py`, `SYSTEM`), tuned on dev only:
   - a fact that matches a listed relation must use that relation, never `OTHER`;
   - facts stated through adjectives count ("a Syrian village" gives country Syria,
     "Hungarian-born American" gives country of citizenship).
2. Run a larger pilot of about 15 batches with at least 9 new batches:
   - the 6 old batches (indices 8, 14, 45, 113, 146, 159, chosen by dev gold density) are re-extracted
     with the new prompt (new cache keys, so they are new calls);
   - the new batches are the next batches by dev gold density, excluding the old 6, chosen by the same
     rule in `scripts/run_m2_pilot.py`;
   - report old and new batches separately. The old batches were used to find the prompt problems, so
     only the new batches give an unbiased estimate. Keep the old prompt's results for comparison.
3. Create a 30-triple hand-check sheet (`results/m2/extraction_handcheck.md`), a seeded random sample
   of extracted triples from the larger pilot, each with its paragraph, to measure true precision
   (2Wiki gold triples cover only what the questions need). Same format as
   `data/single_hop_handcheck.md`; a verdict starting with "ok" counts as correct.
4. Only after the hand check: the full-corpus run (`scripts/run_m2_extraction.py`), then
   `results/m2/extraction_scores.json` on all dev paragraphs. That closes M2.

## 4. Quality bar for the full run (decided)

Quality bar, fixed on 2026-10-04 before seeing any new-batch numbers. Measured on the new (unseen)
pilot batches only, at the primary name threshold 90:
  - recall of at least 0.75 against dev gold triples;
  - slot precision of at least 0.90;
  - at least 90% of the 30 hand-checked triples correct (at least 27 of 30).
If all three pass, the full corpus is extracted. If not, one more prompt fix on dev, then decide.

## 5. Open decisions

- Gemini judge limits: `docs/limits.md` still has TBD for the gemini-3.8-flash free-tier limits. The
  judge runs once, at M5, on about 2,250 answers, so the limits matter then.
- Single-hop set v2: regenerated after the hand check and not checked a second time. Decide whether to
  hand-check it again before M5.
- Reasoning effort for extraction: medium (shared setting with the generator). Most output tokens are
  reasoning. Lowering it would cut cost but may lower recall; only worth testing if cost becomes a
  problem, and it would apply to extraction only.
