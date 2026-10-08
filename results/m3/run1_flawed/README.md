# M3 run 1 (flawed, kept for the record)

Run on 2026-10-08: every candidate pair (17,309) judged by the LLM, cost 0.59 USD
(`resolution_run.json`). Not used further, and its hand-check sheets were deleted before anyone
labelled them, because two defects make the clusters invalid:

1. Judge prompt bug. Each mention was shown with the paragraph it came from, and the LLM judged the
   paragraph's subject instead of the mentioned entity. Examples from `decisions.jsonl`: "Jeff Bezos"
   mentioned in MacKenzie Scott's paragraph vs the Jeff Bezos page: "husband and wife, separate
   persons"; "Albert Capellani" mentioned in a film's paragraph vs his page: "one is a person, the
   other a film". Effect: bridge link recall on the report split fell from 0.83 (exact name matching)
   to 0.51; all 24 lost links were judged "different" (23) or "unsure" (1).
2. No cannot-link check. Union-find blocked only merges between two pages. 2,439 pairs judged
   "different" still ended up in one cluster through chains of "same" verdicts; the largest cluster
   joined United States and United Kingdom (117 mentions).
