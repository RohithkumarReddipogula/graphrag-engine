# GraphRAG Engine: Build Plan (v2)

Replaces the original 21-day build doc (a local PDF, not kept in the repo). The scope below came out of two rounds of
design review. **Core scope is fixed. Dates are not.** Work is planned by milestones, and every milestone
after M1 is reported as a gain (or loss) relative to the M1 baseline.

## 1. Goal

Measure, honestly, when a knowledge graph helps retrieval-augmented QA on multi-hop questions, and when it
does not. The deliverable is a system I understand end to end plus a benchmark anyone can reproduce.

Not a goal: a pre-decided headline number. Results are published whatever they show, including where
the baseline wins.

## 2. Decisions (settled)

| Area | Decision |
|---|---|
| Dataset | 2WikiMultiHopQA, **validation split only** (the public test split has no answers). Our dev and test splits are both carved from it with a fixed seed. |
| Sample | 100 dev + 300 test multi-hop questions, stratified (25 / 75 per type × 4 types: compositional, comparison, bridge_comparison, inference). Plus 100 single-hop questions generated from evidence triples (25 dev / 75 test). |
| Corpus | Gold paragraphs of every sampled question + 4 random distractors per question, pooled into one corpus. 1 paragraph = 1 chunk. Chunk id = paragraph title. Texts of one title that differ only in spacing around punctuation are merged into one chunk; genuinely different texts of one title are all kept as `title#<short hash>`. Every gold paragraph of every sampled question must be in the corpus. |
| Store | Neo4j is the only persistent store (local, Docker via OrbStack): graph + vector index. No Chroma, no FAISS. The BM25 index is built in memory from the Neo4j chunks at startup (about 2k paragraphs, under a second), because the Neo4j full-text index does not expose `k1`/`b` and uses a different tokenizer from the thesis. |
| Baseline | Same retriever as my MSc thesis (github.com/RohithkumarReddipogula/AI-Powered-Rag-System): BM25 (`rank_bm25`, k1 1.5, b 0.75, lowercase + punctuation stripped) + E5 dense, min-max normalised, weighted fusion `alpha * dense + (1 - alpha) * bm25` with alpha 0.70 as the start value, tuned on dev. RRF is kept as an alternative and compared on dev. Then the `bge-reranker-base` cross-encoder. Lives behind its own `Retriever` interface and never imports graph code. |
| Generation | Short answer only (entity, date, or yes/no). Same generator model and same context token budget for every system. |
| Primary metric | EM and F1 using the official 2Wiki answer normalisation. Paired bootstrap 95% CIs. Broken down by question type. |
| Headline subset | Questions the closed-book model gets wrong. All-question numbers are reported too. |
| Secondary metrics | Own LLM judge for answer correctness (no RAGAS, no LangChain), hand-checked on 30 items. Retrieval recall@k of gold paragraphs. |
| Systems compared | closed-book · hybrid baseline · graph-only · chunks-only · graph + chunks · (optional) `neo4j-graphrag` |
| Entity types | 6 coarse types: PERSON, FILM, PLACE, ORG, WORK, OTHER |
| Relations | Fixed list of the 34 relations that occur in 2Wiki, plus `OTHER` (stored, never scored). Date relations (`date of birth`, `publication date`, …) are stored as node properties, not nodes. |
| Entity IDs | Stable surrogate IDs assigned by the resolver. Never a hash of the name. |
| Entity resolution | normalise → candidate pairs by embedding of name + short description, within the same type → LLM check for borderline pairs only. **When unsure, do not merge.** |
| Provenance | Every relationship stores `source_chunk_ids` and `confidence`. Structural edge is `(:Document)-[:HAS_CHUNK]->(:Chunk)`. |
| Graph retrieval | Link question entities via vector index on entity names + rapidfuzz. Expand ≤ 2 hops, cap edges per node. **Score whole paths from the seed entity**, not single facts. |
| Context budget | Graph-facts vs chunks token split is a config value, tuned on dev only. |
| Out of scope (v1) | Query analyser/router, aggregation questions, feedback loop, public ingest endpoint, React dashboard, LaTeX paper, video. |
| Python | 3.14 (verified: every core dependency resolves with arm64 wheels on 3.14). |
| CI | Unit tests only. Evaluation runs manually; result files are committed. |

### Models (pin before M1 starts)

Available on 2026-10-04 (checked against the provider docs):

| Role | Proposed | Notes |
|---|---|---|
| Extractor | `openai/gpt-oss-120b` via OpenRouter, the same pinned `deepinfra/bf16` endpoint as the generator, fallbacks disabled, strict JSON-schema output | Changed from `gemini-3.8-flash` (2026-10-04): the Gemini free tier returned 503 "high demand" on the first pilot call, and its daily limits are unknown, so a ~260-call extraction could stall for days. Paid from the same OpenRouter credit; pilot cost and projection in `results/m2/pilot.json`. Consequence: extractor and generator are the same model (listed as a limitation). |
| Generator | `openai/gpt-oss-120b` via OpenRouter, pinned to the `deepinfra/bf16` endpoint with fallbacks disabled (`allow_fallbacks: false`, `require_parameters: true`). Temperature 0, `reasoning.effort` = `medium` for every system. | `llama-3.3-70b-versatile` is not available on my Groq account (404 `model_not_found`), and Groq Developer upgrades are unavailable. The provider that served each call is stored in the cache entry and in `results/spend/openrouter_calls.jsonl`; a call served by any other provider raises and is not cached. Paid from prepaid OpenRouter credit; balance in `results/spend/openrouter_balance.json`. Cost estimate: `results/m0/generator_cost.json`. |
| Judge | `gemini-3.8-flash` | Never the same model as the generator (or the extractor). |
| Embeddings | `intfloat/e5-base-v2` (768-dim, local) | Same as the thesis. Prefixes are mandatory: `query: ` for questions, `passage: ` for chunks and entity descriptions. |
| Reranker | `BAAI/bge-reranker-base` (local cross-encoder) | |

Gemini free-tier limits are **not published** (AI Studio shows them per account). Read them from AI
Studio and record them in `docs/limits.md`. The generator has no daily cap; it is bounded by OpenRouter
credit instead. Log the model version string returned with every response.

## 3. Budget estimate

Dataset numbers come from `data/stats.json` (built by `scripts/build_data.py`, seed 42):

| Item | Value |
|---|---|
| Paragraphs in corpus | 2,049 |
| Corpus size | 154,414 words |
| Paragraph length | median 47 words, max 900 |
| Same title, different raw text | 6 titles. All 6 are the same paragraph tokenised two ways (spacing around punctuation), not homonyms, and were merged. Titles with genuinely different texts: 0. |
| Single-hop candidates (checkable, single-valued) | 225 dev, 633 test |
| Extraction calls at 8 paragraphs / call | ~260 calls, ~0.6M input tokens incl. prompt overhead |
| Generator calls, one dev iteration (125 q × ~5 systems) | ~625 |
| Generator + judge calls, final test run (375 q × 6 systems) | ~2,250 + ~2,250 |

Extraction is cheap (~260 calls). It fits in 1 to 3 days even at a low daily request cap. **The real quota
risk is evaluation, not extraction**: the final test run needs ~4,500 calls (generator calls are paid from
OpenRouter credit, judge calls count against the Gemini free tier). Mitigations:
- Every LLM call is cached on disk, keyed by `(model, prompt hash)`. Re-runs are free.
- Dev iterations use EM/F1 only. The judge runs once, on test.
- The test run may span several days and resumes from cache.

## 4. Repository layout

```
graphrag-engine/
├── pyproject.toml
├── docker-compose.yml          # neo4j:5 with APOC off, vector + fulltext indexes
├── .env.example
├── src/graphrag/
│   ├── config.py               # pydantic-settings; all tunables live here
│   ├── llm/                    # thin clients + disk cache (gemini, groq)
│   ├── data/                   # 2wiki loading, sampling, corpus, single-hop generation
│   ├── store/                  # neo4j client, schema, indexes
│   ├── retrieval/
│   │   ├── base.py             # Retriever protocol
│   │   ├── hybrid.py           # BM25 + dense + RRF + rerank (no graph imports)
│   │   └── graph.py            # M4
│   ├── extraction/             # M2
│   ├── resolution/             # M3
│   ├── generation.py           # short-answer prompt, budget enforcement
│   └── eval/                   # official normalisation, EM/F1, bootstrap, judge
├── scripts/                    # thin CLI entry points per milestone
├── data/                       # sampled questions + corpus (committed, small)
├── results/                    # committed result files per milestone
└── tests/
```

## 5. Milestones

### M0: Environment (half a day)
- Install OrbStack. `docker compose up` brings up Neo4j 5 locally.
- Python 3.14 venv, `pyproject.toml`, `pytest` runs green on an empty test.
- Record model choices, rate limits and the generator's provider pin in `docs/limits.md`.

**Done when:** `scripts/check_env.py` connects to Neo4j and makes one cached call to each LLM.

### M1: Data + hybrid baseline + closed-book (the yardstick)
Detailed in §6.

**Done when:** `results/m1/` holds EM/F1 with bootstrap CIs, by type, on **dev**, for closed-book and the
hybrid baseline, plus retrieval recall@k.

### M2: Extraction
- Extractor: `openai/gpt-oss-120b` on the pinned OpenRouter endpoint, strict `json_schema` structured
  output (the endpoint supports it, and `require_parameters` forces it). Prompt returns, per chunk,
  entities (name, coarse type, short description) and relations from the fixed 34 + `OTHER` list,
  8 paragraphs per call in a fixed batching (sorted chunk ids), every item tagged with its `chunk_id`.
- Validation: the schema plus "exactly the chunk ids that were sent"; invalid output is retried once
  (as a new, separately cached call), then logged and skipped. Rate limits and temporary 5xx errors are
  retried after a minute; a persistent outage stops the run, which resumes from the cache.
- Pilot first (`scripts/run_m2_pilot.py`): the 6 batches with the most dev gold triples; tokens, cost,
  projection for the full corpus against the remaining credit, and scores. Results in `results/m2/pilot.json`.
- Scoring against 2Wiki gold triples on **dev paragraphs only**. 2Wiki gold triples are only the facts
  the questions need, so plain precision would count correct extra facts as errors. Reported instead:
  recall (share of gold triples matched by some extracted triple) and slot precision (among extracted
  triples whose subject and relation match a gold triple, the share whose object matches). A match
  needs the same relation, names matching by rapidfuzz ratio on normalised text (primary threshold 90,
  fixed before the pilot; 80 to 100 as a sensitivity check), dates matching on year, month and day
  where both state them. True precision: hand check of a sample of extracted triples.

- Quality bar, fixed on 2026-10-04 before seeing any new-batch numbers. Measured on the new (unseen)
pilot batches only, at the primary name threshold 90:
  - recall of at least 0.75 against dev gold triples;
  - slot precision of at least 0.90;
  - at least 90% of the 30 hand-checked triples correct (at least 27 of 30).
If all three pass, the full corpus is extracted. If not, one more prompt fix on dev, then decide.
- Decision rule after pilot v2, fixed on 2026-10-04 before seeing any v3 numbers:
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

- Outcome of the pilots (2026-10-04): prompt v3 on 9 fresh batches passed recall (0.76) and failed
  slot precision (0.76); the manual classification of all 10 failures found 1 real miss and 0 wrong
  facts (`results/m2/pilot_v3_classification.json`), and the hand check found 28 of 30 triples correct
  (`results/m2/extraction_handcheck.md`, bar 27). Under the decision rule the full corpus is extracted
  with prompt v3 and the strict score is reported as it is.
- Chunk-id repair in validation: when exactly one sent chunk id is missing and exactly one unknown id
  came back, and the two are near-identical (prefix, or rapidfuzz ratio of at least 90), the unknown id
  is mapped back. Added after the full run, where one batch failed twice on a truncated id
  ("Saw Hnaung of Saga" for "Saw Hnaung of Sagaing"); the cached responses were re-validated, no new call.
- Deterministic post-processing of every extracted relation (`src/graphrag/extraction/postprocess.py`,
  not a prompt change), with the number of triples each rule changed reported in
  `results/m2/extraction_run.json`:
  1. `OTHER` whose `other_label` is a listed relation name becomes that relation;
  2. a leading `<label>:` prefix is stripped from the object, where `<label>` is the triple's own
     `other_label` or relation name (other colons, as in "Star Wars: A New Hope", are left alone).

**Done when:** extraction for the whole corpus is cached, and `results/m2/extraction_scores.json` exists.

### M3: Entity resolution
Approved on 2026-10-08, before any M3 numbers existed.

Entities:
- Page entities: the subject of each paragraph, identified by its `chunk_id`, never by its name.
  Different paragraph titles are different Wikipedia pages, so two page entities never merge.
- Mentions: every other extracted entity. A mention is linked to a page entity, or clustered with other
  mentions of the same thing that has no page, or left alone.

Method:
1. Normalise names: lowercase, drop the bracketed part, drop punctuation, collapse spaces. Titles such
   as "Dr." or "Sir" are not stripped by rule.
2. Candidate pairs, only between compatible types (PERSON, PLACE, ORG strict; FILM and WORK
   compatible with each other; OTHER compatible with every type), if any of: (a) identical normalised
   names; (b) rapidfuzz ratio of at least 85; (c) among the 10 nearest neighbours by E5 embedding of
   `name: description`, above a lower cutoff `t_low`.
3. Decision per pair:
   - never merge: two page entities; or a merge that would put two different page entities into one
     cluster (checked on every union, so A=B and B=C cannot join two pages through C);
   - auto-merge: identical normalised name, compatible type, description similarity at least
     `t_high`, and the name matches at most one page;
   - LLM check (gpt-oss-120b, pinned endpoint, strict JSON) for everything in between: fuzzy-only names,
     identical names with dissimilar descriptions, mentions whose name matches two or more pages. The
     LLM sees both names, types, descriptions and both source paragraphs and answers same, different
     or unsure. Unsure means don't merge;
   - no merge below `t_low`.
4. Union-find with the never-merge checks; every merge records its reason (rule or LLM) and score.
   Stable surrogate ids and an alias table are kept.
5. After run 1 failed (`results/m3/run1_flawed/README.md`), approved on 2026-10-08: judge prompt v2 states
   each mention's role in its paragraph (subject, or only mentioned in a paragraph about something else)
   and says to judge the named entity, not the paragraph's subject; a cannot-link rule refuses any merge
   that joins two clusters containing a pair judged "different"; merges are applied most confident first
   (E5 cosine, highest first, ties by pair ids). Pilot for v2 in `results/m3/pilot_v2.json`.

Tuning and reporting are kept apart:
- Paragraphs are split by a seeded hash into tune (30%) and report (70%). A pair belongs to the split of
  its mention's paragraph.
- `t_low` and `t_high` are tuned only on tune-split pairs, against labels from **dev question gold
  triples only. Test questions are never used for tuning or for any M3 number.** Positives: dev gold
  triples whose object has its own paragraph (the mention should link to that page). Negatives: pairs
  across the 24 page titles that are shared by more than one paragraph.
- Every reported number comes from the report split only.
- A pilot of about 100 borderline pairs runs first, with the measured cost per pair and a projection for
  the full LLM run, shown together with the tuned thresholds before the full run.

Quality bar (judged on the report split):
- Merge precision: at least 48 of 50 seeded-random merges correct, with every kind of merge represented
  (rule-based and LLM-decided). Each merge is shown with both mentions and their paragraphs.
- Duplicate rate before and after, both reported: 60 seeded-random mentions, each shown with up to 5
  candidates chosen independently of the resolver (by name and embedding similarity); the labeller
  marks which are the same entity. Before = share of mentions with at least one same-entity mention
  elsewhere; after = share where some of those are still in a different cluster. Target: after at most
  half of before.
- When unsure, do not merge.

Decided after the pilot (2026-10-08): auto-merge is off. Dev gold triples give no same-name negatives,
so `t_high` cannot be tuned from gold, and the pilot showed identical-name pairs judged different even at
cosine 1.0 (two different "Louis, Dauphin of France"). Every candidate pair is judged by the LLM.
The merge hand check samples pairs of mentions from inside the final clusters, so merges made
indirectly through union-find chains are checked too. Strata: direct same-name, direct different-name,
indirect, and mention-to-page links; within a stratum a cluster is picked uniformly before a pair.

Reported, not part of the bar:
- Bridge link recall on the report half of the dev questions, before M3 (exact normalised name matching
  only) and after M3: the share of gold bridge facts (for example film -> director, where the director
  has a paragraph) whose mention is linked to the right page entity.

**Done when:** `results/m3/resolution_report.md` shows the merge precision, both duplicate rates, the
bridge link recall before and after, and the thresholds used. Done on 2026-10-09: M3 passed (report in
`results/m3/resolution_report.md`). The hand checks were first drafted by Claude (a different model from
the gpt-oss-120b judge) and reviewed and corrected by Rohith Kumar Reddipogula; the same country across
historical eras counts as one entity.

### M4: Graph build + graph retrieval

**Approved as written on 2026-10-09.** Everything below was fixed before any M4 code or result existed.

Already decided:
- Exact-title links (decided on 2026-10-09, after seeing the report-split bridge link numbers in
  `results/m3/resolution_report.md`: 0.83 with exact name matching, 0.80 after M3, 5 links lost and 3
  gained): the M3 clusters are kept as they are, and the graph also gets exact-title links (a mention
  whose normalised name equals exactly one page title) as a separate, labelled edge type, so either
  source of a link can be told apart and evaluated on its own. No M3 threshold, prompt or merge rule is
  changed.
- Hub nodes: resolved entities that are mentioned very often (countries such as France, Italy, United
  States; concepts such as "suicide") become hubs. Graph retrieval must cap or down-rank hubs, so that
  paths through them do not flood the context.

Measured on the committed M2 and M3 outputs before drafting (no LLM calls): the graph would have 10,830
entity nodes (M3 clusters), 8,033 distinct relation edges (3,021 of them `OTHER`), 2,392 date facts
stored as properties, and 111 exact-title links that M3 did not already merge; 1,965 extracted relation
triples have an endpoint that is not an extracted entity (for example an occupation string) and are
dropped. Node degree: median 1, 99th percentile 12; 8 nodes above 25, one above 50 (United States, 111).
On the 100 dev multi-hop questions, 92 contain their first gold subject's name verbatim.

#### M4.1 Graph schema (Neo4j, same database as the M1 chunks)
- Existing: `(:Document)-[:HAS_CHUNK]->(:Chunk)`.
- `(:Entity {id, name, aliases, type, is_page, page_chunk_id, degree})`: one node per M3 cluster. `id` is
  the M3 cluster id, `name` the most frequent mention name, `aliases` all mention names, `type` the most
  frequent coarse type. Date facts are properties: `date_of_birth`, `date_of_death`,
  `publication_date`, `inception`, each with its source chunk ids.
- `(:Entity)-[:MENTIONED_IN {mention_id}]->(:Chunk)`: every mention of the cluster.
- `(:Entity)-[:REL {type, other_label, source_chunk_ids, n_sources}]->(:Entity)`: M2 relations after
  the M2 post-processing. Endpoints are resolved to the mention in the same paragraph (identical
  normalised name, else rapidfuzz ratio at least 90), then to its M3 cluster. One edge per (subject,
  type, object); `n_sources` counts the paragraphs that state it. No LLM confidence exists, so none is
  stored.
- `(:Entity)-[:EXACT_TITLE {mention_ids}]->(:Entity)`: from a cluster containing a non-page mention to the
  page entity whose title has exactly the same normalised name, only when the two are different
  clusters. A separate, labelled edge type; never merged into the M3 clusters.
- The loader is idempotent and deterministic, like the M1 loader.

#### M4.2 Graph retrieval
- Seed entities (at most 5 per question), in this order:
  1. entity names and aliases (at least 3 characters) found verbatim in the normalised question at word
     boundaries, longest match first; page entities before other entities;
  2. if fewer than 5 seeds, E5 nearest entities to the question (`query: ` prefix), cosine at least
     0.85 (the M3 `t_low`, reused, not tuned again).
- Expansion: up to 2 hops over `REL` edges in either direction. `EXACT_TITLE` edges are identity links:
  following one does not count as a hop. Per node, at most 20 neighbours (most `n_sources` first, then
  name).
- Hubs: an entity with degree above 25 (the 99.9th percentile is 25.4) can be the end of a path but is
  never passed through. Each intermediate node v multiplies the path score by 1 / log2(2 + degree(v)).
- Path score: E5 cosine between the question and the path written out as text (for example "Heat
  (film) -director-> Michael Mann; Michael Mann: date of birth 5 February 1943"), times the hub factors.
  The 10 best paths are kept; ties broken by entity ids.
- Graph context: the kept paths as cited facts ("Fact: ... [source: <chunk ids>]"), then the page
  paragraphs of the entities on those paths, in path-score order.

#### M4.3 Systems compared on dev (same generator, prompt, 1,500-token budget and abstention)
- `closed_book` and `hybrid` from M1 (results reused from `results/m1/`, cached).
- `graph_only`: graph context only.
- `graph_plus_chunks`: graph context first, using a share g of the budget, then the hybrid baseline's
  chunks fill the rest (paragraphs already included are skipped). g in {0.25, 0.5}.
- Ablation: `graph_plus_chunks` at the chosen g without `EXACT_TITLE` edges.
- Selection rule for g, fixed now: highest multi-hop dev EM; ties by multi-hop all-gold-in-context; then
  0.25. The chosen configuration is frozen for M5.

#### M4.4 Dev evaluation
- Questions: the same 125 dev questions as M1 (100 multi-hop, 25 single-hop). Test questions are never
  used in M4. Dev numbers are used for choices, so they are optimistic; M5 on test is the real estimate.
- QA metrics, per question type and on the closed-book-wrong subset: EM and F1 (official 2Wiki v1.1 with
  aliases), unknown rate, EM on answered questions, 95% bootstrap CIs, paired bootstrap differences
  against `hybrid`.
- Retrieval metrics: all gold paragraphs in the packed context (multi-hop); bridge entity recall: for
  multi-hop dev questions whose gold evidence has a bridge entity (the object of one gold triple that is
  the subject of another) with its own paragraph, the share where that paragraph is in the packed
  context. Both are also computed for `hybrid` from its committed per-question contexts.
- Quality bar, fixed now before any M4 result exists (judged for `graph_plus_chunks` at the chosen g):
  1. retrieval gain: multi-hop all-gold-in-context at least 0.10 above `hybrid` (0.53 in
     `results/m1/generation_dev.json`, so at least 0.63);
  2. no QA regression: paired EM difference against `hybrid` over all 125 dev questions has a 95% CI lower
     bound of at least -0.05;
  3. gain where the graph should help: mean EM on bridge_comparison and compositional dev questions
     together (50 questions) above `hybrid`.
  If all three pass, `graph_plus_chunks` goes to M5 as the GraphRAG system. If not, it still goes to M5
  and is reported as it is; at most one fix on dev is allowed, then decide. M5 reports every system.

#### M4.5 Cost and time (estimate)
- Graph build and retrieval use no LLM (E5 and Neo4j locally).
- Generation: 4 new configurations x 125 dev questions = 500 calls. M1's 375 calls cost 0.028 USD
  (`results/m1/generation_dev.json`), so about 0.04 USD; credit left 6.05 USD.
- Time: graph build a few minutes; retrieval about 1 second per question; generation about 5 minutes per
  configuration with 8 workers; under an hour in total.

**Done when:** `results/m4/` has dev numbers for every system above against the M1 baseline, and the
quality bar is judged in `results/m4/graph_report.md`. Done on 2026-10-09: all three criteria passed for
`graph_plus_chunks` at g = 0.5 (chosen by the fixed rule), so it goes to M5 as the GraphRAG system;
numbers in `results/m4/graph_report.md`. On dev, `graph_only` scored higher than `graph_plus_chunks`
on several metrics; the plan did not make `graph_only` eligible, so this is recorded and not acted on.
M5 reports every system, including `graph_only`.

### M5: Final test run (once)

**Approved on 2026-10-09** with three changes to the draft: one primary result and a pre-registered
expectation (M5.3), and a judge on OpenRouter instead of the Gemini free tier (M5.5). Everything below was
fixed before the test split was touched.

Test questions so far: their ids were read only to check that dev and test never overlap; their
paragraphs are part of the shared corpus by design (M1), so M2 extraction and M3 resolution ran on them
without the questions or answers. No system has seen a test question or answer.

#### M5.1 Systems on test (run once, every one with the frozen settings in M5.2)
1. `closed_book`
2. `hybrid` (the M1 baseline)
3. `graph_only`
4. `graph_plus_chunks_g0.5` (the GraphRAG system chosen in M4)
5. `graph_plus_chunks_g0.5_no_exact_title` (ablation)
All on the 375 test questions: 300 multi-hop (75 per type) and 75 single-hop
(`data/questions_test.jsonl`, `data/single_hop_test.jsonl`).

#### M5.2 Frozen settings
The code is frozen at a git tag `m5-frozen` on the commit that adds the M5 script; the script refuses to
run on any other commit or on a working tree with uncommitted changes.
- Data: `data/corpus.jsonl` (2,049 paragraphs), test files above, `data/answer_aliases.json`; seed 42.
- Hybrid retriever: BM25 (`rank_bm25`, k1 1.5, b 0.75, thesis tokenizer) + `intfloat/e5-base-v2` dense
  (exact cosine in Neo4j), weighted min-max fusion with **alpha 0.9, no title** (read from
  `results/m1/retrieval_dev.json`, as in M1 and M4; the `HybridConfig` code defaults are the thesis
  setting and are not used), 50 candidates per retriever, `BAAI/bge-reranker-base` over the fused top 30.
- Generation: `openai/gpt-oss-120b` on OpenRouter endpoint `deepinfra/bf16`, fallbacks disabled,
  temperature 0, reasoning effort medium; prompts `SYSTEM` and `SYSTEM_CLOSED_BOOK` in
  `src/graphrag/generation.py`; 1,500-token context (o200k tokenizer), skip-and-continue packing;
  abstention ("unknown" scores 0).
- Graph: `results/m2/extractions.jsonl` (prompt v3, post-processed), `results/m3/clusters.jsonl` (judge
  v2, cannot-link, `t_low` 0.85), the graph in Neo4j as built by `scripts/build_graph.py`
  (`results/m4/graph_stats.json`); retrieval constants MAX_SEEDS 5, SEED_MIN_COS 0.85,
  MIN_ALIAS_CHARS 3, MAX_NGRAM 12, MAX_NEIGHBOURS 20, HUB_DEGREE 25, TOP_PATHS 10; graph share g = 0.5.
- Scoring: official 2Wiki v1.1 EM and F1 with Wikidata aliases; 95% percentile bootstrap, 10,000
  resamples, seed 0.
- Before running, the script checks that the Neo4j graph matches `results/m4/graph_stats.json` (entity,
  REL, EXACT_TITLE and MENTIONED_IN counts) and stops if not.

#### M5.3 Comparisons
- **Primary result (one only):** the paired EM difference of `graph_plus_chunks_g0.5` vs `hybrid` over
  all 375 test questions, with its 95% bootstrap CI.
- **Pre-registered expectation (fixed before the run):** `graph_plus_chunks_g0.5` beats `hybrid` on EM,
  with the 95% CI of the paired difference above 0. The result is reported as it comes out, whatever it
  is; the expectation is recorded as met or not met.
- Pre-specified secondary result: the same paired difference on the closed-book-wrong test subset (the
  questions `closed_book` gets wrong). It is not a second headline.
- Other secondary comparisons, reported as exploratory (several comparisons, no correction):
  `graph_only` vs `hybrid`; `graph_plus_chunks_g0.5` vs `graph_only`; the ablation vs
  `graph_plus_chunks_g0.5`; every system vs `closed_book`; per question type.

#### M5.4 Metrics
- Per system, per question type, for multi-hop, single-hop and all, and on the closed-book-wrong subset:
  EM and F1 with 95% bootstrap CIs, unknown rate, number answered and EM on answered questions.
- Paired bootstrap differences for the comparisons in M5.3.
- Retrieval: multi-hop all-gold-in-context and bridge entity recall, as in M4.
- Judge correctness (M5.5) as a secondary metric next to EM.
- Dev vs test side by side for every system, to show how optimistic dev was.

#### M5.5 LLM judge
- Changed on 2026-10-09, before the test split was touched: the judge does not use the Gemini free tier.
  It runs on OpenRouter with the prepaid credit only, on a model from a different family than the
  generator and extractor (not gpt-oss), with a pinned provider and fallbacks disabled, like the
  generator.
- Proposed judge: `meta-llama/llama-3.3-70b-instruct` (Meta), endpoint `parasail/fp8`, temperature 0,
  strict JSON-schema output, `allow_fallbacks: false` and `require_parameters: true`. The model, the
  provider and the expected cost and time are shown again before the judge runs; the judge does not run
  without approval.
- No judge scores exist on dev: dev was scored with EM and F1 only, and the judge runs once, on test. So
  there is no dev judge score to compare with; EM stays the primary metric.
- What it judges: every test answer that is not "unknown" ("unknown" is incorrect without a judge call),
  deduplicated by (question, normalised answer) across systems. Projected from dev: about 530 unique
  answers out of 1,875.
- Input per item: the question, the gold answer and its accepted aliases, the system's answer. Output
  (strict JSON): correct / incorrect / unsure and a reason of at most 15 words. Correct means the answer
  names the same entity or value as the gold answer, allowing other wording, spellings, aliases and date
  formats; an answer less specific than the question asks for is incorrect; unsure counts as incorrect
  and is reported separately. The judge does not see which system produced the answer.
- 25 items per call, so about 22 calls; cached; invalid output retried once, then the items are marked
  unsure and logged. Expected cost (at about 4,000 input and 1,000 output tokens per call) under 0.05
  USD; expected time a few minutes.

#### M5.6 Hand check of the judge (30 items)
- 30 seeded-random judged items from the test answers, stratified: 10 where the judge says correct but
  EM is 0, 10 where the judge says incorrect, 10 where judge and EM agree on correct (fewer if a stratum
  is smaller; the rest filled from the others).
- The sheet shows the question, the gold answer and aliases, and the system's answer, but not the
  judge's verdict or the system name; the verdicts and strata are in a separate key file.
- Reported: agreement between the judge and the hand labels, overall and per stratum. Not a pass/fail
  bar: if agreement is below 27 of 30, the judge metric is reported with that warning and EM stays the
  primary metric.

#### M5.7 BENCHMARK.md
Generated from the M5 results files (no number typed by hand), plain ASCII, numbered contents:
1. setup: data, systems, frozen settings and the `m5-frozen` commit;
2. primary result: `graph_plus_chunks_g0.5` vs `hybrid` on all 375 test questions, and whether the
   pre-registered expectation was met; then the closed-book-wrong subset as a secondary result;
3. full results table per system and per question type (EM, F1, unknown rate, answered, EM on answered,
   CIs);
4. secondary comparisons and the exact-title ablation;
5. retrieval metrics (all-gold-in-context, bridge entity recall);
6. judge results and hand-check agreement;
7. dev vs test;
8. error analysis with real examples (seeded sample of graph failures, classified);
9. limitations (from section 7 of this plan, M2 to M4 notes, judge agreement) and cost.

#### M5.8 Run-once safeguard
- `scripts/run_m5_test.py` is the only script that reads test questions for answering. It creates
  `results/m5/TEST_RUN.lock` atomically (fails if it exists) with the start time, the `m5-frozen` commit
  and a hash of every frozen setting.
- If the lock exists with status "finished", the script refuses to run. If it exists with status
  "running" (an interrupted run), it may resume only with the same commit and settings hash; every LLM
  call is cached, so a resume cannot change any answer.
- When the run ends, the lock is set to "finished" and committed together with the results.
- Tests cover the lock: creation, refusal after "finished", refusal on a different commit or hash.

#### M5.9 Cost and time
- Generation: 5 systems x 375 = 1,875 calls; at the dev rate (`results/m1/generation_dev.json`, 375 calls
  for 0.028 USD) about 0.14 USD; credit left 5.89 USD. Retrieval a few minutes; generation about 30
  minutes with 8 workers.
- Judge: 0 USD on the free tier; minutes to 2 days depending on the limits.
- Hand checks: about 30 minutes of labelling.

**Done when:** `results/m5/` holds the test results, the judge results and the hand-check agreement,
`results/m5/TEST_RUN.lock` is "finished", and `BENCHMARK.md` is generated.

### M6: Packaging
- README: problem, design decisions, how to reproduce, results, limitations (aggregation not handled,
  single-hop wording favours BM25, pooled-corpus setting differs from published 2Wiki setting).
- Read-only FastAPI: `POST /query`, `GET /graph/{entity}`, `GET /health`. Simple graph view.
- Optional: public read-only demo.

### M7: Optional: `neo4j-graphrag` as a third system
Same corpus, same generator, same budget, same eval.

## 6. Milestone 1 in detail

### 6.1 Data (`src/graphrag/data/`)
1. Download `validation` parquet of 2WikiMultihopQA (HF mirror `framolfese/2WikiMultihopQA`). Keep the file
   checksum in `data/README.md`.
2. Stratified sample with `seed=42`: per type, 25 dev + 75 test. Store as `data/questions_{dev,test}.jsonl`
   with `id, question, answer, type, evidences, gold_titles`.
3. Corpus: for each sampled question, its gold paragraphs + 4 seeded-random distractors. Paragraph text =
   sentences joined with spaces. Identical (title, text) pairs are stored once.
   **Collision rule:** texts are compared with a key that ignores spacing around punctuation. 2Wiki
   contains some paragraphs in two tokenisations (`Silverstein (born` vs `Silverstein( born`); those are
   merged into one chunk, storing the variant most questions use, and every raw variant maps to that id.
   Genuinely different texts of one title are all kept, with id `title#` + the first 8 hex chars of the
   SHA-1 of the comparison key. Merges and kept-apart titles are logged in `data/collisions.jsonl`. Each
   question stores the chunk ids of its gold paragraphs (`gold_chunk_ids`), resolved from its own
   context, so gold never points to a different same-title paragraph. Store `data/corpus.jsonl`. Dev and test share one corpus, as in a real deployment.
4. Single-hop questions: from evidence triples of the sampled questions (dev triples → dev, test triples →
   test), one template per single-valued relation (16 templates; multi-valued relations such as
   `award received` or `child` are excluded because they have several correct answers). The full rule
   list is `SINGLE_HOP_RULES` in `src/graphrag/data/build.py`, recorded in `data/stats.json`.
   Hand check of 30 questions of the first sample (`data/single_hop_handcheck.md`): 4 errors. Rules
   added after it: drop `country of citizenship`, keep a bracketed title disambiguation in the question,
   skip subjects whose name matches more than one corpus paragraph, keep the paragraph's casing in the
   answer, and exclude one fact with a wrong 2Wiki evidence triple. The regenerated sample has not been
   hand-checked again.

### 6.2 Store (`src/graphrag/store/`)
- `(:Document {id, title})-[:HAS_CHUNK]->(:Chunk {id, title, text, text_hash, embedding_title, embedding_text})`.
  One document per paragraph for now.
- Two E5 embeddings per chunk (`passage: ` prefix): one of `title. text`, one of the text alone, so
  whether to index the title is decided on dev. Unique constraints on `Chunk.id` and `Document.id`;
  one vector index per embedding (768, cosine).
- Idempotent loader (`scripts/load_corpus.py`): only new or changed chunks are embedded, chunks no longer
  in the corpus are deleted, and a second run changes nothing.

### 6.3 Hybrid retriever (`src/graphrag/retrieval/hybrid.py`)
- BM25 top-50 (in-memory `rank_bm25`, thesis tokenizer and parameters), dense top-50 by **exact** cosine
  computed in Neo4j (question embedded with the `query: ` prefix). The HNSW vector index is approximate:
  on 20 dev questions its top-50 missed up to 3 exact hits in the tail, and fusion uses the whole top-50.
  At about 2k chunks an exact scan is fast and deterministic. (`db.index.vector.queryNodes` is also
  deprecated in this Neo4j version in favour of the `SEARCH` clause, which works if the corpus grows.)
- Fusion, set in config:
  - `weighted` (thesis): min-max normalise each score list, missing score counts as 0,
    `alpha * dense + (1 - alpha) * bm25`.
  - `rrf`: Reciprocal Rank Fusion with `k=60`.
- Top-30 after fusion, then `bge-reranker-base` rerank (on `title. text`), then top-k.
- Recall@k is reported both before and after the reranker, so the reranker's effect is visible.
- Implements `Retriever.retrieve(question, k) -> list[Passage]`. No graph imports (enforced by a test).
- Dev sweep (`scripts/run_m1_retrieval.py`): title on/off x (weighted alpha 0.0 to 1.0 step 0.1, rrf).
  Selection rule, fixed before the run: max post-rerank `all_gold@10` on multi-hop dev, then
  `recall@5`, then grid order. Chosen config and all numbers: `results/m1/retrieval_dev.json`.

### 6.4 Generation (`src/graphrag/generation.py`)
- One prompt for every retrieval system: answer only from the context with the shortest possible span
  (an entity, a date, or yes/no), or `unknown`. No explanation. Closed-book gets the same instructions
  without the context sentence ("If you do not know, reply unknown").
- Context is packed to 1,500 tokens (o200k tokenizer) in rank order; a passage that does not fit is
  skipped and the next one is tried.
- Temperature 0, reasoning effort medium. All calls cached; provider pinned (see section 2).

### 6.5 Evaluation (`src/graphrag/eval/`)
- Answer scoring copied from the official 2Wiki evaluation script v1.1 (Apache 2.0, notice kept): EM and
  F1 as the max over the gold answer plus the Wikidata aliases and demonyms of the answer entity
  (`data/answer_aliases.json`, built by `scripts/build_answer_aliases.py` from the official
  `data_ids.zip`, checksum pinned). For single-hop questions the aliases come from the evidence object's
  entity id when the official ids line up with the evidence list.
- 95% percentile bootstrap CIs (10,000 resamples, fixed seed); paired bootstrap for differences.
- Retrieval recall@k (k = 2, 5, 10) against `gold_chunk_ids`. No LLM needed, so it is the fastest
  signal while tuning.
- Output: `results/m1/generation_dev_<system>.jsonl` (per question) and `results/m1/generation_dev.json`.

### 6.6 Tests
- Sampling is deterministic for a fixed seed.
- Every gold paragraph (exact text) of every sampled question is in the corpus under its `gold_chunk_ids`.
- Title collisions keep both texts with distinct ids, and both are logged.
- E5 prefixes: questions get `query: `, passages get `passage: ` (tested on the encode wrapper).
- Weighted fusion with alpha 1.0 equals dense-only ranking, with alpha 0.0 equals BM25-only ranking.
- Loader is idempotent.
- `hybrid.py` imports nothing from `graphrag.retrieval.graph` / `graphrag.extraction`.
- EM/F1 matches the official script on a handful of known cases.

### 6.7 Exit criteria
- Dev numbers for closed-book and hybrid baseline, with CIs, by type.
- Recall@k of the baseline. If recall@10 for multi-hop is already very high, write that down: it means the
  graph has to win on reasoning, not on finding paragraphs.
- Size of the closed-book-wrong subset on dev. If it is tiny, revisit the sample size before M2.

## 7. Known limitations (to repeat in the README)
- Pooled corpus over ~2k paragraphs differs from the standard 2Wiki distractor setting, so absolute
  numbers are not directly comparable with published leaderboards; the normalisation is.
- The models have likely seen Wikipedia; the closed-book baseline and the closed-book-wrong headline
  subset exist to account for that.
- Template single-hop questions overlap lexically with their source paragraph, which favours BM25.
- Aggregation questions are out of scope.
- Baseline retrieval config: all 24 dev configurations (title on/off x 11 alphas + RRF) are within noise
  on the selection metric (spread in `results/m1/retrieval_dev.json`, `spread_note`). The chosen config
  (no title, alpha 0.9) follows a rule fixed before the run; the thesis setting (title, alpha 0.7) is
  reported next to it.
- Scoring: answers count as correct when they match the gold answer or a Wikidata alias. 57 of 100
  single-hop questions have no official entity ids that line up with their evidence, so they are scored
  against the gold answer only.
