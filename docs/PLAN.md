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
| Extractor | `gemini-3.8-flash` (stable) | Fallback choice if the free daily quota is too low: `gemini-3.5-flash-lite`. **No cross-provider fallback**: on quota exhaustion the run stops and resumes from cache. |
| Generator | `openai/gpt-oss-120b` via OpenRouter, pinned to the `deepinfra/bf16` endpoint with fallbacks disabled (`allow_fallbacks: false`, `require_parameters: true`). Temperature 0, `reasoning.effort` = `medium` for every system. | `llama-3.3-70b-versatile` is not available on my Groq account (404 `model_not_found`), and Groq Developer upgrades are unavailable. The provider that served each call is stored in the cache entry and in `results/spend/openrouter_calls.jsonl`; a call served by any other provider raises and is not cached. Paid from prepaid OpenRouter credit; balance in `results/spend/openrouter_balance.json`. Cost estimate: `results/m0/generator_cost.json`. |
| Judge | `gemini-3.8-flash` | Never the same model as the generator. |
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
- Prompt that returns JSON entities (name, coarse type, short description) and relations from the fixed
  34 + `OTHER` list, 8 paragraphs per call, with the paragraph title attached to every item.
- Strict schema validation; invalid output is retried once, then logged and skipped.
- Score against 2Wiki gold triples on **dev paragraphs only**: relation must match, both entities must
  fuzzy-match after normalisation (rapidfuzz, threshold tuned on dev). Report precision/recall per relation.

**Done when:** extraction for the whole corpus is cached, and `results/m2/extraction_scores.json` exists.

### M3: Entity resolution
- Normalise (case, punctuation, suffixes like "(film)", "Jr.").
- Candidate pairs: same coarse type, embedding of `name + description`, top-k neighbours above a threshold.
- LLM check only for pairs in a borderline band. Unsure → keep separate.
- Assign stable surrogate IDs; keep an alias table.
- Measure: duplicate rate on a hand-labelled sample of ~100 entities (before/after), and **precision on a
  hand-checked sample of ~50 merges**.

**Done when:** `results/m3/resolution_report.md` shows both numbers and the thresholds used.

### M4: Graph build + graph retrieval
- Load resolved entities and relations into Neo4j with `source_chunk_ids` and `confidence`.
- Entity linking from question → seed nodes.
- ≤ 2-hop expansion with per-node edge cap; path scoring from the seed; serialise top paths as cited facts.
- Three variants: graph-only, chunks-only, graph + chunks. Tune budget split and caps **on dev**.

**Done when:** `results/m4/` has dev numbers for all three variants vs the M1 baseline.

### M5: Final test run (once)
- Freeze all config. Run every system on **test**, once.
- EM/F1 + CIs by type, headline on the closed-book-wrong subset, judge scores, 30-item hand check.
- `BENCHMARK.md`: tables, per-type breakdown, error analysis with real examples, limitations.

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
- One prompt for every system: answer with the shortest possible span (an entity, a date, or yes/no)
  or `unknown`. No explanation.
- Context is packed to a fixed token budget (config, e.g. 1,500 tokens) in rank order. Closed-book gets
  an empty context and the same instructions.
- Temperature 0. All calls cached.

### 6.5 Evaluation (`src/graphrag/eval/`)
- Vendor the answer-normalisation and EM/F1 functions from the official 2Wiki eval script (check and keep
  its licence notice).
- Paired bootstrap (10,000 resamples) for differences between systems.
- Retrieval recall@k (k = 2, 5, 10) against `gold_chunk_ids`. No LLM needed, so it is the fastest signal
  while tuning.
- Output `results/m1/{system}_{split}.jsonl` (per-question) and `results/m1/summary.json`.

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
