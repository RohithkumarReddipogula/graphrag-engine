# GraphRAG Engine

Does a knowledge graph built from the same documents help a language model answer multi-hop questions,
compared with a strong hybrid retriever, when the comparison is measured honestly? This project answers
that on 2WikiMultihopQA: it builds a knowledge graph from the corpus with an LLM, resolves entities,
retrieves facts and paragraphs through the graph, and compares the answers with a closed-book model and
with the hybrid retriever from my MSc thesis, on a test split that was run once with a pre-registered
primary result.

## Contents

1. Headline result
2. How the result was protected
3. What went wrong and how I fixed it
4. Architecture
5. How to reproduce
6. Limitations
7. Cost and LLM providers
8. Data, code and licences

## 1. Headline result

<!-- BEGIN m5-headline -->
Source: `results/m5/test_summary.json`. Test split, 375 questions, run once. EM = exact match (official 2Wiki scoring with aliases), 95% bootstrap CI. Unknown = the system abstained (scores 0).

| System | EM, all 375 [95% CI] | EM, multi-hop 300 [95% CI] | Unknown rate |
|---|---|---|---|
| Closed-book (no retrieval) | 0.31 [0.26, 0.35] | 0.34 [0.29, 0.39] | 0.30 |
| Hybrid retriever (baseline) | 0.57 [0.51, 0.62] | 0.48 [0.43, 0.54] | 0.36 |
| GraphRAG: graph + chunks (pre-registered system) | 0.79 [0.74, 0.83] | 0.76 [0.71, 0.81] | 0.15 |
| Graph only | 0.79 [0.75, 0.83] | 0.76 [0.71, 0.81] | 0.12 |

Primary result (one, fixed before the run): paired EM difference, GraphRAG minus hybrid, over all 375 test questions: **+0.22 [+0.17, +0.27]**. Pre-registered expectation (95% CI above 0): **met**.

Pre-specified secondary result, the 260 questions the model gets wrong without retrieval: +0.22 [+0.16, +0.28].
<!-- END m5-headline -->

The full benchmark, with per-type results, the retrieval metrics, an LLM judge with its hand check, dev vs
test and an error analysis, is in `BENCHMARK.md`.

## 2. How the result was protected

- Pre-registered before the test run: one primary result (the paired EM difference above) and its
  expected direction (GraphRAG beats the hybrid retriever, with the 95% CI above 0), recorded in
  `docs/PLAN.md` before any test answer existed. The result is reported as it came out.
- Every choice (retrieval configuration, entity-resolution threshold, graph share of the context) was
  made on the dev split. Quality bars for each milestone were written down before their results existed.
- The test split was run once, at the git tag `m5-frozen`. The run script refuses to start if code or
  data differ from that tag, and a run-once lock (`results/m5/TEST_RUN.lock`, status finished) stops it
  from running again.
- Every system answers with the same generator, prompt and context budget, and may answer "unknown"; the
  unknown rate is reported next to every score.

## 3. What went wrong and how I fixed it

Entity resolution decides which mentions are the same real-world entity, with an LLM judging candidate
pairs. The first full run (run 1) looked fine on cost and speed, but two checks against gold data showed
it was wrong:

1. The judge prompt showed each mention together with its paragraph, and the LLM judged the paragraph's
   subject instead of the mentioned entity. "Jeff Bezos" mentioned in MacKenzie Scott's paragraph was
   judged "husband and wife, separate persons" when compared with the Jeff Bezos page.
2. Clustering had no cannot-link rule, so chains of "same" verdicts joined entities that the LLM had
   explicitly judged different.

The fix: a judge prompt that states each mention's role (subject of its paragraph, or only mentioned in
it), a cannot-link rule checked on whole clusters, a pilot that tested both directions (known links must
be judged "same", run 1's false merges must be judged "different"), and then a full re-run. Run 1 is
kept in `results/m3/run1_flawed/` with an explanation.

<!-- BEGIN m3-run1 -->
Sources: `results/m3/run1_flawed/resolution_run.json`, `results/m3/resolution_run.json`, `results/m3/handcheck_scores.json`.

|  | Run 1 (flawed) | Run 2 (fixed) |
|---|---|---|
| Bridge links found (report split; exact name matching finds 57 of 69) | 35 of 69 | 55 of 69 |
| Merges refused by cannot-link | rule did not exist | 1,673 |
| Largest cluster | United States + United Kingdom | United States |
| Hand-checked merges correct | not checked (sheets deleted) | 49 of 50 |
<!-- END m3-run1 -->

## 4. Architecture

```
Ingestion (once, over the whole corpus)

  Wikipedia paragraphs --> entity and relation --> entity resolution --> Neo4j graph
  (2WikiMultihopQA)        extraction (LLM,        (LLM judge on          Entity, REL, EXACT_TITLE,
                           fixed relation list)    candidate pairs,       MENTIONED_IN, Chunk
                                                   cannot-link rule)

Answering a question

  question --+--> hybrid retriever: BM25 + E5 dense, weighted fusion, reranker --> paragraphs --+
             |                                                                                 |
             +--> graph retrieval: seed entities, 2-hop paths, hubs capped --> cited facts ---+
                                                                               + page paragraphs
                                                                                               |
                                    context (fixed token budget) <-----------------------------+
                                                 |
                                                 v
                                    generator LLM --> short answer or "unknown"
```

<!-- BEGIN data-stats -->
Source: `data/stats.json`. Corpus: 2,049 Wikipedia paragraphs (2WikiMultihopQA, HF revision fe713bf). Dev: 100 multi-hop + 25 single-hop questions. Test: 300 multi-hop + 75 single-hop questions.
<!-- END data-stats -->

## 5. How to reproduce

Requirements: Python 3.14, Docker (for Neo4j) and an OpenRouter API key with prepaid credit.

```
python3.14 -m venv .venv
.venv/bin/pip install -e ".[dev]"
cp .env.example .env                              # fill in the OpenRouter key and a Neo4j password
docker compose up -d                              # Neo4j
.venv/bin/python scripts/check_env.py
.venv/bin/python scripts/build_data.py            # pinned download, sample, corpus, single-hop set
.venv/bin/python scripts/build_answer_aliases.py  # official alias file (pinned)
.venv/bin/python scripts/load_corpus.py           # paragraphs and E5 embeddings into Neo4j
.venv/bin/python scripts/run_m2_extraction.py     # entity and relation extraction
.venv/bin/python scripts/run_m3_resolution.py     # entity resolution
.venv/bin/python scripts/build_graph.py           # graph into Neo4j
.venv/bin/python scripts/run_m4_eval.py           # dev evaluation of all systems
.venv/bin/python scripts/render_tables.py         # re-render the README number blocks
.venv/bin/pytest
```

Read-only API (runs from the venv, localhost only; Neo4j must be running):

```
.venv/bin/uvicorn graphrag.api:app --host 127.0.0.1 --port 8000
curl -s 127.0.0.1:8000/health
curl -s -X POST 127.0.0.1:8000/ask -H 'content-type: application/json' \
     -d '{"question": "Which film has the director born later, Freakin'"'"' Beautiful World or Billy Two Hats?"}'
curl -s "127.0.0.1:8000/entity/Ted%20Kotcheff"
```

- `GET /health`: Neo4j reachable and the graph equal to the one the results were produced with.
- `POST /ask`: the pre-registered GraphRAG system with its frozen settings; returns the answer (or
  "unknown"), the graph facts it used with their source paragraphs, and the paragraphs in its context.
  Each new question makes one call to the pinned generator on OpenRouter; repeated questions come from
  the cache.
- `GET /entity/{id}`: an entity's names, type, dated facts with sources, relations and page paragraph.
- No ingestion and no writes. `tests/test_pipeline_parity.py` checks that the API builds exactly the
  contexts of the committed dev run.

Static graph view for a website (no backend): `docs/graph_view/subgraph.json` holds the graph paths the
GraphRAG system used as facts for 20 dev questions (5 per question type), with their answers from the
committed dev run, as nodes and links. `docs/graph_view/example.html` shows it with a force-graph library:

```
.venv/bin/python scripts/export_graph_view.py              # re-export (needs Neo4j; no LLM calls)
python3 -m http.server 8001 --directory docs/graph_view    # then open http://localhost:8001/example.html
```

`scripts/run_m5_test.py` refuses to run again by design (run-once lock); the committed files in
`results/m5/` are the test run. Every LLM call is cached on disk, so re-running a step costs nothing for
calls that were already made.

## 6. Limitations

- The corpus pools the gold and distractor paragraphs of all sampled questions, unlike the standard 2Wiki
  distractor setting, so absolute numbers are not comparable with published leaderboards. The answer
  scoring is the official one.
- The same model extracts the graph and answers the questions, so their errors may be correlated.
- The language models have likely seen Wikipedia; the closed-book baseline and the closed-book-wrong
  subset account for that.
- Per-type results rest on small samples; secondary comparisons are exploratory.
- Aggregation questions are out of scope.
- More, with numbers: `BENCHMARK.md`, section 9.

## 7. Cost and LLM providers

Every LLM call behind a result (entity and relation extraction, entity resolution, answer generation and
the test-answer judge) went through OpenRouter with prepaid credit, each role pinned to one upstream
provider with fallbacks disabled. Three early calls to a free Gemini tier (two connectivity checks and one
pilot extraction batch, before the extractor was moved to OpenRouter; listed in
`results/spend/gemini_audit.json`) contributed to no result; the Gemini path has since been removed.

<!-- BEGIN cost -->
Source: `results/m5/spend_snapshot.json`, taken from the spend ledger `results/spend/openrouter_calls.jsonl` (one row per paid call, with provider, tokens and cost).

| Item | Cost (USD) |
|---|---|
| All OpenRouter calls, M0 to M5 | 1.82 (6,148 calls: DeepInfra 6,126, Parasail 22) |
| of which the single test run | 0.112 |
| of which the test-answer judge | 0.017 |
<!-- END cost -->

## 8. Data, code and licences

- This project: MIT License (see `LICENSE`). The MIT licence covers the code and documentation written
  for this project. It does not cover the third-party material below, which keeps its own terms.
- Dataset: 2WikiMultihopQA. Xanh Ho, Anh-Khoa Duong Nguyen, Saku Sugawara and Akiko Aizawa,
  "Constructing A Multi-hop QA Dataset for Comprehensive Evaluation of Reasoning Steps", COLING 2020.
  Repository: github.com/Alab-NII/2wikimultihop (Apache License 2.0). The dataset is built from
  Wikipedia and Wikidata. This repository commits a small derived sample under `data/` (sampled
  questions and their paragraphs); the full dataset is downloaded by the build scripts and not
  committed. The question files are loaded from the Hugging Face mirror `framolfese/2WikiMultihopQA`
  at a pinned revision; the answer aliases come from the official `data_ids.zip` release.
- Scoring code: `src/graphrag/eval/answers.py` is copied from the official 2Wiki evaluation script
  v1.1 under the Apache License 2.0, with the notice kept.
- Models: `intfloat/e5-base-v2`, `BAAI/bge-reranker-base`, `openai/gpt-oss-120b` and
  `meta-llama/llama-3.3-70b-instruct` are used under their own licences and terms; see their model
  cards and provider terms.
- Retriever design: from my MSc thesis repository, github.com/RohithkumarReddipogula/AI-Powered-Rag-System.

---

Rohith Kumar Reddipogula, 2026. MIT License.
