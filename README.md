# GraphRAG Engine

Status: work in progress. Milestones M0 and M1 are complete: data, a strong hybrid retrieval baseline,
and a closed-book baseline, all measured on the dev split. The graph milestones (M2 to M5) are next.
No test-split numbers exist yet; the test split is run once, at the end.

## Contents

1. What this is
2. Plan and milestones
3. Data
4. Baseline system (M1)
5. M1 results on the dev split
6. Reproducing
7. Limitations so far
8. Data, code and licences
9. Work in progress

## 1. What this is

This project measures when a knowledge graph helps retrieval-augmented question answering on multi-hop
questions, and when it does not. It compares, on the same questions, corpus, generator and context
budget:

- a closed-book model (no retrieval),
- a strong hybrid retrieval baseline (BM25 + dense + reranker, the retriever from my MSc thesis),
- graph-based retrieval over a knowledge graph built from the same corpus (next milestones).

Design choices that keep the comparison honest:

- The benchmark is 2WikiMultihopQA, which has real multi-hop questions with checked answers.
- The language models have likely seen Wikipedia. The headline numbers are therefore reported on the
  questions the closed-book model gets wrong, next to the numbers for all questions.
- Every system may answer "unknown" when the context does not contain the answer. That scores 0. Every
  results table shows the unknown rate and the accuracy on answered questions next to EM.
- All tuning happens on the dev split. The test split is run once, with frozen settings.
- Every number in this README is rendered from a committed file under `data/` or `results/` by
  `scripts/render_tables.py`.

## 2. Plan and milestones

The full plan and every design decision are in `docs/PLAN.md`.

| Milestone | Content | Status |
|---|---|---|
| M0 | Environment: Neo4j in Docker, cached LLM clients, cost tracking | done |
| M1 | Data, hybrid baseline, closed-book baseline, dev results | done |
| M2 | Entity and relation extraction, scored against 2Wiki evidence triples | next |
| M3 | Entity resolution (merge only when sure), measured by hand-checked samples | planned |
| M4 | Graph build and graph retrieval (path scoring from linked entities) | planned |
| M5 | One final run on the test split, judge scores, error analysis | planned |
| M6 | Packaging: read-only API and a simple graph view | planned |
| M7 | Optional: `neo4j-graphrag` as a third system | optional |

## 3. Data

<!-- BEGIN data-stats -->
Source: `data/stats.json`.

| Item | Value |
|---|---|
| Source | 2WikiMultihopQA validation split (12,576 questions), HF revision fe713bf |
| Multi-hop questions | dev 100 (compositional 25, comparison 25, bridge_comparison 25, inference 25); test 300 |
| Single-hop questions | dev 25, test 75 (generated from evidence triples) |
| Corpus | 2,049 paragraphs, 154,414 words, median 47 words |
| Distractors | 4 per question, pooled into one corpus |
| Same paragraph in two tokenisations | 6 titles, merged |
| Single-hop hand check | 4 errors in 30 checked questions (13%); rules tightened afterwards |
<!-- END data-stats -->

Details:

- Dev and test questions are both drawn from the 2Wiki validation split, because the public test split
  has no answers. Sampling is stratified by question type with a fixed seed.
- Each question contributes its gold paragraphs and 4 random distractor paragraphs from its own
  context; all of them are pooled into one corpus that every question is answered against.
- Some paragraphs appear in 2Wiki in two tokenisations (spacing around punctuation). Those are merged
  into one chunk, and gold paragraph ids point to the merged chunk.
- Single-hop questions are generated from 2Wiki evidence triples with one template per single-valued
  relation, and are only kept when the answer appears verbatim in the paragraph.

## 4. Baseline system (M1)

- Store: Neo4j (local, Docker). One chunk per paragraph, with E5 embeddings.
- Sparse retrieval: BM25 (`rank_bm25`, k1 1.5, b 0.75), the tokenizer from my MSc thesis.
- Dense retrieval: `intfloat/e5-base-v2` with the `query: ` and `passage: ` prefixes, exact cosine
  search in Neo4j.
- Fusion: min-max normalised weighted sum, `alpha * dense + (1 - alpha) * bm25`, with RRF as an
  alternative. The configuration was chosen on dev by a rule fixed before the run.
- Reranker: `BAAI/bge-reranker-base` over the fused top 30.
- Generation: `openai/gpt-oss-120b` through OpenRouter, pinned to one upstream provider with fallbacks
  disabled, temperature 0, the same reasoning effort for every system. Short answers only (an entity, a
  date, yes/no, or "unknown"). Context budget: 1,500 tokens.
- Scoring: exact match (EM) and F1 from the official 2Wiki evaluation script v1.1, with the gold answer
  plus its Wikidata aliases accepted. 95% bootstrap confidence intervals.
- Every LLM call is cached on disk and every paid call is logged with its provider, tokens and cost in
  `results/spend/`.

## 5. M1 results on the dev split

<!-- BEGIN m1-tables -->
Source: `results/m1/retrieval_dev.json`, dev split, 100 multi-hop + 25 single-hop questions.

| Retriever | Multi-hop recall@5 | Multi-hop recall@10 | Multi-hop all gold in top 10 | Single-hop recall@2 |
|---|---|---|---|---|
| Chosen (no title, alpha 0.9), before rerank | 0.74 | 0.79 | 0.56 | 1.00 |
| Chosen (no title, alpha 0.9), after rerank | 0.74 | 0.79 | 0.56 | 1.00 |
| Thesis setting (title, alpha 0.7), before rerank | 0.74 | 0.78 | 0.54 | 1.00 |
| Thesis setting (title, alpha 0.7), after rerank | 0.75 | 0.79 | 0.54 | 1.00 |

Across all 24 retrieval configurations tried on dev, multi-hop all-gold-in-top-10 after rerank ranges from 0.48 to 0.56. With 100 questions, one question is 0.01, so these differences are within noise.

Source: `results/m1/generation_dev.json`, dev split. Unknown = the system abstained (scores 0).

Table: all dev questions.

| System | Questions | n | EM [95% CI] | F1 | Unknown rate | Answered n | EM on answered |
|---|---|---|---|---|---|---|---|
| Closed-book (no retrieval) | multi-hop | 100 | 0.38 [0.28, 0.48] | 0.42 | 0.28 | 72 | 0.53 |
| Closed-book (no retrieval) | single-hop | 25 | 0.20 [0.04, 0.36] | 0.47 | 0.00 | 25 | 0.20 |
| Closed-book (no retrieval) | all | 125 | 0.34 [0.26, 0.43] | 0.43 | 0.22 | 97 | 0.44 |
| Hybrid baseline (chosen: no title, alpha 0.9) | multi-hop | 100 | 0.44 [0.34, 0.54] | 0.47 | 0.49 | 51 | 0.86 |
| Hybrid baseline (chosen: no title, alpha 0.9) | single-hop | 25 | 0.96 [0.88, 1.00] | 0.98 | 0.00 | 25 | 0.96 |
| Hybrid baseline (chosen: no title, alpha 0.9) | all | 125 | 0.54 [0.46, 0.63] | 0.57 | 0.39 | 76 | 0.89 |
| Hybrid, thesis setting (title, alpha 0.7) | multi-hop | 100 | 0.44 [0.35, 0.54] | 0.48 | 0.48 | 52 | 0.85 |
| Hybrid, thesis setting (title, alpha 0.7) | single-hop | 25 | 0.96 [0.88, 1.00] | 0.98 | 0.00 | 25 | 0.96 |
| Hybrid, thesis setting (title, alpha 0.7) | all | 125 | 0.54 [0.46, 0.63] | 0.58 | 0.38 | 77 | 0.88 |

Table: headline subset, questions the closed-book model gets wrong (n = 82).

| System | Questions | n | EM [95% CI] | F1 | Unknown rate | Answered n | EM on answered |
|---|---|---|---|---|---|---|---|
| Closed-book (no retrieval) | multi-hop | 62 | 0.00 [0.00, 0.00] | 0.06 | 0.45 | 34 | 0.00 |
| Closed-book (no retrieval) | single-hop | 20 | 0.00 [0.00, 0.00] | 0.34 | 0.00 | 20 | 0.00 |
| Closed-book (no retrieval) | all | 82 | 0.00 [0.00, 0.00] | 0.13 | 0.34 | 54 | 0.00 |
| Hybrid baseline (chosen: no title, alpha 0.9) | multi-hop | 62 | 0.32 [0.21, 0.44] | 0.37 | 0.56 | 27 | 0.74 |
| Hybrid baseline (chosen: no title, alpha 0.9) | single-hop | 20 | 0.95 [0.85, 1.00] | 0.97 | 0.00 | 20 | 0.95 |
| Hybrid baseline (chosen: no title, alpha 0.9) | all | 82 | 0.48 [0.37, 0.59] | 0.52 | 0.43 | 47 | 0.83 |
| Hybrid, thesis setting (title, alpha 0.7) | multi-hop | 62 | 0.31 [0.19, 0.42] | 0.36 | 0.56 | 27 | 0.70 |
| Hybrid, thesis setting (title, alpha 0.7) | single-hop | 20 | 0.95 [0.85, 1.00] | 0.97 | 0.00 | 20 | 0.95 |
| Hybrid, thesis setting (title, alpha 0.7) | all | 82 | 0.46 [0.35, 0.57] | 0.51 | 0.43 | 47 | 0.81 |

Table: multi-hop questions by type.

| System | Type | n | EM [95% CI] | Unknown rate | EM on answered |
|---|---|---|---|---|---|
| Closed-book (no retrieval) | comparison | 25 | 0.64 [0.44, 0.84] | 0.20 | 0.80 |
| Closed-book (no retrieval) | inference | 25 | 0.32 [0.16, 0.52] | 0.20 | 0.40 |
| Closed-book (no retrieval) | compositional | 25 | 0.16 [0.04, 0.32] | 0.44 | 0.29 |
| Closed-book (no retrieval) | bridge_comparison | 25 | 0.40 [0.20, 0.60] | 0.28 | 0.56 |
| Hybrid baseline (chosen: no title, alpha 0.9) | comparison | 25 | 0.96 [0.88, 1.00] | 0.04 | 1.00 |
| Hybrid baseline (chosen: no title, alpha 0.9) | inference | 25 | 0.56 [0.36, 0.76] | 0.28 | 0.78 |
| Hybrid baseline (chosen: no title, alpha 0.9) | compositional | 25 | 0.24 [0.08, 0.40] | 0.64 | 0.67 |
| Hybrid baseline (chosen: no title, alpha 0.9) | bridge_comparison | 25 | 0.00 [0.00, 0.00] | 1.00 | - |
| Hybrid, thesis setting (title, alpha 0.7) | comparison | 25 | 0.96 [0.88, 1.00] | 0.04 | 1.00 |
| Hybrid, thesis setting (title, alpha 0.7) | inference | 25 | 0.52 [0.32, 0.72] | 0.32 | 0.76 |
| Hybrid, thesis setting (title, alpha 0.7) | compositional | 25 | 0.24 [0.08, 0.40] | 0.60 | 0.60 |
| Hybrid, thesis setting (title, alpha 0.7) | bridge_comparison | 25 | 0.04 [0.00, 0.12] | 0.96 | 1.00 |

Paired difference in EM, hybrid baseline minus closed-book, all 125 dev questions: +0.20 [+0.09, +0.31]. Hybrid baseline minus thesis setting: +0.00 [-0.02, +0.02].

All gold paragraphs inside the packed context (multi-hop, hybrid baseline): 0.53. Mean context size: 1484 tokens.
<!-- END m1-tables -->

What these numbers say so far:

- Retrieval clearly helps over closed-book, including on the questions the model cannot answer from
  memory.
- Comparison questions and the template single-hop questions are already handled by the baseline.
- Bridge-comparison questions are where the baseline fails: the question names two films, but the
  answer depends on their directors, whose pages are not named in the question and rarely reach the
  context. The baseline then correctly abstains. This is the case graph traversal is meant to fix.
- The reranker changes little, and the chosen retrieval configuration and the thesis setting give the
  same QA results.

## 6. Reproducing

Requirements: Python 3.14, Docker, a Gemini API key and an OpenRouter API key.

```
python3.14 -m venv .venv
.venv/bin/pip install -e ".[dev]"
cp .env.example .env                    # fill in the keys and a Neo4j password
docker compose up -d
.venv/bin/python scripts/check_env.py
.venv/bin/python scripts/build_data.py           # download (pinned), sample, corpus, single-hop set
.venv/bin/python scripts/build_answer_aliases.py # official alias file (pinned)
.venv/bin/python scripts/load_corpus.py          # Neo4j + E5 embeddings
.venv/bin/python scripts/run_m1_retrieval.py     # dev retrieval sweep
.venv/bin/python scripts/run_m1_generation.py    # dev QA: closed-book and hybrid
.venv/bin/python scripts/render_tables.py        # re-render the README tables
.venv/bin/pytest
```

## 7. Limitations so far

- Pooled corpus: questions are answered against one pooled corpus of gold and distractor paragraphs.
  This differs from the standard 2Wiki distractor setting, so absolute numbers are not directly
  comparable with published leaderboards. The answer scoring is the official one.
- Small samples: per-type results rest on 25 dev questions each, so the confidence intervals are wide.
- Memorisation: the models have likely seen Wikipedia. The closed-book baseline and the
  closed-book-wrong subset exist to account for that.
- Single-hop questions are generated from templates, so their wording overlaps with the source
  paragraph. That favours lexical retrieval such as BM25. The regenerated single-hop set has not been
  hand-checked a second time.
- Retrieval configuration: all configurations tried on dev are within noise of each other (section 5).
- Answer aliases: some single-hop questions have no official entity ids that line up with their
  evidence, so they are scored against the gold answer only.
- Abstention: "unknown" scores 0. On two-way comparison questions a guess would often be right, so a
  system that abstains can score below a system that guesses. The unknown rate is reported for that
  reason.
- Aggregation questions are out of scope.

## 8. Data, code and licences

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
  `gemini-3.8-flash` are used under their own licences and terms; see their model cards and provider
  terms.
- Retriever design: from my MSc thesis repository, github.com/RohithkumarReddipogula/AI-Powered-Rag-System.

## 9. Work in progress

Next: M2 (entity and relation extraction with a fixed relation list taken from 2Wiki, scored against
the 2Wiki evidence triples), then M3 (entity resolution) and M4 (graph retrieval). Graph results will be
reported next to the M1 baseline in the same table format, including the unknown rate.

---

Rohith Kumar Reddipogula, 2026. Work in progress.
