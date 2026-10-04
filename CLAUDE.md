# CLAUDE.md

GraphRAG engine benchmarked on 2WikiMultiHopQA. The plan and every settled design decision live in
`docs/PLAN.md`. Read it before starting work, and do not reopen settled decisions without asking.

## Writing rules (docs, README, comments, commit messages)

- No em dashes and no emojis anywhere.
- README: plain ASCII only, numbered table of contents, no collapsible sections (`<details>`), every
  results table fully visible, short plain footer.

## Secrets

- Never print, cat, grep or otherwise display `.env` or any `.env.*` file.
- Never put API keys in code, logs, test fixtures, result files or commit messages.
- `.env` and `.env.*` are in `.gitignore`. Only `.env.example` (no real values) is committed.

## Numbers

- Never invent numbers. Every number in a doc or README must come from a committed file under
  `results/` (or `data/` for dataset statistics), and should name that file.

## Workflow

- Run `pytest` before every commit. Do not commit if tests fail.
- Tune only on the dev split. The test split is run once, at M5, with frozen config.
- Every LLM call goes through the disk cache in `src/graphrag/llm/`.
