# Agent instructions

This repository is optimized for coding agents doing analytics. Favor small, reused,
readable code over new code.

## Before writing code

1. Read this file, then the README of the analysis you are working on.
2. Read `docs/data/<source>.md` before inspecting a dataset; it lists columns, types and quirks.
   Skim `docs/lessons.md` for harness and method pitfalls already paid for.
3. Check `src/analytics/` for a function that already does the job. Reuse it.

## Rules

- **One entry point.** Run things with `uv run analytics ...`. Do not add new runner scripts.
- **Thin analyses.** An analysis is `README.md`, `config.toml`, `run.py` (aim for ~100 lines or
  fewer), optionally `explore.py`, and committed `figures/`. Anything generic (a chart type, a
  clustering method) goes to `src/analytics/` with tests. Analysis-specific rules that don't fit in
  ~100 lines may stay in `run.py` plus at most one sibling module (e.g. `checks.py` for a
  validation suite); say why in the README. Logic used by a second analysis moves to
  `src/analytics/` (rule of two: not before).
- **Every tunable value goes in `config.toml`.** No magic numbers in code. Functions take
  dataframes and config values, and return dataframes or arrays.
- **No notebooks.** Explore in `.py` files with `# %%` cells. Throwaway work goes in `scratch/`
  (git-ignored).
- **No data or secrets in git; images only.** Raw snapshots live in `data/` and run outputs in
  `analyses/*/outputs/` (both git-ignored). The only results committed are chart images in
  `analyses/*/figures/`. Never commit parquet, csv, npz, pickles or API dumps; the one exception
  is tiny synthetic fixtures in `tests/fixtures/`. Keys come from the environment (`.env.example`
  lists the names).
- **No new dependencies without asking the user.** Heavy ones go in an optional extra.
- **Look at every chart you make.** Open the PNG and check labels, overlap and counts before
  reporting.
- **Findings go in the analysis README** as question → answer → evidence → how to rerun.
  Update it when results change.
- **Update the data dictionary** when a source gains or changes a column.

## Before reporting completion

```bash
uv run ruff check . && uv run ruff format --check . && uv run pytest
```
