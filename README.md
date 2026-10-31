# analytics

Analytical projects for David's work and systems. One repository, many analyses:
shared code for getting and shaping data lives in `src/analytics/`; each question
gets a thin folder in `analyses/`.

## Quick start

```bash
uv sync                                    # core deps (+ dev tools)
export LINEAR_KEY=...                      # or put it in .env (git-ignored)
uv run analytics fetch linear              # snapshot raw data into data/raw/linear/
uv run analytics run linear-issue-map      # run an analysis
uv run analytics list                      # list analyses
```

Override any config value for one run without editing files:

```bash
uv run analytics run linear-issue-map --set tfidf.min_df=3 --set cluster.method=kmeans
```

## Layout

```
src/analytics/      shared library: config, CLI, sources/, features/, viz/
analyses/<name>/    one folder per question: README (question → answer), config.toml, run.py
analyses/_template/ copy this to start a new analysis
docs/data/          data dictionaries, one per source
data/               git-ignored raw snapshots (data/raw/<source>/<dataset>/<date>/)
scratch/            git-ignored throwaway exploration
tests/              fast tests on tiny fixtures
```

## Dependencies

Core: pandas, numpy, pyarrow, scikit-learn, duckdb, matplotlib, httpx.
Optional extras: `embeddings` (sentence-transformers, pulls in PyTorch), `viz` (plotly).
Dev: pytest, ruff, pre-commit.

## Checks

```bash
uv run ruff check . && uv run ruff format --check . && uv run pytest
uv run pre-commit install                  # run the checks on every commit
```
