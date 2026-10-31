# %% [markdown]
# Exploration for linear-issue-map. Run cell by cell (VS Code / Cursor), after
# `uv run analytics run linear-issue-map`. Keep it short; move anything reusable to src/.

# %%
from pathlib import Path

import duckdb

OUT = Path(__file__).parent / "outputs"
issues = f"'{OUT / 'issues.parquet'}'"

# %% Clusters by size and dominant team
duckdb.sql(f"""
    select tfidf_cluster, any_value(tfidf_cluster_terms) as terms, count(*) as n, mode(team) as top_team
    from {issues} where tfidf_cluster >= 0 group by 1 order by n desc
""").show()

# %% Titles in one cluster
duckdb.sql(f"select id, team, title from {issues} where tfidf_cluster = 1 limit 20").show()

# %% Issues that couldn't be placed
duckdb.sql(f"select id, title from {issues} where tfidf_nnz = 0").show()
