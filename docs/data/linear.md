# Linear issues

Raw snapshot from `uv run analytics fetch linear`, one row per issue.

- **Path:** `data/raw/linear/issues/<YYYY-MM-DD>/issues.parquet` (`latest` points at the newest);
  `fetch_meta.json` beside it records when it ran, the options and the row count.
- **Load:** `analytics.sources.linear.load_issues(snapshot="latest")`.
- **Query with SQL:** `duckdb -c "select team, count(*) from 'data/raw/linear/issues/latest/issues.parquet' group by 1"`
  (or `duckdb.sql(...)` in Python).
- **Scope:** all teams (XYZ = work, ABC = automation output, MaiN = personal); archived issues
  are excluded unless fetched with `--include-archived`.

| Column | Type | Notes |
|---|---|---|
| `id` | str | Issue identifier, e.g. `XYZ-1`. Unique. |
| `title` | str | |
| `status` | str | Workflow state name, e.g. `Done`, `In Review`. Names differ by team. |
| `statusType` | str | Normalized state: `triage`, `backlog`, `unstarted`, `started`, `completed`, `canceled`, `duplicate`. Prefer this for analysis. |
| `team` | str | Team name (`XYZ`, `ABC`, `MaiN`). |
| `team_key` | str | Team key (`XYZ`, `ABC`, `MAIN`). |
| `project` | str, nullable | Project name. ~11% of issues have none. |
| `milestone` | str, nullable | Project milestone name. Sparse (~23%). |
| `labels` | list[str] | Sorted label names; empty list when unlabeled. |
| `priority` | str | `Urgent`, `High`, `Medium`, `Low`, `No priority`. |
| `priority_value` | int | 0 = none, 1 = urgent … 4 = low. Not monotonic in urgency because of 0. |
| `estimate` | float, nullable | Points. Sparse (~33%). |
| `estimate_size` | str, nullable | T-shirt name (XS…XXXL) when the team uses t-shirt estimates. |
| `assignee` | str, nullable | Display name. |
| `parentId` | str, nullable | Parent issue identifier for sub-issues. |
| `cycle` | Int64, nullable | Cycle number within the team. |
| `createdAt`, `updatedAt` | datetime (UTC) | Always set. |
| `startedAt`, `completedAt`, `canceledAt` | datetime (UTC), nullable | Set when the issue entered that state type. |
| `archivedAt` | datetime (UTC), nullable | Only set with `--include-archived`. |
| `dueDate` | datetime (UTC), nullable | Date only (midnight UTC). Sparse (~9%). |

## Quirks

- ABC is dominated by templated automation issues (`Debrief - <date>`, ~140 near-identical titles);
  they can swamp text analyses. Filter by `team` or labels when that matters.
- Some titles are bare hashes (e.g. `ABC-62`) or single words, and carry no usable text.
