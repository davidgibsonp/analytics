"""Linear issues via the GraphQL API.

Snapshots land in data/raw/linear/issues/<YYYY-MM-DD>/issues.parquet with a `latest` link.
Columns are documented in docs/data/linear.md; keep that file in step with flatten().
"""

from __future__ import annotations

import os
import sys
import time
from datetime import UTC, datetime

import httpx
import pandas as pd

from analytics.config import DATA_DIR, save_run_config

API_URL = "https://api.linear.app/graphql"
API_KEY_ENV = "LINEAR_KEY"
PAGE_SIZE = 100
ISSUES_DIR = DATA_DIR / "raw" / "linear" / "issues"
DATE_COLUMNS = ["createdAt", "updatedAt", "startedAt", "completedAt", "canceledAt", "archivedAt", "dueDate"]
TSHIRT = {0: "No estimate", 1: "XS", 2: "S", 3: "M", 5: "L", 8: "XL", 13: "XXL", 21: "XXXL"}

QUERY = """
query Issues($first: Int!, $after: String, $filter: IssueFilter, $archived: Boolean) {
  issues(first: $first, after: $after, filter: $filter, includeArchived: $archived, orderBy: createdAt) {
    pageInfo { hasNextPage endCursor }
    nodes {
      identifier title priority priorityLabel estimate
      createdAt updatedAt startedAt completedAt canceledAt archivedAt dueDate
      state { name type }
      team { key name issueEstimationType }
      project { name }
      projectMilestone { name }
      labels { nodes { name } }
      assignee { name }
      parent { identifier }
      cycle { number }
    }
  }
}"""


def flatten(node: dict) -> dict:
    """One API issue node -> one flat row of raw values."""

    def get(obj: dict | None, key: str):
        return (obj or {}).get(key)

    est = node["estimate"]
    tshirt = get(node["team"], "issueEstimationType") == "tShirt"
    return {
        "id": node["identifier"],
        "title": node["title"],
        "status": get(node["state"], "name"),
        "statusType": get(node["state"], "type"),
        "team": get(node["team"], "name"),
        "team_key": get(node["team"], "key"),
        "project": get(node["project"], "name"),
        "milestone": get(node["projectMilestone"], "name"),
        "labels": sorted(label["name"] for label in node["labels"]["nodes"]),
        "priority": node["priorityLabel"],
        "priority_value": node["priority"],
        "estimate": est,
        "estimate_size": TSHIRT.get(int(est)) if est is not None and tshirt else None,
        "assignee": get(node["assignee"], "name"),
        "parentId": get(node["parent"], "identifier"),
        "cycle": get(node["cycle"], "number"),
        **{c: node[c] for c in DATE_COLUMNS},
    }


def to_frame(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    for c in DATE_COLUMNS:
        df[c] = pd.to_datetime(df[c], utc=True, errors="coerce")
    df["estimate"] = pd.to_numeric(df["estimate"], errors="coerce")
    df["cycle"] = df["cycle"].astype("Int64")
    return df.sort_values("createdAt").reset_index(drop=True)


def fetch_issues(
    api_key: str, teams: list[str] | None = None, include_archived: bool = False
) -> pd.DataFrame:
    gql_filter = {"team": {"key": {"in": teams}}} if teams else None
    rows: list[dict] = []
    after = None
    with httpx.Client(timeout=60, headers={"Authorization": api_key}) as client:
        while True:
            variables = {
                "first": PAGE_SIZE,
                "after": after,
                "filter": gql_filter,
                "archived": include_archived,
            }
            for attempt in range(4):
                r = client.post(API_URL, json={"query": QUERY, "variables": variables})
                if r.status_code < 500 and r.status_code != 429:
                    break
                time.sleep(2 ** (attempt + 1))
            r.raise_for_status()
            body = r.json()
            if body.get("errors"):
                raise RuntimeError(f"Linear API error: {body['errors']}")
            conn = body["data"]["issues"]
            rows += [flatten(n) for n in conn["nodes"]]
            print(f"  fetched {len(rows)} issues", file=sys.stderr)
            if not conn["pageInfo"]["hasNextPage"]:
                return to_frame(rows)
            after = conn["pageInfo"]["endCursor"]


def fetch(args) -> None:
    """CLI handler for `analytics fetch linear`."""
    key = os.environ.get(API_KEY_ENV)
    if not key:
        sys.exit(f"Set {API_KEY_ENV} (environment or .env) to a Linear API key.")
    df = fetch_issues(key, args.teams, args.include_archived)
    now = datetime.now(UTC)
    out = ISSUES_DIR / now.strftime("%Y-%m-%d")
    out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out / "issues.parquet", index=False)
    save_run_config(
        {"teams": args.teams, "include_archived": args.include_archived},
        out / "fetch_meta.json",
        fetched_at=now.isoformat(),
        rows=len(df),
    )
    latest = ISSUES_DIR / "latest"
    latest.unlink(missing_ok=True)
    latest.symlink_to(out.name)
    print(f"wrote {len(df)} issues -> {out / 'issues.parquet'}")


def load_issues(snapshot: str = "latest") -> pd.DataFrame:
    """Load a raw issues snapshot by date folder name, or "latest"."""
    path = ISSUES_DIR / snapshot / "issues.parquet"
    if not path.exists():
        sys.exit(f"No Linear snapshot at {path}. Run: uv run analytics fetch linear")
    return pd.read_parquet(path)
