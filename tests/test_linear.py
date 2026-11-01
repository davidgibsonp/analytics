import json
from pathlib import Path

from analytics.sources.linear import flatten, to_frame

PAGE = json.loads((Path(__file__).parent / "fixtures" / "linear_issues_page.json").read_text())
NODES = PAGE["data"]["issues"]["nodes"]


def test_flatten_full_issue():
    row = flatten(NODES[0])
    assert row["id"] == "XYZ-1"
    assert row["statusType"] == "completed"
    assert row["labels"] == ["Improvement", "Research"]
    assert row["estimate_size"] == "M"
    assert (row["project"], row["milestone"], row["cycle"]) == ("stack", "v0.1", 4)


def test_flatten_sparse_issue():
    row = flatten(NODES[1])
    assert row["project"] is None and row["assignee"] is None and row["estimate_size"] is None
    assert row["labels"] == [] and row["parentId"] == "ABC-1"


def test_to_frame_types_and_order():
    df = to_frame([flatten(n) for n in NODES])
    assert list(df["id"]) == ["ABC-7", "XYZ-1"]
    assert str(df["createdAt"].dt.tz) == "UTC"
    assert df["completedAt"].isna().sum() == 1
    assert str(df["cycle"].dtype) == "Int64"
