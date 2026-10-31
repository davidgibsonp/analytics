"""linear-issue-map: place Linear issues by text similarity, cluster them, and plot the map.

Outputs (in outputs/): issues.parquet (raw columns + derived), tfidf.npz + tfidf_meta.json,
random_scatter.png, tfidf_scatter.png, run_config.json.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from analytics.config import save_run_config
from analytics.features import text
from analytics.sources.linear import load_issues
from analytics.viz.scatter import scatter

PRIORITY_ORDER = ["Urgent", "High", "Medium", "Low", "No priority"]


def select(df: pd.DataFrame, f: dict) -> pd.DataFrame:
    if f["teams"]:
        df = df[df["team"].isin(f["teams"]) | df["team_key"].isin(f["teams"])]
    if f["status_types"]:
        df = df[df["statusType"].isin(f["status_types"])]
    if f["created_after"]:
        df = df[df["createdAt"] >= pd.Timestamp(f["created_after"], tz="UTC")]
    df = df.reset_index(drop=True)
    for c in ["team", "status", "statusType"]:
        df[c] = df[c].astype("category")
    df["priority"] = pd.Categorical(df["priority"], categories=PRIORITY_ORDER, ordered=True)
    df["n_labels"] = df["labels"].str.len()
    df["days_to_complete"] = (df["completedAt"] - df["createdAt"]).dt.total_seconds() / 86400
    return df


def main(cfg: dict) -> None:
    out = Path(cfg["paths"]["outputs"])
    out.mkdir(parents=True, exist_ok=True)
    df = select(load_issues(cfg["data"]["snapshot"]), cfg["filter"])

    rng = np.random.default_rng(cfg["random"]["seed"])
    df["x_random"], df["y_random"] = rng.uniform(0, 1, len(df)), rng.uniform(0, 1, len(df))

    t = cfg["tfidf"]
    X, terms = text.tfidf(text.document_text(df, t["text_fields"]), **t)
    xy, lsa = text.layout_2d(X, **cfg["embed"])
    labels = text.cluster(xy, lsa, **cfg["cluster"])
    names = text.cluster_names(X, labels, terms, cfg["cluster"]["terms_n"])
    df["tfidf_nnz"] = X.getnnz(axis=1)
    df["tfidf_top_terms"] = text.top_terms(X, terms, t["top_terms_n"])
    df["x_tfidf"], df["y_tfidf"] = xy[:, 0], xy[:, 1]
    df["tfidf_cluster"] = labels
    df["tfidf_cluster_terms"] = df["tfidf_cluster"].map(names)

    df.to_parquet(out / "issues.parquet", index=False)
    sparse.save_npz(out / "tfidf.npz", X)
    (out / "tfidf_meta.json").write_text(json.dumps({"row_ids": df["id"].tolist(), "vocab": terms.tolist()}))

    p, fields, method = cfg["plot"], " + ".join(t["text_fields"]), cfg["embed"]["method"]
    scatter(
        df,
        "x_random",
        "y_random",
        p["color_by"],
        out / "random_scatter.png",
        title=f"All {len(df):,} Linear issues at random positions, colored by {p['color_by']}",
        xlabel="random x",
        ylabel="random y",
    )
    placed = int(df["x_tfidf"].notna().sum())
    scatter(
        df,
        "x_tfidf",
        "y_tfidf",
        p["color_by"],
        out / "tfidf_scatter.png",
        title=f"{placed:,} Linear issues placed by TF-IDF similarity of {fields}, colored by {p['color_by']}",
        xlabel=f"{method} 1 (no units: nearby = similar wording)",
        ylabel=f"{method} 2",
        hide_ticks=True,
        group_by="tfidf_cluster",
        group_names=names,
        label_groups=p["label_clusters"],
    )

    n_clusters = len(set(labels.tolist()) - {-1, -2})
    stats = {
        "rows": len(df),
        "terms": len(terms),
        "placed": placed,
        "clusters": n_clusters,
        "unclustered": int((labels == -1).sum()),
    }
    save_run_config(cfg, out / "run_config.json", stats=stats)
    print(f"{stats} -> {out}")
