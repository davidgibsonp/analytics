"""Text features: TF-IDF, a 2D similarity layout, and clusters named by their top terms.

Typical flow (all values come from an analysis's config.toml):

    text = document_text(df, ["title"])
    X, terms = tfidf(text, **cfg["tfidf"])
    xy, lsa = layout_2d(X, **cfg["embed"])        # rows with no terms get NaN
    labels = cluster(xy, lsa, **cfg["cluster"])
    names = cluster_names(X, labels, terms, n=3)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE

LETTER_WORDS = r"(?u)\b[a-zA-Z][a-zA-Z]+\b"  # skips dates, ids and numbers


def document_text(df: pd.DataFrame, fields: list[str]) -> pd.Series:
    """Join text fields per row; list columns (e.g. labels) are space-joined."""
    parts = []
    for field in fields:
        col = df[field]
        if col.map(lambda v: isinstance(v, (list, np.ndarray))).any():
            parts.append(col.map(lambda v: " ".join(v) if isinstance(v, (list, np.ndarray)) else ""))
        else:
            parts.append(col.astype("string").fillna(""))
    return pd.concat(parts, axis=1).agg(" ".join, axis=1).str.strip()


def tfidf(
    text: pd.Series,
    *,
    ngram_max: int = 2,
    min_df: int | float = 2,
    max_df: int | float = 1.0,
    sublinear_tf: bool = True,
    stop_words: str = "english",
    token_pattern: str = LETTER_WORDS,
    **_: object,
) -> tuple[sparse.csr_matrix, np.ndarray]:
    """TF-IDF matrix (rows = text rows) and its vocabulary."""
    vec = TfidfVectorizer(
        ngram_range=(1, ngram_max),
        min_df=min_df,
        max_df=max_df,
        sublinear_tf=sublinear_tf,
        stop_words=stop_words or None,
        token_pattern=token_pattern,
    )
    X = vec.fit_transform(text).tocsr()
    return X, vec.get_feature_names_out()


def top_terms(X: sparse.csr_matrix, terms: np.ndarray, n: int = 5) -> list[list[str]]:
    """The n highest-weighted terms of each row."""
    out = []
    for i in range(X.shape[0]):
        row = X.getrow(i)
        out.append([str(terms[j]) for j in row.indices[np.argsort(row.data)[::-1][:n]]])
    return out


def layout_2d(
    X: sparse.csr_matrix,
    *,
    method: str = "tsne",
    svd_components: int = 50,
    perplexity: float = 30,
    seed: int = 42,
    **_: object,
) -> tuple[np.ndarray, np.ndarray]:
    """2D coordinates (n x 2) and unit-length LSA vectors (n x k).

    Rows with no terms can't be placed: they get NaN in both outputs.
    method: "tsne" (neighborhoods preserved, distances between groups meaningless)
            or "svd" (first two LSA components, global structure, more overlap).
    """
    placed = np.flatnonzero(X.getnnz(axis=1) > 0)
    Xp = X[placed]
    k = max(2, min(svd_components, Xp.shape[1] - 1, Xp.shape[0] - 1))
    Z = TruncatedSVD(n_components=k, random_state=seed).fit_transform(Xp)
    Z /= np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-9)
    if method == "svd":
        xy = Z[:, :2]
    elif method == "tsne":
        xy = TSNE(
            n_components=2,
            metric="cosine",
            perplexity=min(perplexity, (len(placed) - 1) / 3),
            init="pca",
            random_state=seed,
        ).fit_transform(Z)
    else:
        raise ValueError(f"layout method must be 'tsne' or 'svd', got {method!r}")
    full_xy = np.full((X.shape[0], 2), np.nan)
    full_z = np.full((X.shape[0], k), np.nan)
    full_xy[placed], full_z[placed] = xy, Z
    return full_xy, full_z


def cluster(
    xy: np.ndarray,
    lsa: np.ndarray,
    *,
    method: str = "hdbscan",
    space: str = "2d",
    min_cluster_size: int = 15,
    k: int = 12,
    seed: int = 42,
    **_: object,
) -> np.ndarray:
    """Cluster ids per row: -1 = unclustered (hdbscan noise), -2 = unplaced (NaN input)."""
    points = xy if space == "2d" else lsa
    if space not in ("2d", "lsa"):
        raise ValueError(f"cluster space must be '2d' or 'lsa', got {space!r}")
    ok = ~np.isnan(points).any(axis=1)
    labels = np.full(len(points), -2)
    if method == "none":
        labels[ok] = -1
    elif method == "hdbscan":
        labels[ok] = HDBSCAN(min_cluster_size=min_cluster_size, copy=True).fit_predict(points[ok])
    elif method == "kmeans":
        labels[ok] = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(points[ok])
    else:
        raise ValueError(f"cluster method must be hdbscan, kmeans or none, got {method!r}")
    return labels


def cluster_names(X: sparse.csr_matrix, labels: np.ndarray, terms: np.ndarray, n: int = 3) -> dict[int, str]:
    """Name each cluster by the top-n terms of its mean TF-IDF vector."""
    names = {-1: "unclustered", -2: "unplaced"}
    for c in sorted(set(labels.tolist()) - {-1, -2}):
        weights = np.asarray(X[labels == c].mean(axis=0)).ravel()
        names[c] = ", ".join(terms[weights.argsort()[::-1][:n]])
    return names
