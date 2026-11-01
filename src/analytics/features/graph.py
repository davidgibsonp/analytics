"""Graph features: weighted edge layers, Louvain communities, their consensus, and nested splits.

Typical flow (all values come from an analysis's config.toml; rows are items, e.g. issues):

    layers = {"text_knn": text_knn(X, k=10, min_cos=0.2), "project": clique(df["project"], 1)}
    A = combine(layers, cfg["edges"]["weights"])              # one weighted adjacency matrix
    G = to_graph(A)
    runs = [louvain(G, resolution, seed) for seed in seeds]
    groups = consensus(runs, share=0.5)                         # 0..k-1, largest first
    groups = absorb_small(A, groups, min_size)                  # tiny groups join their best neighbour
    subgroups = split_within(A, groups, **cfg["planets"])       # Louvain inside each group

Louvain and modularity need networkx: `uv sync --extra graph`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform


def _nx():
    try:
        import networkx
    except ImportError as e:
        raise ImportError("Graph communities need networkx: run `uv sync --extra graph`.") from e
    return networkx


def onehot(lab: np.ndarray) -> sparse.csr_matrix:
    """Membership matrix (rows = items, columns = labels 0..max) of non-negative integer labels."""
    return sparse.csr_matrix((np.ones(len(lab)), (np.arange(len(lab)), lab)))


def incidence(keys) -> sparse.csr_matrix:
    """Item x value matrix; each key is a value, a list of values, or missing."""
    rows, cols, ids = [], [], {}
    for i, v in enumerate(keys):
        for k in v if isinstance(v, (list, np.ndarray)) else ([] if pd.isna(v) else [v]):
            rows.append(i)
            cols.append(ids.setdefault(k, len(ids)))
    return sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(keys), len(ids)))


def clique(keys, w: float) -> sparse.csr_matrix:
    """Shared-value edges; each member's strength from one group is w (pair weight w / (n - 1))."""
    B = incidence(keys)
    n = np.asarray(B.sum(0)).ravel()
    A = (B @ sparse.diags(np.where(n > 1, w / np.maximum(n - 1, 1), 0)) @ B.T).tolil()
    A.setdiag(0)
    return A.tocsr()


def text_knn(X: sparse.csr_matrix, k: int, min_cos: float) -> sparse.csr_matrix:
    """Symmetrized cosine kNN of L2-normalized rows (e.g. TF-IDF), rescaled so the mean strength
    of linked items is 1. Neighbours below min_cos are dropped."""
    S = (X @ X.T).toarray()
    np.fill_diagonal(S, 0)
    r = np.repeat(np.arange(len(S)), k)
    c = np.argsort(-S, axis=1, kind="stable")[:, :k].ravel()
    keep = S[r, c] >= min_cos
    A = sparse.csr_matrix((S[r, c][keep], (r[keep], c[keep])), shape=S.shape)
    A = A.maximum(A.T)
    s = np.asarray(A.sum(1)).ravel()
    return A / s[s > 0].mean()


def combine(layers: dict, weights: dict) -> sparse.csr_matrix:
    """Weighted sum of edge layers; layers missing from `weights` get weight 0."""
    return sum(v * weights.get(k, 0) for k, v in layers.items()).tocsr()


def group_weights(A: sparse.spmatrix, lab: np.ndarray) -> np.ndarray:
    """Total edge weight between every pair of groups (dense, groups x groups)."""
    M = onehot(lab)
    return (M.T @ A @ M).toarray()


def affinity(W: np.ndarray) -> np.ndarray:
    """Normalized group affinity W_ab / sqrt(S_a S_b), S = each group's total edge weight."""
    S = W.sum(1)
    return W / np.sqrt(np.maximum(np.outer(S, S), 1e-12))


def to_graph(A: sparse.spmatrix):
    """networkx graph of a symmetric weighted adjacency matrix (upper triangle, no self-loops)."""
    return _nx().from_scipy_sparse_array(sparse.triu(A, 1).tocsr(), edge_attribute="weight")


def louvain(G, res: float, seed: int) -> np.ndarray:
    """Louvain community per node, numbered largest first."""
    parts = _nx().community.louvain_communities(G, weight="weight", resolution=res, seed=seed)
    lab = np.empty(G.number_of_nodes(), int)
    for k, p in enumerate(sorted(parts, key=len, reverse=True)):
        lab[list(p)] = k
    return lab


def relabel_by_size(lab: np.ndarray) -> np.ndarray:
    """Renumber labels 0..k-1, largest group first."""
    order = pd.Series(lab).value_counts().index
    return pd.Series(lab).map({g: k for k, g in enumerate(order)}).to_numpy()


def consensus(runs: list[np.ndarray], share: float) -> np.ndarray:
    """Average-linkage clusters of the co-assignment share across runs, cut so every group has a
    mean co-assignment of at least `share` (no single-link chaining). One run returns itself."""
    D = 1 - sum((lambda M: M @ M.T)(onehot(r)) for r in runs).toarray() / len(runs)
    np.fill_diagonal(D, 0)
    return relabel_by_size(fcluster(linkage(squareform(D, checks=False), "average"), 1 - share, "distance"))


def absorb_small(A: sparse.spmatrix, lab: np.ndarray, min_size: int) -> np.ndarray:
    """Groups under min_size join the large group with the highest normalized affinity.
    Groups with no edge to a large group stay as they are."""
    aff, n = affinity(group_weights(A, lab)), np.bincount(lab)
    aff[:, n < min_size] = 0
    target = np.where((n < min_size) & (aff.max(1) > 0), aff.argmax(1), np.arange(len(n)))
    return target[lab]


def modularity(A: sparse.spmatrix, lab: np.ndarray, res: float = 1.0) -> float:
    G = to_graph(A)
    return _nx().community.modularity(
        G, [set(np.flatnonzero(lab == k)) for k in np.unique(lab)], "weight", res
    )


def split(
    sub: sparse.spmatrix, *, resolution: float, min_size: int, seed: int, min_q: float = -1.0, **_: object
) -> np.ndarray:
    """Louvain groups of one (sub)graph, small ones absorbed; kept whole (all 0) when the split's
    modularity is below min_q."""
    if sub.shape[0] < 2:
        return np.zeros(sub.shape[0], int)
    local = relabel_by_size(absorb_small(sub, louvain(to_graph(sub), resolution, seed), min_size))
    return local if modularity(sub, local) >= min_q else np.zeros(sub.shape[0], int)


def split_within(
    A: sparse.spmatrix,
    groups: np.ndarray,
    *,
    resolution: float,
    min_size: int,
    max_size: int,
    min_split_modularity: float,
    seed: int,
) -> np.ndarray:
    """Subgroups of each group (labels >= 0) from Louvain on its own subgraph; a subgroup over
    max_size is split once more if that split reaches min_split_modularity. Numbered
    consecutively, largest first within each group; items with a negative group get -1."""
    kw = dict(resolution=resolution, min_size=min_size, seed=seed)
    out, nxt = np.full(len(groups), -1), 0
    for g in np.unique(groups[groups >= 0]):
        nodes = np.flatnonzero(groups == g)
        sub = A[nodes][:, nodes]
        local = split(sub, **kw).copy()
        for big in [k for k in np.unique(local) if (local == k).sum() > max_size]:
            inner = np.flatnonzero(local == big)
            local[inner] = split(sub[inner][:, inner], **kw, min_q=min_split_modularity) + local.max() + 1
        local = relabel_by_size(local)
        out[nodes] = local + nxt
        nxt += local.max() + 1
    return out
