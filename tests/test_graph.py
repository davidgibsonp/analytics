import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from analytics.features import graph

SIZES = [30, 20, 12]


@pytest.fixture
def nx():
    """Louvain and modularity need the graph extra; skip those tests without it."""
    return pytest.importorskip("networkx")


@pytest.fixture
def planted():
    """Three dense blocks with a few weak links between them; returns (adjacency, true labels)."""
    rng = np.random.default_rng(0)
    truth = np.repeat(np.arange(len(SIZES)), SIZES)
    same = truth[:, None] == truth[None, :]
    upper = np.triu(rng.random((len(truth),) * 2) < np.where(same, 0.6, 0.02), 1)
    W = (upper | upper.T).astype(float)
    return sparse.csr_matrix(W), truth


def test_consensus_recovers_blocks_deterministically(planted, nx):
    A, truth = planted

    def run():  # a fresh graph and fresh Louvain runs with the same seeds
        G = graph.to_graph(A)
        return graph.consensus([graph.louvain(G, 1.0, seed) for seed in range(5)], 0.5)

    first, again = run(), run()
    assert np.array_equal(first, again)
    assert np.array_equal(first, truth)  # blocks are numbered largest first, like the truth


def test_consensus_of_one_run_is_that_run():
    run = np.array([1, 1, 0, 0, 0, 2])
    assert np.array_equal(graph.consensus([run], 0.5), graph.relabel_by_size(run))


def test_relabel_by_size_orders_largest_first():
    assert graph.relabel_by_size(np.array([5, 7, 7, 7, 5, 9])).tolist() == [1, 0, 0, 0, 1, 2]


def test_absorb_small_joins_best_linked_large_group(planted):
    A, truth = planted
    lab = truth.copy()
    lab[:3] = 3  # a tiny group carved out of block 0: most of its edges go to block 0
    out = graph.absorb_small(A, lab, min_size=10)
    assert np.array_equal(out, truth)


def test_absorb_small_keeps_unlinked_group():
    A = sparse.csr_matrix(np.kron(np.eye(2), np.ones((3, 3))) - np.eye(6))
    lab = np.array([0, 0, 0, 1, 1, 1])
    assert np.array_equal(graph.absorb_small(A, lab, min_size=4), lab)


def test_modularity_sanity(planted, nx):
    A, truth = planted
    assert graph.modularity(A, truth) > 0.4
    assert graph.modularity(A, np.zeros(len(truth), int)) == pytest.approx(0.0, abs=1e-12)


def test_split_gate_keeps_group_whole_below_min_q(planted, nx):
    A, truth = planted
    kw = dict(resolution=1.0, min_size=5, seed=0)
    assert np.array_equal(graph.split(A, **kw), truth)
    assert not graph.split(A, **kw, min_q=0.99).any()


def test_split_within_numbers_subgroups_consecutively(planted, nx):
    A, truth = planted
    groups = np.where(truth == 2, -1, 0)  # blocks 0 and 1 form one group; block 2 is left out
    out = graph.split_within(
        A, groups, resolution=1.0, min_size=5, max_size=100, min_split_modularity=0.2, seed=0
    )
    assert (out[truth == 2] == -1).all()
    assert np.array_equal(out[truth < 2], truth[truth < 2])


def test_clique_gives_each_member_strength_w():
    A = graph.clique(pd.Series(["a", "a", "a", "b", None, "b"]), 2.0)
    assert np.allclose(np.asarray(A.sum(1)).ravel(), [2, 2, 2, 2, 0, 2])
    assert A.diagonal().sum() == 0


def test_text_knn_symmetric_with_unit_mean_strength():
    rng = np.random.default_rng(1)
    X = sparse.csr_matrix(rng.random((20, 6)))
    X = sparse.csr_matrix(X / np.linalg.norm(X.toarray(), axis=1, keepdims=True))
    A = graph.text_knn(X, k=3, min_cos=0.0)
    assert (A != A.T).nnz == 0
    s = np.asarray(A.sum(1)).ravel()
    assert s[s > 0].mean() == pytest.approx(1.0)


def test_combine_ignores_unweighted_layers():
    one = sparse.csr_matrix(np.ones((2, 2)))
    assert graph.combine({"a": one, "b": one * 5}, {"a": 2.0}).toarray().tolist() == [[2, 2], [2, 2]]
