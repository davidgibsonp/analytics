import numpy as np
import pandas as pd
import pytest

from analytics.features.text import cluster, cluster_names, document_text, layout_2d, tfidf, top_terms

TOPICS = ["daily debrief notes", "fix login auth bug", "research clustering method", "plan garden beds"]


@pytest.fixture
def corpus():
    titles = [f"{t} {i}" for t in TOPICS for i in range(12)] + ["zzz", "2026-10-31"]
    return pd.DataFrame({"title": titles, "labels": [["a"]] * (len(titles) - 1) + [[]]})


def test_document_text_joins_lists_and_strings(corpus):
    text = document_text(corpus, ["title", "labels"])
    assert text.iloc[0] == "daily debrief notes 0 a"
    assert text.iloc[-1] == "2026-10-31"


def test_tfidf_ignores_numbers_and_rare_words(corpus):
    X, terms = tfidf(corpus["title"], min_df=2)
    assert X.shape[0] == len(corpus)
    assert "zzz" not in terms and not any(t[0].isdigit() for t in terms)
    assert X[-2:].nnz == 0
    assert top_terms(X, terms, n=2)[0][0] in {"daily", "debrief", "notes", "daily debrief", "debrief notes"}


@pytest.mark.parametrize("method", ["tsne", "svd"])
def test_layout_marks_unplaced_rows_nan(corpus, method):
    X, _ = tfidf(corpus["title"], min_df=2)
    xy, lsa = layout_2d(X, method=method, svd_components=5, perplexity=5, seed=0)
    assert xy.shape == (len(corpus), 2)
    assert np.isnan(xy[-2:]).all() and not np.isnan(xy[:-2]).any()


def test_kmeans_recovers_topics_and_names_them(corpus):
    X, terms = tfidf(corpus["title"], min_df=2)
    xy, lsa = layout_2d(X, method="svd", svd_components=5, seed=0)
    labels = cluster(xy, lsa, method="kmeans", space="lsa", k=4, seed=0)
    assert (labels[-2:] == -2).all()
    groups = [set(labels[i * 12 : (i + 1) * 12]) for i in range(4)]
    assert all(len(g) == 1 for g in groups) and len(set.union(*groups)) == 4
    names = cluster_names(X, labels, terms, n=2)
    assert any("debrief" in name for name in names.values())


def test_bad_method_raises(corpus):
    X, _ = tfidf(corpus["title"], min_df=2)
    with pytest.raises(ValueError):
        layout_2d(X, method="umap")
