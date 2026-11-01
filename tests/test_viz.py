import numpy as np
import pandas as pd

from analytics.viz.scatter import scatter
from analytics.viz.style import fold_categories


def test_fold_categories_keeps_top_and_folds_rest():
    values, order = fold_categories(pd.Series(list("aaabbcd") + [None]), 2)
    assert order == ["a", "b", "Other"]
    assert values.tolist().count("Other") == 3


def test_fold_categories_no_fold_when_few():
    _, order = fold_categories(pd.Series(["x", "y"]), 3)
    assert order == ["x", "y"]


def test_scatter_writes_png_and_skips_nan(tmp_path):
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {"x": rng.random(40), "y": rng.random(40), "team": list("abcde") * 8, "g": [0] * 20 + [1] * 20}
    )
    df.loc[0, "x"] = np.nan
    out = scatter(
        df,
        "x",
        "y",
        "team",
        tmp_path / "s.png",
        title="t",
        group_by="g",
        group_names={0: "zero", 1: "one"},
        label_groups=2,
    )
    assert out.exists() and out.stat().st_size > 10_000
