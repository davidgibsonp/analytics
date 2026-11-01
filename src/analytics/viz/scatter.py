"""Category-colored scatter with optional group labels."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from analytics.viz.style import (
    CATEGORICAL,
    GRID,
    OTHER,
    SCATTER_MAX_CATEGORIES,
    SURFACE,
    TEXT,
    finish,
    fold_categories,
    new_figure,
)


def scatter(
    df: pd.DataFrame,
    x: str,
    y: str,
    color_by: str,
    path: Path | str,
    *,
    title: str,
    xlabel: str = "",
    ylabel: str = "",
    hide_ticks: bool = False,
    group_by: str | None = None,
    group_names: dict | None = None,
    label_groups: int = 0,
) -> Path:
    """Scatter of df[x] vs df[y], one color per category of `color_by` (max 3, rest -> Other).

    Rows with NaN x/y are skipped. If `group_by` is set, the `label_groups` largest groups
    (ids >= 0) get a label above them: group_names[id] plus the group size.
    """
    df = df.dropna(subset=[x, y])
    values, order = fold_categories(df[color_by], SCATTER_MAX_CATEGORIES)
    colors = dict(zip(order, [*CATEGORICAL[:SCATTER_MAX_CATEGORIES], OTHER], strict=False))
    fig, ax = new_figure()
    for name in order:
        d = df[values == name]
        ax.scatter(
            d[x],
            d[y],
            s=30,
            c=colors[name],
            edgecolors=SURFACE,
            linewidths=1.1,
            alpha=0.9,
            label=f"{name} ({len(d)})",
            zorder=3,
        )
    if group_by and label_groups:
        groups = df[df[group_by] >= 0]
        for g in groups[group_by].value_counts().index[:label_groups]:
            d = groups[groups[group_by] == g]
            text = f"{(group_names or {}).get(g, g)}  (n={len(d)})"
            ax.annotate(
                text,
                (d[x].mean(), d[y].max()),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8.5,
                color=TEXT,
                zorder=5,
                bbox={"boxstyle": "round,pad=0.25", "fc": SURFACE, "ec": GRID, "alpha": 0.88},
            )
    if hide_ticks:
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        ax.grid(color=GRID, lw=0.8)
        ax.set_axisbelow(True)
    path = Path(path)
    finish(fig, ax, title, xlabel, ylabel, path, legend_columns=len(order))
    return path
