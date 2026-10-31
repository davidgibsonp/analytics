"""Shared chart style: surfaces, text colors, and the categorical palette.

Categorical colors are assigned in this fixed order and never cycled. Scatter-type charts
(every pair of colors can sit side by side) are validated for the first three slots only:
past three, fold the rest into "Other" (see fold_categories).
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402  (installed with matplotlib)

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e8e7e3"
OTHER = "#9a9893"

CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SCATTER_MAX_CATEGORIES = 3


def fold_categories(
    values: pd.Series, max_categories: int, other: str = "Other"
) -> tuple[pd.Series, list[str]]:
    """Keep the most common categories, fold the rest into `other`. Returns (values, ordered names)."""
    values = values.astype("string").fillna("(none)")
    order = values.value_counts().index.tolist()
    if len(order) <= max_categories:
        return values, order
    keep = order[:max_categories]
    return values.where(values.isin(keep), other), [*keep, other]


def new_figure(width: float = 12, height: float = 9, dpi: int = 150):
    fig, ax = plt.subplots(figsize=(width, height), dpi=dpi, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY, length=0)
    return fig, ax


def finish(fig, ax, title: str, xlabel: str, ylabel: str, path, legend_columns: int | None = None) -> None:
    """Title top-left, legend in one row under it, recessive axes; save and close."""
    ax.set_title(title, loc="left", color=TEXT, fontsize=13, pad=30)
    ax.set_xlabel(xlabel, color=TEXT_SECONDARY)
    ax.set_ylabel(ylabel, color=TEXT_SECONDARY)
    if ax.get_legend_handles_labels()[0]:
        legend = ax.legend(
            loc="lower left",
            bbox_to_anchor=(0, 1.0),
            ncol=legend_columns or 6,
            frameon=False,
            fontsize=10,
            handletextpad=0.3,
            columnspacing=1.5,
            borderaxespad=0.3,
        )
        for t in legend.get_texts():
            t.set_color(TEXT_SECONDARY)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def optimize_png(path, max_kb: int = 1000) -> float:
    """Shrink a PNG over max_kb in place by quantizing it to a 256-color palette. Median cut with
    one k-means pass keeps flat fills, text and markers exact; only anti-aliased edge pixels move.
    Returns the size in KB (a chart PNG typically drops to about a third); warns if it is still
    over max_kb (then lower the chart's dpi or size)."""
    path = Path(path)
    if path.stat().st_size > max_kb * 1024:
        with Image.open(path) as im:
            small = im.convert("RGB").quantize(colors=256, method=Image.Quantize.MEDIANCUT, kmeans=1)
        small.save(path, optimize=True)
    kb = path.stat().st_size / 1024
    if kb > max_kb:
        warnings.warn(f"{path.name} is {kb:.0f} KB after quantizing, over max_kb = {max_kb}", stacklevel=2)
    return kb
