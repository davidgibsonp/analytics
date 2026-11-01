"""Shared chart style: surfaces, text colors, and the categorical palette.

Categorical colors are assigned in this fixed order and never cycled. Scatter-type charts
(every pair of colors can sit side by side) are validated for the first three slots only:
past three, fold the rest into "Other" (see fold_categories).

Magnitude uses one hue light -> dark (SEQUENTIAL, by step); states use the reserved STATUS colors,
never a series color, and always with an icon or label beside them.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import matplotlib
import matplotlib.text

matplotlib.use("Agg")
# The rc font list ends in DejaVu Sans as a glyph fallback; matplotlib warns that it has no semibold on
# every semibold text, although Inter SemiBold is the face that renders. Keep real errors only.
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from PIL import Image  # noqa: E402  (installed with matplotlib)

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e8e7e3"
OTHER = "#9a9893"

CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SCATTER_MAX_CATEGORIES = 3

BASELINE = "#c3c2b7"  # axis and baseline ink
HAIRLINE = "#e1e0d9"  # hairline gridlines, cut faces
# Sequential blue, light -> dark (validated steps 100-700). 25 and 50 are surface tints below the
# validated ramp, only for "near zero" (a floor, the lowest band), never for a discrete mark.
SEQUENTIAL = {25: "#eef3f9", 50: "#e6f0fc", 100: "#cde2fb", 150: "#b7d3f6", 200: "#9ec5f4"}
SEQUENTIAL |= {250: "#86b6ef", 300: "#6da7ec", 350: "#5598e7", 400: "#3987e5", 450: "#2a78d6"}
SEQUENTIAL |= {500: "#256abf", 550: "#1c5cab", 600: "#184f95", 650: "#104281", 700: "#0d366b"}
STATUS = {"good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a", "critical": "#d03b3b"}


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


def sequential_cmap(steps, at=None, name: str = "sequential"):
    """A continuous colormap through these SEQUENTIAL steps (evenly spaced, or at the given 0-1 stops)."""
    at = np.linspace(0, 1, len(steps)) if at is None else at
    return LinearSegmentedColormap.from_list(
        name, [(a, SEQUENTIAL[s]) for a, s in zip(at, steps, strict=True)]
    )


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


def texts_outside(fig) -> list[str]:
    """Texts (first 40 characters) that run off the canvas: a check to run before saving a chart."""
    fig.canvas.draw()
    r, W, H = fig.canvas.get_renderer(), fig.bbox.width, fig.bbox.height
    out = []
    for t in fig.findobj(matplotlib.text.Text):
        if t.get_visible() and t.get_text().strip():
            b = t.get_window_extent(r)
            if b.width and (b.x0 < 0 or b.y0 < 0 or b.x1 > W or b.y1 > H):
                out.append(t.get_text()[:40])
    return out


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
