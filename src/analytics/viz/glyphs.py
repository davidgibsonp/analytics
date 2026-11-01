"""Glyphs for terrain charts: flags (summits), tents (stalled camps), cairns (stacked stones) and
cracks (crevasses), in neutral ink; only the tent takes a reserved status color.

Two renderings of one look:
- matplotlib markers (FLAG, TENT, CRACK, and cairn_stack(n) for a cairn of n stones) for scatter
  plots in data coordinates. ANCHOR says how far below a lane centre each one stands (in lanes);
  EXTENT / HALF give the flag and tent boxes in marker sizes, cairn_stack the stack's box in stone
  widths, for collision checks and dodging;
- pixel drawers (pole, stones, tent) for an overlay axes in pixel units, e.g. over a 3D projection.
  Each returns the box it covers; glyph_box gives the same box without drawing.

stone() draws one flat stone (a cairn ledger) and stone_count() maps a weight to a stack height (one
more stone = twice the weight; both maps use it).
"""

from __future__ import annotations

import numpy as np
from matplotlib.markers import MarkerStyle
from matplotlib.patches import Ellipse, Polygon
from matplotlib.path import Path as MPath

from analytics.viz.style import STATUS, SURFACE, TEXT, TEXT_SECONDARY


def marker(polys, close: bool = True) -> MarkerStyle:
    """MarkerStyle from polygons in marker units; (0, 0) stays the anchor (matplotlib only scales)."""
    verts, codes = [], []
    for poly in polys:
        verts += [*poly, poly[0]] if close else poly
        codes += [MPath.MOVETO, *[MPath.LINETO] * (len(poly) - 1)] + ([MPath.CLOSEPOLY] if close else [])
    return MarkerStyle(MPath(verts, codes))


def ellipse(cx, cy, rx, ry, n: int = 24) -> list:
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return list(zip(cx + rx * np.cos(t), cy + ry * np.sin(t), strict=True))


# flag: pole from the anchor up to y = 1, pennant to the right (max |vertex| = 1, so pole = size / 2)
FLAG = marker([[(-0.06, 0), (0.06, 0), (0.06, 1), (-0.06, 1)], [(0.06, 1), (0.7, 0.81), (0.06, 0.6)]])
# tent: base on the anchor, a door cut out (opposite winding)
TENT = marker([[(-1, 0), (0, 0.95), (1, 0)], [(0.24, 0), (0, 0.42), (-0.24, 0)]])
# crevasse: a short open zigzag crack
CRACK = marker([[(-1, 0.55), (-0.25, -0.35), (0.2, 0.3), (1, -0.6)]], close=False)
ANCHOR = {"flag": 0.47, "tent": 0.34, "cairn": 0.3, "crack": 0.0}  # lanes below the lane centre
EXTENT = {"flag": (-0.03, 0.36, 0, 0.5), "tent": (-0.5, 0.5, 0, 0.48), "crack": (-0.5, 0.5, -0.3, 0.28)}
HALF = {"flag": 0.36, "tent": 0.5}  # half-width for dodging (a flag leans right)
# cairn stones in bottom-stone widths, the proportions of stones() below: height, rise per stone and
# the width each higher stone loses
STONE_H, STONE_STEP, STONE_SHRINK = 0.48, 0.385, 0.123


def cairn_stack(n: int, stone_pt: float):
    """A cairn of n stacked stones whose bottom stone is stone_pt wide, standing on the anchor.
    Returns (marker, size, box): the scatter size (pt) that keeps every stone stone_pt wide whatever
    n (matplotlib scales a marker to its largest vertex) and the box (x0, x1, y0, y1) in pt."""
    top = STONE_H + (n - 1) * STONE_STEP
    polys = [
        ellipse(0, STONE_H / 2 + k * STONE_STEP, (1 - k * STONE_SHRINK) / 2, STONE_H / 2) for k in range(n)
    ]
    box = np.array([-0.5, 0.5, 0.0, top]) * stone_pt
    return marker(polys), 2 * stone_pt * max(0.5, top), box


def stone_count(weight) -> int:
    """Stones in a cairn stack: one more stone = twice the weight (1 -> 1, 2-3 -> 2, 4-7 -> 3, 8-15 -> 4)."""
    return 1 + int(np.floor(np.log2(max(float(weight), 1.0))))


def stone(ax, xc, ybottom, w, h, color, z: float = 3) -> None:
    """One flat stone (an ellipse) whose bottom sits on ybottom, for an axes with y pointing down."""
    ax.add_patch(Ellipse((xc, ybottom - h / 2), w, h, facecolor=color, edgecolor="none", zorder=z))


# ------------------------------------------------------------------- pixel drawers (y points up)
POLE_CLEAR = 24  # px kept free under a pole (keep_clear), so a cairn there never reads as its base


def pole(ax, x, y, n: int, s: float = 1.0, z: float = 10, ink: str = TEXT) -> np.ndarray:
    """A summit pole standing on (x, y) with n stacked pennants; returns its box."""
    h = (22 + 8.5 * n) * s
    ax.plot([x, x], [y, y + h], color=ink, lw=1.4 * s, solid_capstyle="butt", zorder=z)
    ax.add_patch(Ellipse((x, y), 6 * s, 3 * s, color=ink, zorder=z))
    for i in range(n):
        top = y + h - i * 8.5 * s
        tri = [(x, top), (x + 14 * s, top - 3.4 * s), (x, top - 6.8 * s)]
        ax.add_patch(Polygon(tri, closed=True, fc=ink, ec=SURFACE, lw=0.7 * s, zorder=z))
    return np.array([x - 3 * s, y - 2 * s, x + 15 * s, y + h + 1])


def stones(ax, x, y, n: int, s: float = 1.0, z: float = 10, ink: str = TEXT_SECONDARY) -> np.ndarray:
    """A cairn of n stacked stones standing on (x, y), widest at the bottom; returns its box."""
    top = y
    for k in range(n):
        w, h = (13 - 1.6 * k) * s, 6.2 * s
        cy = y + 2.8 * s + k * 5.0 * s
        ax.add_patch(Ellipse((x, cy), w, h, fc=ink, ec=SURFACE, lw=1.0 * s, zorder=z))
        top = cy + h / 2
    return np.array([x - 7 * s, y - 1, x + 7 * s, top + 1])


def tent(
    ax, x, y, s: float = 1.0, z: float = 10, fill: str = STATUS["warning"], ink: str = TEXT
) -> np.ndarray:
    """A tent standing on (x, y) in a status color with its door cut out (the TENT marker's shape);
    returns its box."""
    w, h = 15 * s, 12 * s
    ax.add_patch(
        Polygon([(x - w / 2, y), (x + w / 2, y), (x, y + h)], closed=True, fc=fill, ec=ink, lw=s, zorder=z)
    )
    door = [(x - 0.24 * w / 2, y), (x + 0.24 * w / 2, y), (x, y + 0.42 / 0.95 * h)]
    ax.add_patch(Polygon(door, closed=True, fc=SURFACE, ec=ink, lw=0.7 * s, zorder=z + 0.1))
    return np.array([x - w / 2 - 1, y - 1, x + w / 2 + 1, y + h + 1])


DRAW = {"pole": pole, "cairn": stones}
WIDTH = {"pole": 20.0, "tent": 16.0, "cairn": 14.0}  # footprint in px (a pole with its pennant)


def glyph_box(kind: str, x, y, n: int, keep_clear: bool = False, pad: float = 2) -> np.ndarray:
    """The box pole / tent / cairn (n pennants or stones) would cover at (x, y), plus pad px."""
    if kind == "pole":
        b = np.array([x - 3, y - 2 - (POLE_CLEAR if keep_clear else 0), x + 15, y + 22 + 8.5 * n + 1])
    elif kind == "tent":
        b = np.array([x - 8.5, y - 1, x + 8.5, y + 13])
    else:
        b = np.array([x - 7, y - 1, x + 7, y + 2.8 + 5.0 * (n - 1) + 3.1 + 1])
    return b + [-pad, -pad, pad, pad]
