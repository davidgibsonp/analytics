"""Cairn chart: how well each route is marked (an index) and the most-followed markers (a register).

(a) One row per route: a cairn of flat stones (one stone per marker, width proportional to its
    weight, oldest at the bottom, shade = kind; past max_stones only the heaviest are drawn, with a
    '+N' tag on top), a traffic bar and the marking ratio (markers /
    traffic) on a dot strip, with the busy and barely-marked cut-offs drawn in. Flagged rows (busy yet
    barely marked) get a band, a dark bar and a short note. Rows that are not routes come last.
(b) The register: the heaviest markers as a ruled ledger, each with its stone at the scale of (a).

    stats = cairn_chart(rows, register, cuts, text, pc, path)   # size, hidden stones, collisions, KB

    rows     : name, sub, traffic, cairns, ratio, route (bool), fold (a total over several; hollow dot),
               flag (bool), stones (a DataFrame per row: id, weight, kind; oldest first)
    register : rank order; id, title, project, completedAt, kind, kinds ('a+b'), inbound, children,
               bonus, weight
    cuts     : (busy_min traffic, barely_max ratio), drawn over the route rows
    text     : every string, ready to print (title, subtitle, panels, rule, legend, heads, notes), except
               the per-row templates frac ({cairns}, {traffic}), flag_note ({what}, {traffic}) and
               more ({n}: stones not drawn)
    pc       : sizes in px, fonts, columns {name: x px}, shades {kind: color}

The page is one axes in pixel units (y down), so stone widths and text positions are exact, and every
text is kept for a collision check (Page.collisions): close side-by-side pairs and text off the page.
"""

from __future__ import annotations

import textwrap

import numpy as np
from matplotlib.patches import FancyBboxPatch, Rectangle

from analytics.viz.glyphs import stone
from analytics.viz.style import GRID, SURFACE, TEXT, TEXT_SECONDARY, optimize_png, plt


class Page:
    """One axes over the whole figure in pixel units, y pointing down; every text is kept for the
    collision check."""

    def __init__(self, width, height, dpi):
        self.fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi, facecolor=SURFACE)
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set_xlim(0, width)
        self.ax.set_ylim(height, 0)
        self.ax.axis("off")
        self.texts = []

    def text(self, x, y, s, size, color=TEXT, **kw):
        t = self.ax.text(x, y, s, fontsize=size, color=color, **kw)
        self.texts.append(t)
        return t

    def box(self, t):
        """Text extent in page px (x0, y0 top, x1, y1 bottom)."""
        b = t.get_window_extent(self.fig.canvas.get_renderer())
        h = self.fig.get_figheight() * self.fig.dpi
        return b.x0, h - b.y1, b.x1, h - b.y0

    def collisions(self, xgap=4, ytol=3):
        """Text pairs closer than xgap px side by side while sharing more than ytol px of height (line
        boxes are tall, so stacked lines touching by a pixel or two are not hits), and texts off the page."""
        r = self.fig.canvas.get_renderer()
        boxes = [(t, t.get_window_extent(r)) for t in self.texts if t.get_text()]

        def hit(a, b):
            return min(a.x1, b.x1) - max(a.x0, b.x0) > -xgap and min(a.y1, b.y1) - max(a.y0, b.y0) > ytol

        hits = [
            (ta.get_text()[:40], tb.get_text()[:40])
            for i, (ta, a) in enumerate(boxes)
            for tb, b in boxes[i + 1 :]
            if hit(a, b)
        ]
        w, h = self.fig.get_figwidth() * self.fig.dpi, self.fig.get_figheight() * self.fig.dpi
        out = [t.get_text()[:40] for t, b in boxes if b.x0 < 0 or b.x1 > w or b.y0 < 0 or b.y1 > h]
        return hits, out


def kind_label(kind: str, kinds: str, names: dict) -> str:
    """'parent tree + mention target': the main kind first, then the other qualifying kinds ('a+b')."""
    return " + ".join(names[k] for k in [kind, *[k for k in kinds.split("+") if k != kind]])


def shown(stones, max_stones):
    """The stones drawn (the heaviest max_stones, kept oldest first) and how many are left out."""
    if len(stones) <= max_stones:
        return stones, 0
    keep = stones.sort_values("weight", ascending=False, kind="stable").index[:max_stones]
    return stones.loc[stones.index.isin(keep)], len(stones) - max_stones


def stack(P, xc, ground, stones, pc, right_limit, more="") -> None:
    """Stones bottom-up (oldest first) on the ground line; ids on the heavy ones: right of the stone,
    else above it when it is the top stone, else left of it; `more` (e.g. '+8') above the top stone."""
    y, placed, fs = ground, [], pc["font"]

    def clash(t):
        x0, y0, x1, y1 = P.box(t)
        return x1 > right_limit or any(
            min(x1, b[2]) - max(x0, b[0]) > -4 and min(y1, b[3]) - max(y0, b[1]) > -2 for b in placed
        )

    for j, s in enumerate(stones.itertuples()):
        w = pc["unit"] * s.weight
        stone(P.ax, xc, y, w, pc["stone_h"], pc["shades"][s.kind])
        if s.weight >= pc["label_weight"]:
            yc = y - pc["stone_h"] / 2
            t = P.text(xc + w / 2 + 6, yc, s.id, fs["id"], TEXT_SECONDARY, va="center")
            if clash(t) and j == len(stones) - 1:
                t.set_position((xc, y - pc["stone_h"] - 3))
                t.set_ha("center")
                t.set_va("bottom")
            if clash(t):
                t.set_position((xc - w / 2 - 6, yc))
                t.set_ha("right")
                t.set_va("center")
            placed.append(P.box(t))
        y -= pc["stone_h"] + pc["gap"]
    if more:
        P.text(xc, y + pc["gap"] - 1, more, fs["id"], TEXT_SECONDARY, ha="center", va="bottom")


def bar(ax, x0, ycenter, length, h, color) -> None:
    """Horizontal bar, 4 px rounded data end, square at the baseline."""
    if length <= 0:
        return
    r = min(4, length / 2, h / 2)
    ax.add_patch(
        FancyBboxPatch(
            (x0, ycenter - h / 2),
            length,
            h,
            boxstyle=f"round,pad=0,rounding_size={r}",
            facecolor=color,
            edgecolor="none",
            zorder=2,
        )
    )
    ax.add_patch(
        Rectangle((x0, ycenter - h / 2), min(length, r + 1), h, facecolor=color, edgecolor="none", zorder=2)
    )


def legend_row(P, x, y, items, size, stone_h) -> float:
    """Items left to right on one line: str = text, (width_px, color) = a stone. Returns the end x."""
    for it in items:
        if isinstance(it, str):
            x = P.box(P.text(x, y, it, size, TEXT_SECONDARY, va="center"))[2] + 8
        else:
            stone(P.ax, x + it[0] / 2, y + stone_h / 2, it[0], stone_h, it[1])
            x += it[0] + 6
    return x


def pct(v) -> str:
    return "0%" if v == 0 else f"{100 * v:.1f}%" if v < 0.1 else f"{100 * v:.0f}%"


def cairn_chart(rows, register, cuts, text, pc, path) -> dict:
    fs, X, XR, sh = pc["font"], pc["columns"], pc["register_columns"], pc["shades"]
    busy_min, barely_max = cuts
    step = pc["stone_h"] + pc["gap"]
    cut = [shown(s, pc["max_stones"]) for s in rows.stones]
    tag = pc["font"]["id"] * 2  # px for a '+N' tag on top
    rows = rows.assign(
        stones=[c[0] for c in cut],
        more=[c[1] for c in cut],
        h=[max(pc["row_min"], len(c[0]) * step + pc["row_pad"] + (tag if c[1] else 0)) for c in cut],
    )
    y_title, y_a, section_gap = 34, 150, 46
    y_rows = y_a + 150
    y_b = y_rows + rows.h.sum() + section_gap + 124
    y_reg = y_b + 96
    height = int(y_reg + len(register) * pc["register_row_h"] + 100)
    W, M, u = pc["width"], pc["margin"], pc["unit"]
    P = Page(W, height, pc["dpi"])
    ax, rule_x = P.ax, (M - 12, W - M + 12)
    small = fs["small"]

    P.text(M, y_title, text["title"], fs["title"], va="top")
    P.text(M, y_title + 40, text["subtitle"], fs["sub"], TEXT_SECONDARY, va="top", linespacing=1.5)
    P.text(M, y_a, text["panel_a"], fs["panel"], va="top", weight="semibold")
    P.text(M, y_a + 32, text["rule"], small + 0.5, TEXT_SECONDARY, va="top")
    ly = y_a + 70
    sizes = [it for w in pc["legend_weights"] for it in ((w * u, sh["parent"]), str(w))]
    x = legend_row(P, M, ly, [text["legend_size"], *sizes], small, pc["stone_h"])
    kinds = [it for k, name in text["kinds_legend"].items() for it in ((28, sh[k]), name)]
    x = legend_row(P, x + 30, ly, [text["legend_kind"], *kinds], small, pc["stone_h"])
    legend_row(P, x + 30, ly, [text["legend_ids"], text["legend_maps"]], small, pc["stone_h"])
    yh = y_rows - 12
    for col, (a, b) in text["heads"].items():
        ha = "center" if col == "cairn" else "left"
        if b:
            P.text(X[col], yh - 18, a, fs["head"], TEXT_SECONDARY, ha=ha, va="bottom")
        P.text(X[col], yh, b or a, fs["head"] - (0.5 if b else 0), TEXT_SECONDARY, ha=ha, va="bottom")

    y = y_rows
    r_lo, r_hi = X["ratio"], X["ratio"] + pc["ratio_max"] * pc["ratio_px"]
    route_top, route_bottom = y, None
    for r in rows.itertuples():
        if not r.route and route_bottom is None:
            route_bottom = y
            y += section_gap
            P.text(
                X["name"], y - 10, text["section"], fs["head"], TEXT_SECONDARY, va="bottom", style="italic"
            )
        g = y + r.h  # ground line
        bg = pc["band"] if r.flag else SURFACE
        if r.flag:
            ax.add_patch(
                Rectangle(
                    (rule_x[0], y + 2),
                    rule_x[1] - rule_x[0],
                    r.h - 2,
                    facecolor=bg,
                    edgecolor="none",
                    zorder=0,
                )
            )
        ax.plot(rule_x, [g, g], color=GRID, lw=0.8, solid_capstyle="butt", zorder=1)
        P.text(
            X["name"],
            g - 24,
            r.name,
            fs["name"],
            TEXT,
            va="bottom",
            weight="semibold" if r.flag else "normal",
        )
        P.text(X["name"], g - 6, r.sub, small, TEXT_SECONDARY, va="bottom")
        if len(r.stones):
            more = text["more"].format(n=r.more) if r.more else ""
            stack(P, X["cairn"], g - 4, r.stones, pc, X["bar"] - 8, more)
        else:
            P.text(
                X["cairn"],
                g - 6,
                text["none"],
                small,
                TEXT_SECONDARY,
                va="bottom",
                ha="center",
                style="italic",
            )
        yc, length = g - 15, r.traffic * pc["traffic_px"]
        bar(ax, X["bar"], yc, length, pc["bar_h"], pc["bar_flag"] if r.flag else pc["bar"])
        P.text(
            X["bar"] + length + 8,
            yc,
            f"{r.traffic}",
            small + 0.5,
            TEXT_SECONDARY,
            va="center",
            bbox=dict(facecolor=bg, edgecolor="none", pad=1.5),
        )
        clip, filled = r.ratio > pc["ratio_max"], r.route and not r.fold
        xd = r_lo + min(r.ratio, pc["ratio_max"]) * pc["ratio_px"]
        ax.scatter(
            [xd],
            [yc],
            s=pc["dot"] ** 2 if filled else (pc["dot"] - 1) ** 2,
            marker=">" if clip else "o",
            facecolor=(TEXT if r.flag else TEXT_SECONDARY) if filled else bg,
            edgecolor=bg if filled else TEXT_SECONDARY,
            linewidth=2 if filled else 1.2,
            zorder=4,
        )
        P.text(
            X["pct"],
            yc,
            pct(r.ratio),
            small + 0.5,
            TEXT if r.flag else TEXT_SECONDARY,
            ha="right",
            va="center",
        )
        P.text(
            X["frac"],
            yc,
            text["frac"].format(cairns=r.cairns, traffic=r.traffic),
            small,
            TEXT_SECONDARY,
            va="center",
        )
        if r.flag:
            what = text["no_cairn"] if r.cairns == 0 else f"{r.cairns} cairn" + ("s" if r.cairns > 1 else "")
            P.text(X["note"], g - 24, text["flag"], fs["name"], TEXT, va="bottom", weight="semibold")
            P.text(
                X["note"],
                g - 6,
                text["flag_note"].format(what=what, traffic=r.traffic),
                small,
                TEXT_SECONDARY,
                va="bottom",
            )
        y = g
    route_bottom = y if route_bottom is None else route_bottom
    xb, xr = X["bar"] + busy_min * pc["traffic_px"], r_lo + barely_max * pc["ratio_px"]
    for xx in (xb, xr):  # cut-off hairlines over the route rows, behind every mark
        ax.plot([xx, xx], [route_top + 4, route_bottom], color=TEXT_SECONDARY, lw=0.8, zorder=0.5, alpha=0.55)
    P.text(xb, route_bottom + 5, text["busy"], small - 0.5, TEXT_SECONDARY, va="top", ha="center")
    P.text(xr, route_bottom + 5, text["barely"], small - 0.5, TEXT_SECONDARY, va="top", ha="center")
    ya = y + 10  # ratio axis under the last row
    ax.plot([r_lo, r_hi], [ya, ya], color=GRID, lw=0.8)
    for v in np.arange(0, pc["ratio_max"] + 1e-9, pc["ratio_tick"]):
        xv = r_lo + v * pc["ratio_px"]
        ax.plot([xv, xv], [ya, ya + 4], color=GRID, lw=0.8)
        P.text(xv, ya + 6, f"{100 * v:.0f}%", small - 0.5, TEXT_SECONDARY, va="top", ha="center")
    P.text(r_hi + 26, ya + 6, text["clip"], small - 0.5, TEXT_SECONDARY, va="top")
    P.text(M, ya + 34, text["note_a"], small, TEXT_SECONDARY, va="top")

    ax.plot(rule_x, [y_b - 16, y_b - 16], color=GRID, lw=1.2)
    P.text(M, y_b, text["panel_b"], fs["panel"], va="top", weight="semibold")
    P.text(M, y_b + 32, text["register_sub"], small + 0.5, TEXT_SECONDARY, va="top")
    align = dict(rank="right", stone="center", mentions="right", subs="right", retro="right", weight="right")
    for col, label in text["register_heads"].items():
        P.text(XR[col], y_reg - 9, label, fs["head"], TEXT_SECONDARY, ha=align.get(col, "left"), va="bottom")
    ax.plot(rule_x, [y_reg - 2, y_reg - 2], color=TEXT_SECONDARY, lw=0.8)
    rh = pc["register_row_h"]
    for j, c in enumerate(register.itertuples()):
        g = y_reg + (j + 1) * rh
        ym = g - rh / 2
        ax.plot(rule_x, [g, g], color=GRID, lw=0.8)
        P.text(XR["rank"], ym, f"{j + 1}", fs["name"], TEXT_SECONDARY, ha="right", va="center")
        stone(ax, XR["stone"], ym + pc["stone_h"] / 2, u * c.weight, pc["stone_h"], sh[c.kind])
        P.text(XR["id"], ym, c.id, fs["name"], TEXT, va="center", weight="semibold")
        title = (
            c.title
            if len(c.title) <= pc["title_chars"]
            else textwrap.shorten(c.title, pc["title_chars"], placeholder=" …")
        )
        P.text(XR["title"], ym, title, fs["name"], TEXT, va="center")
        P.text(XR["project"], ym, c.project, small + 0.5, TEXT_SECONDARY, va="center")
        P.text(XR["done"], ym, c.completedAt.strftime("%-d %b %Y"), small + 0.5, TEXT_SECONDARY, va="center")
        P.text(
            XR["kind"],
            ym,
            kind_label(c.kind, c.kinds, text["kind_names"]),
            small + 0.5,
            TEXT_SECONDARY,
            va="center",
        )
        for col, v, s in (
            ("mentions", c.inbound, f"{c.inbound}"),
            ("subs", c.children, f"{c.children}"),
            ("retro", c.bonus, f"+{c.bonus}"),
        ):
            P.text(
                XR[col],
                ym,
                s if v else "–",
                fs["name"],
                TEXT if v else TEXT_SECONDARY,
                ha="right",
                va="center",
            )
        P.text(XR["weight"], ym, f"{c.weight}", fs["name"], TEXT, ha="right", va="center", weight="semibold")
    yf = y_reg + len(register) * rh + 14
    P.text(M, yf, text["register_foot"], small, TEXT_SECONDARY, va="top")
    P.text(M, yf + 26, text["source"], small - 0.5, TEXT_SECONDARY, va="top")
    hits, out = P.collisions()
    P.fig.savefig(path, facecolor=SURFACE, dpi=pc["dpi"])
    plt.close(P.fig)
    kb = round(optimize_png(path, pc["max_kb"]))
    return dict(size_px=(W, height), hidden=int(rows.more.sum()), collisions=hits, outside=out, kb=kb)
