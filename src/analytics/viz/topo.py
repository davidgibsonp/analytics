"""Topographic plan view of a terrain grid: lanes top to bottom, time left to right.

A (lanes, weeks) height grid (features.terrain) is drawn as filled contours in one sequential hue
with recessive contour lines, cut at a right edge (e.g. a snapshot). Around it: lane labels under
bold block headers, light vertical bands with staggered tags (e.g. sprints) and month lines, both
under the terrain (they show on bare ground only), month ticks, a bar per lane in a side panel, a
legend row and a footnote. Glyphs (viz.glyphs) stand on their lanes: flags, tents with a dotted trail
to the edge, cairns of one stone per doubling of weight (glyphs.stone_count, as on the 3D relief),
and cracks under them.

    stats = topo_map(height, weeks, lanes, blocks, marks, bands, side, edge, text, pc, path)

    lanes  : lane, label (lanes without a label are gaps)
    blocks : header_lane, header, stats, note ("" for none); the header sits on header_lane
    marks  : kind (flag, tent, cairn, crack), lane, x, weight (cairn stones), label ("" = none),
             group (flags of one group share one label, set off the latest flag), trail (tent: bool)
    bands  : x0, x1, tag
    side   : lane, value
    edge   : x of the right edge; text: title, subtitle, footnote ({shift_days}, {crack_days}), edge,
             band_note, side_title, side_note ({total}, {lanes}), scale, legend {band, flag, tent,
             trail, cairn, crack, side, edge} (a key left out drops that item, e.g. one labelled in
             place); pc: the [map2d]-style options (sizes, fonts, levels, ramp steps)

x is fractional weeks since weeks[0] (features.terrain.week_float). Glyphs on one lane and day sit
side by side (pack.dodge); a flag or crack that still touches a glyph slides along its lane. Labels
are placed greedily: each tries offsets in order and keeps the first that touches no glyph, label or
frame and crosses no trail or leader (tents first, on their own lane: along their own trail, else
left of the tent, else off the lane with a leader; then flag groups; then cairns), and two cairn
labels whose leaders cross trade places. Returns label misses, glyph clashes (cracks included), the
largest shift per kind (days), counts drawn and the PNG size in KB.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.transforms import Bbox

from analytics.features.terrain import month_starts, upsample
from analytics.viz.glyphs import ANCHOR, CRACK, EXTENT, FLAG, HALF, TENT, cairn_stack, stone_count
from analytics.viz.pack import crosses, cuts, dodge
from analytics.viz.style import (
    GRID,
    OTHER,
    SEQUENTIAL,
    STATUS,
    SURFACE,
    TEXT,
    TEXT_SECONDARY,
    optimize_png,
    plt,
    texts_outside,
)

STANDING = ("flag", "tent", "cairn")
TAG_ROWS = (1.006, 1.023)  # band tags alternate between these heights (axes fraction)


def topo_map(height, weeks, lanes, blocks, marks, bands, side, edge, text, pc, path) -> dict:
    fs, n_lanes = pc["font"], height.shape[0]
    fig = plt.figure(figsize=(pc["width"], pc["height"]), dpi=pc["dpi"], facecolor=SURFACE)
    ax = fig.add_axes(pc["map_box"])
    ax.set_facecolor(SURFACE)
    ax.set_xlim(0, edge + pc["edge_pad"])
    ax.set_ylim(n_lanes - 0.5, -0.5)
    for s in ax.spines.values():
        s.set_visible(False)
    levels, cmap = terrain(ax, height, edge, pc)
    fixed = axis_text(fig, ax, weeks, lanes, blocks, bands, edge, text, pc)
    marks = marks.reset_index(drop=True)
    stand, boxes, clash = arrange(fig, ax, marks[marks.kind.isin(STANDING)].copy(), pc)
    cracks, crack_clash = slide_cracks(fig, ax, marks[(marks.kind == "crack") & (marks.lane >= 0)], boxes, pc)
    status = STATUS[pc["status"]]
    ax.scatter(
        cracks.xd,
        cracks.lane,
        s=pc["crack_pt"] ** 2,
        marker=CRACK,
        facecolors="none",
        edgecolors=TEXT_SECONDARY,
        linewidths=0.7,
        alpha=0.75,
        zorder=3,
    )
    for r in stand[(stand.kind == "tent") & stand.trail.astype(bool)].itertuples():
        ax.plot(
            [r.xd, edge],
            [r.lane, r.lane],
            color=status,
            lw=1.3,
            ls=(0, (1, 1.6)),
            zorder=4,
            solid_capstyle="butt",
        )
    cairn = dict(c=TEXT_SECONDARY, edgecolors=SURFACE, linewidths=0.6, zorder=5)
    for n, m in stand[stand.kind == "cairn"].groupby("n"):  # one marker per stack height
        ax.scatter(m.xd, m.lane + ANCHOR["cairn"], s=m["size"] ** 2, marker=cairn_stack(n, 1)[0], **cairn)
    style = {
        "flag": dict(marker=FLAG, c=TEXT, edgecolors=SURFACE, linewidths=0.6, zorder=6),
        "tent": dict(marker=TENT, c=status, edgecolors=TEXT, linewidths=0.7, zorder=6),
    }
    for kind, kw in style.items():
        m = stand[stand.kind == kind]
        ax.scatter(m.xd, m.lane + ANCHOR[kind], s=m["size"] ** 2, **kw)
    misses = Labels(fig, ax, boxes, fixed, stand, edge, fs).run()
    side_panel(fig, ax, side, text, pc)
    fig.text(0.008, 0.985, text["title"], ha="left", va="top", fontsize=fs["title"], color=TEXT)
    fig.text(
        0.008, 0.952, text["subtitle"], ha="left", va="top", fontsize=fs["subtitle"], color=TEXT_SECONDARY
    )
    legend_row(fig, levels, cmap, text, pc)
    moved = pd.concat([stand[["kind", "x", "xd"]], cracks[["kind", "x", "xd"]]])
    shift = (moved.xd - moved.x).abs().groupby(moved.kind).max().mul(7).round(1).to_dict()  # days
    most = max((v for k, v in shift.items() if k != "crack"), default=0)
    foot = text["footnote"].format(
        shift_days=int(np.ceil(most)), crack_days=int(np.ceil(shift.get("crack", 0)))
    )
    fig.text(
        0.008,
        0.008,
        foot,
        ha="left",
        va="bottom",
        fontsize=fs["footnote"],
        color=TEXT_SECONDARY,
        linespacing=1.45,
    )
    outside = texts_outside(fig)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    drawn = stand.kind.value_counts().to_dict() | {"crack": len(cracks), "side": len(side)}
    return dict(
        misses=misses,
        outside=outside,
        clashes=clash + crack_clash,
        shift_days=shift,
        drawn=drawn,
        kb=round(optimize_png(path, pc["max_kb"])),
    )


def terrain(ax, height, edge, pc):
    """Filled contours of the upsampled grid, cut at the edge; the last band runs to the peak."""
    up, n = pc["upsample"], height.shape[0]
    fx = np.linspace(0, edge, int(edge * up) + 1)
    fy = np.linspace(-0.5, n - 0.5, n * up + 1)
    Z = upsample(height, fy, fx)
    peak = max(float(height.max()), pc["levels"][0])
    levels = [v for v in pc["levels"] if v < peak] + [peak + 1e-6]  # the top band ends at the peak
    cmap = ListedColormap([SEQUENTIAL[s] for s in pc["ramp"][: len(levels) - 1]])
    ax.contourf(fx, fy, Z, levels=levels, cmap=cmap, norm=BoundaryNorm(levels, cmap.N), zorder=0)
    ax.contour(
        fx,
        fy,
        Z,
        levels=levels[:-1],
        colors=SEQUENTIAL[pc["line_step"]],
        linewidths=pc["line_width"],
        alpha=pc["line_alpha"],
        zorder=1,
    )
    return levels, cmap


def axis_text(fig, ax, weeks, lanes, blocks, bands, edge, text, pc) -> list:
    """Bands with tags, the edge line, month ticks, lane labels, block headers with stats and notes.
    Returns the texts inside the map (obstacles for labels)."""
    fs, fixed = pc["font"], []
    tag_tf = ax.get_xaxis_transform()
    bands = bands.sort_values("x0").reset_index(drop=True)
    for k, b in bands.iterrows():
        # under the terrain (zorder -1): a band shows on bare ground only, so it never tints the ramp
        ax.axvspan(b.x0, b.x1, color=TEXT, alpha=pc["band_alpha"][k % 2], lw=0, zorder=-1)
        fixed.append(
            ax.text(
                (b.x0 + b.x1) / 2,
                TAG_ROWS[k % 2],
                b.tag,
                transform=tag_tf,
                ha="center",
                va="bottom",
                fontsize=fs["tag"],
                color=TEXT_SECONDARY,
            )
        )
    if len(bands):
        fixed.append(
            ax.text(
                bands.x0.min() - 0.6,
                TAG_ROWS[0],
                text["band_note"],
                transform=tag_tf,
                ha="right",
                va="bottom",
                fontsize=fs["tag"] + 0.5,
                color=TEXT_SECONDARY,
                style="italic",
            )
        )
    ax.axvline(edge, color=TEXT_SECONDARY, lw=0.9, zorder=6)
    fixed.append(
        ax.text(
            edge + 0.25,
            -0.012,
            text["edge"],
            transform=tag_tf,
            ha="right",
            va="top",
            fontsize=fs["lane"],
            color=TEXT,
            linespacing=1.15,
        )
    )
    months, mx = month_starts(weeks, weeks[0] + pd.Timedelta(weeks=edge))
    labels = [
        m.strftime("%b") + (f"\n{m.year}" if m.month == 1 or i == 0 else "") for i, m in enumerate(months)
    ]
    ax.set_xticks(mx, labels)
    ax.tick_params(axis="x", length=3, color=GRID, labelsize=fs["legend"], labelcolor=TEXT_SECONDARY, pad=3)
    for v in mx:  # under the terrain too: no seams through the contours
        ax.axvline(v, color=GRID, lw=0.5, zorder=-0.5)
    ridge = lanes[lanes.label.fillna("") != ""]
    ax.set_yticks(ridge.lane, ridge.label)
    ax.tick_params(axis="y", length=0, labelsize=fs["lane"], labelcolor=TEXT_SECONDARY, pad=4)
    ytf = ax.get_yaxis_transform()
    left = -(pc["map_box"][0] - 0.006) / pc["map_box"][2]
    heads = [
        ax.text(
            left,
            b.header_lane,
            b.header,
            transform=ytf,
            ha="left",
            va="center",
            fontsize=fs["block"],
            color=TEXT,
            weight="bold",
        )
        for b in blocks.itertuples()
    ]
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for t, b in zip(heads, blocks.itertuples(), strict=True):  # stats measured right after the header
        x = ax.transAxes.inverted().transform((t.get_window_extent(renderer).x1 + 5, 0))[0]
        ax.text(
            x,
            b.header_lane,
            b.stats,
            transform=ytf,
            ha="left",
            va="center",
            fontsize=fs["lane"] - 0.5,
            color=TEXT_SECONDARY,
        )
        if b.note:
            fixed.append(
                ax.text(
                    pc["note_x"],
                    b.header_lane,
                    b.note,
                    ha="left",
                    va="center",
                    fontsize=fs["lane"],
                    color=TEXT_SECONDARY,
                    style="italic",
                    zorder=8,
                )
            )
    return fixed


def arrange(fig, ax, stand, pc):
    """Marker sizes, positions dodged along each lane, and collision boxes (px). A flag is taller than a
    lane: one that still touches a glyph slides along its lane to the nearest free spot."""
    stand = stand[stand.lane >= 0].copy()
    to_px = ax.transData.transform
    pt_x = (to_px((1, 0)) - to_px((0, 0)))[0] * 72 / fig.dpi  # points per week
    pt_lane = (to_px((0, 0)) - to_px((0, 1)))[1] * 72 / fig.dpi  # points per lane (lanes run down)
    sizes = {"flag": pc["flag_pt"] * 2, "tent": pc["tent_pt"] * 2}
    rows = []
    for k, w in zip(stand.kind, stand.weight.fillna(1), strict=True):
        if k == "cairn":
            n = stone_count(w)
            _, size, ext = cairn_stack(n, pc["stone_pt"])
            rows.append((n, size, *ext, pc["stone_pt"] / 2))
        else:
            rows.append((0, sizes[k], *(np.array(EXTENT[k]) * sizes[k]), HALF[k] * sizes[k]))
    cols = ["n", "size", "ex0", "ex1", "ey0", "ey1", "half"]
    stand[cols] = pd.DataFrame(rows, index=stand.index, columns=cols)
    stand["half"] /= pt_x
    stand["yc"] = stand.lane + stand.kind.map(ANCHOR) - (stand.ey0 + stand.ey1) / 2 / pt_lane  # centre
    stand["xd"] = stand.x
    for _, sub in stand.groupby("lane"):
        stand.loc[sub.index, "xd"] = dodge(sub.x.to_numpy(), sub.half.to_numpy(), gap=1.2 / pt_x)
    rows = list(stand.itertuples())
    boxes = [glyph_box(fig, ax, r, r.xd) for r in rows]
    for i, r in enumerate(rows):
        if r.kind == "flag":
            stand.loc[r.Index, "xd"], boxes[i] = slide(fig, ax, r, r.xd, boxes[:i] + boxes[i + 1 :], pt_x)
    clash = [
        (f"{a.kind}@{a.lane}", f"{b.kind}@{b.lane}")
        for (i, a), (j, b) in combinations(enumerate(rows), 2)
        if boxes[i].padded(-1.5).overlaps(boxes[j].padded(-1.5))
    ]
    return stand, boxes, clash


def glyph_box(fig, ax, r, xd) -> Bbox:
    """The glyph's box (px) with its anchor at (xd, lane); r carries kind, lane and the box in pt."""
    cx, cy = ax.transData.transform((xd, r.lane + ANCHOR[r.kind]))
    px = fig.dpi / 72
    return Bbox.from_extents(cx + r.ex0 * px, cy + r.ey0 * px, cx + r.ex1 * px, cy + r.ey1 * px).padded(1.5)


def slide(fig, ax, r, xd, others, pt_x, steps: int = 24):
    """xd, or the nearest position along the lane (one point at a time, up to `steps` either way) whose
    box touches none of `others`; (position, box)."""
    for k in (0, *(s * k for k in range(1, steps + 1) for s in (1, -1))):
        b = glyph_box(fig, ax, r, xd + k / pt_x)
        if not any(b.overlaps(o) for o in others):
            return xd + k / pt_x, b
    return xd, glyph_box(fig, ax, r, xd)


def slide_cracks(fig, ax, cracks, boxes, pc):
    """Cracks are the lowest glyphs: same-day cracks sit side by side (pack.dodge), then one that
    touches a standing glyph slides along its lane; standing glyphs never move for a crack. Returns
    the cracks with xd and any crack still touching a standing glyph."""
    to_px = ax.transData.transform
    pt_x = (to_px((1, 0)) - to_px((0, 0)))[0] * 72 / fig.dpi
    ext = np.array(EXTENT["crack"]) * pc["crack_pt"]
    cracks = cracks.assign(ex0=ext[0], ex1=ext[1], ey0=ext[2], ey1=ext[3], xd=cracks.x)
    for _, sub in cracks.groupby("lane"):
        half = np.full(len(sub), pc["crack_pt"] / 2 / pt_x)
        cracks.loc[sub.index, "xd"] = dodge(sub.x.to_numpy(), half, gap=1.5 / pt_x)
    clash, laid = [], []
    for r in cracks.itertuples():  # off the standing glyphs and the cracks laid so far
        xd, b = slide(fig, ax, r, r.xd, boxes + laid, pt_x, steps=pc["crack_slide"])
        cracks.loc[r.Index, "xd"] = xd
        laid.append(b)
        if any(b.padded(-1.5).overlaps(o.padded(-1.5)) for o in boxes):
            clash.append((f"crack@{r.lane} week {xd:.1f}", "glyph"))
    return cracks, clash


def offsets(base, factors=(1, 1.8, 2.8, 4.2)) -> list:
    return [(dx * f, dy * f, ha, va) for f in factors for dx, dy, ha, va in base]


TENT_AROUND = offsets([(6, 9, "left", "bottom"), (6, -6, "left", "top")])  # off the lane: the last resort
TENT_STEP, TENT_LEFT = 6, 4  # pt: slide along the trail in these steps; air between a left label and its tent
FLAG_OPTS = offsets(
    [
        (7, -1, "left", "center"),
        (4, 4, "left", "bottom"),
        (-3, 4, "right", "bottom"),
        (4, -9, "left", "top"),
        (-3, -9, "right", "top"),
    ],
    (1, 1.8, 2.8, 4.2, 6),
)
CAIRN_OPTS = offsets(
    [
        (7, -7, "left", "top"),
        (7, 7, "left", "bottom"),
        (-7, -7, "right", "top"),
        (-7, 7, "right", "bottom"),
        (0, -10, "center", "top"),
        (0, 10, "center", "bottom"),
    ]
)
LEADER_MIN = 11  # pt: a label set off its glyph farther than this (and not on its own trail) gets a leader


class Labels:
    """Greedy label placement in pixel space (see the module docstring)."""

    def __init__(self, fig, ax, boxes, fixed, stand, edge, fs):
        self.fig, self.ax, self.stand, self.fs = fig, ax, stand, fs
        fig.canvas.draw()
        self.r = fig.canvas.get_renderer()
        f = ax.get_window_extent(self.r)
        self.frame = Bbox.from_extents(f.x0 + 2, f.y0 + 2, f.x1 - 2, f.y1 - 2)
        self.obstacles = list(boxes) + [t.get_window_extent(self.r).padded(2) for t in fixed if t.get_text()]
        self.px, self.to_px = fig.dpi / 72, ax.transData.transform
        tents = stand[(stand.kind == "tent") & stand.trail.astype(bool)]
        self.trails = {
            i: (self.to_px((r.xd, r.lane)), self.to_px((edge, r.lane))) for i, r in tents.iterrows()
        }
        self.halo = dict(boxstyle="round,pad=0.15", fc=SURFACE, ec="none", alpha=0.86)
        self.placed, self.leaders, self.misses = [], [], []

    def run(self) -> list:
        ink = dict(fontsize=self.fs["label"], color=TEXT)
        st = self.stand[self.stand.label.fillna("") != ""]
        to_pt = 72 / self.fig.dpi
        for i, r in st[st.kind == "tent"].iterrows():  # on its own lane: along its trail, else left of it
            room = (self.trails[i][1][0] - self.trails[i][0][0]) * to_pt if i in self.trails else 0
            half = r.ex1
            opts = [(dx, 0, "left", "center") for dx in np.arange(half, max(room, half + 1), TENT_STEP)]
            opts += [(-half - TENT_LEFT, 0, "right", "center"), *TENT_AROUND]
            self.place((r.xd, r.lane), r.label, opts, ("tent", r.label), required=True, own=i, **ink)
        flags = st[st.kind == "flag"]
        for _, sub in flags.groupby("group"):
            top = sub.loc[sub.xd.idxmax()]  # off the latest flag of the group
            xy = (top.xd, top.lane + ANCHOR["flag"] - 1.0)
            self.place(xy, top.label, FLAG_OPTS, ("flag", top.label), **ink)
        notes = []
        for r in st[st.kind == "cairn"].sort_values("weight", ascending=False).itertuples():
            rx, ry = r.ex1 * 0.6, (r.ey1 - r.ey0) / 2  # around the stack's centre
            opts = [(dx + np.sign(dx) * rx, dy + np.sign(dy) * ry, ha, va) for dx, dy, ha, va in CAIRN_OPTS]
            notes.append(
                self.place(
                    (r.xd, r.yc),
                    r.label,
                    opts,
                    ("cairn", r.label),
                    shrink=max(rx, ry) * 0.8,
                    fontsize=self.fs["label"],
                    color=TEXT_SECONDARY,
                )
            )
        self.uncross([n for n in notes if n and n["lead"] is not None])
        return self.misses

    def place(self, xy, text, opts, key, required=False, own=None, shrink=2, **kw):
        best = None
        for dx, dy, ha, va in opts:
            a = self.ax.annotate(
                text,
                xy,
                xytext=(dx, dy),
                textcoords="offset points",
                ha=ha,
                va=va,
                zorder=9,
                bbox=self.halo,
                **kw,
            )
            bb = a.get_window_extent(self.r).padded(2.5)
            ext = (bb.x0, bb.y0, bb.x1, bb.y1)
            hard = sum(bb.overlaps(o) for o in self.placed + self.obstacles)
            hard += (not self.frame.contains(bb.x0, bb.y0)) + (not self.frame.contains(bb.x1, bb.y1))
            soft = sum(
                bb.y0 < p[1] < bb.y1 and bb.x1 > p[0] and bb.x0 < q[0]
                for k, (p, q) in self.trails.items()
                if k != own
            )
            soft += sum(cuts(ext, *seg) for seg in self.leaders)  # an earlier leader runs under this label
            if dy != 0 and np.hypot(dx, dy) > LEADER_MIN:  # needs a leader: it must cross nothing
                seg = (self.to_px(xy), self.to_px(xy) + np.array([dx, dy]) * self.px)
                soft += sum(crosses(*seg, *other) for other in [*self.leaders, *self.trails.values()])
                soft += sum(cuts(o.extents, *seg) for o in self.placed)
                soft += sum(cuts(o.extents, *seg) for o in self.obstacles if not o.contains(*seg[0]))
            score = (hard, soft)
            if best is None or score < best[0]:
                if best:
                    best[1].remove()
                best = (score, a, bb, (dx, dy))
            else:
                a.remove()
            if score == (0, 0):
                break
        (hard, _), a, bb, (dx, dy) = best
        if hard and not required:
            a.remove()
            self.misses.append((key, "skipped"))
            return None
        if hard:
            self.misses.append((key, f"overlap {hard}"))
        self.placed.append(bb)
        lead = None
        if dy != 0 and np.hypot(dx, dy) > LEADER_MIN:
            self.leaders.append((self.to_px(xy), self.to_px(xy) + np.array([dx, dy]) * self.px))
            lead = self.leader(xy, (dx, dy), shrink)
        return {"ann": a, "lead": lead, "xy": xy, "off": np.array([dx, dy], float), "shrink": shrink}

    def leader(self, xy, off, shrink):
        props = dict(arrowstyle="-", lw=0.5, color=TEXT_SECONDARY, shrinkA=0, shrinkB=shrink, patchA=None)
        return self.ax.annotate(
            "", xy, xytext=tuple(off), textcoords="offset points", arrowprops=props, zorder=8
        )

    def uncross(self, notes) -> None:
        """Two labels whose leaders cross trade texts (same-width ids, so their boxes stay clear)."""
        for A, B in combinations(notes, 2):
            pa, pb = (self.to_px(n["xy"]) + n["off"] * self.px for n in (A, B))
            if crosses(self.to_px(A["xy"]), pa, self.to_px(B["xy"]), pb):
                ta, tb = A["ann"].get_text(), B["ann"].get_text()
                A["ann"].set_text(tb)
                B["ann"].set_text(ta)
                A["ann"], B["ann"] = B["ann"], A["ann"]
                for n, p in ((A, pb), (B, pa)):
                    n["lead"].remove()
                    n["off"] = (p - self.to_px(n["xy"])) / self.px
                    n["lead"] = self.leader(n["xy"], n["off"], n["shrink"])
                self.misses.append((("cairn", ta, tb), "swapped to uncross leaders"))


def side_panel(fig, ax, side, text, pc) -> None:
    """Bars per lane right of the map (sharing its lanes), values on the long ones."""
    fs = pc["font"]
    gx = fig.add_axes(pc["side_box"], sharey=ax)
    gx.set_facecolor(SURFACE)
    gx.barh(side.lane, side.value, height=0.62, color=OTHER, lw=0)
    for r in side[side.value >= pc["side_label_min"]].itertuples():
        gx.text(
            r.value + 0.4,
            r.lane,
            str(r.value),
            va="center",
            ha="left",
            fontsize=fs["lane"] - 0.5,
            color=TEXT_SECONDARY,
        )
    gx.set_xlim(0, max(side.value.max(), 1) * 1.25)
    gx.set_xticks(pc["side_ticks"])
    gx.tick_params(axis="x", length=3, color=GRID, labelsize=fs["lane"], labelcolor=TEXT_SECONDARY)
    gx.tick_params(axis="y", length=0, labelleft=False)
    for name, s in gx.spines.items():
        s.set_visible(name == "left")
        s.set_color(GRID)
    gx.text(
        0,
        1.006,
        text["side_title"],
        transform=gx.transAxes,
        ha="left",
        va="bottom",
        fontsize=fs["block"],
        color=TEXT,
        weight="bold",
    )
    note = text["side_note"].format(total=int(side.value.sum()), lanes=len(side))
    gx.text(
        0,
        -0.035,
        note,
        transform=gx.transAxes,
        ha="left",
        va="top",
        fontsize=fs["lane"],
        color=TEXT_SECONDARY,
    )


def legend_row(fig, levels, cmap, text, pc) -> None:
    """One row under the subtitle: the elevation scale, then every glyph with its label."""
    fs, lg, y = pc["font"], text["legend"], 0.915
    status = STATUS[pc["status"]]
    x0 = 0.008 + fs["lane"] * 1.2 / 72 / fig.get_figwidth()  # room for half the first tick label
    cax = fig.add_axes([x0, y - 0.006, 0.16, 0.012])
    n = len(levels) - 1
    cax.imshow(np.arange(n)[None, :], cmap=cmap, aspect="auto", extent=(0, n, 0, 1))
    for k in range(1, n):
        cax.axvline(k, color=SEQUENTIAL[pc["line_step"]], lw=pc["line_width"], alpha=pc["line_alpha"])
    cax.set_xticks(range(n + 1), [f"{v:g}" for v in levels[:-1]] + [f"{round(levels[-1], 1):g}"])  # peak
    cax.tick_params(axis="x", length=0, labelsize=fs["lane"], labelcolor=TEXT_SECONDARY, pad=1.5)
    cax.set_yticks([])
    for s in cax.spines.values():
        s.set_visible(False)
    fig.text(
        x0 + 0.164, y, text["scale"], ha="left", va="center", fontsize=fs["legend"], color=TEXT_SECONDARY
    )
    lx = fig.add_axes([0.27, y - 0.02, 0.73, 0.04])
    lx.set_xlim(0, 1)
    lx.set_ylim(-1, 1)
    lx.axis("off")
    x = 0.0

    def add(draw, key, width):  # an item whose key is not in the legend text is left out
        nonlocal x
        if key not in lg:
            return
        draw(x)
        t = lx.text(
            x + width + 0.006, 0, lg[key], ha="left", va="center", fontsize=fs["legend"], color=TEXT_SECONDARY
        )
        fig.canvas.draw()
        x = (
            lx.transAxes.inverted().transform((t.get_window_extent(fig.canvas.get_renderer()).x1, 0))[0]
            + 0.022
        )

    unit = 72 * fig.get_figwidth() * lx.get_position().width  # points per legend-axes unit
    pt_y = 72 * fig.get_figheight() * lx.get_position().height / 2  # points per legend-axes y unit
    stacks = [cairn_stack(stone_count(w), pc["stone_pt"]) for w in pc["legend_weights"]]
    step = (pc["stone_pt"] + 4) / unit
    add(
        lambda x0: lx.axvspan(x0, x0 + 0.012, ymin=0.2, ymax=0.8, color=TEXT, alpha=0.075, lw=0),
        "band",
        0.012,
    )
    add(
        lambda x0: lx.scatter(
            [x0 + 0.004],
            [-0.45],
            s=(pc["flag_pt"] * 2) ** 2 * 0.8,
            marker=FLAG,
            c=TEXT,
            edgecolors=SURFACE,
            linewidths=0.6,
        ),
        "flag",
        0.008,
    )
    add(
        lambda x0: lx.scatter(
            [x0 + 0.0095],
            [-0.4],
            s=(pc["tent_pt"] * 2) ** 2,
            marker=TENT,
            c=status,
            edgecolors=TEXT,
            linewidths=0.7,
        ),
        "tent",
        0.019,
    )
    add(lambda x0: lx.plot([x0, x0 + 0.02], [0, 0], color=status, lw=1.3, ls=(0, (1, 1.6))), "trail", 0.02)

    def cairns(x0):
        base = -(stacks[-1][2][3] / 2) / pt_y  # the tallest stack is centred on the row
        for k, (m, size, _) in enumerate(stacks):
            lx.scatter(
                [x0 + step * (k + 0.5)],
                [base],
                s=size**2,
                marker=m,
                c=TEXT_SECONDARY,
                edgecolors=SURFACE,
                linewidths=0.6,
            )

    add(cairns, "cairn", step * len(stacks))
    add(
        lambda x0: lx.scatter(
            [x0 + 0.005],
            [0],
            s=pc["crack_pt"] ** 2 * 1.4,
            marker=CRACK,
            facecolors="none",
            edgecolors=TEXT_SECONDARY,
            linewidths=0.8,
        ),
        "crack",
        0.01,
    )
    add(lambda x0: lx.barh([0], [0.014], left=x0, height=0.7, color=OTHER), "side", 0.014)
    add(lambda x0: lx.plot([x0 + 0.004] * 2, [-0.8, 0.8], color=TEXT_SECONDARY, lw=0.9), "edge", 0.006)
