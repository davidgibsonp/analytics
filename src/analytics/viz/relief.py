"""3D relief of a terrain grid: a shaded surface in one sequential hue with glyphs standing on it.

relief() draws a static matplotlib view: the (lanes, weeks) height grid (features.terrain), upsampled
by PCHIP (no overshoot), coloured by height in one hue, with a hillshade that moves along the same
colour ramp, contour lines (left out on `flat` blocks, e.g. a capped strip) and dashed valley lines
on the floor between blocks all baked into the texture (so ridges in front hide them), a cut face at
the time end facing the viewer, block labels with brackets along one lane edge (pushed outward until
clear of every bracket) and month ticks along one week edge. Glyphs (viz.glyphs: poles with
pennants, tents, stone cairns) are drawn in a pixel overlay at their projected positions, slid along
the time direction until clear of each other (at most max_shift_weeks; past that a glyph takes the
least crowded spot and is reported as touching), and a few callouts are placed by cost (text on text
near-forbidden; leaders through glyphs or across leaders, labels across the frame and leader stubs
shorter than leader_min penalised).

relief_html() writes the same terrain as an interactive plotly page (drag to rotate, hover for titles).
plotly is optional (the viz extra), imported only there.

    blocks : first_lane, last_lane, mid_lane, label, flat (bool)
    glyphs : kind (pole, tent, cairn), key, lane, x, n (pennants or stones); relief_html also takes
             kind "route" (a line through the points in order) and hover (a list per pennant for poles)
    callouts (relief): [{"keys": [glyph keys], "text": str}] or [{"at": (lane, x), "text": str}] for a
             point on the surface (e.g. a peak)
    text   : title, subtitle, scale, shift ({shift_days}: under the legend), legend {pole, cairn,
             tent: [line, second line]};
             relief_html: title, subtitle, scale, gap (hover over gap lanes), names {kind: legend name}
    pc     : the [terrain3d]-style options (view, box, ramp, contour step, valley dash, light, sizes,
             max_shift_weeks, leader_min)

relief() returns counts drawn, the largest shift per kind (days), clashes (label on label or on a
bracket, glyphs left touching) and the PNG size in KB.

Coordinates: x = week_float (a vertex at week centre k + 0.5), y = (n_lanes - 1) - lane, so lane 0
is at the back, z = height.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.colors import LightSource, Normalize, to_rgb
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.interpolate import RegularGridInterpolator

from analytics.features.terrain import month_starts, upsample
from analytics.viz import glyphs as gl
from analytics.viz.pack import crosses, cuts, nearest_on_box, overlap, segment_in_box
from analytics.viz.style import (
    BASELINE,
    HAIRLINE,
    SEQUENTIAL,
    STATUS,
    SURFACE,
    TEXT,
    TEXT_SECONDARY,
    optimize_png,
    plt,
    sequential_cmap,
    texts_outside,
)

ORDER = {"pole": 0, "tent": 1, "cairn": 2}  # placement order: poles keep their spot, cairns move first


def fine_grid(height, factor):
    """Lane and week positions of the upsampled grid, and the heights there."""
    n_lanes, n_weeks = height.shape
    lane_f = np.linspace(0, n_lanes - 1, (n_lanes - 1) * factor + 1)
    x_f = np.linspace(0.5, n_weeks - 0.5, (n_weeks - 1) * factor + 1)
    return lane_f, x_f, upsample(height, lane_f, x_f)


def month_ticks(weeks, n_weeks):
    """(x, label) per month start on the axis; the year on the first tick and on January."""
    months, mx = month_starts(weeks, weeks[0] + pd.Timedelta(weeks=n_weeks))
    labels = [
        m.strftime("%b %Y") if (i == 0 or m.month == 1) else m.strftime("%b") for i, m in enumerate(months)
    ]
    return [(x, lab) for x, lab in zip(mx, labels, strict=True) if x <= n_weeks - 0.5]


# ------------------------------------------------------------------------------------------ geometry
class Scene:
    """3D axes plus a pixel overlay; projects data points to overlay pixels; tracks occupied boxes."""

    def __init__(self, fig, ax, box):
        self.fig, self.ax, self.box = fig, ax, box
        ax.apply_aspect()
        self.M = ax.get_proj()
        self.r = fig.canvas.get_renderer()
        self.W, self.H = fig.bbox.width, fig.bbox.height
        self.ov = fig.add_axes([0, 0, 1, 1], zorder=5)
        self.ov.set_xlim(0, self.W)
        self.ov.set_ylim(0, self.H)
        self.ov.axis("off")
        self.ov.patch.set_alpha(0)
        self.boxes: list[np.ndarray] = []  # occupied [x0, y0, x1, y1] in pixels
        self.texts: list[np.ndarray] = []  # the boxes that hold text
        self.leaders: list[tuple[np.ndarray, np.ndarray]] = []
        self.frame: list[tuple[np.ndarray, np.ndarray]] = []  # outline edges callouts must not cross
        self.clashes: list[tuple[str, str]] = []

    def px(self, x, y, z) -> np.ndarray:
        return project(self.ax, x, y, z, self.M)

    def depth(self, x, y, z) -> np.ndarray:
        """Distance toward the viewer in box-scaled units (bigger = nearer)."""
        e, a = np.radians(self.ax.elev), np.radians(self.ax.azim)
        v = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
        lim = [self.ax.get_xlim3d(), self.ax.get_ylim3d(), self.ax.get_zlim3d()]
        s = [self.box[i] / (lim[i][1] - lim[i][0]) for i in range(3)]
        return sum(v[i] * s[i] * np.asarray(c, float) for i, c in enumerate((x, y, z)))

    def bbox(self, t) -> np.ndarray:
        b = t.get_window_extent(self.r)
        return np.array([b.x0, b.y0, b.x1, b.y1])


def project(ax, x, y, z, M=None) -> np.ndarray:
    x, y, z = (np.atleast_1d(np.asarray(v, float)) for v in (x, y, z))
    xs, ys, _ = proj3d.proj_transform(x, y, z, ax.get_proj() if M is None else M)
    return ax.transData.transform(np.column_stack([xs, ys]))


def fit(fig, ax, pts, target, box, zoom=1.0) -> None:
    """Zoom and shift the 3D axes so the projected points fill `target` (figure fractions)."""
    W, H = fig.bbox.width, fig.bbox.height
    tx0, ty0, tx1, ty1 = target[0] * W, target[1] * H, target[2] * W, target[3] * H
    for _ in range(4):
        ax.apply_aspect()
        p = project(ax, *pts)
        zoom *= min((tx1 - tx0) / np.ptp(p[:, 0]), (ty1 - ty0) / np.ptp(p[:, 1]))
        ax.set_box_aspect(box, zoom=zoom)
        ax.apply_aspect()
        p = project(ax, *pts)
        shift = np.array([(tx0 + tx1) / 2, (ty0 + ty1) / 2]) - (p.min(0) + p.max(0)) / 2
        pos = ax.get_position(original=True)
        ax.set_position([pos.x0 + shift[0] / W, pos.y0 + shift[1] / H, pos.width, pos.height])


def outward(sc: Scene, edge_pt, inside_pt):
    """Alignment for a label just outside an edge, from the outward screen direction."""
    d = sc.px(*edge_pt)[0] - sc.px(*inside_pt)[0]
    if abs(d[0]) >= abs(d[1]):
        return ("left" if d[0] > 0 else "right"), "center"
    return "center", ("bottom" if d[1] > 0 else "top")


def repel_along(sc: Scene, items, u, pad=3.0) -> None:
    """Shift text artists along screen direction u until their boxes stop overlapping."""
    u = np.asarray(u, float) / np.hypot(*u)
    items.sort(key=lambda it: it["pos"] @ u)
    for _ in range(60):
        moved = False
        for a, b in zip(items, items[1:], strict=False):
            ba, bb = sc.bbox(a["t"]), sc.bbox(b["t"])
            if overlap(ba + [-pad, -pad, pad, pad], bb) > 0:
                ext = [abs((x[2] - x[0]) * u[0]) + abs((x[3] - x[1]) * u[1]) for x in (ba, bb)]
                need = (ext[0] + ext[1]) / 2 + pad - (b["pos"] - a["pos"]) @ u
                for it, sgn in ((a, -0.5), (b, 0.5)):
                    it["pos"] = it["pos"] + sgn * max(need, 1.0) * u
                    it["t"].set_position(it["pos"])
                moved = True
        if not moved:
            break


# ------------------------------------------------------------------------------------------ static
def texture(lane_f, x_f, Z, zmax, pc, no_lines=None, valleys=()):
    """Sequential colour by height; a hillshade and contour lines that move a face along the same ramp
    (a slope turned from the light, or a contour, reads a little darker, never grey); dashed valley
    lines on the floor along each lane in `valleys`; all baked into the texture."""
    box = pc["box"]
    sx, sy, sz = box[0] / (x_f[-1] - x_f[0]), box[1] / (lane_f[-1] - lane_f[0]), box[2] / zmax
    light = LightSource(azdeg=pc["light_az"], altdeg=pc["light_alt"]).direction
    # LightSource.hillshade's normals and light, without its contrast stretch, so flat ground keeps
    # exactly its height colour (rows run like an image: first row on top, hence -dy)
    dy, dx = np.gradient(Z * sz, sy * (lane_f[1] - lane_f[0]), sx * (x_f[1] - x_f[0]))
    normal = np.stack([-dx, dy, np.ones_like(Z)], axis=-1)
    shade = (normal / np.linalg.norm(normal, axis=-1, keepdims=True)) @ light - light[2]  # 0 when flat
    level = Normalize(0, zmax)(Z) - pc["shade"] * shade
    band = np.floor(Z / pc["contour_step"] + 1e-6)
    corners = np.stack([band[:-1, :-1], band[1:, :-1], band[:-1, 1:], band[1:, 1:]])
    line = np.zeros(Z.shape, bool)  # face (r, c) spans vertices r..r+1, c..c+1
    line[:-1, :-1] = np.ptp(corners, axis=0) > 0
    if no_lines is not None:
        line[no_lines] = False
    level[line] += pc["contour_shift"]
    rgb = sequential_cmap(pc["ramp"], pc["ramp_at"])(np.clip(level, 0, 1))[..., :3]
    for v in valleys:  # face rows start at a vertex: the row starting nearest the valley lane
        r = int(np.argmin(np.abs(lane_f - v)))
        dash = np.floor(x_f / pc["valley_dash"]) % 2 == 0
        rgb[r, dash] += pc["valley_alpha"] * (np.array(to_rgb(BASELINE)) - rgb[r, dash])
    return np.concatenate([rgb, np.ones(Z.shape + (1,))], axis=-1)


def relief(height, weeks, blocks, glyphs, callouts, text, pc, path) -> dict:
    n_lanes, n_weeks = height.shape
    lane_f, x_f, Z = fine_grid(height, pc["upsample"])
    zmax = float(np.ceil(height.max()))
    Y = (n_lanes - 1) - lane_f
    zat = RegularGridInterpolator((lane_f, x_f), Z, bounds_error=False, fill_value=None)
    fig = plt.figure(figsize=(pc["width"], pc["height"]), dpi=pc["dpi"], facecolor=SURFACE)
    ax = fig.add_axes([0, 0, 1, 1], projection="3d", computed_zorder=False)
    ax.set_proj_type("ortho")
    ax.view_init(pc["elev"], pc["azim"])
    box = pc["box"]
    ax.set_box_aspect(box)
    ax.set_axis_off()
    ax.set_facecolor((0, 0, 0, 0))
    ax.set_xlim(0, n_weeks)
    ax.set_ylim(-0.5, n_lanes - 0.5)
    ax.set_zlim(0, zmax)
    Xg, Yg = np.meshgrid(x_f, Y)
    sub = (slice(None, None, 2), slice(None, None, 2))
    corners = np.array([[0, -0.5], [n_weeks, -0.5], [0, n_lanes - 0.5], [n_weeks, n_lanes - 0.5]])
    pts = (
        np.r_[Xg[sub].ravel(), corners[:, 0]],
        np.r_[Yg[sub].ravel(), corners[:, 1]],
        np.r_[Z[sub].ravel(), [0] * 4],
    )
    fit(fig, ax, pts, pc["target"], box)
    no_lines = np.zeros(len(lane_f), bool)
    for b in blocks[blocks.flat].itertuples():  # a contour on a capped strip would only trace its rim
        no_lines |= (lane_f > b.first_lane - 1) & (lane_f < b.last_lane + 1)
    B = blocks.sort_values("first_lane")
    valleys = (B.last_lane.to_numpy()[:-1] + B.first_lane.to_numpy()[1:]) / 2  # mid gap between blocks
    surf = ax.plot_surface(
        Xg,
        Yg,
        Z,
        facecolors=texture(lane_f, x_f, Z, zmax, pc, no_lines, valleys),
        rstride=1,
        cstride=1,
        linewidth=0,
        antialiased=False,
        shade=False,
        zorder=1,
    )
    surf.set_clip_on(False)
    (outline,) = ax.plot(
        [0, n_weeks, n_weeks, 0, 0],
        [-0.5, -0.5, n_lanes - 0.5, n_lanes - 0.5, -0.5],
        [0] * 5,
        color=BASELINE,
        lw=0.8,
        zorder=0,
    )
    outline.set_clip_on(False)
    facing = np.cos(np.radians(pc["azim"]))  # the cut face at the time end that faces the viewer
    if abs(facing) > 0.5:  # edge-on faces only add slivers
        j = -1 if facing > 0 else 0
        xs = x_f[j] + (0.5 if j == -1 else -0.5)
        verts = [(xs, Y[0], 0), *[(xs, y, z) for y, z in zip(Y, Z[:, j], strict=True)], (xs, Y[-1], 0)]
        skirt = Poly3DCollection([verts], facecolor=HAIRLINE, edgecolor=BASELINE, linewidth=0.6, zorder=2)
        ax.add_collection3d(skirt)
        skirt.set_clip_on(False)
    sc = Scene(fig, ax, box)
    rim = sc.px([0, n_weeks, n_weeks, 0], [-0.5, -0.5, n_lanes - 0.5, n_lanes - 0.5], [0] * 4)
    sc.frame = [(rim[k], rim[(k + 1) % 4]) for k in range(4)]
    edge_labels(sc, blocks, weeks, n_lanes, n_weeks, pc)

    def z_on(lane, x):
        return zat(np.column_stack([np.asarray(lane, float), np.clip(np.asarray(x, float), x_f[0], x_f[-1])]))

    G = place_glyphs(sc, glyphs, n_lanes, z_on, pc)
    shift = ((G.moved.abs() / G.px_per_week) * 7).groupby(G.kind).max().round(1).to_dict()  # days
    top = sc.H - 24
    sc.ov.text(30, top, text["title"], fontsize=pc["font"]["title"], color=TEXT, va="top")
    t = sc.ov.text(
        30,
        top - 36,
        text["subtitle"],
        fontsize=pc["font"]["subtitle"],
        color=TEXT_SECONDARY,
        va="top",
        linespacing=1.5,
    )
    sc.boxes.append(sc.bbox(t) + [-6, -6, 6, 40])
    note = text["shift"].format(shift_days=int(np.ceil(max(shift.values(), default=0))))
    sc.boxes.append(legend(sc, zmax, text, pc, x0=sc.W - pc["legend_width"], y0=top + 2, note=note))
    sc.texts += sc.boxes[-2:]
    for c in callouts:
        if "keys" in c:
            gboxes = [np.asarray(G.loc[G.key == k].iloc[0].box, float) for k in c["keys"]]
        else:  # a point on the surface
            lane, x = c["at"]
            p = sc.px(x, (n_lanes - 1) - lane, z_on([lane], [x])[0])[0]
            gboxes = [np.r_[p - 3, p + 3]]
        place_callout(sc, gboxes, c["text"], pc["font"]["callout"], pc["leader_min"])
    sc.clashes += [(t, "off the canvas") for t in texts_outside(fig)]
    fig.savefig(path, facecolor=SURFACE, dpi=pc["dpi"])
    plt.close(fig)
    drawn = G.kind.value_counts().to_dict() | {"pennants": int(G.n[G.kind == "pole"].sum())}
    kb = round(optimize_png(path, pc["max_kb"]))
    return dict(drawn=drawn, shift_days=shift, clashes=sc.clashes, kb=kb)


def edge_labels(sc, blocks, weeks, n_lanes, n_weeks, pc) -> None:
    """Block labels (with a bracket over their lanes) along one lane edge, month ticks along one week
    edge; a tick that would land on a block label is dropped."""
    ov, fs = sc.ov, pc["font"]
    x0_edge = pc["label_edge"] == "x0"
    xe, xi = (-1.2, 2.0) if x0_edge else (n_weeks + 1.2, n_weeks - 2.0)
    xb = -0.45 if x0_edge else n_weeks + 0.45
    labels, brackets = [], []
    for b in blocks.itertuples():
        y0, y1 = (n_lanes - 1) - b.first_lane + 0.35, (n_lanes - 1) - b.last_lane - 0.35
        p = sc.px([xb, xb], [y0, y1], [0, 0])
        ov.plot(p[:, 0], p[:, 1], color=TEXT_SECONDARY, lw=1.6, solid_capstyle="butt", zorder=4)
        brackets.append((p[0], p[1]))
        ym = (n_lanes - 1) - b.mid_lane
        ha, va = outward(sc, (xe, ym, 0), (xi, ym, 0))
        anchor = sc.px(xe, ym, 0)[0]
        t = ov.text(*anchor, b.label, ha=ha, va=va, fontsize=fs["block"], color=TEXT, zorder=6)
        labels.append({"t": t, "pos": anchor.copy(), "anchor": anchor, "bracket": p.mean(0)})
    along = sc.px(xe, n_lanes, 0)[0] - sc.px(xe, 0, 0)[0]
    out = sc.px(xe, 0, 0)[0] - sc.px(xi, 0, 0)[0]
    out = out / np.hypot(*out)

    def on_bracket(b, pad=3):  # the text box crosses or holds a bracket line (not just its bounding box)
        return any(segment_in_box(b + [-pad, -pad, pad, pad], *k) for k in brackets)

    for _ in range(3):  # slide apart along the edge, then step outward off any bracket; repeat
        repel_along(sc, labels, along, pad=pc["label_gap"])
        for it in labels:
            for _ in range(40):
                if not on_bracket(sc.bbox(it["t"])):
                    break
                it["pos"] = it["pos"] + 2 * out
                it["t"].set_position(it["pos"])
    for i, it in enumerate(labels):
        b = sc.bbox(it["t"])
        name = it["t"].get_text()
        sc.clashes += [(name, "bracket")] if on_bracket(b, pad=0) else []
        sc.clashes += [(name, o["t"].get_text()) for o in labels[i + 1 :] if overlap(b, sc.bbox(o["t"])) > 0]
        sc.boxes.append(b)
        sc.texts.append(b)
        if np.hypot(*(it["pos"] - it["anchor"])) > 4:  # moved: point back at its bracket
            end = nearest_on_box(it["bracket"], b + [-2, -2, 2, 2])
            ov.plot(*zip(it["bracket"], end, strict=True), color=BASELINE, lw=0.8, zorder=4)
    y0_edge = pc["tick_edge"] == "y0"
    ye, yi = (-1.6, 1.0) if y0_edge else (n_lanes + 0.6, n_lanes - 2.0)
    yt = -0.5 if y0_edge else n_lanes - 0.5
    ticks = []
    for x, lab in month_ticks(weeks, n_weeks):
        ha, va = outward(sc, (x, ye, 0), (x, yi, 0))
        a, b = sc.px([x, x], [yt, ye + (0.5 if y0_edge else -0.5)], [0, 0])
        ov.plot([a[0], b[0]], [a[1], b[1]], color=BASELINE, lw=0.9, zorder=4)
        anchor = sc.px(x, ye, 0)[0]
        t = ov.text(*anchor, lab, ha=ha, va=va, fontsize=fs["tick"], color=TEXT_SECONDARY, zorder=6)
        ticks.append({"t": t, "pos": anchor.copy()})
    repel_along(sc, ticks, sc.px(n_weeks, ye, 0)[0] - sc.px(0, ye, 0)[0], pad=4)
    taken = list(sc.boxes)
    for it in ticks:
        b = sc.bbox(it["t"])
        if any(overlap(b + [-3, -3, 3, 3], o) > 0 for o in taken):
            it["t"].remove()
        else:
            sc.boxes.append(b)
            sc.texts.append(b)


def place_glyphs(sc, glyphs, n_lanes, z_on, pc) -> pd.DataFrame:
    """Anchor glyphs on the surface, slide each along the time direction (smallest shift first, either
    way, at most max_shift_weeks) until clear of earlier glyphs on its lane and of every pole and tent
    (cairns on different lanes may touch: their surface ring separates them); a glyph with no clear
    spot within the cap takes the least crowded one and is reported. Then draw back to front."""
    G = glyphs[glyphs.kind.isin(ORDER)].reset_index(drop=True).copy()
    G["n"] = G.n.astype(int)
    G["z"] = z_on(G.lane.to_numpy(float), G.x.to_numpy(float))
    G["y"] = (n_lanes - 1) - G.lane
    P = sc.px(G.x, G.y, G.z)
    G["px"], G["py"] = P[:, 0], P[:, 1]
    u = sc.px(1, 0, 0)[0] - sc.px(0, 0, 0)[0]
    G["px_per_week"] = np.hypot(*u)
    u = u / np.hypot(*u)  # screen direction of time
    G["t"], G["moved"] = G.px * u[0] + G.py * u[1], 0.0
    cap = pc["max_shift_weeks"] * G.px_per_week.iloc[0] if len(G) else 0
    steps = [0, *[k * s for k in range(2, int(cap) + 1, 2) for s in (1, -1)]]
    placed = []
    for i in G.assign(order=G.kind.map(ORDER)).sort_values(["order", "t"]).index:
        kind, lane, n = G.at[i, "kind"], G.at[i, "lane"], G.at[i, "n"]
        best = None
        for step in steps:
            b = gl.glyph_box(kind, G.at[i, "px"] + step * u[0], G.at[i, "py"] + step * u[1], n)
            hit = sum(overlap(b, o) for ln, kd, o in placed if ln == lane or kd != "cairn" or kind != "cairn")
            if best is None or hit < best[0]:
                best = (hit, step)
            if hit == 0:
                break
        hit, step = best
        if hit > 0:
            sc.clashes.append((f"{kind} {G.at[i, 'key']}", "touching (no clear spot within max_shift_weeks)"))
        x, y = G.at[i, "px"] + step * u[0], G.at[i, "py"] + step * u[1]
        G.at[i, "px"], G.at[i, "py"], G.at[i, "moved"] = x, y, step
        placed.append((lane, kind, gl.glyph_box(kind, x, y, n, keep_clear=True)))
    G["depth"] = sc.depth(G.x, G.y, G.z)
    G["box"] = None
    fill = STATUS[pc["status"]]
    for i in G.sort_values("depth").index:
        g = G.loc[i]
        if g.kind == "tent":
            G.at[i, "box"] = gl.tent(sc.ov, g.px, g.py, z=10 + 0.001 * len(sc.boxes), fill=fill)
        else:
            G.at[i, "box"] = gl.DRAW[g.kind](sc.ov, g.px, g.py, g.n)
    sc.boxes += list(G.box)
    return G


def legend(sc: Scene, zmax, text, pc, x0, y0, note="") -> np.ndarray:
    """Three glyph rows (two lines each), the height scale and a note under it; returns its box."""
    ov, fs, y = sc.ov, pc["font"], y0
    draw = {
        "pole": lambda y: gl.pole(ov, x0 + 6, y - 38, 1, s=0.95),
        "cairn": lambda y: gl.stones(ov, x0 + 10, y - 36, 3),
        "tent": lambda y: gl.tent(ov, x0 + 10, y - 34, fill=STATUS[pc["status"]]),
    }
    for kind, (a, b) in text["legend"].items():
        draw[kind](y)
        ov.text(x0 + 32, y, a, fontsize=fs["legend"], color=TEXT, va="top")
        ov.text(x0 + 32, y - 21, b, fontsize=fs["legend"] - 0.5, color=TEXT_SECONDARY, va="top")
        y -= 48
    bw, bh, by = 180, 9, y - 14
    ov.imshow(
        np.linspace(0, 1, 256)[None, :],
        extent=(x0 + 32, x0 + 32 + bw, by, by + bh),
        cmap=sequential_cmap(pc["ramp"], pc["ramp_at"]),
        aspect="auto",
        zorder=6,
    )
    ov.add_patch(Rectangle((x0 + 32, by), bw, bh, fc="none", ec=BASELINE, lw=0.6, zorder=7))
    for v in range(int(zmax) + 1):
        ov.text(
            x0 + 32 + bw * v / zmax,
            by - 3,
            str(v),
            fontsize=fs["tick"] - 0.5,
            color=TEXT_SECONDARY,
            ha="center",
            va="top",
        )
    ov.text(
        x0 + 32 + bw + 10, by + bh / 2, text["scale"], fontsize=fs["tick"], color=TEXT_SECONDARY, va="center"
    )
    bottom = by - 20
    if note:
        t = ov.text(x0 + 32, by - 24, note, fontsize=fs["legend"] - 0.5, color=TEXT_SECONDARY, va="top")
        bottom = sc.bbox(t)[1] - 4
    ov.set_xlim(0, sc.W)
    ov.set_ylim(0, sc.H)
    return np.array([x0 - 6, bottom, sc.W, sc.H])


def place_callout(sc: Scene, gboxes, text, size, leader_min) -> None:
    """Try positions on rings around the glyph boxes; cost = text overlap (near-forbidden) + glyph
    overlap (heavy; within 8 px of a glyph, light) + distance + leaders through other glyphs or labels
    or across other leaders + the label across the frame + a leader stub shorter than leader_min px.
    Leaders start on the glyph box, at the side facing the label."""
    centre = np.mean([[(b[0] + b[2]) / 2, b[3]] for b in gboxes], axis=0)
    others = [o for o in sc.boxes if not any(np.array_equal(o, gb) for gb in gboxes)]
    best = None
    for r in (40, 60, 85, 115, 150, 190, 240, 290, 340):
        for ang in range(90, -270, -15):
            a = np.radians(ang)
            pos = centre + r * np.array([np.cos(a), np.sin(a)])
            ha = "left" if np.cos(a) > 0.35 else "right" if np.cos(a) < -0.35 else "center"
            va = "bottom" if np.sin(a) > 0.3 else "top" if np.sin(a) < -0.3 else "center"
            t = sc.ov.text(*pos, text, ha=ha, va=va, fontsize=size, linespacing=1.3)
            b = sc.bbox(t) + [-5, -4, 5, 4]
            t.remove()
            text_hit = sum(overlap(b, o) for o in sc.texts)
            starts = [nearest_on_box(((b[0] + b[2]) / 2, (b[1] + b[3]) / 2), gb) for gb in gboxes]
            cost = (
                (1e6 + 100 * text_hit if text_hit > 0 else 0)
                + sum(60 * overlap(b, o) for o in sc.boxes)  # on a glyph: heavy
                + sum(4 * (overlap(b + [-8, -8, 8, 8], o) - overlap(b, o)) for o in sc.boxes)  # near: light
                + r
                + sum(leader_cost(sc, p, b, others) for p in starts)
                + 2000.0 * sum(cuts(b, a, e) for a, e in sc.frame)
                + 1000.0 * sum(0 < np.hypot(*(nearest_on_box(p, b) - p)) < leader_min for p in starts)
            )
            if b[0] < 8 or b[1] < 8 or b[2] > sc.W - 8 or b[3] > sc.H - 8:
                cost += 1e7
            if best is None or cost < best[0]:
                best = (cost, pos, ha, va, b, starts)
    _, pos, ha, va, b, starts = best
    sc.ov.text(
        *pos,
        text,
        ha=ha,
        va=va,
        fontsize=size,
        color=TEXT,
        linespacing=1.3,
        zorder=12,
        bbox=dict(fc=SURFACE, ec="none", alpha=0.9, pad=2.0),
    )
    sc.boxes.append(b)
    sc.texts.append(b)
    for p in starts:
        end = nearest_on_box(p, b)
        if np.hypot(*(end - p)) > 0:
            sc.ov.plot([p[0], end[0]], [p[1], end[1]], color=TEXT_SECONDARY, lw=0.8, zorder=11)
            sc.leaders.append((p, end))


def leader_cost(sc: Scene, anchor, b, boxes) -> float:
    """Penalise a leader that crosses glyphs, labels or earlier leaders (sampled every ~4 px)."""
    end = nearest_on_box(anchor, b)
    n = max(14, int(np.hypot(*(end - anchor)) / 4))
    pts = anchor + np.linspace(0.08, 0.97, n)[:, None] * (end - anchor)
    hits = sum(
        ((pts[:, 0] > o[0]) & (pts[:, 0] < o[2]) & (pts[:, 1] > o[1]) & (pts[:, 1] < o[3])).sum()
        for o in boxes
    )
    crossings = sum(crosses(anchor, end, a, e) for a, e in sc.leaders)
    for a, e in sc.leaders:  # an earlier leader through this label box
        seg = a + np.linspace(0.05, 0.95, 14)[:, None] * (e - a)
        hits += ((seg[:, 0] > b[0]) & (seg[:, 0] < b[2]) & (seg[:, 1] > b[1]) & (seg[:, 1] < b[3])).sum()
    return hits * 3000.0 + crossings * 5000.0


# ----------------------------------------------------------------------------------- plotly html
def relief_html(height, count, weeks, lane_names, blocks, glyphs, text, pc, path) -> float:
    """Interactive version (plotly, the viz extra): surface with hover per lane and week, poles with
    one diamond per pennant, stacked stones, tents, and an optional route (legend only). Returns KB."""
    import plotly.graph_objects as go

    n_lanes, n_weeks = height.shape
    lane_f, x_f, Z = fine_grid(height, pc["html_upsample"])
    zat = RegularGridInterpolator((lane_f, x_f), Z, bounds_error=False, fill_value=None)
    Y, zmax, ink, status = (n_lanes - 1) - lane_f, float(np.ceil(height.max())), TEXT, STATUS[pc["status"]]
    near_l = np.clip(np.rint(lane_f).astype(int), 0, n_lanes - 1)
    near_w = np.clip(np.floor(x_f).astype(int), 0, n_weeks - 1)
    hover = np.array(
        [
            [
                f"{lane_names[li] or text['gap']}<br>week of {weeks[wj]:%Y-%m-%d}<br>"
                f"{int(count[li, wj])} completed · height {height[li, wj]:.1f}"
                for wj in near_w
            ]
            for li in near_l
        ],
        dtype=object,
    )
    scale = [[a, SEQUENTIAL[s]] for a, s in zip(pc["ramp_at"], pc["ramp"], strict=True)]
    fig = go.Figure(
        go.Surface(
            x=x_f,
            y=Y,
            z=Z,
            colorscale=scale,
            cmin=0,
            cmax=zmax,
            text=hover,
            hovertemplate="%{text}<extra></extra>",
            showlegend=False,
            colorbar=dict(
                title=dict(text=text["scale"], side="right"), thickness=12, len=0.45, outlinewidth=0
            ),
            contours=dict(
                z=dict(
                    show=True,
                    start=pc["contour_step"],
                    end=zmax,
                    size=pc["contour_step"],
                    color="rgba(13,54,107,0.28)",
                    width=1,
                )
            ),
            lighting=dict(ambient=1.0, diffuse=0.3, specular=0.0, roughness=1.0, fresnel=0.0),
            lightposition=dict(x=-1e5, y=1e5, z=8e4),
        )
    )

    def z_on(lane, x):
        return zat(np.column_stack([np.asarray(lane, float), np.clip(np.asarray(x, float), x_f[0], x_f[-1])]))

    G = glyphs.assign(y=(n_lanes - 1) - glyphs.lane, z=z_on(glyphs.lane, glyphs.x))
    lift, step = zmax * 0.02, zmax * 0.06
    poles = G[G.kind == "pole"]
    lines, diamonds = [], []
    for g in poles.itertuples():
        top = g.z + zmax * 0.22 + step * g.n
        lines += [(g.x, g.y, g.z), (g.x, g.y, top), (None, None, None)]
        diamonds += [(g.x, g.y, top - k * step, h) for k, h in enumerate(g.hover)]
    if len(poles):
        lx, ly, lz = zip(*lines, strict=True)
        fig.add_scatter3d(
            x=lx,
            y=ly,
            z=lz,
            mode="lines",
            line=dict(color=ink, width=3),
            hoverinfo="skip",
            legendgroup="pole",
            showlegend=False,
        )
        dx, dy, dz, dh = zip(*diamonds, strict=True)
        fig.add_scatter3d(
            x=dx,
            y=dy,
            z=dz,
            mode="markers",
            marker=dict(symbol="diamond", size=3.5, color=ink),
            hovertext=dh,
            hovertemplate="%{hovertext}<extra></extra>",
            name=text["names"]["pole"],
            legendgroup="pole",
        )
    K = G[G.kind == "cairn"]
    stack = [
        (g.x, g.y, g.z + lift + k * zmax * 0.035, 8 - 0.9 * k, g.hover)
        for g in K.itertuples()
        for k in range(g.n)
    ]
    if stack:
        cx, cy, cz, cs, ch = zip(*stack, strict=True)
        fig.add_scatter3d(
            x=cx,
            y=cy,
            z=cz,
            mode="markers",
            hovertext=ch,
            hovertemplate="%{hovertext}<extra></extra>",
            marker=dict(size=list(cs), color=TEXT_SECONDARY, line=dict(color=SURFACE, width=1)),
            name=text["names"]["cairn"],
        )
    T = G[G.kind == "tent"]
    fig.add_scatter3d(
        x=T.x,
        y=T.y,
        z=T.z + lift,
        mode="markers",
        hovertext=T.hover,
        hovertemplate="%{hovertext}<extra></extra>",
        name=text["names"]["tent"],
        marker=dict(symbol="square", size=6, color=status, line=dict(color=ink, width=1)),
    )
    R = G[G.kind == "route"]
    fig.add_scatter3d(
        x=R.x,
        y=R.y,
        z=R.z + lift,
        mode="lines+markers",
        hovertext=R.hover,
        hovertemplate="%{hovertext}<extra></extra>",
        visible="legendonly",
        name=text["names"]["route"],
        line=dict(color=TEXT_SECONDARY, width=2, dash="dot"),
        marker=dict(size=3, color=TEXT_SECONDARY),
    )
    ticks = month_ticks(weeks, n_weeks)
    e, a, eye, k = np.radians(pc["elev"]), np.radians(pc["azim"]), 3.1, pc["html_scale"]
    axis = dict(title="", showbackground=False, showgrid=False)  # gridlines would only cage the empty air
    fig.update_layout(
        title=dict(
            text=f"{text['title']}<br><sup>{text['subtitle']}</sup>", x=0.02, font=dict(color=TEXT, size=17)
        ),
        paper_bgcolor=SURFACE,
        font=dict(family="Inter, system-ui, sans-serif", color=TEXT_SECONDARY, size=12),
        legend=dict(x=0.99, xanchor="right", y=0.97, yanchor="top", bgcolor="rgba(252,252,251,0.85)"),
        margin=dict(l=0, r=0, t=96, b=0),
        scene=dict(
            xaxis=axis
            | dict(tickvals=[t[0] for t in ticks], ticktext=[t[1] for t in ticks], range=[0, n_weeks]),
            yaxis=axis
            | dict(
                tickvals=list((n_lanes - 1) - blocks.mid_lane),
                ticktext=list(blocks.label),
                range=[-0.5, n_lanes - 0.5],
            ),
            # no z ticks: on the flat box they stack into the first month's label; the colorbar has height
            zaxis=axis | dict(showticklabels=False, range=[0, zmax * 1.6], showspikes=False),
            aspectmode="manual",
            aspectratio=dict(x=pc["box"][0] * k, y=pc["box"][1] * k, z=pc["box"][2] * 1.6 * k),
            camera=dict(
                eye=dict(x=eye * np.cos(e) * np.cos(a), y=eye * np.cos(e) * np.sin(a), z=eye * np.sin(e)),
                projection=dict(type="orthographic"),
            ),
        ),
    )
    fig.write_html(path, include_plotlyjs=True, full_html=True)
    return path.stat().st_size / 1024
