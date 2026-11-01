"""Universe chart: galaxies of planets of moons, drawn as nested circles.

Inputs are a hierarchy (from any grouping), plus display names and plot options from config:

    moons   : one row per item: galaxy_id, planet_id (-1 = not drawn), state ("done", "open" or
              "dropped") and a sortable column (order_by) that puts the oldest moon at the centre
    planets : index planet_id; galaxy_id, n, open, pinned, glow (bools), role and target:
              role "" = packed in its galaxy; in the hub galaxy "core" (centre), "belt" (an
              annulus around the core), "orbit" (outside the belt, facing galaxy `target`, with a
              dotted arrow to it; target -1 = no arrow, placed in the widest free direction)

The hub galaxy sits in the centre; the others sit on a ring around it, ordered so the most
affine pairs are neighbours. Labels are placed greedily (each tries offsets in order and keeps the
first that touches nothing); the layout is re-packed with more room where a label found no free
spot, then the PNG is palette-quantized to stay under plot.max_kb.

At most len(CATEGORICAL) = 8 galaxies, hub included: one palette color each, never cycled. This
also bounds the ring search, which tries every cyclic order (n - 2)! of the n - 1 ring galaxies.

    stats = universe(moons, planets, hub=7, galaxy_names=..., planet_names=..., details=...,
                     affinity=aff, links=[(0, 1, 5)], legend=cfg["legend"], pc=cfg["plot"], path=...)
"""

from __future__ import annotations

import textwrap
from itertools import permutations

import numpy as np
import pandas as pd
from matplotlib.legend_handler import HandlerBase, HandlerPatch
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch
from matplotlib.transforms import Bbox

from analytics.viz.pack import pack, spread
from analytics.viz.style import CATEGORICAL, OTHER, SURFACE, TEXT, TEXT_SECONDARY, optimize_png, plt


def layout(moons, P, aff, hub, pc, order=None, order_by="createdAt"):
    """Moon xy (sunflowers, oldest at the centre), planets P with r/x/y/r_in, galaxies G (r/x/y).
    Planets are packed inside galaxies; the hub's core in the centre, its belt around the core,
    orbit planets outside the belt facing their target galaxy. The other galaxies sit on a ring
    around the hub in affinity order, or in `order` when given."""
    s, gap = pc["moon_spacing"], pc["hub_gap"]
    golden = np.pi * (3 - np.sqrt(5))
    P["r"] = s * np.sqrt(P.n + 0.5) + pc["planet_rim"] * s
    P[["x", "y"]] = 0.0
    R = {}
    for g, sub in P[P.galaxy_id != hub].groupby("galaxy_id"):
        c = pack((sub.r + sub.room).to_numpy(), pc["planet_pad"])
        P.loc[sub.index, ["x", "y"]] = c
        R[g] = np.max(np.linalg.norm(c, axis=1) + sub.r + sub.room) + pc["galaxy_pad"]
    H = P[P.galaxy_id == hub]
    centre = H[H.role == "core"]
    P.loc[centre.index, ["x", "y"]] = pack((centre.r + centre.room).to_numpy(), gap)
    r_in = np.max(np.linalg.norm(P.loc[centre.index, ["x", "y"]], axis=1) + centre.r + centre.room)
    r_in += pc["belt_inner"]
    belt = H[H.role == "belt"]
    r_out = np.sqrt(r_in**2 + belt.n.sum() * s**2)
    P.loc[belt.index, "r"] = r_out
    orbit = H[H.role == "orbit"]
    rho = r_out + gap + (orbit.r + orbit.room).to_numpy()  # room moves an orbit planet off the belt
    R[hub] = max([r_out, *(rho + orbit.r)]) + pc["galaxy_pad"]
    # ring: cyclic order with the largest summed affinity of neighbours, a free slot above the hub
    # for its title, rotated so the largest galaxy sits nearest the horizontal (the canvas is wide)
    ring = sorted(R.keys() - {hub}, key=lambda g: -R[g])
    cyc = max(
        ([ring[0], *p] for p in permutations(ring[1:])),
        key=lambda o: sum(aff.loc[a, b] for a, b in zip(o, o[1:] + o[:1], strict=True)),
    )
    room = P.groupby("galaxy_id").galaxy_room.first()  # extra spacing where a galaxy title collided
    d = pd.Series({g: R[hub] + room[hub] + pc["ring_gap"] + R[g] + room[g] for g in cyc})
    slot = np.radians(pc["title_slot"]) / 2
    while True:
        half = np.arcsin(np.minimum(1, (pd.Series(R)[cyc] + room[cyc] + pc["ring_gap"] / 2) / d))
        if 2 * (half.sum() + slot) <= 2 * np.pi:
            break
        d += pc["ring_step"]
    slack = (2 * np.pi - 2 * (half.sum() + slot)) / (len(cyc) + 1)

    def angles(order):
        h = half[order].to_numpy()
        return pd.Series(
            np.pi / 2 - slot - slack - h[0] - np.r_[0, np.cumsum(h[:-1] + h[1:] + slack)], index=order
        )

    turns = [o for i in range(len(cyc)) for o in (cyc[i:] + cyc[:i], (cyc[i:] + cyc[:i])[::-1])]
    theta = angles(order) if order else min((angles(o) for o in turns), key=lambda t: abs(np.sin(t[ring[0]])))
    G = pd.DataFrame(
        {
            "r": [R[g] for g in theta.index],
            "x": d[theta.index] * np.cos(theta) * pc["stretch"],
            "y": d[theta.index] * np.sin(theta),
        }
    )
    G.loc[hub] = [R[hub], 0.0, 0.0]
    # orbit planets face their target galaxy; untargeted ones take the widest free direction
    tgt = orbit.target.to_numpy()
    phi = np.array([np.arctan2(G.y[t], G.x[t]) if t >= 0 else np.nan for t in tgt])
    if np.isnan(phi).any():
        a = np.sort(phi[~np.isnan(phi)])
        gaps = np.diff(np.r_[a, a[0] + 2 * np.pi])
        phi[np.isnan(phi)] = a[np.argmax(gaps)] + gaps.max() / 2
    phi = spread(phi, np.arcsin(((orbit.r + orbit.room).to_numpy() + gap / 2) / rho))
    P.loc[orbit.index, ["x", "y"]] = np.c_[rho * np.cos(phi), rho * np.sin(phi)]
    P[["x", "y"]] += G.loc[P.galaxy_id, ["x", "y"]].to_numpy()
    xy = np.zeros((len(moons), 2))
    for p, sub in moons[moons.planet_id >= 0].groupby("planet_id"):
        k = np.arange(len(sub))
        idx = sub.sort_values(order_by).index
        if P.role[p] == "belt":  # an annulus with the same density, oldest on the inside
            rad = np.sqrt(r_in**2 + (r_out**2 - r_in**2) * (k + 0.5) / len(k))
        else:
            rad = s * np.sqrt(k + 0.5)
        xy[idx] = (
            P.loc[p, ["x", "y"]].to_numpy(float) + np.c_[rad * np.cos(k * golden), rad * np.sin(k * golden)]
        )
    P["r_in"] = np.where(P.role == "belt", r_in, 0.0)
    return xy, P, G


def entry(p0, p1, bb):
    """Parameter t in [0, 1] where segment p0 -> p1 enters box bb (Liang-Barsky); None if it misses."""
    d, t0, t1 = p1 - p0, 0.0, 1.0
    for p, q in (
        (-d[0], p0[0] - bb.x0),
        (d[0], bb.x1 - p0[0]),
        (-d[1], p0[1] - bb.y0),
        (d[1], bb.y1 - p0[1]),
    ):
        if p == 0 and q < 0:
            return None
        if p != 0:
            t0, t1 = (max(t0, q / p), t1) if p < 0 else (t0, min(t1, q / p))
    return t0 if t0 <= t1 else None


def plot(moons, P, G, hub, gnames, pnames, details, links, legend, pc, path):
    """Draw one laid-out universe to `path`. Returns label misses [(key, cost)], labels that spilled
    out of their own galaxy {planet: px} and the scale in points per data unit."""
    hue = {g: CATEGORICAL[k] for k, g in enumerate(sorted(G.index))}
    xy, r = G[["x", "y"]].to_numpy(), G[["r"]].to_numpy()
    fig = plt.figure(figsize=(pc["width"], pc["height"]), dpi=pc["dpi"], facecolor=SURFACE)
    ax = fig.add_axes(pc["axes_box"])
    ax.set_facecolor(SURFACE)
    ax.set_aspect("equal", anchor="N")
    ax.axis("off")
    ax.set_xlim((xy - r).min(0)[0] - pc["margin"], (xy + r).max(0)[0] + pc["margin"])
    ax.set_ylim((xy - r).min(0)[1] - pc["margin"], (xy + r).max(0)[1] + pc["title_room"])
    fig.canvas.draw()
    to_px = ax.transData.transform
    unit = (to_px([1, 0]) - to_px([0, 0]))[0] * 72 / fig.dpi  # points per data unit

    def disc(row, r, **kw):
        ax.add_patch(Circle((row.x, row.y), r, **kw))

    for g, row in G.iterrows():
        disc(row, row.r, fc=hue[g], alpha=0.05, lw=0, zorder=0)
        disc(row, row.r, fc="none", ec=hue[g], alpha=0.55, lw=1.0, zorder=0)
    for _, row in P.iterrows():
        col = hue[row.galaxy_id]
        for f in (1.0, 0.66, 0.33) if row.glow else ():
            disc(row, row.r + pc["glow_width"] * f, fc=col, alpha=0.09, lw=0, zorder=1)
        if row.role == "belt":  # an annulus: one thick stroke between the inner and outer radius
            width = (row.r - row.r_in + pc["moon_spacing"]) * unit
            disc(row, (row.r + row.r_in) / 2, fc="none", ec=col, alpha=0.13, lw=width, zorder=2)
            continue
        disc(row, row.r, fc=col, alpha=0.13, lw=0, zorder=2)
        if row.pinned:
            disc(row, row.r, fc="none", ec=TEXT_SECONDARY, lw=0.9, ls=(0, (3, 2)), zorder=2)
    msize = (pc["moon_size"] * pc["moon_spacing"] * unit) ** 2
    d = moons[moons.planet_id >= 0]
    col = d.galaxy_id.map(hue)
    for state, kw in {
        "done": dict(c=col[d.state == "done"], edgecolors=SURFACE, linewidths=0.3),
        "open": dict(facecolors=SURFACE, edgecolors=col[d.state == "open"], linewidths=0.9),
        "dropped": dict(c=OTHER, edgecolors=SURFACE, linewidths=0.3),
    }.items():
        sel = (d.state == state).to_numpy()
        ax.scatter(d.x[sel], d.y[sel], s=msize * (0.55 if state == "dropped" else 1), zorder=3, **kw)

    # ---- label placement: try offsets in order, keep the first that touches nothing
    renderer = fig.canvas.get_renderer()
    px = unit * fig.dpi / 72
    frame = Bbox([[6, ax.get_window_extent(renderer).y0], [fig.bbox.x1 - 6, fig.bbox.y1 * pc["frame_top"]]])
    glow = (P.r + pc["glow_width"] * P.glow) * px  # labels clear a planet's glow too
    discs = [(to_px((r.x, r.y)), gr, r.r_in * px) for r, gr in zip(P.itertuples(), glow, strict=True)]
    rims = {g: (to_px((r.x, r.y)), r.r * px) for g, r in G.iterrows()}
    boxes, keepout, lines, ends, misses, spill = [], [], [], [], [], {}

    def gap(bb, c):  # distance from a box to a point (0 inside)
        return np.hypot(max(bb.x0 - c[0], 0, c[0] - bb.x1), max(bb.y0 - c[1], 0, c[1] - bb.y1))

    def cost(bb, own, mine=None):
        """overlap_cost per label overlap or off-canvas; disc_cost per planet disc covered (never
        over moons) or arrow tip crowded; galaxy_cost per other galaxy entered; cross_cost per
        arrow crossed. For a planet label (mine = its row in P): keepout_cost per padded title
        block of another galaxy entered, owner_cost when another planet is about as near as its
        own, outside_cost when it leaves its own galaxy (a preference only: it never triggers a
        repair). Returns cost and spill (px)."""
        inside = frame.x0 <= bb.x0 and bb.x1 <= frame.x1 and frame.y0 <= bb.y0 and bb.y1 <= frame.y1
        corners = [(x, y) for x in (bb.x0, bb.x1) for y in (bb.y0, bb.y1)]
        far = lambda c: max(np.hypot(x - c[0], y - c[1]) for x, y in corners)  # noqa: E731
        circ = sum(gap(bb, c) < r and far(c) > r_in for c, r, r_in in discs)
        gal = sum(gap(bb, c) < r for g, (c, r) in rims.items() if g != own)
        cross = sum(entry(a, b, bb) is not None for a, b in lines)
        crowd = sum(gap(bb, e) < pc["arrow_end_px"] for e in ends)
        hard = sum(bb.overlaps(o) for o in boxes) + (not inside)
        total = pc["overlap_cost"] * hard + pc["disc_cost"] * (circ + crowd)
        total, out = total + pc["galaxy_cost"] * gal + pc["cross_cost"] * cross, 0.0
        if mine is not None:
            near = (own, P.target.iloc[mine])  # an orbit planet's label may sit by its target's header
            total += pc["keepout_cost"] * sum(bb.overlaps(o) for g, o in keepout if g not in near)
            rim = lambda c, r, r_in: r_in - far(c) if 0 < far(c) < r_in else gap(bb, c) - r  # noqa: E731
            d = [rim(*disc) for disc in discs]
            d_own = d.pop(mine)
            total += pc["owner_cost"] * (min(d) < d_own + pc["owner_margin"] * bb.height)
            out = max(far(rims[own][0]) - rims[own][1], 0.0)
        return total + pc["outside_cost"] * (out > 0), out

    def place(x, y, text, options, key, own=None, mine=None, **kw):
        tried = []
        for dx, dy, va, ha in options:
            t = ax.text(x + dx, y + dy, text, ha=ha, va=va, zorder=7, linespacing=1.1, **kw)
            bb = t.get_window_extent(renderer).expanded(1.03, 1.08)
            tried.append((*cost(bb, own, mine), len(tried), t, bb))
            if tried[-1][0] == 0:
                break
        best = min(tried, key=lambda z: (z[0], z[2]))
        for z in tried:
            if z is not best:
                z[3].remove()
        boxes.append(best[4])
        if best[1] > 0:  # spilled out of its own galaxy (no free spot inside)
            spill[key[1]] = round(best[1])
        hard = best[0] - pc["outside_cost"] * (best[1] > 0)
        if hard:
            misses.append((key, int(hard)))
        return best[3]

    halo = dict(boxstyle="round,pad=0.12", fc=SURFACE, ec="none", alpha=0.85)
    gray = dict(fontsize=pc["stats_font"], color=TEXT_SECONDARY, bbox=halo)
    steps = [k * pc["stats_slide"] for k in range(pc["stats_slides"] + 1)]
    slide = [(dx, 0, "bottom", "center") for st in steps for dx in (st, -st)][1:]
    label_min = lambda g: pc["hub_label_min"] if g == hub else pc["label_min"]  # noqa: E731
    title_box = {}
    for g, row in G.iterrows():  # galaxy name over a gray stats line, slid along the rim if needed
        sub = moons[moons.galaxy_id == g]
        n_pl, small = (P.galaxy_id == g).sum(), P[(P.galaxy_id == g) & (P.n < label_min(g))]
        stats = f"{len(sub):,} {pc['moon_noun']} · {n_pl} planets · {sub.state.eq('open').mean():.0%} open · "
        stats += details[g]
        if len(small):
            also = ", ".join(f"{pnames[p]} {int(r.n)}" for p, r in small.iterrows())
            for long, short in pc["also_abbrev"].items():
                also = also.replace(", " + long, ", " + short)
            stats += "\n" + textwrap.fill("also " + also, pc["stats_wrap"])
        x = place(row.x, row.y + row.r + pc["stats_lift"], stats, slide, ("galaxy", g), g, **gray)
        x = x.get_position()[0]
        y = ax.transData.inverted().transform((0, boxes[-1].y1 + 3))[1]  # 3 px above the stats line
        bold = dict(fontsize=pc["galaxy_font"], color=TEXT, weight="bold", bbox=halo)
        place(x, y, gnames[g], [(0, 0, "bottom", "center")], ("galaxy", g), g, **bold)
        title_box[g] = Bbox.union(boxes[-2:])
        keepout.append((g, title_box[g].padded(pc["title_pad_px"])))

    def segment(a, b, ra, rb):  # rim to rim, and the unit normal for label offsets
        u = (b - a) / np.linalg.norm(b - a)
        return a + u * ra, b - u * rb, np.array([-u[1], u[0]])

    along = lambda nrm: [(*(k * nrm), "center", "center") for k in (0, 1.5, -1.5, 3, -3)]  # noqa: E731
    centre = lambda row: row[["x", "y"]].to_numpy(float)  # noqa: E731
    for a, b, n in links:  # (1) links between galaxies, width = count
        pa, pb, nrm = segment(centre(G.loc[a]), centre(G.loc[b]), G.r[a], G.r[b])
        lw = pc["link_width"][0] + pc["link_width"][1] * n / max(m for *_, m in links)
        ax.plot(*np.c_[pa, pb], color=OTHER, lw=lw, alpha=0.5, solid_capstyle="round", zorder=0)
        place(*(pa + pb) / 2, str(n), along(nrm), ("link", a, b), **gray)
    dash = (0, tuple(pc["arrow_dash"]))
    shaft = dict(arrowstyle="-", lw=pc["arrow_lw"], ls=dash, color=TEXT_SECONDARY, shrinkA=0, shrinkB=0)
    head = dict(shaft, arrowstyle="-|>", ls="-", mutation_scale=pc["arrow_head"])  # solid, never dashed
    arrows = P[P.role.eq("orbit") & (P.target >= 0)]
    for p, row in arrows.iterrows():  # (2) orbit planet -> target galaxy arrows
        tgt = G.loc[row.target]
        pa, pb, nrm = segment(centre(row), centre(tgt), row.r, tgt.r)
        a, b = to_px(pa), to_px(pb)
        t = entry(a, b, title_box[row.target].expanded(1.0, 1.15))  # stop at the target's title
        pb = ax.transData.inverted().transform(b if t is None else a + t * (b - a))
        u = (pb - pa) / np.linalg.norm(pb - pa)
        ax.add_patch(FancyArrowPatch(pa, pb, zorder=4, **shaft))
        ax.add_patch(FancyArrowPatch(pb - u / unit, pb, zorder=4, **head))  # a 1 pt stub sets the direction
        ends.append(to_px(pb))  # the count sits mid-shaft, clear of the head
        place(*(pa + pb) / 2, str(int(row.n)), along(nrm), ("arrow", p), hub, **gray)
        lines.append((to_px(pa), to_px(pb)))
    around = []  # label directions in order of preference: below, above, sides, diagonals
    for a in np.radians([-90, 90, 0, 180, -45, -135, 45, 135, -65, -115, 65, 115, -25, -155, 25, 155]):
        c, s = np.cos(a), np.sin(a)
        around.append((c, s, "bottom" if s > 0.3 else "top" if s < -0.3 else "center",
                       "left" if c > 0.3 else "right" if c < -0.3 else "center"))  # fmt: skip
    dark = dict(fontsize=pc["label_size"], color=TEXT, bbox=dict(halo, alpha=0.8))
    for p, row in P[P.n >= P.galaxy_id.map(label_min)].sort_values("n", ascending=False).iterrows():
        opts = [
            (f * c, f * s, va, ha)
            for f in row.r + pc["glow_width"] * row.glow + pc["label_gap"] + pc["label_far"] * np.arange(3)
            for c, s, va, ha in around
        ]
        text = f"{pnames[p]}\n{int(row.n)} · {int(row.open)} open"
        place(row.x, row.y, text, opts, ("planet", p), row.galaxy_id, P.index.get_loc(p), **dark)

    # ---- title, subtitle, legend rows, footnote
    title = pc["title"].format(galaxies=len(G), planets=len(P), moons=f"{len(moons):,}")
    fig.text(0.01, 0.988, title, ha="left", va="top", fontsize=17, color=TEXT)
    fig.text(0.01, 0.962, pc["subtitle"], ha="left", va="top", fontsize=10, color=TEXT_SECONDARY)
    dot = dict(ls="", marker="o", ms=7)

    class Swatch(HandlerBase):
        """Legend swatches drawn like the chart: a glowing planet, and the two-part orbit arrow."""

        def create_artists(self, legend, orig, xd, yd, width, height, fontsize, trans):
            cx, cy = width / 2 - xd, height / 2 - yd
            if orig is glow_handle:  # core disc plus three glow discs, as on the planets
                core = height * pc["legend_glow"]
                arts = [Circle((cx, cy), core, fc=TEXT_SECONDARY, alpha=0.13, lw=0)]
                arts += [
                    Circle((cx, cy), core * f, fc=TEXT_SECONDARY, alpha=0.09, lw=0) for f in (1, 1.33, 1.66)
                ]
            else:  # dotted shaft over the full width, solid head at the right end
                a, b = np.array([-xd, cy]), np.array([width - xd, cy])
                arts = [FancyArrowPatch(a, b, **shaft), FancyArrowPatch(b - [1, 0], b, **head)]
            for art in arts:
                art.set_transform(trans)
            return arts

    swatch = Swatch()
    glow_handle = Circle((0, 0), 1, label=legend["glow"])
    arrow_handle = Circle((0, 0), 1, label=legend["arrow"])
    rows = [
        [Line2D([], [], mfc=hue[g], mec=hue[g], label=gnames[g], **dot) for g in G.index.sort_values()],
        [
            Line2D([], [], mfc=TEXT_SECONDARY, mec=TEXT_SECONDARY, label=legend["done"], **dot),
            Line2D([], [], mfc=SURFACE, mec=TEXT_SECONDARY, mew=1.2, label=legend["open"], **dot),
            Line2D([], [], mfc=OTHER, mec=OTHER, label=legend["dropped"], **dict(dot, ms=5)),
            glow_handle,
            Circle((0, 0), 1, fc="none", ec=TEXT_SECONDARY, ls=(0, (3, 2)), label=legend["pinned"]),
            *[Line2D([], [], color=OTHER, lw=3, alpha=0.6, label=legend["links"])] * bool(links),
            *[arrow_handle] * bool(len(arrows)),
        ],
    ]
    circle = HandlerPatch(
        patch_func=lambda xdescent, ydescent, width, height, **_: Circle(
            (width / 2 - xdescent, height / 2 - ydescent), height * 0.65
        )
    )
    for k, handles in enumerate(rows):
        leg = fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.005, 0.942 - 0.024 * k),
                         ncol=len(handles), frameon=False, fontsize=9.5, handletextpad=0.4, columnspacing=1.6,
                         handler_map={Circle: circle, glow_handle: swatch, arrow_handle: swatch})  # fmt: skip
        for t in leg.get_texts():
            t.set_color(TEXT_SECONDARY)
    foot = dict(ha="left", va="bottom", fontsize=8.5, color=TEXT_SECONDARY, linespacing=1.4)
    fig.text(0.01, 0.006, pc["footnote"], **foot)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return misses, spill, unit


def universe(
    moons: pd.DataFrame,
    planets: pd.DataFrame,
    *,
    hub: int,
    galaxy_names: dict,
    planet_names: dict,
    details: dict,
    affinity: pd.DataFrame,
    links: list[tuple[int, int, int]],
    legend: dict,
    pc: dict,
    path,
    order_by: str = "createdAt",
) -> dict:
    """Lay out and draw the universe to `path` (see the module docstring for the inputs).

    details: galaxy -> text after its stats line (e.g. its top projects). affinity: galaxy x
    galaxy (non-hub) DataFrame that orders the ring. links: (galaxy, galaxy, count) lines.
    legend: labels for done, open, dropped, glow, pinned, links, arrow. pc: plot options, incl.
    title (may use {galaxies}, {planets}, {moons}), subtitle, footnote and max_kb.
    Re-packs up to pc["repair_rounds"] times, giving more room where a label found no free spot.
    Returns moon xy, the label misses and spills of the final drawing, its scale and size.
    Raises ValueError for more galaxies than the palette has colors (see the module docstring)."""
    n_gal = len(set(planets.galaxy_id) | {hub})
    if n_gal > len(CATEGORICAL):
        raise ValueError(
            f"universe() draws at most {len(CATEGORICAL)} galaxies (one palette color each, hub "
            f"included), got {n_gal}. Make fewer, larger galaxies upstream, e.g. a larger "
            "galaxies.min_size or a smaller galaxies.resolution in the analysis config."
        )
    moons = moons.reset_index(drop=True)
    P = planets.assign(room=0.0, galaxy_room=0.0)
    order, rounds = None, []
    for i in range(pc["repair_rounds"] + 1):
        if i == pc["repair_rounds"] and rounds:  # none was clean: redraw the least bad round
            P[["room", "galaxy_room"]] = min(rounds, key=lambda z: z[:2])[2]
        xy, L, G = layout(moons, P.copy(), affinity, hub, pc, order, order_by)
        misses, spill, unit = plot(
            moons.assign(x=xy[:, 0], y=xy[:, 1]), L, G, hub, galaxy_names, planet_names, details, links,
            legend, pc, path,
        )  # fmt: skip
        print(f"round {i}: {unit:.2f} pt per unit; label collisions: {misses or 'none'}")
        grow = [k[1] for k, c in misses if k[0] == "planet" and c >= pc["repair_min"]]
        tight = [k[1] for k, _ in misses if k[0] == "galaxy"]
        if not (grow or tight) or i == pc["repair_rounds"]:  # what is left (an arrow crossed) room won't fix
            break
        rounds.append((sum(c for _, c in misses), -unit, P[["room", "galaxy_room"]].copy()))
        order = G.index[:-1].tolist()  # keep the ring order of the first round
        P.loc[grow, "room"] += pc["repair_step"]
        P.loc[P.galaxy_id.isin(tight), "galaxy_room"] += pc["repair_step"]
    kb = optimize_png(path, pc["max_kb"])
    return {
        "xy": xy,
        "label_collisions": misses,
        "spill": spill,
        "pt_per_unit": unit,
        "repair_rounds_used": i,
        "kb": round(kb),
    }
