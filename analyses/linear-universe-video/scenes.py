"""Scenes of linear-universe-video (ManimCE, no LaTeX) and the helpers they share.

Top: style S (from config.toml [style]/[frame]; run.py calls use(cfg) so --set overrides apply),
data access, text, the Swarm (1,143 dots drawn as a few batched paths), discs, labels, legends,
arrows and VideoScene. Bottom: one scene class per entry of [scenes].order.

    sw = dots_from(positions("random"), team_colors())                # a Swarm
    self.play(FadeIn(sw))
    self.play(sw.morph(xy=positions("tsne"), lag=0.3), run_time=3)     # NaN target -> fades out in place
    self.play(sw.morph(color=galaxy_colors(), opacity=np.where(mask, 1, S.dim)))
    self.play(sw.morph(xy=positions("universe"), **universe_style()))  # moons: hue, done/open/dropped
    self.play(self.zoom(**data()["tour"][0]["camera"]))                # pin() captions before zooming

Move and restyle a Swarm only with morph() / set_state() (they keep its per-dot arrays in step);
FadeIn/FadeOut work as usual; never sw.animate. Every number on screen must be in metrics.json, the
READMEs or the data: caption() and title_card() raise otherwise (style.strict_numbers).
"""

from __future__ import annotations

import json
import re
import textwrap
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from manim import (
    DL,
    DOWN,
    LEFT,
    ORIGIN,
    RIGHT,
    UL,
    UP,
    UR,
    Animation,
    Annulus,
    BackgroundRectangle,
    Circle,
    Create,
    DashedLine,
    DashedVMobject,
    FadeIn,
    FadeOut,
    GrowFromEdge,
    LaggedStart,
    Line,
    ManimColor,
    MovingCameraScene,
    Polygon,
    Rectangle,
    ReplacementTransform,
    RoundedRectangle,
    ShowPassingFlash,
    Text,
    Triangle,
    UpdateFromAlphaFunc,
    VGroup,
    VMobject,
    linear,
    smooth,
)

from analytics.config import load_config

HERE = Path(__file__).resolve().parent
S = SimpleNamespace()  # style, filled by use(): S.bg, S.ink, S.hues, S.caption, S.dot, ...
_CACHE: dict = {}


def use(cfg: dict) -> None:
    """Load style and paths from a config dict (run.py passes its own, with --set overrides)."""
    st, fr = cfg["style"], cfg["frame"]
    S.cfg, S.bg, S.ink, S.ink2 = cfg, st["background"], st["ink"], st["ink2"]
    S.muted, S.grid, S.baseline = st["muted"], st["grid"], st["baseline"]
    S.hues, S.teams, S.font, S.mono = st["hues"], st["team_order"], st["font"], st["mono"]
    S.title, S.subtitle, S.caption = st["title_size"], st["subtitle_size"], st["caption_size"]
    S.label, S.small, S.kicker = st["label_size"], st["small_size"], st["kicker_size"]
    S.dot, S.hollow, S.dropped = st["dot_radius"], st["hollow_stroke"], st["dropped_scale"]
    S.dim, S.levels, S.strict = st["dim_opacity"], st["color_levels"], st["strict_numbers"]
    S.disc_fill, S.disc_stroke, S.planet_fill = st["disc_fill"], st["disc_stroke"], st["planet_fill"]
    S.W, S.H, S.caption_y, S.caption_w = fr["width"], fr["height"], fr["caption_y"], fr["caption_width"]
    S.kicker_xy, S.durations = fr["kicker_xy"], cfg["durations"]
    S.data_path = Path(cfg["paths"]["outputs"]) / "video_data.json"
    _CACHE.clear()


use(load_config(HERE / "config.toml"))


# ---- data ---------------------------------------------------------------------------------------
def data() -> dict:
    """outputs/video_data.json (written by run.py's export), loaded once. Schema: see run.py."""
    if "data" not in _CACHE:
        if not S.data_path.exists():
            raise FileNotFoundError(
                f"{S.data_path} is missing: run `analytics run linear-universe-video` once"
            )
        _CACHE["data"] = json.loads(S.data_path.read_text())
    return _CACHE["data"]


def positions(layout: str) -> np.ndarray:
    """(n, 3) frame positions for 'random', 'tsne', 'universe' or 'graph'; NaN where not placed."""
    return np.array([[*p, 0.0] if p else [np.nan] * 3 for p in data()["issues"][layout]])


def column(name: str) -> np.ndarray:
    """One per-issue column of data()['issues'] as an array (id, title, team, galaxy_id, ...)."""
    return np.asarray(data()["issues"][name])


def hue(galaxy_id: int) -> str:
    return S.hues[galaxy_id] if galaxy_id >= 0 else S.muted


def team_colors() -> np.ndarray:
    return np.array([S.hues[t] if t >= 0 else S.muted for t in column("team_idx")], dtype=object)


def galaxy_colors(labels=None) -> np.ndarray:
    """Hue per issue from galaxy ids (default: the consensus; pass a seed's labels for S5)."""
    return np.array([hue(g) for g in (column("galaxy_id") if labels is None else labels)], dtype=object)


def universe_style() -> dict:
    """Swarm kwargs for the universe chart's moons: galaxy hue, filled = done, hollow = open,
    small gray = canceled/duplicate; moon-sized.
    Use as sw.morph(xy=positions("universe"), **universe_style())."""
    life = column("lifecycle")
    color = np.where(life == "dropped", S.muted, galaxy_colors())
    r = data()["meta"]["moon_radius"] * np.where(life == "dropped", S.dropped, 1.0)
    return {"color": color, "radius": r, "hollow": life == "open", "opacity": 1.0}


def galaxy(name_or_id) -> dict:
    gs = data()["galaxies"]
    return gs[name_or_id] if isinstance(name_or_id, int) else next(g for g in gs if g["name"] == name_or_id)


def planet(name: str) -> dict:
    return next(p for p in data()["planets"] if p["name"] == name)


def p3(xy) -> np.ndarray:
    """[x, y] -> [x, y, 0] for manim."""
    return np.array([xy[0], xy[1], 0.0])


# ---- numbers guard and text ---------------------------------------------------------------------
def unknown_numbers(text: str) -> list[str]:
    """Numbers in text that are not in metrics.json, the READMEs or the data (run.py's allowed set)."""
    allowed = set(data()["allowed_numbers"])
    found = re.findall(r"\d[\d,]*(?:\.\d+)?", text)
    return [x for x in found if f"{float(x.replace(',', '')):.4f}".rstrip("0").rstrip(".") not in allowed]


def verify(text: str) -> str:
    bad = unknown_numbers(text)
    if bad and S.strict:
        raise ValueError(
            f"{text!r}: {bad} not in metrics.json, the READMEs or the data (style.strict_numbers)"
        )
    return text


def txt(
    s: str, size: float | None = None, color: str | None = None, weight: str = "NORMAL", mono: bool = False
) -> Text:
    """Text in the video's font. weight: NORMAL, MEDIUM, SEMIBOLD, BOLD."""
    return Text(
        s, font=S.mono if mono else S.font, font_size=size or S.label, color=color or S.ink, weight=weight
    )


def wrap(s: str, size: float, width: float) -> list[str]:
    """Split s into lines of at most `width` frame units at font `size` (Inter: ~0.0066 units/char/pt)."""
    return [ln for part in s.split("\n") for ln in textwrap.wrap(part, max(8, int(width / (0.0066 * size))))]


def lines(
    s: str, size: float, color: str, width: float, weight: str = "NORMAL", spacing: float = 0.35
) -> VGroup:
    return VGroup(*[txt(ln, size, color, weight) for ln in wrap(s, size, width)]).arrange(
        DOWN, buff=spacing * size / 30
    )


def caption(text: str, sub: str | None = None) -> VGroup:
    """Bottom-band caption: primary ink, optional secondary line(s) under it; wrapped, centred."""
    g = lines(verify(text), S.caption, S.ink, S.caption_w)
    if sub:
        g.add(*lines(verify(sub), S.label, S.ink2, S.caption_w))
        g.arrange(DOWN, buff=0.14)
    return g.move_to([0, S.caption_y, 0]).shift(UP * max(0, g.height - 0.9) / 2)


def title_card(title: str, sub: str | None = None) -> VGroup:
    """Centred title (primary ink) with an optional subtitle (secondary ink)."""
    g = VGroup(lines(verify(title), S.title, S.ink, S.W - 1.5, "SEMIBOLD"))
    if sub:
        g.add(lines(verify(sub), S.subtitle, S.ink2, S.W - 2.5))
    return g.arrange(DOWN, buff=0.4).move_to([0, 0.3, 0])


def kicker(text: str) -> Text:
    """Small top-left scene label, e.g. '2 · Words → map'."""
    t = txt(text, S.kicker, S.ink2)
    return t.move_to(p3(S.kicker_xy), aligned_edge=LEFT)


def fmt(n: float) -> str:
    """1143 -> '1,143'."""
    return f"{n:,.0f}" if float(n).is_integer() else f"{n:,}"


# ---- the swarm: many dots as a few batched paths ------------------------------------------------
_UNIT = Circle(radius=1, num_components=5).points.copy()  # 4 cubic curves: a unit circle at the origin


def _vec(v, n, dtype=float):
    a = np.asarray(v, dtype=dtype)
    return np.full(n, a, dtype=dtype) if a.ndim == 0 else a.copy()


def _xyz(xy, n):
    a = np.asarray(xy, float)
    return np.c_[a[:, :2], np.zeros(n)] if a.ndim == 2 else np.tile(np.r_[a[:2], 0.0], (n, 1))


class Swarm(VGroup):
    """n dots with per-dot position, color, radius, opacity and hollow (open) state, drawn as one
    VMobject per distinct style: ~4x faster than a VGroup of Dots (36 vs 150 ms/frame at -ql) and
    smooth at 1,143 dots. Dots with NaN position or zero opacity are not drawn. The current per-dot
    values are in sw.dots: xy (n, 3), color, radius, opacity, hollow."""

    def __init__(self, xy, color, radius=None, opacity=1.0, hollow=False, stroke_width=None):
        super().__init__()
        n = self.n = len(xy)
        self.stroke = S.hollow if stroke_width is None else stroke_width
        self.dots = {
            "xy": _xyz(xy, n),
            "color": _vec(color, n, object),
            "radius": _vec(S.dot if radius is None else radius, n),
            "opacity": _vec(opacity, n),
            "hollow": _vec(hollow, n, bool),
        }
        self._styles, self._pool = [], {}
        self._settle()

    @property
    def xy(self) -> np.ndarray:
        return self.dots["xy"]

    def at(self, idx) -> np.ndarray:
        """Current positions of dots idx (frame units, (k, 3))."""
        return self.dots["xy"][np.atleast_1d(idx)]

    def _codes(self, color, opacity, hollow):
        """Style code per dot; a style is (color, opacity, hollow)."""
        index = {k: i for i, k in enumerate(self._styles)}
        out = np.array(
            [index.setdefault(k, len(index)) for k in zip(color, np.round(opacity, 3), hollow, strict=True)]
        )
        self._styles = list(index)
        return out

    def _style(self, a, b, lev):
        (c0, o0, h0), (c1, o1, h1) = self._styles[a], self._styles[b]
        t = lev / S.levels if a != b else 0.0
        color = ManimColor(c0).interpolate(ManimColor(c1), t) if c0 != c1 else ManimColor(c0)
        fill = o0 * (not h0) * (1 - t) + o1 * (not h1) * t
        line = o0 * h0 * (1 - t) + o1 * h1 * t
        return color, fill, line, self.stroke if (h0 or h1) else 0.0

    def _reset(self):
        """Forget styles and batches. Old batches may still be in the scene's list of moving
        mobjects for the current play, so they lose their points instead of just being dropped."""
        for batch in self._pool.values():
            batch.points = np.zeros((0, 3))
        self._styles, self._pool = [], {}

    def _settle(self):
        d = self.dots
        self._reset()
        return self._draw(d["xy"], d["radius"], self._codes(d["color"], d["opacity"], d["hollow"]), None, 0.0)

    def _draw(self, xy, r, code0, code1, e):
        """Draw dots at xy with radius r; style goes from code0 to code1 at per-dot progress e."""
        code1 = code0 if code1 is None else code1
        L = S.levels
        lev = np.where(code0 == code1, 0, np.rint(np.asarray(e) * L)).astype(int)
        C = len(self._styles) + 1
        key = (code0 * C + code1) * (L + 1) + lev
        visible = ~np.isnan(xy[:, 0])
        used = []
        for k in np.unique(key[visible]):
            a, rest = divmod(int(k), C * (L + 1))
            b, lv = divmod(rest, L + 1)
            color, fill, line, sw = self._style(a, b, lv)
            if fill <= 0.002 and line <= 0.002:
                continue
            m = np.flatnonzero(visible & (key == k))
            batch = self._pool.setdefault(int(k), VMobject())
            batch.set_fill(color, fill, family=False).set_stroke(color, sw, line, family=False)
            batch.points = (_UNIT[None] * r[m, None, None] + xy[m, None, :]).reshape(-1, 3)
            used.append((fill + line, int(k)))
        keep = {k for _, k in used}
        for k, batch in self._pool.items():
            if k not in keep and len(batch.points):
                batch.points = np.zeros((0, 3))
        self.submobjects = [self._pool[k] for _, k in sorted(used)]  # faint dots under bright ones
        return self

    def set_state(self, xy=None, color=None, radius=None, opacity=None, hollow=None) -> Swarm:
        """Change per-dot values instantly (each a scalar or a per-dot array); returns self."""
        new = dict(xy=xy, color=color, radius=radius, opacity=opacity, hollow=hollow)
        for k, v in new.items():
            if v is not None:
                self.dots[k] = _xyz(v, self.n) if k == "xy" else _vec(v, self.n, self.dots[k].dtype)
        return self._settle()

    def morph(self, xy=None, color=None, radius=None, opacity=None, hollow=None, lag=0.0, order=None, **kw):
        """Animate to new per-dot values (each a scalar or a per-dot array). lag in [0, 1): dots
        start one after another by `order` (a per-dot sort key, e.g. x; default random). NaN target
        positions fade out in place; dots without a position fade in at their target.
        kw: run_time, rate_func (default smooth)."""
        end = dict(xy=xy, color=color, radius=radius, opacity=opacity, hollow=hollow)
        return _Morph(self, end, lag, order, **kw)


class _Morph(Animation):
    def __init__(self, swarm: Swarm, end: dict, lag: float, order, **kw):
        self.end, self.lag, self.order = end, lag, order
        kw.setdefault("rate_func", smooth)
        super().__init__(swarm, **kw)

    def create_starting_mobject(self):
        return self.mobject  # no deep copy: the swarm holds its own start state

    def begin(self):
        sw, n = self.mobject, self.mobject.n
        d = {k: v.copy() for k, v in sw.dots.items()}
        f = {
            k: d[k].copy() if v is None else (_xyz(v, n) if k == "xy" else _vec(v, n, d[k].dtype))
            for k, v in self.end.items()
        }
        gone = np.isnan(f["xy"][:, 0]) & ~np.isnan(d["xy"][:, 0])
        new = np.isnan(d["xy"][:, 0]) & ~np.isnan(f["xy"][:, 0])
        f["xy"][gone], f["opacity"][gone] = d["xy"][gone], 0.0  # vanish where they are
        d["xy"][new], d["opacity"][new] = f["xy"][new], 0.0  # appear where they land
        sw._reset()
        self.code0 = sw._codes(d["color"], d["opacity"], d["hollow"])
        self.code1 = sw._codes(f["color"], f["opacity"], f["hollow"])
        self.start, self.final = d, f
        rank = (
            np.random.default_rng(0).permutation(n)
            if self.order is None
            else np.argsort(np.argsort(self.order))
        )
        self.u = rank / max(n - 1, 1)
        self.interpolate(0)

    def interpolate_mobject(self, alpha):
        if self.lag:
            t = np.clip((alpha - self.lag * self.u) / (1 - self.lag), 0, 1)
            e = np.array([self.rate_func(x) for x in t])
        else:
            e = np.full(self.mobject.n, self.rate_func(alpha))
        d, f = self.start, self.final
        xy = d["xy"] + (f["xy"] - d["xy"]) * e[:, None]
        self.mobject._draw(xy, d["radius"] + (f["radius"] - d["radius"]) * e, self.code0, self.code1, e)

    def finish(self):
        self.interpolate(1)
        self.mobject.dots = self.final
        self.mobject._settle()


def dots_from(xy, colors, radius=None, opacity=1.0, hollow=False) -> Swarm:
    """A Swarm of dots at frame positions xy ((n, 2|3), NaN = hidden) with one color each."""
    return Swarm(xy, colors, radius=radius, opacity=opacity, hollow=hollow)


# ---- shapes ---------------------------------------------------------------------------------------
def galaxy_disc(g) -> Circle:
    """A galaxy's disc: hue wash + outline. g: dict, name or id. Label it with galaxy_labels()."""
    g = galaxy(g) if not isinstance(g, dict) else g
    return Circle(
        radius=g["radius"],
        stroke_color=g["color"],
        stroke_opacity=S.disc_stroke,
        stroke_width=2,
        fill_color=g["color"],
        fill_opacity=S.disc_fill,
    ).move_to(p3(g["centre"]))


def galaxy_label(g, stats: bool = False, size: float | None = None, width: float = 99) -> VGroup:
    """Galaxy name (primary ink, semibold; wrapped to `width`) with an optional stats line under it."""
    g = galaxy(g) if not isinstance(g, dict) else g
    size = size or S.label
    out = VGroup(*lines(g["name"], size, S.ink, width, "SEMIBOLD", spacing=0.15))
    if stats:
        out.add(txt(g["stats"], size * S.small / S.label, S.ink2))
    return out.arrange(DOWN, buff=0.05)


def place_labels(items, circles=(), avoid=(), bounds=None, gap: float = 0.08) -> VGroup:
    """Greedy placement, largest circle first. items: (label, centre, radius[, preferred angle in
    degrees]). Each label goes to the spot around its circle that least covers other circles, labels
    already placed, `avoid` mobjects (e.g. a legend) and the outside of bounds [x0, y0, x1, y1]
    (default: the frame above the caption band), stays nearer its own circle than any other, and is
    close to its preferred angle (default 90 = above). circles: extra (centre, radius) obstacles.
    Moves the labels; returns them as a VGroup in the order given."""
    x0, y0, x1, y1 = bounds or (-S.W / 2 + 0.1, S.caption_y + 0.55, S.W / 2 - 0.1, S.H / 2 - 0.1)
    obstacles = [(p3(it[1])[:2], it[2]) for it in items] + [(p3(c)[:2], r) for c, r in circles]
    placed = [(*m.get_corner(DOWN + LEFT)[:2], *m.get_corner(UP + RIGHT)[:2]) for m in avoid]

    def inter(a, q):
        return max(0, min(a[2], q[2]) - max(a[0], q[0])) * max(0, min(a[3], q[3]) - max(a[1], q[1]))

    def edge(box, oc, r):  # gap between a box and a circle (0 when they touch or overlap)
        return max(0.0, np.linalg.norm(oc - np.clip(oc, box[:2], box[2:])) - r)

    def cost(box, own):
        gx, gy = np.meshgrid(np.linspace(box[0], box[2], 9), np.linspace(box[1], box[3], 5))
        pts, area = np.c_[gx.ravel(), gy.ravel()], (box[2] - box[0]) * (box[3] - box[1])
        others = [o for i, o in enumerate(obstacles) if i != own]
        cover = sum(np.mean(np.linalg.norm(pts - oc, axis=1) < r) for oc, r in others)
        mine = edge(box, *obstacles[own])
        owner = sum(edge(box, oc, r) < mine + 0.12 for oc, r in others)  # reads as another circle's label
        crowd = sum(inter(box, q) for q in placed) / area
        return 8 * cover + owner + 4 * crowd + 10 * (1 - inter(box, (x0, y0, x1, y1)) / area)

    for k in np.argsort([-it[2] for it in items], kind="stable"):
        label, centre, r, *pref = items[k]
        c, w, h, pref = p3(centre), label.width, label.height, pref[0] if pref else 90
        best = None
        for step, far in enumerate((0, 0.25, 0.55)):
            for deg in range(0, 360, 15):
                u = np.array([np.cos(np.radians(deg)), np.sin(np.radians(deg))])
                # the box edge or corner nearest the circle sits on the ray at angle deg
                mid = (
                    c[:2] + u * (r + gap + far) + np.clip(u / max(abs(u).max(), 1e-9), -1, 1) * [w / 2, h / 2]
                )
                box = (mid[0] - w / 2, mid[1] - h / 2, mid[0] + w / 2, mid[1] + h / 2)
                turn = abs((deg - pref + 180) % 360 - 180) / 180
                cand = (cost(box, k) + 0.1 * step + 0.6 * turn, box)
                best = cand if best is None or cand[0] < best[0] else best
        box = best[1]
        label.move_to([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2, 0])
        placed.append(box)
    return VGroup(*[it[0] for it in items])


def galaxy_labels(
    stats: bool = False, ids=None, size: float | None = None, avoid=(), hub_width: float = 1.9
) -> VGroup:
    """Labels of the universe galaxies (all, or ids), placed clear of every galaxy disc and `avoid`:
    ring galaxies label outward from the hub; the hub's name wraps to hub_width and sits above it."""
    gs = [g for g in data()["galaxies"] if ids is None or g["id"] in ids]
    others = [(g["centre"], g["radius"]) for g in data()["galaxies"] if g not in gs]
    hub_id = data()["meta"]["hub_id"]
    hub = np.array(galaxy(hub_id)["centre"])
    items = []
    for g in gs:
        away = np.array(g["centre"]) - hub
        outward = 90.0 if g["id"] == hub_id else float(np.degrees(np.arctan2(away[1], away[0])))
        label = galaxy_label(g, stats, size, hub_width if g["id"] == hub_id else 99)
        items.append((label, g["centre"], g["radius"], outward))
    return place_labels(items, others, avoid)


def planet_disc(p, color: str | None = None) -> VMobject:
    """A planet's disc (hue wash; belt = annulus; dashed outline when pinned). p: dict or name."""
    p = planet(p) if not isinstance(p, dict) else p
    col, c = color or hue(p["galaxy_id"]), p3(p["centre"])
    if p["role"] == "belt":
        return Annulus(
            inner_radius=p["r_in"],
            outer_radius=p["radius"],
            fill_color=col,
            fill_opacity=S.planet_fill,
            stroke_width=0,
        ).move_to(c)
    disc = Circle(radius=p["radius"], fill_color=col, fill_opacity=S.planet_fill, stroke_width=0).move_to(c)
    if p["pinned"]:
        ring = DashedVMobject(
            Circle(radius=p["radius"], stroke_color=S.ink2, stroke_width=1.5), num_dashes=28
        ).move_to(c)
        return VGroup(disc, ring)
    return disc


def hull_outline(
    points, pad: float = 0.08, color: str | None = None, corner: float = 0.06, width: float = 1.5
) -> VMobject:
    """Rounded outline around hull vertices [[x, y], ...] (e.g. data()['clusters']['items'][k]['hull'])."""
    pts = np.asarray(points, float)
    c = pts.mean(0)
    d = pts - c
    pts = pts + pad * d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)
    poly = Polygon(
        *[p3(p) for p in pts], stroke_color=color or S.ink2, stroke_width=width, stroke_opacity=0.7
    )
    return poly.round_corners(corner)


def delegate_arrow(arrow: dict, color: str | None = None) -> VGroup:
    """Dotted shaft + solid head + count, for one entry of data()['arrows'] (hub -> galaxy)."""
    a, b, col = p3(arrow["start"]), p3(arrow["end"]), color or S.ink2
    u = (b - a) / np.linalg.norm(b - a)
    shaft = DashedLine(a, b - u * 0.09, dash_length=0.05, dashed_ratio=0.5, stroke_color=col, stroke_width=2)
    head = Triangle(fill_color=col, fill_opacity=1, stroke_width=0).scale(0.06)
    head.rotate(np.arctan2(u[1], u[0]) - np.pi / 2).move_to(b - u * 0.05)
    count = txt(str(arrow["count"]), S.small, S.ink2).move_to((a + b) / 2 + np.array([-u[1], u[0], 0]) * 0.17)
    return VGroup(shaft, head, count)


def legend(rows, size: float | None = None) -> VGroup:
    """Legend rows [(kind, color, label)]; kind: dot, ring, small, disc, line, dashed, arrow.
    Labels stay in secondary ink; the swatch carries the color. Position it with .to_corner(...)."""
    size = size or S.label
    out = VGroup()
    for kind, col, label in rows:
        if kind in ("dot", "ring", "small"):
            sw = Circle(
                radius=0.07 * (0.7 if kind == "small" else 1),
                stroke_color=col,
                stroke_width=2 if kind == "ring" else 0,
                fill_color=col,
                fill_opacity=0 if kind == "ring" else 1,
            )
        elif kind == "disc":
            sw = Circle(radius=0.11, fill_color=col, fill_opacity=0.35, stroke_color=col, stroke_width=1.5)
        elif kind == "line":
            sw = Line(LEFT * 0.2, RIGHT * 0.2, stroke_color=col, stroke_width=3)
        elif kind == "dashed":
            sw = DashedLine(LEFT * 0.2, RIGHT * 0.2, dash_length=0.06, stroke_color=col, stroke_width=3)
        elif kind == "arrow":
            sw = VGroup(
                DashedLine(LEFT * 0.2, RIGHT * 0.12, dash_length=0.04, stroke_color=col, stroke_width=2),
                Triangle(fill_color=col, fill_opacity=1, stroke_width=0)
                .scale(0.05)
                .rotate(-np.pi / 2)
                .move_to(RIGHT * 0.16),
            )
        else:
            raise ValueError(f"legend kind {kind!r}")
        out.add(VGroup(sw, txt(label, size, S.ink2)).arrange(RIGHT, buff=0.15))
    return out.arrange(DOWN, aligned_edge=LEFT, buff=0.16)


# ---- base scene ---------------------------------------------------------------------------------
class VideoScene(MovingCameraScene):
    """Dark surface, the data, and a target length from [durations] (self.target, seconds).
    The camera starts at the default frame (centre 0, width S.W); zoom() moves it, pin() keeps
    captions, kickers and legends fixed on screen while it moves."""

    def setup(self):
        super().setup()
        self.camera.background_color = S.bg
        self.data = data()
        self.target = S.durations.get(type(self).__name__, 0)

    def zoom(self, centre, width: float):
        """Camera move as an animation, e.g. self.play(self.zoom(**stop["camera"]), run_time=2)."""
        return self.camera.frame.animate.move_to(p3(centre)).set(width=width)

    def pin(self, *mobs):
        """Keep mobjects where they are on screen (laid out for the default camera) whatever the
        camera does; returns them. Unpin with mob.clear_updaters()."""
        frame = self.camera.frame
        for m in mobs:
            home, w0 = m.get_center().copy(), max(m.width, 1e-6)

            def follow(mob, home=home, w0=w0):
                k = frame.width / S.W
                mob.set(width=w0 * k).move_to(frame.get_center() + home * k)

            m.add_updater(follow)
            follow(m)
        return mobs[0] if len(mobs) == 1 else mobs

    def finish(self, fade_out: bool = False, run_time: float = 0.6) -> None:
        """Optionally fade everything out, then wait until the scene reaches self.target seconds."""
        if fade_out and self.mobjects:
            self.play(*[FadeOut(m) for m in self.mobjects if m is not self.camera.frame], run_time=run_time)
        left = self.target - self.renderer.time
        if left > 1 / 60:
            self.wait(left)

    def placeholder(self, title: str, point: str) -> None:
        card = title_card(title, point)
        self.play(FadeIn(card))
        self.finish()


# ==== scenes ======================================================================================

# ---- S1-S3 helpers --------------------------------------------------------------------------------
# Tunables not in config.toml yet (read from it when present): move them to [s1]/[s2]/[s3] on merge.
S1_SIDES = [(-1, -0.6), (1, -0.6), (-1, 0.6)]  # [s1] callout direction per [s2] example, at random positions
S2_TAGGED = 3  # [s2] tagged_clusters: the largest text clusters (not the ring) shown with their top words
# [s3] examples: my-system titles shown, each with its box centre in t-SNE frame units (in free corners)
S3_EXAMPLES = [["XYZ-105", [3.75, 2.6]], ["XYZ-683", [-5.2, 2.15]], ["XYZ-130", [4.3, -1.95]]]


def fade_out_all(scene, run_time: float = 0.6) -> None:
    """Hold until the scene's target length minus run_time, then fade everything out (clean last frame)."""
    left = scene.target - scene.renderer.time - run_time
    if left > 1 / 60:
        scene.wait(left)
    mobs = [m for m in scene.mobjects if m is not scene.camera.frame]
    if mobs:
        scene.play(*[FadeOut(m) for m in mobs], run_time=run_time)
    scene.finish()


def shorten(s: str, n: int) -> str:
    return s if len(s) <= n else s[: n - 1].rstrip(" ,;:—–-") + "…"


def panel(mob, buff: float = 0.12, opacity: float = 0.88) -> RoundedRectangle:
    """A background-colored rounded box behind a label, so it reads over dots."""
    return RoundedRectangle(
        width=mob.width + 2 * buff,
        height=mob.height + 2 * buff,
        corner_radius=0.08,
        fill_color=S.bg,
        fill_opacity=opacity,
        stroke_color=S.baseline,
        stroke_width=1,
    ).move_to(mob)


def callout(
    xy, ident: str, title: str, direction=None, at=None, width: float = 4.0, chars: int = 60
) -> VGroup:
    """Ring on one dot, a leader line and a boxed label (id in mono, title in primary ink).
    The box goes at `at` (its centre) or else next to the dot, its nearest corner `direction` (dx, dy)
    away. Returns VGroup(ring, leader, box, label)."""
    c = p3(xy)
    ring = Circle(radius=0.11, stroke_color=S.ink, stroke_width=2.5).move_to(c)
    rows = [txt(ln, S.label, S.ink) for ln in wrap(shorten(title, chars), S.label, width)]
    label = VGroup(txt(ident, S.small, S.ink2, mono=True), *rows).arrange(DOWN, aligned_edge=LEFT, buff=0.1)
    if at is None:
        u = np.array([*direction, 0.0], float) / np.linalg.norm(direction)
        label.move_to(c + u * 0.5 + np.sign(u) * [label.width / 2 + 0.14, label.height / 2 + 0.14, 0])
    else:
        label.move_to(p3(at))
    box = panel(label)
    end = np.clip(c, box.get_corner(DOWN + LEFT), box.get_corner(UP + RIGHT))  # nearest point of the box
    u = (end - c) / max(np.linalg.norm(end - c), 1e-9)
    leader = Line(c + u * 0.11, end, stroke_color=S.ink2, stroke_width=1.5)
    return VGroup(ring, leader, box, label)


def marker(xy, ident: str) -> VGroup:
    """Ring on one dot plus a small boxed id above-right; follows the dot with follow()."""
    ring = Circle(radius=0.1, stroke_color=S.ink, stroke_width=2.5).move_to(p3(xy))
    label = txt(ident, S.small, S.ink, mono=True).next_to(ring, UP + RIGHT, buff=0.02)
    return VGroup(ring, panel(label, 0.06, 0.8), label)


def follow(group, start, end, n: int, idx, lag: float, order, run_time: float) -> UpdateFromAlphaFunc:
    """Move each child of group (whose first submobject sits on a dot) in step with
    sw.morph(xy=..., lag=lag, order=order): the same per-dot timing, so a ring stays on its dot."""
    u = np.argsort(np.argsort(order))[idx] / max(n - 1, 1)
    start, end = np.asarray(start, float), np.asarray(end, float)

    def update(g, alpha):
        for k, m in enumerate(g):
            e = smooth(float(np.clip((alpha - lag * u[k]) / (1 - lag), 0, 1)))
            m.shift(start[k] + (end[k] - start[k]) * e - m[0].get_center())

    return UpdateFromAlphaFunc(group, update, rate_func=linear, run_time=run_time)


def term_card(ex: dict, width: float, size: float, bar_len: float):
    """S2: an example title (id above it) whose top TF-IDF terms can be lit, and its weight bars.
    Returns (head, title rows, glyphs to dim, underlines, bars); bars are VGroup(label, bar) rows."""
    color = S.hues[S.teams.index(ex["team"])]
    title, rows, starts, pos = ex["title"], VGroup(), [], 0
    for ln in wrap(title, size, width):
        starts.append(title.find(ln, pos))
        pos = starts[-1] + len(ln)
        rows.add(Text(ln, font=S.font, font_size=size, color=S.ink, disable_ligatures=True))
    rows.arrange(DOWN, aligned_edge=LEFT, buff=0.14)
    head = txt(ex["id"], S.small, S.ink2, mono=True)
    VGroup(head, rows).arrange(DOWN, aligned_edge=LEFT, buff=0.12)
    # every occurrence of each term (the export keeps only the first span), as TF-IDF counts them all
    spans = [
        m.span()
        for term in ex["terms"]
        for m in re.finditer(r"\b" + r"\W+".join(map(re.escape, term["term"].split())) + r"\b", title, re.I)
    ]
    lit, unders = np.zeros(len(title), bool), VGroup()
    for a, b in spans:
        lit[a:b] = True
    for row, s0 in zip(rows, starts, strict=True):
        for a, b in spans:
            a, b = max(a - s0, 0), min(b - s0, len(row.original_text))
            glyphs = VGroup(*[row[j] for j in range(a, b) if not row.original_text[j].isspace()])
            if len(glyphs):
                y = row.get_bottom()[1] - 0.06
                unders.add(Line([glyphs.get_left()[0], y, 0], [glyphs.get_right()[0], y, 0]))
    unders.set_stroke(color, 3)
    dim = VGroup(
        *[
            row[j]
            for row, s0 in zip(rows, starts, strict=True)
            for j in range(len(row.original_text))
            if not row.original_text[j].isspace() and not lit[s0 + j]
        ]
    )
    bars = VGroup()
    for term in ex["terms"]:
        lab = txt(term["term"], S.small, S.ink2)
        bar = Rectangle(width=max(bar_len * term["weight"], 0.03), height=0.17, stroke_width=0)
        bars.add(VGroup(lab, bar.set_fill(color, 0.95)))
    return head, rows, dim, unders, bars


# ---- S1-S3 scenes ---------------------------------------------------------------------------------
class S1Cloud(VideoScene):
    """1,143 issues as a cloud of dots at random positions, colored by team."""

    def construct(self):
        d = self.data
        n = d["meta"]["n"]
        card = title_card(S.cfg["text"]["title"], f"One Linear workspace, snapshot {d['meta']['snapshot']}")
        self.play(FadeIn(card, shift=UP * 0.15), run_time=1.0)
        self.wait(1.8)
        self.play(FadeOut(card, shift=UP * 0.3), run_time=0.7)

        k = kicker("1 · Cloud")
        leg = legend([("dot", S.hues[t["hue"]], f"{t['name']} {t['n']:,}") for t in d["teams"]])
        leg.arrange(RIGHT, buff=0.5).move_to([0, S.kicker_xy[1], 0]).to_edge(RIGHT, buff=0.47)
        sw = dots_from(np.full((n, 3), np.nan), team_colors())
        self.play(FadeIn(k), FadeIn(leg), sw.morph(xy=positions("random"), lag=0.75), run_time=2.4)
        cap = caption(
            "Each dot is one issue, placed at random",
            "Colored by team: XYZ work, ABC automation output, MaiN personal",
        )
        self.play(FadeIn(cap), run_time=0.5)
        self.wait(0.6)

        sides = S.cfg.get("s1", {}).get("sides", S1_SIDES)
        calls = [
            callout(sw.at(ex["i"])[0], ex["id"], ex["title"], direction=side, chars=50)
            for ex, side in zip(d["s2"]["examples"], sides, strict=True)
        ]
        self.play(
            LaggedStart(*[FadeIn(c, scale=0.9) for c in calls], lag_ratio=0.45),
            run_time=2.2,
        )
        self.wait(1.4)
        cap2 = caption("1,143 issues look like a loose cloud of titles", "Which of them belong together?")
        self.play(*[FadeOut(c) for c in calls], FadeOut(cap), FadeIn(cap2), run_time=0.7)
        fade_out_all(self)


class S2WordsToMap(VideoScene):
    """Titles become TF-IDF word weights; t-SNE places similar titles together; 25 loose clusters."""

    def construct(self):
        d = self.data
        n, cl = d["meta"]["n"], d["clusters"]
        k = kicker("2 · Words → map")
        cap = caption(
            "Each title becomes a set of weighted words (TF-IDF)",
            "Bar length = weight; words that few titles share count more",
        )
        self.play(FadeIn(k), FadeIn(cap), run_time=0.6)

        # three example titles, top terms lit and their weights as bars
        tops, bar_x, cards = [2.35, 0.55, -1.45], 3.55, []
        for ex, top in zip(d["s2"]["examples"], tops, strict=True):
            head, rows, dim, unders, bars = term_card(ex, width=7.6, size=26, bar_len=2.9)
            VGroup(head, rows, unders).align_to([-6.45, top, 0], UP + LEFT)
            for j, row in enumerate(bars):
                row[0].move_to([bar_x - row[0].width / 2, top - 0.42 - 0.34 * j, 0])
                row[1].move_to(row[0]).align_to([bar_x + 0.15, 0, 0], LEFT)
            cards.append((head, rows, dim, unders, bars))
        for j, (head, rows, dim, unders, bars) in enumerate(cards):
            self.play(FadeIn(head), FadeIn(rows), run_time=0.6)
            self.play(
                dim.animate.set_color(S.muted),
                *[Create(u) for u in unders],
                LaggedStart(*[FadeIn(b[0]) for b in bars], lag_ratio=0.15),
                LaggedStart(*[GrowFromEdge(b[1], LEFT) for b in bars], lag_ratio=0.15),
                run_time=1.2,
            )
            self.wait(1.0 if j < 2 else 0.6)
        cap_b = caption(
            "A debrief template has one usable word: debrief",
            "Dates are not words, so every Debrief - <date> title looks the same",
        )
        self.play(FadeOut(cap), FadeIn(cap_b), run_time=0.5)
        self.wait(2.6)

        # dots at random positions; the examples' ids move onto their dots
        ex_idx = [ex["i"] for ex in d["s2"]["examples"]]
        rand, tsne = positions("random"), positions("tsne")
        sw = dots_from(rand, team_colors(), opacity=0.0)
        marks = [marker(rand[i], ex["id"]) for i, ex in zip(ex_idx, d["s2"]["examples"], strict=True)]
        self.add(sw)
        self.bring_to_front(*[c[0] for c in cards])
        cap_c = caption(
            "t-SNE moves titles with similar word weights close together",
            "Watch where the three examples land",
        )
        self.play(
            *[FadeOut(VGroup(rows, unders, bars)) for _, rows, _, unders, bars in cards],
            *[ReplacementTransform(c[0], m[2]) for c, m in zip(cards, marks, strict=True)],
            *[FadeIn(part) for m in marks for part in m[:2]],
            sw.morph(opacity=1.0),
            FadeOut(cap_b),
            FadeIn(cap_c),
            run_time=1.0,
        )
        self.remove(*[part for m in marks for part in m])
        mk = VGroup(*marks)
        self.add(mk)
        self.wait(0.8)
        order = np.random.default_rng(3).random(n)
        lag, fly = 0.35, 3.8
        self.play(
            sw.morph(xy=tsne, lag=lag, order=order),
            follow(mk, rand[ex_idx], tsne[ex_idx], n, ex_idx, lag, order, fly),
            run_time=fly,
        )
        cap_d = caption("Nearby dots have similar wording", "Distances between far-apart groups mean nothing")
        self.play(FadeOut(cap_c), FadeIn(cap_d), run_time=0.5)
        self.wait(2.8)

        # the debrief ring: an artifact of near-identical titles
        ring = cl["ring"]
        members = np.flatnonzero(np.asarray(d["issues"]["cluster"]) == ring["cluster"])
        r = float(np.percentile(np.linalg.norm(tsne[members, :2] - ring["centre"], axis=1), 95)) + 0.16
        halo = DashedVMobject(Circle(radius=r, stroke_color=S.ink, stroke_width=2), num_dashes=40)
        halo.move_to(p3(ring["centre"]))
        note = VGroup(
            txt(verify(f"{len(members)} near-identical"), S.label, S.ink2),
            txt("debrief / braindump titles", S.label, S.ink2),
        ).arrange(DOWN, buff=0.08)
        note.next_to(halo, DOWN, buff=0.16)
        cap_e = caption(
            "Titles that can't be told apart spread into a ring",
            "An artifact of t-SNE, not a shape in the data",
        )
        self.play(
            *[FadeOut(m) for m in mk[:2]],
            *[FadeOut(part) for part in mk[2][1:]],
            Create(halo),
            FadeIn(note),
            FadeOut(cap_d),
            FadeIn(cap_e),
            run_time=1.0,
        )
        self.wait(3.4)

        # wording alone: 25 loose clusters, 280 issues in none
        by_x = sorted(cl["items"], key=lambda c: c["centroid"][0])
        hulls = [hull_outline(c["hull"]) for c in by_x]
        ring_hull = hulls[[c["id"] for c in by_x].index(ring["cluster"])]
        loose = np.asarray(d["issues"]["cluster"]) == -1
        cap_f = caption(
            f"Wording alone gives {cl['n']} loose clusters",
            f"{cl['unclustered']} issues fall in none; {cl['unplaced']} titles share no word with any other",
        )
        self.play(
            FadeOut(halo),
            FadeOut(note),
            FadeOut(mk[2][0]),
            LaggedStart(*[Create(h) for h in hulls], lag_ratio=0.06),
            sw.morph(opacity=np.where(loose, 0.3, 1.0)),
            FadeOut(cap_e),
            FadeIn(cap_f),
            run_time=2.4,
        )
        self.wait(2.0)

        # cluster names are just their top words; some are loose catch-alls
        big = sorted((c for c in cl["items"] if c["id"] != cl["ring"]["cluster"]), key=lambda c: -c["n"])
        named = big[: S.cfg["s2"].get("tagged_clusters", S2_TAGGED)]

        def reach(c):
            return float(np.linalg.norm(np.asarray(c["hull"]) - c["centroid"], axis=1).max())

        items = []
        for c in named:
            label = txt(c["terms"], S.small, S.ink)
            items.append((VGroup(panel(label, 0.07, 0.85), label), c["centroid"], reach(c)))
        others = [(c["centroid"], reach(c)) for c in cl["items"] if c not in named]
        box = (-S.W / 2 + 0.2, -2.75, S.W / 2 - 0.2, 3.35)
        tags = place_labels(items, circles=others, avoid=[k, ring_hull], bounds=box)
        cap_g = caption(
            "Each cluster is named by its top words",
            f"Some are loose catch-alls, like {named[0]['terms']}",
        )
        self.play(
            LaggedStart(*[FadeIn(t) for t in tags], lag_ratio=0.25),
            FadeOut(cap_f),
            FadeIn(cap_g),
            run_time=1.2,
        )
        fade_out_all(self)


class S3WordingNotEnough(VideoScene):
    """One big project (my-system) scatters across most text clusters."""

    def construct(self):
        d = self.data
        n, s3, cl = d["meta"]["n"], d["s3"], d["clusters"]
        k = kicker("3 · Wording isn't enough")
        sw = dots_from(positions("tsne"), team_colors())
        hulls = {c["id"]: hull_outline(c["hull"]) for c in cl["items"]}
        self.play(FadeIn(k), FadeIn(sw), *[FadeIn(h) for h in hulls.values()], run_time=1.0)

        mine = np.isin(np.arange(n), s3["idx"])
        cap = caption(f"Take one big project: {s3['project']}", f"{s3['n']} issues, all in the same project")
        leg = legend([("dot", S.hues[0], s3["project"]), ("dot", S.muted, "other issues")])
        leg.arrange(RIGHT, buff=0.5).move_to([0, S.kicker_xy[1], 0]).to_edge(RIGHT, buff=0.47)
        self.play(
            sw.morph(
                color=np.where(mine, S.hues[0], S.muted),
                opacity=np.where(mine, 1.0, 0.35),
                radius=np.where(mine, S.dot * 1.25, S.dot),
            ),
            FadeIn(cap),
            FadeIn(leg),
            run_time=1.6,
        )
        self.wait(1.4)

        cluster = np.asarray(d["issues"]["cluster"])
        hit = sorted({int(c) for c in cluster[mine] if c >= 0}, key=lambda c: hulls[c].get_center()[0])
        assert len(hit) == s3["clusters_spanned"], (len(hit), s3["clusters_spanned"])
        cap2 = caption(
            f"Its issues scatter across {len(hit)} of {s3['of']} text clusters",
            "Same project, different wording",
        )
        self.play(
            *[h.animate.set_stroke(opacity=0.18) for c, h in hulls.items() if c not in hit],
            LaggedStart(*[hulls[c].animate.set_stroke(S.ink, 2.5, 1.0) for c in hit], lag_ratio=0.08),
            FadeOut(cap),
            FadeIn(cap2),
            run_time=1.8,
        )
        self.wait(0.6)

        ids = list(d["issues"]["id"])
        picks = S.cfg["s3"].get("examples", S3_EXAMPLES)
        calls = [
            callout(sw.at(ids.index(i))[0], i, d["issues"]["title"][ids.index(i)], at=at, width=3.2)
            for i, at in picks
        ]
        self.play(LaggedStart(*[FadeIn(c, scale=0.9) for c in calls], lag_ratio=0.4), run_time=1.8)
        self.wait(4.4)
        cap3 = caption(
            "Titles alone can't tell which work belongs together",
            "Next: tie issues together with projects, milestones and sub-issue links",
        )
        self.play(*[FadeOut(c) for c in calls], FadeOut(cap2), FadeIn(cap3), run_time=0.7)
        fade_out_all(self)


EXTRA_MANIM = ["BackgroundRectangle", "Create", "LaggedStart", "ReplacementTransform"]  # not in scenes.py yet


# ---- local helpers (not in scenes.py) -----------------------------------------------------------
def hold_until(scene, t: float) -> None:
    """Wait until the scene clock reaches t seconds (no-op if already past)."""
    left = t - scene.renderer.time
    if left > 1 / 60:
        scene.wait(left)


def swap_caption(scene, old, new, run_time: float = 0.6):
    """Replace the bottom caption: the old one fades out first (no overlapping text); returns new."""
    if old is not None:
        scene.play(FadeOut(old), run_time=run_time * 0.4)
    scene.play(FadeIn(new), run_time=run_time * 0.6)
    return new


def edge_lines(pos: np.ndarray, pairs, color: str, width: float = 2.4, opacity: float = 0.95) -> VGroup:
    """Straight edges between node positions pos[i] and pos[j] for each (i, j) in pairs."""
    return VGroup(
        *[
            Line(pos[i], pos[j], stroke_color=color, stroke_width=width, stroke_opacity=opacity)
            for i, j, *_ in pairs
        ]
    )


def boxed(mob, opacity: float = 0.82, buff: float = 0.07) -> VGroup:
    """A label on a background-colored plate, readable on top of dots."""
    plate = BackgroundRectangle(mob, color=S.bg, fill_opacity=opacity, buff=buff)
    return VGroup(plate, mob)


def note(text: str, sub: str | None = None, size: float | None = None) -> VGroup:
    """Annotation: a line in primary ink, optionally a second line in secondary ink (left-aligned)."""
    g = VGroup(txt(text, size or S.small, S.ink, "MEDIUM"))
    if sub:
        g.add(txt(sub, S.small, S.ink2))
    return g.arrange(DOWN, aligned_edge=LEFT, buff=0.08)


def group_circle(pts: np.ndarray, pad: float = 0.12):
    """(centre, radius) of a loose circle around points (frame units)."""
    c = pts[:, :2].mean(0)
    return c, float(np.max(np.linalg.norm(pts[:, :2] - c, axis=1)) + pad)


def edge_samples(pos: np.ndarray, pairs, step: float = 0.12) -> np.ndarray:
    """Points every `step` along straight edges (i, j): obstacles for labels."""
    out = [np.zeros((0, 2))]
    for i, j, *_ in pairs:
        a, b = pos[i, :2], pos[j, :2]
        t = np.linspace(0, 1, max(2, int(np.linalg.norm(b - a) / step) + 1))[:, None]
        out.append(a + (b - a) * t)
    return np.vstack(out)


def free_spot(label, centre, r: float, pts: np.ndarray, boxes=(), bounds=None, pref: float = 90.0):
    """Move `label` beside the circle (centre, r) to the spot covering the fewest obstacle points `pts`
    (nodes, edge samples), no box in `boxes` [x0, y0, x1, y1] and nothing outside `bounds`, near the
    preferred angle `pref` (degrees). Returns the box it took."""
    w, h = label.width, label.height
    c, best = np.asarray(centre, float)[:2], None
    for ring, far in enumerate((0.1, 0.35, 0.65, 1.0)):
        for deg in range(0, 360, 10):
            u = np.array([np.cos(np.radians(deg)), np.sin(np.radians(deg))])
            mid = c + u * (r + far) + u / max(abs(u).max(), 1e-9) * [w / 2, h / 2]
            box = (mid[0] - w / 2, mid[1] - h / 2, mid[0] + w / 2, mid[1] + h / 2)
            inside = (pts[:, 0] > box[0] - 0.06) & (pts[:, 0] < box[2] + 0.06)
            inside &= (pts[:, 1] > box[1] - 0.06) & (pts[:, 1] < box[3] + 0.06)
            hits = sum(_boxes_meet(box, q) for q in boxes)
            out = bounds is not None and (
                box[0] < bounds[0] or box[1] < bounds[1] or box[2] > bounds[2] or box[3] > bounds[3]
            )
            turn = abs((deg - pref + 180) % 360 - 180) / 180
            cost = inside.sum() + 60 * hits + 200 * out + 2.0 * turn + 1.5 * ring
            if best is None or cost < best[0]:
                best = (cost, mid, box)
    label.move_to(p3(best[1]))
    return best[2]


def _boxes_meet(a, b, pad: float = 0.08) -> bool:
    return a[0] < b[2] + pad and b[0] < a[2] + pad and a[1] < b[3] + pad and b[1] < a[3] + pad


def leader(box, members: np.ndarray, clear: float, longer: float = 0.3):
    """A thin line from a label box to its nearest member point when that is further than `longer`."""
    near = np.clip(members, box[:2], box[2:])
    dist = np.linalg.norm(members - near, axis=1)
    i = int(np.argmin(dist))
    if dist[i] <= longer:
        return None
    a, b = near[i], members[i]
    b = b - (b - a) / dist[i] * clear
    return Line(p3(a), p3(b), stroke_color=S.ink2, stroke_width=1.4, stroke_opacity=0.8)


def bbox(m) -> tuple:
    (x0, y0), (x1, y1) = m.get_corner(DOWN + LEFT)[:2], m.get_corner(UP + RIGHT)[:2]
    return (x0, y0, x1, y1)


# ==== S4 · The graph ===============================================================================
class S4Graph(VideoScene):
    """A readable subset (two projects, 65 issues) gains its six edge layers one at a time."""

    LAYER_HUES = {"text_knn": 0, "project": 1, "milestone": 2, "parent": 3, "sibling": 4, "repo": 6}

    def construct(self):
        d, s4 = self.data, self.data["s4"]
        n, idx = d["meta"]["n"], np.array(s4["idx"])
        pos = np.array([p3(p) for p in s4["pos"]])
        ids, titles = list(s4["ids"]), list(s4["titles"])
        inside = np.zeros(n, bool)
        inside[idx] = True
        layers, order = s4["layers"], s4["order"]
        color = {k: S.hues[self.LAYER_HUES.get(k, i)] for i, k in enumerate(order)}

        # 1 · the whole text map, then only the subset, flown to its own layout
        k = kicker("4 · The graph")
        tsne = positions("tsne")
        sw = dots_from(tsne, S.ink2, opacity=np.where(inside, 1.0, S.dim))
        self.play(FadeIn(k), FadeIn(sw), run_time=1.0)
        projects = " and ".join(s4["projects"])
        cap = swap_caption(
            self,
            None,
            caption(f"Zoom in on two projects: {projects}", f"{len(idx)} issues, few enough to read"),
        )
        target = np.full_like(tsne, np.nan)
        target[idx] = pos
        self.play(sw.morph(xy=target, radius=np.where(inside, S.dot * 1.55, S.dot), lag=0.25), run_time=2.6)

        # legend on the right, rows revealed with their layer
        legend_rows = legend([("line", color[k], layers[k]["label"]) for k in order], size=S.small)
        head = txt("Kinds of link", S.label, S.ink, "SEMIBOLD")
        leg = VGroup(head, legend_rows).arrange(DOWN, aligned_edge=LEFT, buff=0.22)
        leg.to_edge(RIGHT, buff=0.3).align_to(p3([0, 3.05]), UP)
        self.play(FadeIn(head), run_time=0.4)
        hold_until(self, 5.6)

        # annotations avoid every node (weighted), the edges on screen, the legend and the caption band
        bounds = (-S.W / 2 + 0.15, S.caption_y + 0.62, leg.get_left()[0] - 0.25, S.H / 2 - 0.5)

        def annotate(spec, pairs):
            """spec: (label mobject, node indices, preferred angle); pairs: edges on screen."""
            pts = np.vstack([np.repeat(pos[:, :2], 4, axis=0), edge_samples(pos, pairs)])
            boxes, out = [bbox(leg), bbox(k)], VGroup()
            for label, members, deg in spec:
                c, r = group_circle(pos[members])
                plate = boxed(label)
                box = free_spot(plate, c, r, pts, boxes, bounds, deg)
                boxes.append(box)
                out.add(plate)
                lead = leader(box, pos[members, :2], S.dot * 2.8)
                if lead is not None:
                    out.add(lead)
            return out

        def mark(members, col):
            return VGroup(
                *[
                    Circle(radius=S.dot * 2.6, stroke_color=col, stroke_width=2.2).move_to(pos[m])
                    for m in members
                ]
            )

        digest = [i for i, t in enumerate(titles) if "digest" in t.lower() and ids[i].startswith("ABC")]
        machine = layers["project"]["groups"].get("machine", [])
        salley = layers["project"]["groups"].get("salley", [])
        ms_key, ms_members = next(iter(layers["milestone"]["groups"].items()))
        parent_pairs = layers["parent"]["display"]
        heads = [i for i in {p[0] for p in parent_pairs} if sum(p[0] == i for p in parent_pairs) > 1]
        tree_root = heads[0] if heads else parent_pairs[0][0]
        kids = [j for i, j in parent_pairs if i == tree_root]
        repo_groups = sorted(layers["repo"]["groups"].items(), key=lambda kv: -len(kv[1]))
        top_salley = salley[int(np.argmax(pos[salley, 1]))] if salley else 0
        ms_name = ms_key.split("::", 1)[-1]

        # per layer: caption, annotation spec
        plan = {
            "text_knn": (
                "Title text: each issue links to its most similar titles",
                [(note("Salley digest — <date>", "near-identical titles"), digest, 135)],
            ),
            "project": (
                "Same project: every issue in a project is linked",
                [
                    (note("machine", "project"), machine, 270),
                    (note("salley", "project"), [top_salley], 90),
                ],
            ),
            "milestone": (
                "Same milestone: a sub-topic inside a project",
                [(note(ms_name, f"milestone · {len(ms_members)} issues"), ms_members, 60)],
            ),
            "parent": (
                f"Parent / sub-issue: one parent and its {len(kids)} sub-issues",
                [(note(ids[tree_root], titles[tree_root]), [tree_root], 150)],
            ),
            "sibling": (
                "Siblings: sub-issues of the same parent",
                [(note("siblings", f"children of {ids[tree_root]}"), kids, 60)],
            ),
            "repo": (
                "Same repo label: links that cross projects",
                [(note(g, "repo label"), m, 270 if i else 120) for i, (g, m) in enumerate(repo_groups[:2])],
            ),
        }
        marks = {
            "text_knn": digest,
            "project": [],
            "milestone": ms_members,
            "parent": [tree_root],
            "sibling": kids,
            "repo": [m for _, ms in repo_groups[:2] for m in ms],
        }

        drawn, old_note, old_mark, shown = [], None, None, []
        step = (29.2 - 6.0) / len(order)  # layers share 6 s .. 29.2 s
        for li, name in enumerate(order):
            t0 = 6.0 + li * step
            hold_until(self, t0)
            lines_ = edge_lines(pos, layers[name]["display"], color[name])
            self.add(lines_)
            self.bring_to_front(sw)
            text, spec = plan.get(name, (layers[name]["label"], []))
            new_cap = caption(text)
            shown += layers[name]["display"]
            ann = annotate(spec, shown) if spec else VGroup()
            mk = mark(marks.get(name, []), S.ink)
            fades = [FadeOut(m) for m in (old_note, old_mark) if m is not None]
            dims = [m.animate.set_stroke(width=1.4, opacity=0.38) for m in drawn]
            row = legend_rows[li]
            prev = [legend_rows[li - 1][1].animate.set_color(S.ink2)] if li else []
            row[1].set_color(S.ink)
            self.play(*fades, *dims, *prev, FadeOut(cap), run_time=0.35)
            self.play(
                FadeIn(new_cap),
                FadeIn(row),
                LaggedStart(*[Create(m) for m in lines_], lag_ratio=0.6 / max(len(lines_), 1)),
                run_time=1.2,
            )
            cap = new_cap
            self.play(FadeIn(ann), FadeIn(mk), run_time=0.5)
            drawn.append(lines_)
            old_note, old_mark = ann, mk

        # all layers together
        hold_until(self, 29.2)
        final = caption(
            "Six kinds of link, weighted and added, make one graph",
            f"built on all {sum(r != 'hub' for r in d['issues']['route']):,} human issues",
        )
        self.play(FadeOut(old_note), FadeOut(old_mark), FadeOut(cap), run_time=0.35)
        self.play(
            *[m.animate.set_stroke(width=1.8, opacity=0.7) for m in drawn],
            legend_rows[-1][1].animate.set_color(S.ink2),
            FadeIn(final),
            run_time=0.8,
        )
        hold_until(self, self.target - 0.65)
        self.finish(fade_out=True, run_time=0.6)


# ==== S5 · Hub and communities =====================================================================
class S5HubCommunities(VideoScene):
    """Automation items set aside by rule; Louvain seeds differ a little; the consensus: 8 galaxies."""

    def construct(self):
        d, s5 = self.data, self.data["s5"]
        n, hub_id = d["meta"]["n"], d["meta"]["hub_id"]
        hub = np.zeros(n, bool)
        hub[s5["hub_idx"]] = True
        tsne, graph = positions("tsne"), positions("graph")
        human_n = int((~hub).sum())

        k = kicker("5 · Hub and communities")
        sw = dots_from(tsne, S.ink2, opacity=0.85)
        self.play(FadeIn(k), FadeIn(sw), run_time=1.0)

        # 1 · the hub by rule
        cap = swap_caption(
            self,
            None,
            caption(
                "First, a rule sets the automation hub aside",
                f"debrief templates and project-less ABC items: {int(hub.sum())} issues",
            ),
        )
        red = hue(hub_id)
        self.play(sw.morph(color=np.where(hub, red, S.ink2), opacity=np.where(hub, 1.0, 0.45)), run_time=1.0)
        # hub items go to their belt; the rest of the text map slides right to make room
        room = tsne.copy()
        room[~hub, 0] += graph[hub, 0].max() + 0.9 - np.nanmin(tsne[~hub, 0])
        belt_xy = np.where(hub[:, None], graph, room)
        rank = np.full(n, -1.0)
        rank[s5["hub_idx"]] = np.arange(len(s5["hub_idx"]))
        self.play(sw.morph(xy=belt_xy, lag=0.35, order=rank), run_time=2.4)
        hub_g = galaxy(hub_id)
        hub_pts = graph[hub]
        hub_label = VGroup(
            txt(hub_g["name"], S.label, S.ink, "SEMIBOLD"), txt("set aside by rule", S.small, S.ink2)
        ).arrange(DOWN, buff=0.08)
        hub_label.next_to(p3([np.nanmean(hub_pts[:, 0]), np.nanmin(hub_pts[:, 1])]), DOWN, buff=0.22)
        self.play(FadeIn(hub_label), run_time=0.5)
        hold_until(self, 7.0)

        # 2 · the other issues, pulled together by their links
        cap = swap_caption(
            self,
            cap,
            caption(
                f"The other {human_n:,} issues are pulled together by their links",
                "a spring layout of the issue graph: linked issues sit close",
            ),
        )
        self.play(sw.morph(xy=graph, opacity=np.where(hub, 1.0, 0.9), lag=0.3), run_time=3.0)
        hold_until(self, 12.6)

        # 3 · single Louvain seeds, with the issues each one places differently
        cap = swap_caption(
            self,
            cap,
            caption(
                "Louvain finds communities, but each seed differs a little",
                "ring: an issue this seed places differently from the consensus",
            ),
        )
        rings = Swarm(graph, S.ink, radius=S.dot * 2.3, opacity=0.0, hollow=True, stroke_width=2.4)
        self.add(rings)
        tag = None
        seeds = s5["seeds"]
        t_seed = self.renderer.time
        for si, s in enumerate(seeds):
            moved = np.zeros(n, bool)
            moved[s["moved_idx"]] = True
            if tag is not None:
                self.play(FadeOut(tag), run_time=0.25)
            tag = self.seed_tag(f"seed {s['seed']}", f"ARI {s['ari']} vs the consensus")
            self.play(
                sw.morph(color=galaxy_colors(s["galaxy_id"]), opacity=1.0),
                rings.morph(opacity=np.where(moved, 1.0, 0.0)),
                FadeIn(tag),
                run_time=1.1,
            )
            hold_until(self, t_seed + (si + 1) * 3.4)

        # 4 · the consensus of all runs
        runs = s5["runs"]
        self.play(FadeOut(tag), run_time=0.25)
        tag = self.seed_tag(f"consensus of {runs} runs", "grouped together in at least half the runs")
        self.play(
            sw.morph(color=galaxy_colors(), opacity=1.0), rings.morph(opacity=0.0), FadeIn(tag), run_time=1.3
        )
        cap = swap_caption(
            self,
            cap,
            caption(
                f"The consensus: {len(d['galaxies'])} galaxies, the hub among them",
                f"single seeds agree with it at ARI {s5['seed_ari_mean']} (1 = identical)",
            ),
        )
        labels = self.galaxy_names(d["galaxies"], graph, hub_id, avoid=[tag, hub_label, k])
        self.play(LaggedStart(*[FadeIn(m) for m in labels], lag_ratio=0.12), run_time=1.6)
        hold_until(self, self.target - 0.65)
        self.remove(rings)
        self.finish(fade_out=True, run_time=0.6)

    @staticmethod
    def seed_tag(head: str, sub: str) -> VGroup:
        g = VGroup(txt(head, S.label + 4, S.ink, "SEMIBOLD"), txt(verify(sub), S.small, S.ink2))
        g.arrange(DOWN, aligned_edge=LEFT, buff=0.08)
        return g.move_to(p3([S.kicker_xy[0], 2.95]), aligned_edge=UP + LEFT)

    @staticmethod
    def galaxy_names(galaxies, graph: np.ndarray, hub_id: int, avoid=()) -> VGroup:
        """Human galaxy names at their clump in the graph layout, each on a plate."""
        gid = column("galaxy_id")
        out = []
        for g in [g for g in galaxies if g["id"] != hub_id]:
            pts = graph[gid == g["id"]][:, :2]
            c = np.median(pts, axis=0)
            swatch = Circle(radius=0.075, fill_color=g["color"], fill_opacity=1, stroke_width=0)
            label = boxed(
                VGroup(swatch, txt(g["name"], S.label, S.ink, "SEMIBOLD")).arrange(RIGHT, buff=0.12)
            )
            out.append(label.move_to(p3(c)))
        group = VGroup(*out)
        for m in avoid:  # nudge labels off the corner tag and the hub label
            for lb in group:
                if _boxes_meet(bbox(lb), bbox(m), 0.0):
                    lb.shift(DOWN * (lb.get_top()[1] - m.get_bottom()[1] + 0.1))
        return group


# ==== S6 · Planets and names =======================================================================
class S6PlanetsNames(VideoScene):
    """Inside one galaxy, Louvain splits planets out; each is named by the first rule that fits."""

    RADIUS = 2.2  # the galaxy is drawn enlarged to this radius, left of the naming panel (frame units)

    def construct(self):
        d, s6 = self.data, self.data["s6"]
        g = galaxy(s6["galaxy_id"])
        gid = column("galaxy_id")
        mine = gid == g["id"]
        planets = sorted([p for p in d["planets"] if p["galaxy_id"] == g["id"]], key=lambda p: -p["n"])
        skip = {g["id"], d["meta"]["hub_id"]}
        others = [p for p in d["planets"] if p["galaxy_id"] not in skip]

        # the naming panel first (at full size), so the galaxy can take the room left of it
        def here(source):
            return [p for p in planets if p["name_source"] == source]

        def examples(source, k):
            ps = [p for p in others if p["name_source"] == source and not p["mixed"]]
            return [p["name"] for p in sorted(ps, key=lambda p: -p["n"])[:k]]

        def head(t):
            return txt(verify(t), S.label, S.ink, "MEDIUM")

        def sub(t):
            return txt(t, S.small, S.ink2)

        share = dict(zip(["milestone", "ritual label", "project"], s6["cascade"], strict=False))
        pct = {k: f"  ≥ {round(v * 100)}%" for k, v in share.items()}
        loose = sorted([p["name"] for p in d["planets"] if p["mixed"]], key=len)[:2]
        override = here("override")
        rows = [  # (source, panel block, planets named by it here)
            ("milestone", [head("1 · Milestone" + pct["milestone"]),
                           sub("here: " + ", ".join(p["name"] for p in here("milestone")))]),
            ("ritual label", [head("2 · Ritual label" + pct["ritual label"]),
                              sub("e.g. " + ", ".join(examples("ritual label", 2)))]),
            ("project", [head("3 · Project" + pct["project"]),
                         sub("e.g. " + ", ".join(examples("project", 3)))]),
            ("distinctive terms", [head("4 · Distinctive title terms"),
                                   sub("here: " + ", ".join(p["name"] for p in here("distinctive terms")))]),
            ("override", [head("… or a short override map")]
             + [sub(f"{p['name_evidence']}  →  {p['name']}") for p in override[:1]]),
            ("mixed", [head("(mixed) when titles are loose"), sub("e.g. " + ", ".join(loose))]),
        ]  # fmt: skip
        title = txt("How a planet gets its name", S.label + 4, S.ink, "SEMIBOLD")
        blocks = [VGroup(*b).arrange(DOWN, aligned_edge=LEFT, buff=0.08) for _, b in rows]
        panel = VGroup(title, *blocks).arrange(DOWN, aligned_edge=LEFT, buff=0.26)
        panel.to_edge(RIGHT, buff=0.35).align_to(p3([0, 3.2]), UP)

        # the galaxy, enlarged, centred in the room left of the panel
        R, x0 = self.RADIUS, -S.W / 2 + 0.15
        ctr = np.array([(x0 + panel.get_left()[0] - 0.3) / 2, -0.05])
        kz = R / g["radius"]

        def to(xy):
            return (np.asarray(xy, float)[..., :2] - np.asarray(g["centre"])) * kz + ctr

        # start: the galaxy's issues as one clump (their graph-layout positions, recentred)
        gp = positions("graph")[mine][:, :2]
        m0 = np.median(gp, axis=0)
        rel = (gp - m0) * (0.82 * R / np.quantile(np.linalg.norm(gp - m0, axis=1), 0.95))
        norm = np.linalg.norm(rel, axis=1, keepdims=True)
        rel = np.where(norm > 0.93 * R, rel * 0.93 * R / np.maximum(norm, 1e-9), rel)
        start = np.full((len(gid), 3), np.nan)
        start[mine] = np.c_[rel + ctr, np.zeros(len(rel))]
        end = np.full_like(start, np.nan)
        end[mine] = np.c_[to(positions("universe")[mine]), np.zeros(int(mine.sum()))]

        k = kicker("6 · Planets and names")
        sw = dots_from(start, g["color"], radius=d["meta"]["moon_radius"] * kz * 1.15)
        disc = galaxy_disc({**g, "centre": ctr.tolist(), "radius": R})
        gname = VGroup(txt(g["name"], S.label + 2, S.ink, "SEMIBOLD"), txt(g["stats"], S.small, S.ink2))
        gname.arrange(DOWN, aligned_edge=LEFT, buff=0.06).next_to(disc, UP, buff=0.12).align_to(disc, LEFT)
        self.play(FadeIn(k), FadeIn(sw), FadeIn(disc), FadeIn(gname), run_time=1.0)
        cap = swap_caption(
            self,
            None,
            caption(
                "Inside each galaxy, Louvain runs again to find planets",
                f"{g['name']}: {g['n']} issues, {g['planets']} planets",
            ),
        )

        # split into planets
        zp = {p["id"]: {**p, "centre": to(p["centre"]).tolist(), "radius": p["radius"] * kz} for p in planets}
        discs = {pid: planet_disc(p) for pid, p in zp.items()}
        for m in discs.values():
            m.set_opacity(0)
        self.add(*discs.values())
        self.bring_to_front(sw)
        self.play(sw.morph(xy=end, lag=0.3, order=column("planet_id")), run_time=2.6)
        self.play(*[m.animate.set_fill(opacity=S.planet_fill) for m in discs.values()], run_time=0.6)

        # planet labels: final names placed clear of every planet, dot, label and the panel;
        # the raw term labels of override planets sit where their final names will land
        def plabel(p, name=None, tag=None):
            tag = tag or {"override": "override map"}.get(p["name_source"], p["name_source"])
            return VGroup(
                txt(name or p["name"], S.small, S.ink, "SEMIBOLD"), txt(tag, S.small, S.ink2)
            ).arrange(DOWN, aligned_edge=LEFT, buff=0.05)

        grid = np.mgrid[-1:1.01:0.1, -1:1.01:0.1].reshape(2, -1).T
        grid = grid[np.linalg.norm(grid, axis=1) <= 1]
        pts = np.vstack([np.asarray(z["centre"]) + grid * z["radius"] for z in zp.values()] + [end[mine, :2]])
        bounds = (x0, S.caption_y + 0.62, panel.get_left()[0] - 0.2, S.H / 2 - 0.45)
        boxes = [bbox(gname), bbox(k), bbox(panel)]
        final, leads = {}, {}
        for p in planets:
            final[p["id"]] = plabel(p)
            z = zp[p["id"]]
            pref = float(np.degrees(np.arctan2(*(np.asarray(z["centre"]) - ctr)[::-1])))
            box = free_spot(final[p["id"]], z["centre"], z["radius"], pts, boxes, bounds, pref)
            boxes.append(box)
            rim = (
                np.asarray(z["centre"])
                + z["radius"] * np.c_[np.cos(np.r_[0:6.3:0.1]), np.sin(np.r_[0:6.3:0.1])]
            )
            leads[p["id"]] = leader(box, rim, 0.03, longer=0.22) or VGroup()
        raw = {
            p["id"]: plabel(p, p["name_evidence"], "distinctive terms").move_to(
                final[p["id"]], aligned_edge=LEFT
            )
            for p in override
        }

        hold_until(self, 6.4)
        cap = swap_caption(
            self,
            cap,
            caption("Each planet is named by the first rule that fits", "the owner's own words come first"),
        )
        self.play(FadeIn(title), run_time=0.4)
        t_rows, step = 7.4, 2.0
        for ri, (source, _) in enumerate(rows):
            hold_until(self, t_rows + ri * step)
            hits = here(source) + (override if source == "distinctive terms" else [])
            anims = [FadeIn(blocks[ri], shift=RIGHT * 0.1)]
            if source == "override":
                anims += [ReplacementTransform(raw[p["id"]], final[p["id"]]) for p in override]
            else:
                anims += [FadeIn(raw.get(p["id"], final[p["id"]])) for p in hits]
                anims += [FadeIn(leads[p["id"]]) for p in hits]
            pulse = [discs[p["id"]].animate.set_fill(opacity=S.planet_fill * 2.6) for p in hits]
            self.play(*anims, *pulse, run_time=0.8)
            if pulse:
                self.play(*[m.animate.set_fill(opacity=S.planet_fill) for m in discs.values()], run_time=0.4)

        hold_until(self, 19.3)
        cap = swap_caption(
            self,
            cap,
            caption(
                "Two-thirds of human issues: planets named in the owner's words",
                "by milestone, ritual label or project",
            ),
        )
        hold_until(self, self.target - 0.65)
        self.finish(fade_out=True, run_time=0.6)


# ---- S7-S10 helpers -------------------------------------------------------------------------------
# Tunables not in config.toml yet (read from it when present): move them to [s7]/[tour]/[s9] on merge.
S7_FLY_ORDER = [
    7,
    0,
    5,
    6,
    4,
    3,
    2,
    1,
]  # [s7] fly_order: galaxies leave the t-SNE map in this order (hub first)
TOUR_DIM = 0.3  # [tour] dim: opacity of a stop's moons outside its focus planets
TOUR_TAGS = 4  # [tour] tags_max: at most this many planet labels per stop (focus first, then the largest)
TOUR_BAND = 0.85  # [tour] band_opacity: background band behind the caption while zoomed in
OVERLAY_Z = 3  # z-index of pinned overlays (kicker, header, legend, caption), above the chart and its arrows


def hold_to(scene, t: float) -> None:
    """Wait until the scene clock reaches t seconds (no-op when already past)."""
    left = t - scene.renderer.time
    if left > 1 / 60:
        scene.wait(left)


def close_scene(scene, run_time: float = 0.6, tail: float = 0.2) -> None:
    """Fade every mobject out so it is gone `tail` seconds before the target (a clean last frame),
    then pad to the target."""
    hold_to(scene, scene.target - run_time - tail)
    mobs = [m for m in scene.mobjects if m is not scene.camera.frame]
    if mobs:
        scene.play(*[FadeOut(m) for m in mobs], run_time=run_time)
    scene.finish()


def handover(old, new, gap: float = 0.1) -> UpdateFromAlphaFunc:
    """Fade `old` out over the first half and `new` in over the second, so two captions never overlap.
    Pinned mobjects stay pinned (their updaters keep running). Add `new` to the scene first; remove
    `old` afterwards (swap() does both)."""
    half = 0.5 - gap / 2

    def step(_, a):
        if old is not None:
            old.set_opacity(float(np.clip(1 - a / half, 0, 1)))
        new.set_opacity(float(np.clip((a - 1 + half) / half, 0, 1)))

    return UpdateFromAlphaFunc(new, step)


def swap(scene, old, new, *anims, run_time: float = 0.8):
    """Replace one caption (or header) with another, alongside other animations; returns new."""
    scene.add(new.set_opacity(0))
    scene.play(handover(old, new), *anims, run_time=run_time)
    if old is not None:
        scene.remove(old)
    return new


def universe_layers():
    """The universe chart without labels: galaxy discs, planet discs, moons (a Swarm) and the delegate
    arrows, z-ordered discs < planets < moons."""
    d = data()
    discs = VGroup(*[galaxy_disc(g) for g in d["galaxies"]]).set_z_index(-2)
    planets = VGroup(*[planet_disc(p) for p in d["planets"]]).set_z_index(-1)
    moons = Swarm(positions("universe"), **universe_style())
    arrows = VGroup(*[delegate_arrow(a) for a in d["arrows"]])
    return discs, planets, moons, arrows


def backing(mob, buff: float = 0.12, opacity: float = 0.85) -> RoundedRectangle:
    """A background-colored plate behind a label or legend, so it reads over discs and moons."""
    return RoundedRectangle(
        width=mob.width + 2 * buff,
        height=mob.height + 2 * buff,
        corner_radius=0.08,
        fill_color=S.bg,
        fill_opacity=opacity,
        stroke_width=0,
    ).move_to(mob)


def universe_legend() -> VGroup:
    """Moon states, pinned outline and delegate arrows on a plate; bottom left, above the caption band."""
    rows = [
        ("dot", S.ink2, "done"),
        ("ring", S.ink2, "open"),
        ("small", S.muted, "canceled / duplicate"),
        ("dashed", S.ink2, "pinned planet"),
        ("arrow", S.ink2, "delegated items"),
    ]
    leg = legend(rows, S.small).to_corner(DL, buff=0.35)
    leg.shift(UP * (S.caption_y + 0.7 - leg.get_bottom()[1]))
    return VGroup(backing(leg), leg)


def stop_header(g: dict) -> VGroup:
    """Galaxy name and stats line on a plate at the top left, under the kicker (default-camera layout)."""
    head = VGroup(txt(g["name"], S.subtitle, S.ink, "SEMIBOLD"), txt(g["stats"], S.label, S.ink2))
    head.arrange(DOWN, aligned_edge=LEFT, buff=0.12)
    head.move_to(p3([S.kicker_xy[0], S.kicker_xy[1] - 0.4]), aligned_edge=UL)
    return VGroup(backing(head, 0.1), head)


def story_caption(story: str) -> VGroup:
    """A tour story as the caption: one line when it fits, else split at its first ': ', ', ' or ' — '
    into a primary line and a secondary one."""
    one = txt(verify(story), S.caption)
    if one.width <= S.caption_w:
        return VGroup(one).move_to([0, S.caption_y, 0])
    for sep in (": ", ", ", " — "):
        if sep in story:
            return caption(*story.split(sep, 1))
    return caption(story)


def keep_stroke(scene, *mobs) -> None:
    """Keep stroke widths constant on screen while the camera zooms (ManimCE's Cairo camera draws
    strokes in frame units, so they thicken as the camera zooms in). Adds an updater to each mob."""
    frame = scene.camera.frame
    for m in mobs:
        base = [sm.get_stroke_width() for sm in m.family_members_with_points()]

        def update(mob, base=base):
            k = frame.width / S.W
            for sm, w in zip(mob.family_members_with_points(), base, strict=False):
                if w:
                    sm.set_stroke(width=w * k, family=False)

        m.add_updater(update)
        update(m)


def keep_moon_stroke(scene, sw) -> None:
    """keep_stroke for a Swarm's hollow (open) moons. Play its morphs with suspend_mobject_updating=False
    while the camera moves."""
    frame = scene.camera.frame

    def update(m):
        w = S.hollow * frame.width / S.W
        if abs(m.stroke - w) > 1e-6:
            m.stroke = w
            for batch in m.submobjects:
                if batch.get_stroke_width() > 0:
                    batch.set_stroke(width=w, family=False)

    sw.add_updater(update)


def zoom_arrows(scene) -> VGroup:
    """delegate_arrow()s whose stroke, head and count keep their on-screen size while the camera zooms."""
    frame, out = scene.camera.frame, VGroup()
    for a in data()["arrows"]:
        arrow = delegate_arrow(a)
        start, end = p3(a["start"]), p3(a["end"])
        u = (end - start) / np.linalg.norm(end - start)

        def update(m, end=end, u=u, mid=(start + end) / 2, hw=arrow[1].width, cw=arrow[2].width, sw=2.0):
            k = frame.width / S.W
            m[0].set_stroke(width=sw * k)
            m[1].scale(hw * k / m[1].width).move_to(end - u * 0.05 * k)
            m[2].scale(cw * k / m[2].width).move_to(mid + np.array([-u[1], u[0], 0]) * 0.17 * k)

        arrow.add_updater(update)
        out.add(arrow)
    return out


def view_bounds(centre, width: float) -> tuple:
    """place_labels bounds of a camera view: the frame above the caption band, scaled by the zoom."""
    k, c = width / S.W, p3(centre)
    x0, y0, x1, y1 = -S.W / 2 + 0.15, S.caption_y + 0.6, S.W / 2 - 0.15, S.H / 2 - 0.3
    return (c[0] + x0 * k, c[1] + y0 * k, c[0] + x1 * k, c[1] + y1 * k)


def tour_planets(g: dict, stop: dict) -> list:
    """Planets labelled at a tour stop: the focus planets, then the largest up to TOUR_TAGS; at the hub
    only its largest core planet joins the focus (the arrows carry the delegated ones). In a crowded
    galaxy (over 2 x TOUR_TAGS planets) only planets in the outer half have room for a label."""
    mine = [p for p in data()["planets"] if p["galaxy_id"] == g["id"]]
    focus = [p for p in mine if p["id"] in stop["focus_planets"]]
    rest = sorted((p for p in mine if p not in focus), key=lambda p: -p["n"])
    if g["id"] == data()["meta"]["hub_id"]:
        return focus + [p for p in rest if p["role"] == "core"][:1]
    if len(mine) > 2 * TOUR_TAGS:
        rest = [p for p in rest if np.linalg.norm(np.subtract(p["centre"], g["centre"])) > g["radius"] / 2]
    return focus + rest[: max(0, TOUR_TAGS - len(focus))]


def planet_tags(planets: list, k: float, bounds, avoid=(), pad: dict | None = None) -> VGroup:
    """Labels (name, then "n · k open") for planets, sized for a camera zoom k = width / S.W and placed
    clear of every planet circle (enlarged by pad[id], e.g. a focus ring) and of the `avoid` mobjects."""
    d, pad = data(), pad or {}
    ids = {p["id"] for p in planets}
    others = [(p["centre"], p["radius"] + pad.get(p["id"], 0)) for p in d["planets"] if p["id"] not in ids]
    items = []
    for p in planets:
        tag = VGroup(txt(p["name"], S.small, S.ink), txt(f"{p['n']} · {p['open']} open", S.small, S.ink2))
        tag.arrange(DOWN, buff=0.05).scale(k)
        out = np.asarray(p["centre"]) - np.asarray(galaxy(p["galaxy_id"])["centre"])
        pref = float(np.degrees(np.arctan2(out[1], out[0]))) if np.linalg.norm(out) > 1e-6 else 90.0
        items.append((tag, p["centre"], p["radius"] + pad.get(p["id"], 0), pref))
    tags = place_labels(items, others, avoid, bounds, gap=0.06 * k)
    for tag in tags:  # the placer only penalizes overlap: push a tag sideways out of any avoid box
        for m in avoid:
            (x0, y0, _), (x1, y1, _) = m.get_corner(DL), m.get_corner(UR)
            (a0, b0, _), (a1, b1, _) = tag.get_corner(DL), tag.get_corner(UR)
            if a0 < x1 and a1 > x0 and b0 < y1 and b1 > y0:
                right = tag.get_center()[0] > m.get_center()[0]
                tag.shift(
                    RIGHT * ((x1 - a0) if right else (x0 - a1)) + RIGHT * (0.06 * k if right else -0.06 * k)
                )
    return tags


def two_dp(v: float) -> str:
    """0.415 -> '0.42': round half up from the decimal value, as the README does (floats give 0.41)."""
    return str(Decimal(str(v)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def focus_ring(centre, r: float, k: float = 1.0) -> Circle:
    """Primary-ink ring around a focus planet, for a camera zoom k (stroke and gap constant on screen)."""
    return Circle(radius=r + 0.05 * k, stroke_color=S.ink, stroke_width=2.5 * k).move_to(p3(centre))


# ---- S7: the universe ---------------------------------------------------------------------------
class S7Universe(VideoScene):
    """Every issue flies from the wording map to its galaxy and planet; discs, names and the delegate
    arrows appear; positions carry no meaning."""

    def construct(self):
        d = self.data
        k = kicker("7 · The universe")
        sw = Swarm(positions("tsne"), galaxy_colors())
        cap = caption("The wording map, coloured by galaxy", "Now each issue moves to its galaxy and planet")
        self.play(FadeIn(k), FadeIn(sw), FadeIn(cap), run_time=0.8)
        hold_to(self, 1.4)

        rank = {g: i for i, g in enumerate(S7_FLY_ORDER)}
        gid = column("galaxy_id")
        order = np.array([rank.get(g, len(rank)) for g in gid]) + np.random.default_rng(1).uniform(
            0, 0.9, len(gid)
        )
        self.play(sw.morph(xy=positions("universe"), **universe_style(), lag=0.55, order=order), run_time=5.0)

        discs, planets, _, arrows = universe_layers()
        leg = universe_legend()
        names = galaxy_labels(avoid=[leg])
        cap = swap(
            self,
            cap,
            caption(
                "8 galaxies, 46 planets, 1,143 issues",
                "Moons are issues; a planet's area grows with its issues",
            ),
            FadeIn(discs),
            FadeIn(planets),
            run_time=1.2,
        )
        self.play(LaggedStart(*[FadeIn(n, shift=UP * 0.08) for n in names], lag_ratio=0.12), run_time=1.4)

        hold_to(self, 12.0)
        pinned = [p for p in d["planets"] if p["pinned"]]
        rings = VGroup(*[focus_ring(p["centre"], p["radius"]) for p in pinned])
        cap = swap(
            self,
            cap,
            caption(
                "A dashed outline marks a pinned planet: a judgement call",
                "Near-ties between galaxies settled by hand; the graph places the rest",
            ),
            FadeIn(leg),
            Create(rings),
            run_time=1.0,
        )
        self.wait(1.6)
        self.play(FadeOut(rings), run_time=0.8)

        hold_to(self, 18.2)
        cap = swap(
            self,
            cap,
            caption(
                "Dotted arrows: what the delegated items are about",
                "Personal OS 49 · Stack 18 · Life & game 16 · Expedition 8 · Salley 5 · Career 5",
            ),
            LaggedStart(*[Create(a) for a in arrows], lag_ratio=0.25),
            run_time=2.4,
        )

        hold_to(self, 24.4)
        swap(
            self,
            cap,
            caption(
                "Positions carry no metric meaning",
                "The ring is ordered so the most-tied galaxies are neighbours",
            ),
        )
        close_scene(self)


# ---- S8: tour -----------------------------------------------------------------------------------
class S8Tour(VideoScene):
    """The camera visits each galaxy in ring order, with its stats and one story; ends zoomed out."""

    def construct(self):
        d, hub = self.data, self.data["meta"]["hub_id"]
        discs, planets, sw, _ = universe_layers()
        arrows = zoom_arrows(self)
        leg = universe_legend()
        names = galaxy_labels(avoid=[leg])
        k0 = kicker("8 · Tour")
        cap = caption("A tour: one story per galaxy", "Filled moon = done, hollow = open")
        band = Rectangle(width=S.W + 0.2, height=S.H / 2 + S.caption_y + 0.62, stroke_width=0)
        band.set_fill(S.bg, TOUR_BAND).move_to([0, -S.H / 2, 0], aligned_edge=DOWN)
        self.play(*[FadeIn(m) for m in (discs, planets, sw, arrows, band, leg, names, k0, cap)], run_time=1.0)
        band.set_z_index(OVERLAY_Z - 1)
        VGroup(leg, k0, cap).set_z_index(OVERLAY_Z)
        self.pin(band, leg, k0, cap)
        keep_stroke(self, discs, planets, leg)
        keep_moon_stroke(self, sw)
        hold_to(self, 3.4)

        ids, gid = column("id"), column("galaxy_id")
        step = (self.target - 3.4 - 4.6) / len(d["tour"])
        head, extras = None, [names, arrows]  # extras: top-level mobjects to fade at the next move
        for i, stop in enumerate(d["tour"]):
            g, cam = galaxy(stop["galaxy_id"]), stop["camera"]
            kz = cam["width"] / S.W
            focus = np.zeros(sw.n, bool)
            focus[stop["focus_idx"]] = True
            dim = (gid == g["id"]) & ~focus & bool(len(stop["focus_idx"]))
            new_head, new_cap = stop_header(g), story_caption(stop["story"])
            self.pin(new_head.set_z_index(OVERLAY_Z), new_cap.set_z_index(OVERLAY_Z))
            self.add(new_head.set_opacity(0), new_cap.set_opacity(0))
            self.play(
                self.zoom(cam["centre"], cam["width"]),
                sw.morph(opacity=np.where(dim, TOUR_DIM, 1.0), suspend_mobject_updating=False),
                *[FadeOut(m) for m in extras if m in self.mobjects],
                handover(head, new_head),
                handover(cap, new_cap),
                run_time=1.6,
            )
            self.remove(*[m for m in (head, cap) if m is not None])
            head, cap, extras, show = new_head, new_cap, [], []

            avoid = [head, leg]
            if g["name"] == "Personal OS":  # its share of all human issues
                bar = self.share_bar(g["n"], d["meta"]["n"] - galaxy(hub)["n"], g["color"], head)
                avoid.append(bar)
                extras += list(bar)
                show.append(
                    LaggedStart(FadeIn(bar[0]), GrowFromEdge(bar[1], LEFT), FadeIn(bar[2]), lag_ratio=0.4)
                )
            pad = {pid: 0.05 * kz + 0.03 * kz for pid in stop["focus_planets"]}
            tags = planet_tags(
                tour_planets(g, stop), kz, view_bounds(cam["centre"], cam["width"]), avoid, pad
            )
            extras.append(tags)
            show.append(FadeIn(tags))
            for pid in stop["focus_planets"]:
                p = d["planets"][pid]
                ring = focus_ring(p["centre"], p["radius"], kz)
                extras.append(ring)
                show.append(Create(ring))
            if g["name"] == "Life & game":  # one sub-issue tree holds the galaxy together
                root = next(j for j in stop["focus_idx"] if ids[j] == "XYZ-712")
                kids = [j for j in stop["focus_idx"] if j != root]
                tree = VGroup(
                    *[
                        Line(sw.at(root)[0], sw.at(j)[0], stroke_color=S.ink2, stroke_width=1.4 * kz)
                        for j in kids
                    ],
                    Circle(radius=0.06 * kz, stroke_color=S.ink, stroke_width=2.5 * kz).move_to(
                        sw.at(root)[0]
                    ),
                )
                extras.append(tree)
                show.append(Create(tree, lag_ratio=0.05))
            self.play(*show, run_time=0.9)
            if g["id"] == hub:  # then where the delegated items go
                self.wait(0.7)
                flashes = [
                    ShowPassingFlash(
                        Line(p3(a["start"]), p3(a["end"]), stroke_color=S.ink, stroke_width=5 * kz),
                        time_width=0.5,
                    )
                    for a in d["arrows"]
                ]
                self.play(sw.morph(opacity=1.0), FadeIn(arrows), run_time=0.8)
                self.play(*flashes, run_time=1.2)
            hold_to(self, 3.4 + (i + 1) * step)

        last = caption(
            "8 galaxies around an automation hub",
            "Automation is a quarter of the universe, and almost all closed",
        )
        self.pin(last.set_z_index(OVERLAY_Z))
        self.add(last.set_opacity(0))
        self.play(
            self.zoom(ORIGIN, S.W),
            sw.morph(opacity=1.0, suspend_mobject_updating=False),
            *[FadeOut(m) for m in (*extras, head) if m in self.mobjects],
            handover(cap, last),
            run_time=2.0,
        )
        self.remove(cap)
        self.play(FadeIn(names), run_time=0.8)
        close_scene(self)

    def share_bar(self, part: int, whole: int, color: str, head) -> VGroup:
        """A share bar under the stop header (track, fill = part / whole, note), placed for the current
        view and not pinned, so GrowFromEdge can grow the fill."""
        w, y = 3.0, head.get_bottom()[1]  # head is pinned: read its default-camera bottom below
        frame = self.camera.frame
        k = frame.width / S.W
        y = (y - frame.get_center()[1]) / k - 0.32
        track = Rectangle(width=w, height=0.14, stroke_width=0, fill_color=S.baseline, fill_opacity=1)
        track.move_to(p3([S.kicker_xy[0] + w / 2, y]))
        fill = Rectangle(
            width=w * part / whole, height=0.14, stroke_width=0, fill_color=color, fill_opacity=1
        )
        fill.move_to(track, aligned_edge=LEFT)
        note = txt("share of human issues", S.small, S.ink2).next_to(
            track, DOWN, buff=0.12, aligned_edge=LEFT
        )
        bar = VGroup(track, fill, note)
        return bar.scale(k, about_point=ORIGIN).shift(frame.get_center())


# ---- S9: what each field adds -------------------------------------------------------------------
class S9Ablation(VideoScene):
    """One-field ablation: galaxy and planet ARI against the full build when one edge type is dropped."""

    def construct(self):
        rows = self.data["ablation"]
        k = kicker("9 · What each field adds")
        top, gap, thick, length = 1.85, 0.72, 0.24, 3.6
        panels = {"galaxy_ari": -3.3, "planet_ari": 2.0}  # x of each panel's baseline (ARI 0)
        ys = [top - i * gap for i in range(len(rows))]
        bottom = ys[-1] - 0.45

        heads, grid = VGroup(), VGroup()
        for key, x0 in panels.items():
            name = "Galaxies" if key == "galaxy_ari" else "Planets"
            heads.add(
                txt(name, S.subtitle, S.ink, "SEMIBOLD").move_to(p3([x0, top + 0.75]), aligned_edge=LEFT)
            )
            for v in (0.0, 0.5, 1.0):
                x = x0 + v * length
                col = S.baseline if v else S.ink2
                grid.add(Line(p3([x, top + 0.3]), p3([x, bottom]), stroke_color=col, stroke_width=1))
                grid.add(txt(f"{v:.1f}", S.small, S.ink2).move_to(p3([x, bottom - 0.25])))
        field = txt("field dropped", S.small, S.ink2).move_to(
            p3([panels["galaxy_ari"] - 0.3, top + 0.75]), aligned_edge=RIGHT
        )
        labels = VGroup(
            *[
                txt(r["label"], S.label, S.ink).move_to(
                    p3([panels["galaxy_ari"] - 0.3, y]), aligned_edge=RIGHT
                )
                for r, y in zip(rows, ys, strict=True)
            ]
        )
        bars, values = {}, {}
        for key, x0 in panels.items():
            bars[key] = VGroup()
            values[key] = VGroup()
            for r, y in zip(rows, ys, strict=True):
                w = r[key] * length
                bar = RoundedRectangle(
                    width=w,
                    height=thick,
                    corner_radius=0.05,
                    stroke_width=0,
                    fill_color=S.ink2,
                    fill_opacity=0.9,
                )
                bars[key].add(bar.move_to(p3([x0 + w / 2, y])))
                values[key].add(txt(verify(two_dp(r[key])), S.label, S.ink2).next_to(bar, RIGHT, buff=0.12))

        cap = caption(
            "Drop one field, rebuild everything, compare",
            "Agreement with the full build (ARI) on the 889 human issues: 1.0 = unchanged",
        )
        self.play(
            FadeIn(k), FadeIn(heads), FadeIn(grid), FadeIn(field), FadeIn(labels), FadeIn(cap), run_time=0.8
        )
        grow = [GrowFromEdge(b, LEFT) for key in panels for b in bars[key]]
        self.play(LaggedStart(*grow, lag_ratio=0.08), run_time=2.6)
        self.play(*[FadeIn(v) for key in panels for v in values[key]], run_time=0.6)

        def focus(i, text, sub, t):
            hold_to(self, t)
            anims = []
            for j in range(len(rows)):
                on = j == i
                for key in panels:
                    anims.append(
                        bars[key][j].animate.set_fill(S.ink if on else S.ink2, opacity=0.95 if on else 0.25)
                    )
                    anims.append(
                        values[key][j]
                        .animate.set_color(S.ink if on else S.ink2)
                        .set_opacity(1 if on else 0.35)
                    )
                anims.append(labels[j].animate.set_opacity(1 if on else 0.35))
            return swap(self, self._cap, caption(text, sub), *anims, run_time=0.8)

        self._cap = cap
        idx = {r["key"]: i for i, r in enumerate(rows)}
        self._cap = focus(
            idx["drop_project"],
            "Project is the backbone",
            "Without it galaxies change almost entirely: ARI 0.28 (planets 0.42)",
            5.4,
        )
        self._cap = focus(
            idx["drop_milestone"],
            "Milestones barely move galaxies but carve planets",
            "Galaxies 0.98, planets 0.58",
            10.0,
        )
        self._cap = focus(
            idx["drop_text_knn"],
            "Title text is the second force for galaxies",
            "Galaxies 0.82; without it planets merge, 46 become 39",
            14.6,
        )
        close_scene(self)


# ---- S10: close ---------------------------------------------------------------------------------
class S10Close(VideoScene):
    """Three takeaways over a faint universe (the dashed, pinned planets light up with the third), then
    the rerun command."""

    def construct(self):
        discs, planets, sw, _ = universe_layers()
        discs.set_stroke(opacity=0.15).set_fill(opacity=0.03)
        planets.set_stroke(opacity=0.25).set_fill(opacity=0.05)
        sw.set_state(opacity=0.16)
        self.play(FadeIn(discs), FadeIn(planets), FadeIn(sw), run_time=0.8)

        said = VGroup(*[txt(verify(s), S.subtitle, S.ink, "MEDIUM") for s in S.cfg["text"]["close"]])
        said.arrange(DOWN, buff=0.36).move_to(UP * 2.2)
        pinned = VGroup(
            *[
                DashedVMobject(
                    Circle(radius=p["radius"], stroke_color=S.ink, stroke_width=2.5), num_dashes=28
                ).move_to(p3(p["centre"]))
                for p in self.data["planets"]
                if p["pinned"]
            ]
        )
        for t, line in zip((1.0, 2.7, 4.4), said, strict=True):
            hold_to(self, t)
            extra = (
                [FadeIn(pinned), sw.morph(opacity=np.where(self.pinned_moons(), 0.9, 0.16))]
                if t == 4.4
                else []
            )
            self.play(FadeIn(line, shift=UP * 0.1), *extra, run_time=0.7)

        hold_to(self, 6.1)
        cmd = txt(verify(S.cfg["text"]["rerun"]), S.label, S.ink, mono=True)
        rerun = VGroup(txt("Rerun", S.small, S.ink2), cmd).arrange(DOWN, buff=0.16)
        box = backing(rerun, 0.22, 0.92).set_stroke(S.baseline, 1.5)
        VGroup(box, rerun).move_to([0, -2.4, 0])
        self.play(FadeIn(box), FadeIn(rerun), run_time=0.6)
        close_scene(self)

    def pinned_moons(self) -> np.ndarray:
        """Moons of the pinned planets (dashed outlines)."""
        ids = [p["id"] for p in self.data["planets"] if p["pinned"]]
        return np.isin(column("planet_id"), ids)
