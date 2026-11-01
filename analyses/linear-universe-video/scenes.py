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
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from manim import (
    DOWN,
    LEFT,
    RIGHT,
    UP,
    Animation,
    Annulus,
    Circle,
    DashedLine,
    DashedVMobject,
    FadeIn,
    FadeOut,
    Line,
    ManimColor,
    MovingCameraScene,
    Polygon,
    Text,
    Triangle,
    VGroup,
    VMobject,
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


# ==== scenes (one per [scenes].order entry; replace each placeholder with the real scene) ===========
class S1Cloud(VideoScene):
    def construct(self):
        self.placeholder("1 · Cloud", "1,143 Linear issues look like a loose cloud of titles")


class S2WordsToMap(VideoScene):
    def construct(self):
        self.placeholder("2 · Words → map", "Wording alone gives loose clusters")


class S3WordingNotEnough(VideoScene):
    def construct(self):
        self.placeholder("3 · Wording isn't enough", "Big projects spread across text clusters")


class S4Graph(VideoScene):
    def construct(self):
        self.placeholder("4 · The graph", "Many kinds of links between issues")


class S5HubCommunities(VideoScene):
    def construct(self):
        self.placeholder("5 · Hub and communities", "Automation set aside by rule; 8 galaxies by consensus")


class S6PlanetsNames(VideoScene):
    def construct(self):
        self.placeholder("6 · Planets and names", "Milestone, ritual label, project, then distinctive terms")


class S7Universe(VideoScene):
    def construct(self):
        self.placeholder("7 · The universe", "8 galaxies around an automation hub")


class S8Tour(VideoScene):
    def construct(self):
        self.placeholder("8 · Tour", "One story per galaxy")


class S9Ablation(VideoScene):
    def construct(self):
        self.placeholder("9 · What each field adds", "Drop one field, rebuild, compare")


class S10Close(VideoScene):
    def construct(self):
        self.placeholder("10 · Close", "Galaxies are project families")
