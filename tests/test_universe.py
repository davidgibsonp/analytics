from itertools import combinations

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from analytics.viz.pack import pack, spread
from analytics.viz.scatter import scatter
from analytics.viz.style import optimize_png
from analytics.viz.universe import universe


def test_pack_has_no_overlaps_and_is_centred():
    radii = np.random.default_rng(0).uniform(0.5, 4, 15)
    c = pack(radii, pad=0.5)
    for i, j in combinations(range(len(radii)), 2):
        assert np.linalg.norm(c[i] - c[j]) >= radii[i] + radii[j] + 0.5 - 1e-6
    lo, hi = (c - radii[:, None]).min(0), (c + radii[:, None]).max(0)
    assert np.allclose(lo + hi, 0)


def test_spread_pushes_neighbours_apart():
    phi = spread(np.array([0.0, 0.1, 0.2, 3.0]), np.full(4, 0.2))
    gaps = np.diff(np.r_[np.sort(phi), np.sort(phi)[0] + 2 * np.pi])
    assert gaps.min() >= 0.4 - 1e-9


def test_optimize_png_quantizes_only_when_over_cap(tmp_path):
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"x": rng.random(200), "y": rng.random(200), "team": list("abcd") * 50})
    path = scatter(df, "x", "y", "team", tmp_path / "s.png", title="anti-aliased markers and text")
    size = path.stat().st_size
    assert optimize_png(path, max_kb=size) * 1024 == size  # under the cap: untouched
    with pytest.warns(UserWarning, match="over max_kb"):  # still over a 1 KB cap: says so
        assert optimize_png(path, max_kb=1) * 1024 < size
    assert Image.open(path).mode == "P"


# Every option universe() reads, small and fast (the values are not linear-universe's).
PC = dict(width=8, height=6, dpi=60, max_kb=25, axes_box=[0.005, 0.03, 0.99, 0.845], frame_top=0.895)
PC |= dict(moon_spacing=1.0, moon_size=0.62, moon_noun="items", planet_rim=0.7, planet_pad=3.0)
PC |= dict(galaxy_pad=1.5, ring_gap=6.5, ring_step=0.5, hub_gap=1.2, stretch=1.05, title_slot=46)
PC |= dict(belt_inner=4.5, margin=2.0, title_room=4.0, glow_width=1.4, link_width=[0.8, 5.0])
PC |= dict(repair_rounds=1, repair_step=1.5, repair_min=10, label_min=6, hub_label_min=2)
PC |= dict(overlap_cost=100, disc_cost=10, galaxy_cost=3, cross_cost=1, owner_cost=20, owner_margin=0.5)
PC |= dict(outside_cost=2, title_pad_px=15, keepout_cost=10, arrow_end_px=30, arrow_dash=[1, 1.6])
PC |= dict(arrow_lw=1.2, arrow_head=12, legend_glow=0.5, label_size=8, label_gap=0.35, label_far=1.8)
PC |= dict(stats_wrap=100, stats_lift=0.4, stats_slide=2.5, stats_slides=5, also_abbrev={"planet ": "p"})
PC |= dict(galaxy_font=12.5, stats_font=8.5, title="{galaxies} galaxies, {planets} planets, {moons} moons")
PC |= dict(subtitle="subtitle", footnote="footnote")
LEGEND = {k: k for k in ("done", "open", "dropped", "glow", "pinned", "links", "arrow")}


@pytest.fixture
def tiny():
    """Galaxy 0 (two planets) and hub galaxy 1 (core, belt, an orbit planet pointing at galaxy 0
    and one pointing nowhere), plus one moon that is not drawn."""
    planets = pd.DataFrame(
        {
            "galaxy_id": [0, 0, 1, 1, 1, 1],
            "n": [12, 8, 5, 20, 6, 3],
            "role": ["", "", "core", "belt", "orbit", "orbit"],
            "target": [-1, -1, -1, -1, 0, -1],
            "pinned": [True, False, False, False, False, False],
            "glow": [False, True, False, False, False, False],
        }
    )
    moons = planets.loc[planets.index.repeat(planets.n), ["galaxy_id"]].reset_index(names="planet_id")
    moons.loc[len(moons)] = [-1, -1]
    moons["state"] = np.resize(["done", "open", "dropped"], len(moons))
    moons["createdAt"] = pd.date_range("2026-01-01", periods=len(moons), freq="D")
    planets["open"] = (
        moons[moons.planet_id >= 0].groupby("planet_id").state.apply(lambda s: (s == "open").sum())
    )
    return moons, planets


def draw(moons, planets, path, hub=1):
    names = {g: f"galaxy {g}" for g in set(planets.galaxy_id) | {hub}}
    return universe(
        moons,
        planets,
        hub=hub,
        galaxy_names=names,
        planet_names={p: f"planet {p}" for p in planets.index},
        details=dict.fromkeys(names, "alpha, beta"),
        affinity=pd.DataFrame([[1.0]]),
        links=[(0, hub, 4)],
        legend=LEGEND,
        pc=PC,
        path=path,
    )


def test_universe_chart_smoke(tiny, tmp_path):
    moons, planets = tiny
    stats = draw(moons, planets, tmp_path / "u.png")
    assert stats["kb"] <= PC["max_kb"]
    assert Image.open(tmp_path / "u.png").mode == "P"  # over max_kb as drawn, so it was quantized
    drawn = stats["xy"][moons.planet_id >= 0]
    assert np.isfinite(drawn).all() and len(np.unique(drawn, axis=0)) == len(drawn)


def test_universe_refuses_more_galaxies_than_colors(tiny, tmp_path):
    moons, planets = tiny
    planets = planets.loc[np.zeros(9, int)].reset_index(drop=True).assign(galaxy_id=range(9))
    with pytest.raises(ValueError, match="at most 8 galaxies"):
        draw(moons, planets, tmp_path / "u.png", hub=8)
