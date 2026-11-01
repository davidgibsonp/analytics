import numpy as np
import pandas as pd
import pytest
from PIL import Image

from analytics.features import terrain
from analytics.viz.cairns import cairn_chart
from analytics.viz.glyphs import cairn_stack, stone_count
from analytics.viz.pack import crosses, cuts, dodge, nearest_on_box, overlap, segment_in_box
from analytics.viz.relief import relief, relief_html
from analytics.viz.topo import topo_map


def test_dodge_spaces_neighbours_and_centres_blocks():
    x = np.array([5.0, 5.0, 5.0, 9.0])
    pos = dodge(x, np.full(4, 0.5), gap=0.2)
    assert np.allclose(np.sort(pos[:3]), [3.8, 5.0, 6.2])  # a block of three, centred on 5
    assert pos[3] == 9.0  # far enough already: untouched
    assert np.allclose(dodge(np.array([1.0, 3.0]), np.array([0.5, 0.5])), [1, 3])


def test_segment_and_box_geometry():
    assert crosses((0, 0), (2, 2), (0, 2), (2, 0))
    assert not crosses((0, 0), (1, 1), (1, 1), (2, 0))  # a shared end point is not a crossing
    assert cuts((1, 1, 2, 2), (0, 1.5), (3, 1.5)) and not cuts((1, 1, 2, 2), (0, 0), (3, 0))
    assert overlap((0, 0, 2, 2), (1, 1, 3, 3)) == 1 and overlap((0, 0, 1, 1), (2, 2, 3, 3)) == 0
    assert np.allclose(nearest_on_box((5, 0.5), (0, 0, 1, 1)), [1, 0.5])
    box = (0, 0, 10, 10)  # a diagonal segment near a box: inside its bounding box, yet not touching
    assert segment_in_box(box, (2, 2), (3, 3)) and segment_in_box(box, (-5, 5), (15, 5))
    assert not segment_in_box(box, (11, 0), (20, 9))


def test_stone_count_doubles_and_stacks_keep_the_stone_width():
    assert [stone_count(w) for w in (1, 2, 3, 4, 7, 8, 20)] == [1, 2, 2, 3, 3, 4, 5]
    (m1, s1, b1), (m5, s5, b5) = cairn_stack(1, 8), cairn_stack(5, 8)
    assert b1[1] - b1[0] == b5[1] - b5[0] == 8 and b5[3] > 3 * b1[3]  # same bottom stone, taller stack
    for m, s in ((m1, s1), (m5, s5)):  # matplotlib scales a marker to its largest vertex: stone stays 8 pt
        v = m.get_path().vertices
        assert np.isclose(np.ptp(v[:, 0]) * 0.5 / np.abs(v).max() * s, 8, rtol=0.02)


@pytest.fixture
def land():
    """Two blocks of two lanes (A: lanes 1-2, B: lanes 4-5, a flat strip), gaps and edges at 0."""
    rng = np.random.default_rng(1)
    L = terrain.lane_table([("A", ["a1", "a2"]), ("B", ["b1", "b2"])], gap=1, edge=1)
    count = rng.poisson(1.5, (len(L), 12)) * L.kind.eq("ridge").to_numpy()[:, None]
    height = terrain.smooth_blocks(count, L.block, 1.0, 0.5)
    height[4:6] = np.minimum(height[4:6], 1.0)
    weeks = pd.date_range("2026-01-05", periods=12, freq="7D")
    return L, count, height, weeks


TOPO = dict(
    width=8, height=5, dpi=60, max_kb=200, map_box=[0.2, 0.12, 0.62, 0.66], side_box=[0.86, 0.12, 0.1, 0.66]
)
TOPO |= dict(edge_pad=0.3, upsample=4, levels=[0.25, 0.5, 1, 2], ramp=[50, 200, 400, 600], line_step=600)
TOPO |= dict(line_width=0.35, line_alpha=0.45, status="warning", flag_pt=11, tent_pt=9, stone_pt=7)
TOPO |= dict(legend_weights=[1, 4, 16], crack_pt=5, crack_slide=48, band_alpha=[0.05, 0.02], note_x=0.3)
TOPO |= dict(
    side_label_min=10,
    side_ticks=[0, 10],
    font=dict(lane=7, block=8, label=7, tag=6, legend=7, title=12, subtitle=8, footnote=6),
)
TOPO_TEXT = dict(
    title="t", subtitle="s", footnote="shifted {shift_days}, cracks {crack_days}", edge="e", band_note="b"
)
TOPO_TEXT |= dict(side_title="side", side_note="{total} on {lanes}", scale="per week")
TOPO_TEXT["legend"] = {k: k for k in ("band", "flag", "tent", "trail", "cairn", "crack", "side", "edge")}


def test_topo_map_draws_every_glyph_and_spreads_same_day_ones(land, tmp_path):
    L, _, height, weeks = land
    marks = pd.DataFrame(
        [
            dict(kind="flag", lane=1, x=3.2, label="A M1–M2", group=0),
            dict(kind="flag", lane=1, x=3.3, label="A M1–M2", group=0),
            dict(kind="tent", lane=4, x=2.0, label="stalled · idle 40 d", trail=True),
            dict(kind="cairn", lane=2, x=5.5, weight=6, label="X-1"),
            dict(kind="cairn", lane=2, x=5.5, weight=1, label=""),
            dict(kind="crack", lane=5, x=7.0),
            dict(kind="crack", lane=1, x=3.3),  # on a flag: slides off it
            dict(kind="crack", lane=-1, x=7.0),  # no lane: not drawn
        ]
    ).fillna({"label": "", "group": -1, "trail": False})
    blocks = pd.DataFrame(
        dict(header_lane=[0, 3], header=["A", "B"], stats=["3 done", "2 done"], note=["", "flat"])
    )
    bands = pd.DataFrame(dict(x0=[1.0, 4.0], x1=[3.0, 6.0], tag=["C1", "C2"]))
    side = pd.DataFrame(dict(lane=[1, 4], value=[3, 12]))
    lanes = L[["lane"]].assign(label=L.member.fillna(""))
    stats = topo_map(
        height, weeks, lanes, blocks, marks, bands, side, 11.6, TOPO_TEXT, TOPO, tmp_path / "m.png"
    )
    assert stats["drawn"] == {"flag": 2, "tent": 1, "cairn": 2, "crack": 2, "side": 2}
    assert stats["shift_days"]["cairn"] > 0 and stats["shift_days"]["crack"] > 0  # same-day glyphs spread
    assert stats["clashes"] == [] and stats["outside"] == []
    assert stats["kb"] <= TOPO["max_kb"] and Image.open(tmp_path / "m.png").size == (480, 300)


RELIEF = dict(
    width=8, height=5.5, dpi=60, max_kb=300, elev=37, azim=-30, box=[2.6, 1.6, 0.42], label_edge="x1"
)
RELIEF |= dict(tick_edge="y0", target=[0.07, 0.08, 0.95, 0.7], upsample=2, html_upsample=2, html_scale=0.72)
RELIEF |= dict(ramp=[25, 100, 400, 700], ramp_at=[0, 0.1, 0.5, 1], contour_step=1.0, contour_shift=0.06)
RELIEF |= dict(shade=0.3, light_az=225, light_alt=40, status="warning", legend_width=300, valley_dash=1.0)
RELIEF |= dict(valley_alpha=0.55, max_shift_weeks=3, leader_min=16, label_gap=12)
RELIEF |= dict(font=dict(title=12, subtitle=8, legend=7, tick=7, block=8, callout=7))
GLYPHS = pd.DataFrame(
    dict(
        kind=["pole", "tent", "cairn", "cairn"],
        key=["p", "t", "c1", "c2"],
        lane=[1, 4, 2, 2],
        x=[3.3, 2.0, 5.5, 5.5],
        n=[2, 1, 3, 1],
    )
)


def relief_blocks():
    return pd.DataFrame(
        dict(first_lane=[1, 4], last_lane=[2, 5], mid_lane=[1.5, 4.5], label=["A", "B"], flat=[False, True])
    )


def test_relief_stands_glyphs_on_the_surface_and_keeps_them_apart(land, tmp_path):
    _, _, height, weeks = land
    text = dict(title="t", subtitle="s", scale="height", shift="slid up to {shift_days} days")
    text["legend"] = {k: [k, "more"] for k in ("pole", "cairn", "tent")}
    callouts = [{"keys": ["p"], "text": "two summits"}, {"keys": ["c1", "c2"], "text": "two cairns"}]
    callouts += [{"at": (2, 6.5), "text": "a peak"}]  # a point on the surface
    stats = relief(height, weeks, relief_blocks(), GLYPHS, callouts, text, RELIEF, tmp_path / "r.png")
    assert stats["drawn"] == {"cairn": 2, "pole": 1, "tent": 1, "pennants": 2}
    assert stats["shift_days"]["cairn"] > 0  # the second cairn on the same lane and day moved along time
    assert stats["clashes"] == []  # no block label on a bracket or on another label, no glyphs touching
    assert stats["kb"] <= RELIEF["max_kb"] and (tmp_path / "r.png").exists()


def test_relief_html_has_hover_for_every_glyph(land, tmp_path):
    pytest.importorskip("plotly")
    L, count, height, weeks = land
    glyphs = GLYPHS.assign(hover=[["summit one", "summit two"], "tent hover", "cairn hover", "cairn two"])
    glyphs.loc[len(glyphs)] = dict(kind="route", key="r", lane=1, x=4.0, n=1, hover="camp hover")
    names = L.member.fillna("").to_numpy()
    text = dict(title="t", subtitle="s", scale="per week", gap="valley")
    text["names"] = {k: f"{k} name" for k in ("pole", "cairn", "tent", "route")}
    kb = relief_html(height, count, weeks, names, relief_blocks(), glyphs, text, RELIEF, tmp_path / "r.html")
    html = (tmp_path / "r.html").read_text()
    assert kb > 0 and all(
        s in html for s in ("summit two", "tent hover", "cairn two", "camp hover", "valley")
    )


CAIRNS = dict(
    width=1400, dpi=100, max_kb=300, margin=40, unit=12, stone_h=7, gap=2, label_weight=5, row_min=52
)
CAIRNS |= dict(
    max_stones=2, row_pad=10, traffic_px=3.2, ratio_max=0.2, ratio_px=900, ratio_tick=0.05, bar_h=12, dot=9
)
CAIRNS |= dict(legend_weights=[1, 5], register_row_h=38, title_chars=40, bar="#c9c7c0", bar_flag="#52514e")
CAIRNS |= dict(band="#f1f0ec", shades=dict(parent="#3f3e3b", mention="#85837d", retro="#b5b3ac"))
CAIRNS |= dict(font=dict(title=14, sub=8, panel=10, head=7, name=8, small=7, id=6.5))
CAIRNS |= dict(columns=dict(name=40, cairn=330, bar=450, ratio=850, pct=1110, frac=1120, note=1200))
CAIRNS |= dict(
    register_columns=dict(
        rank=60,
        stone=150,
        id=240,
        title=320,
        project=700,
        done=800,
        kind=900,
        mentions=1080,
        subs=1160,
        retro=1230,
        weight=1330,
    )
)


def test_cairn_chart_has_no_text_collisions(tmp_path):
    stones = pd.DataFrame(dict(id=["X-1", "X-2", "X-3"], weight=[1, 6, 1], kind=["retro", "parent", "retro"]))
    rows = pd.DataFrame(
        dict(
            name=["alpha", "beta", "no project"],
            sub=["A", "B", "mostly C"],
            traffic=[40, 30, 20],
            cairns=[2, 0, 1],
            route=[True, True, False],
            fold=[False, False, False],
            flag=[False, True, False],
        )
    ).assign(ratio=lambda d: d.cairns / d.traffic)
    rows["stones"] = [stones, stones.iloc[:0], stones.iloc[:1]]
    register = pd.DataFrame(
        dict(
            id=["X-2", "X-1"],
            title=["a parent", "a retro"],
            project=["alpha", "alpha"],
            kind=["parent", "retro"],
            completedAt=pd.to_datetime(["2026-07-01", "2026-08-01"]),
            kinds=["parent", "retro"],
            inbound=[0, 0],
            children=[6, 0],
            bonus=[0, 1],
            weight=[6, 1],
        )
    )
    text = {
        k: k for k in ("title", "subtitle", "panel_a", "rule", "legend_size", "legend_kind", "legend_ids")
    }
    text |= {
        k: k for k in ("section", "none", "flag", "no_cairn", "busy", "barely", "clip", "note_a", "panel_b")
    }
    text |= dict(register_sub="sub", register_foot="foot", source="source", frac="{cairns} of {traffic}")
    text |= dict(legend_maps="maps", more="+{n}")
    text |= dict(flag_note="{what} in {traffic}", kinds_legend=dict(parent="parent", retro="retro"))
    text |= dict(
        kind_names=dict(parent="parent tree", retro="retro"),
        heads=dict(name=["project", ""], cairn=["c", "d"]),
    )
    text["register_heads"] = {k: k for k in CAIRNS["register_columns"]}
    stats = cairn_chart(rows, register, (30, 0.04), text, CAIRNS, tmp_path / "c.png")
    assert stats["collisions"] == [] and stats["outside"] == []
    assert stats["hidden"] == 1  # three stones over max_stones = 2: the lightest folds into a '+1' tag
    assert stats["kb"] <= CAIRNS["max_kb"] and (tmp_path / "c.png").exists()
