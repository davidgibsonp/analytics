import numpy as np
import pandas as pd

from analytics.features import terrain


def ts(*values):
    return pd.Series(pd.to_datetime(list(values), utc=True))


def test_week_start_is_monday_and_keeps_nat():
    w = terrain.week_start(ts("2026-10-25 23:00", "2026-10-26 00:30", None))  # Sunday, Monday, NaT
    assert w.iloc[0] == pd.Timestamp("2026-10-19") and w.iloc[1] == pd.Timestamp("2026-10-26")
    assert pd.isna(w.iloc[2])
    # a timezone moves the week boundary: Monday 00:30 UTC is still Sunday in New York
    assert terrain.week_start(ts("2026-10-26 00:30"), "America/New_York").iloc[0] == pd.Timestamp(
        "2026-10-19"
    )


def test_week_axis_and_index():
    weeks = terrain.week_axis(ts("2026-10-07", "2026-10-30", None))
    assert list(weeks) == list(pd.to_datetime(["2026-10-05", "2026-10-12", "2026-10-19", "2026-10-26"]))
    idx = terrain.week_index(ts("2026-10-05", "2026-10-25", "2026-11-02", "2026-09-01", None), weeks)
    assert idx.tolist() == [0, 2, -1, -1, -1]
    lead = terrain.week_axis(ts("2026-10-07", "2026-10-30"), lead=2)  # two empty weeks in front
    assert lead[0] == pd.Timestamp("2026-09-21") and lead[2:].equals(weeks)


def test_path_order_follows_strongest_neighbours():
    # a chain a - b - c - d with strong links only between neighbours, given shuffled
    labels = ["c", "a", "d", "b"]
    strong = {("a", "b"), ("b", "c"), ("c", "d")}
    aff = pd.DataFrame(
        [
            [1.0 if x == y else (0.5 if (x, y) in strong or (y, x) in strong else 0.01) for y in labels]
            for x in labels
        ],
        index=labels,
        columns=labels,
    )
    order = terrain.path_order(aff)
    assert order in (["a", "b", "c", "d"], ["d", "c", "b", "a"])


def test_massif_order_puts_largest_in_the_middle():
    sizes = pd.Series({"s1": 1, "big": 9, "mid": 5, "small": 3, "s0": 2})
    order = terrain.massif_order(sizes)
    assert order[len(order) // 2] == "big"
    assert set(order[1:4]) == {"big", "mid", "small"}  # the larger ones stay near the centre


def test_lane_table_gaps_and_edges():
    L = terrain.lane_table([("A", ["a1", "a2"]), ("B", ["b1"])], gap=2, edge=1)
    assert L.lane.tolist() == list(range(7))
    assert L.kind.tolist() == ["gap", "ridge", "ridge", "gap", "gap", "ridge", "gap"]
    assert L.member.tolist()[1:3] == ["a1", "a2"] and L.block[5] == "B" and L.pos[5] == 0


def test_count_grid_skips_missing():
    g = terrain.count_grid(np.array([0, 0, 1, -1, 2]), np.array([1, 1, 0, 0, -1]), 3, 2)
    assert g.tolist() == [[0, 2], [1, 0], [0, 0]]


def test_smooth_blocks_keeps_mass_and_never_crosses_gaps():
    L = terrain.lane_table([("A", ["a1", "a2"]), ("B", ["b1", "b2", "b3"])], gap=1)
    grid = np.zeros((len(L), 12))
    grid[1, 6] = 10  # a2, at the edge next to the gap
    grid[3, 0] = 4  # b1, first week
    out = terrain.smooth_blocks(grid, L.block, time_sigma=1.5, lane_sigma=1.0)
    assert out[2].sum() == 0  # the gap lane stays flat
    assert np.isclose(out[:2].sum(), 10) and np.isclose(out[3:].sum(), 4)  # each block keeps its total
    assert out[0, 6] > 0 and out[4, 0] > 0  # smoothing spreads inside a block
    assert np.allclose(terrain.smooth_blocks(grid, L.block, 0, 0), grid)


def test_height_at_returns_nan_off_grid():
    h = np.arange(6.0).reshape(2, 3)
    assert np.allclose(
        terrain.height_at(h, [0, 1, -1, 1], [2, 0, 1, 3]), [2, 3, np.nan, np.nan], equal_nan=True
    )


def test_week_float_places_events_inside_their_week():
    weeks = pd.date_range("2026-10-05", periods=4, freq="7D")
    x = terrain.week_float(ts("2026-10-05 00:00", "2026-10-15 12:00", "2026-09-28 00:00", None), weeks)
    assert np.allclose(x, [0, 1.5, -1, np.nan], equal_nan=True)  # not clipped, unlike week_index


def test_month_starts_on_the_week_axis():
    weeks = pd.date_range("2026-09-21", periods=8, freq="7D")
    months, x = terrain.month_starts(weeks, "2026-11-15")
    assert list(months) == list(pd.to_datetime(["2026-10-01", "2026-11-01"]))
    assert np.allclose(x, [10 / 7, 41 / 7])


def test_time_groups_close_a_group_after_its_window():
    key = pd.Series(["a", "a", "b", "a", "a"])
    t = ts("2026-01-01", "2026-01-06", "2026-01-02", "2026-01-09", "2026-01-12")
    # a: Jan 1 and 6 within 7 days of the group's first; Jan 9 is 8 days after it, so a new group
    # that Jan 12 joins; b is its own group. Ids follow each group's first event.
    assert terrain.time_groups(key, t, days=7).tolist() == [0, 0, 1, 2, 2]


def test_upsample_passes_samples_and_never_overshoots():
    rng = np.random.default_rng(0)
    grid = rng.random((5, 8)) * np.array([0, 1, 1, 0, 1])[:, None] * 4  # lanes 0 and 3 are flat gaps
    fine = terrain.upsample(grid, np.arange(5), np.arange(8) + 0.5)
    assert np.allclose(fine, grid)  # week column k is sampled at its centre k + 0.5
    lanes, weeks = np.linspace(-1, 5, 61), np.linspace(0, 8, 81)
    fine = terrain.upsample(grid, lanes, weeks)
    assert fine.min() >= 0 and fine.max() <= grid.max() + 1e-12  # PCHIP: no new peak, no negative valley
    assert np.allclose(fine[np.isclose(lanes, 3)], 0)  # a flat gap lane stays flat
    capped = terrain.upsample(np.minimum(grid, 1.0), lanes, weeks)
    assert capped.max() <= 1.0 + 1e-12  # a capped strip keeps its cap
