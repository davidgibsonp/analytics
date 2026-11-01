"""Terrain grids: counts of events per (lane, week), smoothed into a height field.

Lanes are rows (y), weeks are columns (x): every array is shaped (lanes, weeks), so it plots
directly with imshow / contour / a plotly surface. Lanes come in blocks (e.g. mountain ranges)
separated by empty gap lanes; smoothing runs inside each block only, so gaps stay at zero and
read as valleys.

    weeks = week_axis(pd.concat([df.createdAt, df.completedAt]))       # Monday of every week
    L = lane_table([("A", ["a1", "a2"]), ("B", ["b1"])], gap=1, edge=1)
    lane = df.member.map(L.dropna(subset="member").set_index("member").lane)
    count = count_grid(lane, week_index(df.completedAt, weeks), len(L), len(weeks))
    height = smooth_blocks(count, L.block, time_sigma=1.5, lane_sigma=0.6)
    z = height_at(height, marker_lane, marker_week)                    # to sit markers on the terrain
    x = week_float(df.completedAt, weeks)                              # fractional week, for charts
    fine = upsample(height, lanes_f, x_f)                              # smooth relief for drawing

Orders for the blocks and their members: path_order (neighbours by affinity) and massif_order
(largest member in the middle, a range profile). time_groups clusters events of one key in time
(e.g. milestones reached together); month_starts gives month ticks on the week axis.
"""

from __future__ import annotations

from itertools import permutations

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import gaussian_filter1d


def week_start(ts: pd.Series, tz: str = "UTC") -> pd.Series:
    """Monday 00:00 of each timestamp's week in `tz`, tz-naive; NaT stays NaT."""
    local = pd.to_datetime(ts, utc=True).dt.tz_convert(tz).dt.tz_localize(None)
    return local.dt.normalize() - pd.to_timedelta(local.dt.day_of_week, unit="D")


def week_axis(ts: pd.Series, tz: str = "UTC", lead: int = 0) -> pd.DatetimeIndex:
    """Every Monday from the week of the earliest timestamp to the week of the latest, with `lead`
    empty weeks in front (so smoothing near the start has empty ground to fall to, as it does in
    the gap lanes, instead of piling up against the edge)."""
    w = week_start(ts.dropna(), tz)
    return pd.date_range(w.min() - pd.Timedelta(weeks=lead), w.max(), freq="7D")


def week_index(ts: pd.Series, weeks: pd.DatetimeIndex, tz: str = "UTC") -> np.ndarray:
    """Column of each timestamp on a week_axis; -1 for NaT or outside the axis."""
    d = (week_start(ts, tz) - weeks[0]).dt.days.to_numpy(dtype=float, na_value=np.nan) // 7
    ok = np.isfinite(d) & (d >= 0) & (d < len(weeks))
    return np.where(ok, np.nan_to_num(d), -1).astype(int)


def week_float(ts: pd.Series, weeks: pd.DatetimeIndex, tz: str = "UTC") -> np.ndarray:
    """Fractional position of each timestamp on a week_axis: week k spans [k, k + 1), so Thursday
    noon of week k sits at k + 0.5. NaN for NaT; not clipped to the axis (unlike week_index)."""
    local = pd.to_datetime(pd.Series(ts), utc=True).dt.tz_convert(tz).dt.tz_localize(None)
    return ((local - weeks[0]) / pd.Timedelta(days=7)).to_numpy(dtype=float, na_value=np.nan)


def month_starts(weeks: pd.DatetimeIndex, end) -> tuple[pd.DatetimeIndex, np.ndarray]:
    """The first day of every month from weeks[0] to `end` (tz-naive), and its week_float position."""
    months = pd.date_range(weeks[0] + pd.offsets.MonthBegin(0), pd.Timestamp(end), freq="MS")
    return months, ((months - weeks[0]) / pd.Timedelta(days=7)).to_numpy(float)


def time_groups(key: pd.Series, ts: pd.Series, days: float) -> np.ndarray:
    """Group id per event: events of one key, in time order, join the key's latest group while they
    fall within `days` of that group's first event. Ids are 0.. in order of each group's first event."""
    t = pd.to_datetime(pd.Series(ts).reset_index(drop=True), utc=True)
    key = pd.Series(key).reset_index(drop=True)
    out, open_, n = np.full(len(t), -1), {}, 0
    for i in t.sort_values(kind="stable").index:
        g = open_.get(key[i])
        if g is None or t[i] - g[1] > pd.Timedelta(days=days):
            g = open_[key[i]] = (n, t[i])
            n += 1
        out[i] = g[0]
    return out


def path_order(aff: pd.DataFrame) -> list:
    """Labels of a symmetric affinity matrix in the linear order with the largest summed affinity
    of neighbours (exhaustive, so keep it to about 9 labels). Of the two directions, the one that
    starts with the label of larger total affinity comes first."""
    labels = list(aff.index)
    if len(labels) < 3:
        return labels
    A = aff.loc[labels, labels].to_numpy(dtype=float)
    best = max(
        (p for p in permutations(range(len(labels))) if p[0] < p[-1]),
        key=lambda p: sum(A[a, b] for a, b in zip(p, p[1:], strict=False)),
    )
    total = A.sum(axis=1) - np.diag(A)
    if total[best[-1]] > total[best[0]]:
        best = best[::-1]
    return [labels[i] for i in best]


def massif_order(sizes: pd.Series) -> list:
    """Labels centre-out like a mountain range profile: the largest in the middle, then the next
    ones alternating right and left (ties keep their input order)."""
    ranked = sizes.sort_values(ascending=False, kind="stable").index.tolist()
    left, right = [], []
    for i, label in enumerate(ranked):
        (right if i % 2 == 0 else left).append(label)
    return left[::-1] + right


def lane_table(blocks: list[tuple[str, list]], gap: int = 1, edge: int = 0) -> pd.DataFrame:
    """One row per lane, numbered 0.. in order: `edge` empty lanes, the members of each block,
    `gap` empty lanes between blocks, `edge` empty lanes. Columns: lane, kind ("ridge" or "gap"),
    block (None for gaps), member (None for gaps), pos (member's position inside its block)."""
    rows = [(None, None, -1)] * edge
    for i, (name, members) in enumerate(blocks):
        rows += [(None, None, -1)] * (gap if i else 0)
        rows += [(name, m, j) for j, m in enumerate(members)]
    rows += [(None, None, -1)] * edge
    L = pd.DataFrame(rows, columns=["block", "member", "pos"], dtype=object)
    L.insert(0, "lane", np.arange(len(L)))
    L.insert(1, "kind", np.where(L.member.isna(), "gap", "ridge"))
    L["pos"] = L.pos.astype(int)
    return L


def count_grid(lane: np.ndarray, week: np.ndarray, n_lanes: int, n_weeks: int) -> np.ndarray:
    """Event counts shaped (lanes, weeks); events with a negative or missing lane or week are skipped."""
    lane = np.nan_to_num(np.asarray(lane, dtype=float), nan=-1).astype(int)
    week = np.nan_to_num(np.asarray(week, dtype=float), nan=-1).astype(int)
    ok = (lane >= 0) & (lane < n_lanes) & (week >= 0) & (week < n_weeks)
    grid = np.zeros((n_lanes, n_weeks))
    np.add.at(grid, (lane[ok], week[ok]), 1)
    return grid


def smooth_blocks(
    grid: np.ndarray, block, time_sigma: float, lane_sigma: float, mode: str = "reflect"
) -> np.ndarray:
    """Gaussian smoothing of a (lanes, weeks) grid inside each run of consecutive lanes with the
    same block label; lanes whose block is missing (gaps) are set to zero. With mode "reflect" each
    block keeps its total (no mass leaks into gaps or off the time axis)."""
    block = pd.Series(block).reset_index(drop=True)
    out = np.zeros_like(grid, dtype=float)
    run = (block != block.shift()).cumsum()
    for _, idx in block.groupby(run).groups.items():
        rows = np.asarray(idx)
        if pd.isna(block[rows[0]]):
            continue
        sub = grid[rows].astype(float)
        if time_sigma > 0:
            sub = gaussian_filter1d(sub, time_sigma, axis=1, mode=mode)
        if lane_sigma > 0 and len(rows) > 1:
            sub = gaussian_filter1d(sub, lane_sigma, axis=0, mode=mode)
        out[rows] = sub
    return out


def height_at(height: np.ndarray, lane, week) -> np.ndarray:
    """Grid value at each (lane, week) pair, e.g. to set a marker on the terrain; NaN where either
    index is negative (no lane, or off the week axis)."""
    lane = np.nan_to_num(np.asarray(lane, dtype=float), nan=-1).astype(int)
    week = np.nan_to_num(np.asarray(week, dtype=float), nan=-1).astype(int)
    ok = (lane >= 0) & (lane < height.shape[0]) & (week >= 0) & (week < height.shape[1])
    return np.where(ok, height[np.where(ok, lane, 0), np.where(ok, week, 0)], np.nan)


def upsample(grid: np.ndarray, lanes, weeks) -> np.ndarray:
    """Grid values at fractional lane positions (rows) and week_float positions (columns; week column
    k is sampled at its centre k + 0.5), shaped (len(lanes), len(weeks)). Separable monotone cubic
    (PCHIP): it passes through the samples and never overshoots them, so it invents no peak, no
    negative valley and no bump on a capped strip; beyond the first and last samples it holds the edge."""
    n_lanes, n_weeks = grid.shape
    lanes = np.clip(np.asarray(lanes, float), 0, n_lanes - 1)
    weeks = np.clip(np.asarray(weeks, float), 0.5, n_weeks - 0.5)
    z = PchipInterpolator(np.arange(n_lanes), grid, axis=0)(lanes)
    return PchipInterpolator(np.arange(n_weeks) + 0.5, z, axis=1)(weeks)
