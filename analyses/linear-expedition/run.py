"""linear-expedition: the Linear workspace as an expedition map. Builds the shared data that the 2D
map, the 3D terrain and the cairn charts draw from.

Ranges = linear-universe's human galaxies, ridges = their planets (one lane each), Basecamp = the
automation hub (last, capped flat). Elevation = completed issues per (Monday week, lane), smoothed
inside each range. Camps = XYZ cycles, summits / stalled high camps = milestones, cached gear =
open issues, crevasses = canceled or duplicate issues, cairns = completed issues later work points
back to. Every threshold is in config.toml.

Outputs (outputs/): grid.npz (count, smooth, height shaped lanes x weeks; weeks; lane labels) and
grid.parquet (the same, long); lanes, ranges, issues, camps, milestones, summits, stalled, gear,
crevasses, cairns, index_project, index_range (.parquet); metrics.json; run_config.json;
terrain3d.html. Every dated table with a lane also has week (grid column, -1 = off the axis), x (the
fractional week the charts place it at) and z (height at lane, week). Figures (figures/): the 2D map,
the 3D relief and the cairn chart, drawn by charts.py (the one sibling module) from these tables.
"""

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from analytics.cli import sibling
from analytics.config import save_run_config
from analytics.features import terrain
from analytics.sources.linear import load_issues

DATES = {"created": "createdAt", "done": "completedAt", "dropped": "canceledAt"}
STATES = ["done", "open", "dropped"]


def load(cfg):
    """Raw issues joined to the universe's galaxy and planet; state, template flag; galaxy affinity."""
    df, U = load_issues(cfg["data"]["snapshot"]), Path(cfg["paths"]["universe"])
    a = pd.read_parquet(U / "assignments.parquet")
    if missing := set(df.id) - set(a.id):
        sys.exit(f"{len(missing)} issues have no universe planet: run `analytics run linear-universe` first")
    df = df.merge(a[["id", "galaxy", "planet"]], on="id", how="left")
    st = cfg["states"]
    df["state"] = np.select(
        [df.statusType.isin(st["done"]), df.statusType.isin(st["dropped"])], STATES[::2], "open"
    )
    df["template"] = df.title.str.match("|".join(f"(?:{p})" for p in cfg["cairns"]["templates"]))
    return df, pd.DataFrame(json.loads((U / "metrics.json").read_text())["galaxy_affinity"])


def lanes(df, aff, lc):
    """Range order by affinity path, ridges massif-ordered by completed issues, Basecamp last:
    its own lanes by size, then 'Delegated -> X' in the order of the ranges X."""
    ranges = terrain.path_order(aff.drop(index=lc["hub"], columns=lc["hub"], errors="ignore"))
    done = df[df.state == "done"].groupby("planet").size()
    blocks = [
        (g, terrain.massif_order(done.reindex(df.planet[df.galaxy == g].unique(), fill_value=0)))
        for g in ranges
    ]
    hub = df.planet[df.galaxy == lc["hub"]].value_counts().index
    rank = {f"{lc['delegated']}{g}": i for i, g in enumerate(ranges)}
    own = [p for p in hub if not p.startswith(lc["delegated"])]
    blocks.append((lc["basecamp"], own + sorted(set(hub) - set(own), key=lambda p: rank.get(p, len(rank)))))
    L = terrain.lane_table(blocks, lc["gap_lanes"], lc["edge_lanes"])
    L = L.rename(columns={"block": "range", "member": "planet"})
    L["galaxy"], L["basecamp"] = L.range.replace({lc["basecamp"]: lc["hub"]}), L.range.eq(lc["basecamp"])
    L["range_order"] = L.range.map({g: i for i, (g, _) in enumerate(blocks)}).astype("Int64")
    return L


def grid(df, L, weeks, tc):
    """Completed issues per (lane, week); smoothed inside each range; Basecamp capped flat."""
    done = df[df.state == "done"]
    count = terrain.count_grid(done.lane, done.week_done, len(L), len(weeks))
    smooth = terrain.smooth_blocks(count, L.range, tc["time_sigma"], tc["lane_sigma"], tc["mode"])
    return (
        count,
        smooth,
        np.where(L.basecamp.to_numpy()[:, None], np.minimum(smooth, tc["basecamp_cap"]), smooth),
    )


def lane_stats(L, df, height, snap, idle):
    """Issues by state, idle days, idle flag (open issues, no new issue on the lane for idle days: a
    lane rule, not the stalled-milestone rule) and peak per lane; one row per range with its lanes."""
    g = df[df.lane >= 0].groupby("lane")
    n = (
        g.state.value_counts()
        .unstack(fill_value=0)
        .reindex(columns=STATES, fill_value=0)
        .assign(issues=g.size())
    )
    L = L.join(n, on="lane").fillna(dict.fromkeys(n, 0)).astype(dict.fromkeys(n, int))
    L["idle_days"] = (snap - L.lane.map(g.createdAt.max())).dt.days.astype("Int64")
    L["idle"] = ((L.idle_days >= idle) & (L.open > 0)).fillna(False).astype(bool)
    L["peak"], L["peak_week"] = height.max(axis=1), np.where(L.kind == "ridge", height.argmax(axis=1), -1)
    agg = dict(first_lane=("lane", "min"), last_lane=("lane", "max"), planets=("lane", "size"))
    agg |= {c: (c, "sum") for c in [*STATES, "issues"]} | {"peak": ("peak", "max")}
    R = L[L.kind == "ridge"].groupby(["range_order", "range", "galaxy", "basecamp"]).agg(**agg).reset_index()
    return L, R.assign(mid_lane=(R.first_lane + R.last_lane) / 2)


def mode(s):
    s = s[s >= 0]
    return int(s.mode().iloc[0]) if len(s) else -1


def camps(df, cc):
    """One camp per cycle of cc.team: span = quantiles of its completed members' completedAt,
    date = their median, lane = the lane most of them sit on, retro = the 'Cycle N retro' issue
    (a done one first, then the earliest; canceled or duplicate retros never count)."""
    c = df[(df.team_key == cc["team"]) & df.cycle.notna()]
    d = c[c.state == "done"].groupby("cycle").completedAt
    C = pd.DataFrame(
        {"start": d.quantile(cc["span"][0]), "date": d.median(), "end": d.quantile(cc["span"][1])}
    )
    C = C.join(
        c.groupby(["cycle", "state"]).size().unstack(fill_value=0).reindex(columns=STATES, fill_value=0)
    )
    C["issues"], C["lane"] = c.groupby("cycle").size(), c[c.state == "done"].groupby("cycle").lane.agg(mode)
    r = df[df.state != "dropped"]
    hit = (
        r.title.str.extract(cc["retro"])
        .astype(float)
        .assign(id=r.id, rank=r.state.ne("done"), at=r.createdAt)
    )
    hit = pd.concat([hit.rename(columns={k: "cycle"}).dropna(subset=["cycle"]) for k in hit.columns[:2]])
    C["retro"] = hit.sort_values(["rank", "at"]).drop_duplicates("cycle").set_index("cycle").id
    return C.reset_index().astype({"cycle": int})


def milestones(df, snap, idle, days):
    """Per project milestone: summit (all done/dropped, one done; date = last completedAt), stalled
    (open issues, no new issue for idle days; date = last activity), climbing or abandoned. Summits of
    one project within `days` of their group's first share a group (one label, one pole)."""
    m = df[df.milestone.notna()].assign(last=df[list(DATES.values())].max(axis=1))
    g = m.groupby(["project", "milestone"], dropna=False)
    M = g.state.value_counts().unstack(fill_value=0).reindex(columns=STATES, fill_value=0)
    M = M.assign(issues=g.size(), last_created=g.createdAt.max(), last_done=g.completedAt.max())
    M["last_activity"], M["lane"], M["range"] = (
        g["last"].max(),
        g.lane.agg(mode),
        g.range.agg(lambda s: s.mode()[0]),
    )
    M["idle_days"] = (snap - M.last_created).dt.days
    cases = [(M.open == 0) & (M.done > 0), M.open == 0, M.idle_days >= idle]
    M["state"] = np.select(cases, ["summit", "abandoned", "stalled"], "climbing")
    M["date"] = M.last_done.where(M.state == "summit", M.last_activity)
    M, top = M.reset_index(), M.reset_index().state.eq("summit")
    M["group"] = -1
    M.loc[top, "group"] = terrain.time_groups(M.project[top], M.date[top], days)
    return M


def cairns(df, cc):
    """Completed non-template issues that later work points back to: targets of title id-mentions,
    parents with min_children sub-issues, Retro-labeled issues. weight = inbound mentions +
    sub-issues + retro_bonus; kind = the qualifying part that adds most (ties: kind_order). A sub-issue
    that names its own parent is already counted as a child: unless child_mentions, it is no mention."""
    created = df.set_index("id").createdAt
    pairs = [
        (i, r)
        for i, t in zip(df.id, df.title, strict=True)
        for r in sorted(set(re.findall(cc["mention"], t)))
    ]
    E = pd.DataFrame(pairs, columns=["src", "dst"])
    E = E[E.dst.isin(created.index) & (E.src != E.dst)]
    later = created[E.src].to_numpy() > created[E.dst].to_numpy()
    child = E.src.map(df.set_index("id").parentId).eq(E.dst).to_numpy()
    edges = {"edges": len(E), "sources": E.src.nunique(), "not_later": int((~later).sum())}
    edges["child_names_parent"] = int(child.sum())
    E = E[(later | (not cc["later_only"])) & (~child | cc["child_mentions"])]
    edges["counted"] = len(E)
    C = df[df.state == "done"].set_index("id")
    for col, s in {"mentioned_by": E.groupby("dst").src, "sub_issues": df.groupby("parentId").id}.items():
        C[col] = s.agg(list).reindex(C.index).map(lambda v: v if isinstance(v, list) else [])
    C["inbound"], C["children"] = C.mentioned_by.str.len(), C.sub_issues.str.len()
    C["retro"], order = C.labels.map(lambda ls: cc["retro_label"] in ls), cc["kind_order"]
    parts = pd.DataFrame({"mention": C.inbound, "parent": C.children, "retro": C.retro * cc["retro_bonus"]})[
        order
    ]
    ok = pd.DataFrame(
        {"mention": C.inbound > 0, "parent": C.children >= cc["min_children"], "retro": C.retro}
    )[order]
    C["weight"], C["kind"] = parts.sum(axis=1), parts.where(ok, -1).idxmax(axis=1)
    C["kinds"], C["refs"] = ok.apply(lambda r: "+".join(r.index[r]), axis=1), C.mentioned_by + C.sub_issues
    cols = [
        "id",
        "title",
        "kind",
        "kinds",
        "weight",
        "inbound",
        "children",
        "retro",
        "refs",
        "mentioned_by",
        "sub_issues",
    ]
    edges["template_cairns_left_out"] = int((ok.any(axis=1) & C.template).sum())
    K = C[ok.any(axis=1) & ~C.template].reset_index()
    return K[[*cols, "project", "range", "planet", "lane", "completedAt"]].sort_values(
        "weight", ascending=False
    ), edges


def index_table(done, cairn, key, ic):
    """Traffic (completed issues), cairns, marking ratio (cairns / traffic) per `key`. Thirds are
    taken over routes only: rows with min_traffic and not in not_routes; other rows are never flagged."""
    T = done.groupby(key).size().rename("traffic").to_frame()
    T = T.join(cairn.groupby(key).agg(cairns=("id", "size"), cairn_weight=("weight", "sum"))).fillna(0)
    T["ratio"], T["weight_ratio"] = T.cairns / T.traffic, T.cairn_weight / T.traffic
    T["route"] = (T.traffic >= ic["min_traffic"]) & ~T.index.isin(ic["not_routes"])
    pool = T[T.route]
    cuts = {
        "busy_min": pool.traffic.quantile(1 - 1 / ic["parts"]),
        "barely_max": pool.ratio.quantile(1 / ic["parts"]),
    }
    T["busy"] = T.route & (T.traffic >= cuts["busy_min"])
    T["barely_marked"] = T.route & (T.ratio <= cuts["barely_max"])
    T["unmarked_busy"] = T.busy & T.barely_marked
    T["on_busy_cut"] = T.route & np.isclose(T.traffic, cuts["busy_min"])  # a tie: borderline
    T["on_barely_cut"] = T.route & np.isclose(T.ratio, cuts["barely_max"])
    T = T.astype({"cairns": int, "cairn_weight": int}).sort_values("traffic", ascending=False).reset_index()
    return T, {k: float(v) for k, v in cuts.items()}


def main(cfg: dict) -> None:
    out, tz, ic = Path(cfg["paths"]["outputs"]), cfg["data"]["timezone"], cfg["index"]
    out.mkdir(parents=True, exist_ok=True)
    df, aff = load(cfg)
    snap, idle = df[list(DATES.values())].max().max(), cfg["milestones"]["idle_days"]  # newest event
    L = lanes(df, aff, cfg["lanes"])
    ridge = L[L.kind == "ridge"].set_index("planet")
    df["lane"], df["range"] = df.planet.map(ridge.lane).fillna(-1).astype(int), df.planet.map(ridge.range)
    on = pd.concat([df[df.lane >= 0][c] for c in DATES.values()])
    weeks = terrain.week_axis(on, tz, cfg["terrain"]["lead_weeks"])
    for k, c in DATES.items():
        df[f"week_{k}"] = terrain.week_index(df[c], weeks, tz)
    count, smooth, height = grid(df, L, weeks, cfg["terrain"])
    df["z_done"] = terrain.height_at(height, df.lane, df.week_done)
    L, R = lane_stats(L, df, height, snap, idle)

    def place(T, date_col):  # week = grid column of date_col, x = fractional week, z = height there
        T["week"] = terrain.week_index(T[date_col], weeks, tz)
        T["x"] = terrain.week_float(T[date_col], weeks, tz)
        return T.assign(z=terrain.height_at(height, T.lane, T.week))

    M = place(milestones(df, snap, idle, cfg["milestones"]["summit_days"]), "date")
    M["x_created"] = terrain.week_float(M.last_created, weeks, tz)  # tents stand at the last new issue
    gear = L[(L.kind == "ridge") & (L.open > 0)][["lane", "range", "planet", "open"]]
    gear = gear.join(
        df[df.state == "open"].groupby(["lane", "statusType"]).size().unstack(fill_value=0), on="lane"
    )
    gear = gear.assign(week=len(weeks) - 1, z=height[gear.lane, -1])
    crev = df[df.state == "dropped"][
        ["id", "title", "statusType", "project", "range", "planet", "lane", "canceledAt"]
    ]
    K, edges = cairns(df, cfg["cairns"])
    left_out = edges.pop("template_cairns_left_out")
    K = place(K, "completedAt")
    traffic = df[(df.state == "done") & (ic["templates_in_traffic"] | ~df.template)]
    traffic, K["project"] = traffic.fillna({"project": ic["no_project"]}), K.project.fillna(ic["no_project"])
    IP, cut_p = index_table(traffic, K, "project", ic)
    IP.insert(1, "range", IP.project.map(traffic.groupby("project").range.agg(lambda s: s.mode()[0])))
    IR, cut_r = index_table(traffic, K, "range", ic)
    IR = IR.merge(R[["range", "range_order"]], on="range")
    C = place(camps(df, cfg["camps"]), "date")
    C["x_start"], C["x_end"] = (terrain.week_float(C[c], weeks, tz) for c in ("start", "end"))

    cols = ["id", "title", "team_key", "project", "milestone", "statusType", "state", "template", "galaxy"]
    cols += ["range", "planet", "lane", "cycle", *DATES.values(), *[f"week_{k}" for k in DATES], "z_done"]
    tables = {"lanes": L, "ranges": R, "issues": df[cols]}
    tables |= {"camps": C, "milestones": M, "gear": gear}
    tables |= {"summits": M[M.state == "summit"], "stalled": M[M.state == "stalled"], "cairns": K}
    tables |= {"crevasses": place(crev, "canceledAt"), "index_project": IP, "index_range": IR}
    for name, T in tables.items():
        T.to_parquet(out / f"{name}.parquet", index=False)
    labels = {c: L[c].fillna("").to_numpy(dtype=str) for c in ["range", "planet", "kind"]}
    days = weeks.to_numpy().astype("datetime64[D]")
    np.savez_compressed(out / "grid.npz", count=count, smooth=smooth, height=height, weeks=days, **labels)
    lane, week = np.divmod(np.arange(count.size), len(weeks))
    long = dict(lane=lane, week=week, week_start=weeks[week], count=count.ravel(), smooth=smooth.ravel())
    pd.DataFrame(long | {"height": height.ravel()}).to_parquet(out / "grid.parquet", index=False)

    metrics = {
        "snapshot": str(snap),
        "weeks": [str(weeks[0].date()), str(weeks[-1].date()), len(weeks)],
        "lanes": L.kind.value_counts().to_dict(),
        "range_order": R.range.tolist(),
        "issues_without_lane": df.id[df.lane < 0].tolist(),
        "completed_on_grid": int(count.sum()),
        "camps": len(tables["camps"]),
        "milestones": M.state.value_counts().to_dict(),
        "stalled": (M.project + " · " + M.milestone)[M.state == "stalled"].tolist(),
        "idle_lanes": L.planet[L.idle].tolist(),  # lane rule (no new issue on the lane), not a milestone
        "gear_open": int(gear.open.sum()),
        "crevasses": len(crev),
        "cairns": K.kind.value_counts().to_dict() | {"total": len(K), "weight": int(K.weight.sum())},
        "mentions": edges,
        "templates": {  # left out of cairns and traffic: the debrief template
            "issues": int(df.template.sum()),
            "done": int((df.template & (df.state == "done")).sum()),
            "cairns_left_out": left_out,
        },
        "unmarked_busy_projects": IP.project[IP.unmarked_busy].tolist(),
        "unmarked_busy_ranges": IR.range[IR.unmarked_busy].tolist(),
        "index_cuts": {"project": cut_p, "range": cut_r},
    }
    if cfg["figures"]["draw"]:
        terrain_ = dict(count=count, height=height, weeks=weeks, snapshot=snap, cuts=cut_p, cuts_range=cut_r)
        metrics["figures"] = sibling(__file__, "charts").draw(cfg, tables, terrain_)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1, default=str, ensure_ascii=False))
    save_run_config(cfg, out / "run_config.json", stats=metrics)
    print(json.dumps(metrics, default=str, ensure_ascii=False))
