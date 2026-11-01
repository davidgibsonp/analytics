"""linear-expedition figures: which table becomes which glyph, and what everything is called.

The chart types are generic (analytics.viz.topo, viz.relief, viz.cairns). This module holds the
Linear-specific choices: milestone short names ('autoresearch M1–M4'), the callouts on the relief,
the folded rows of the cairn index, and the titles, legends and footnotes. The text templates are in
config.toml and are filled with the run's own thresholds and counts, so `--set` reruns stay truthful.

    stats = draw(cfg, tables, grid)   # run.py calls this after the outputs are written
"""

import importlib.util
import math
import re
import textwrap
from pathlib import Path

import pandas as pd

from analytics.features import terrain
from analytics.viz.cairns import cairn_chart, kind_label, pct
from analytics.viz.glyphs import stone_count
from analytics.viz.relief import relief, relief_html
from analytics.viz.topo import topo_map

CODE = re.compile(r"^(v\d[\w.]*|[A-Z]{1,3}\d+)$")  # a bare milestone code: v0.3.0, M1, FT2
PART = {2: "half", 3: "third", 4: "quarter", 5: "fifth"}


def head(milestone: str) -> str:
    """The milestone name before ' — ' or ' (': 'M1 — Remote orchestration' -> 'M1'."""
    return re.split(r" — | \(", milestone)[0].strip()


def short_name(project: str, milestone: str) -> str:
    """The milestone head; a bare code or a single word gets its project in front ('envoy v0.3.0')."""
    h = head(milestone)
    return f"{project} {h}" if CODE.match(h) or " " not in h else h


def compress(heads: list[str], sep: str) -> str:
    """['M1', 'M2', 'M3', 'M5'] -> 'M1–M3<sep>M5': consecutive numbers of one prefix become a run."""
    parts = sorted((m[1], int(m[2])) for h in heads if (m := re.match(r"^(\D+)(\d+)$", h)))
    if len(parts) < len(heads):
        return sep.join(heads)
    runs = []
    for p, n in parts:
        if runs and runs[-1][-1] == (p, n - 1):
            runs[-1].append((p, n))
        else:
            runs.append([(p, n)])
    return sep.join(f"{r[0][0]}{r[0][1]}" + (f"–{r[-1][0]}{r[-1][1]}" if len(r) > 1 else "") for r in runs)


def group_label(S: pd.DataFrame) -> str:
    """One label per summit group: 'autoresearch M1–M4', 'autoresearch FT1–FT3 · M5', 'game M0'."""
    if len(S) == 1:
        return short_name(S.project.iloc[0], S.milestone.iloc[0])
    return f"{S.project.iloc[0]} {compress([head(m) for m in S.milestone], ' · ')}"


def plural(n: int, word: str) -> str:
    return f"{n} {word}{'' if n == 1 else 's'}"


def facts(cfg, T, G) -> dict:
    """Values the text templates use: the run's thresholds and headline counts."""
    snap, weeks = G["snapshot"].tz_convert("UTC").tz_localize(None), G["weeks"]
    tc, cc, kc = cfg["terrain"], cfg["camps"], cfg["cairns"]
    R, iss = T["ranges"], T["issues"]
    done = iss.completedAt[(iss.state == "done") & (iss.lane >= 0)]
    first = done.min().tz_convert("UTC").tz_localize(None)
    return dict(
        snapshot=snap,
        start=weeks[0],  # the grid's first Monday (lead_weeks before the first event)
        first_done=first,
        months=round((snap - first).days / 30.44),
        done=int(G["count"].sum()),
        lanes=int((T["lanes"].kind == "ridge").sum()),
        ranges=int((~R.basecamp).sum()),
        time_sigma=tc["time_sigma"],
        lane_sigma=tc["lane_sigma"],
        cap=tc["basecamp_cap"],
        team=cc["team"],
        span0=f"{100 * cc['span'][0]:.0f}th",
        span1=f"{100 * cc['span'][1]:.0f}th",
        idle=cfg["milestones"]["idle_days"],
        summit_days=cfg["milestones"]["summit_days"],
        bonus=kc["retro_bonus"],
        label=kc["retro_label"],
        min_children=kc["min_children"],
        ids=cfg["figures"]["cairn_ids"],
    )


def draw(cfg, T, G) -> dict:
    fc = cfg["figures"]
    figs, out = Path(cfg["paths"]["figures"]), Path(cfg["paths"]["outputs"])
    figs.mkdir(parents=True, exist_ok=True)
    f = facts(cfg, T, G)
    stats = {
        "map2d": map2d(cfg, T, G, f, figs / fc["map2d"]),
        "terrain3d": terrain3d(cfg, T, G, f, figs / fc["terrain3d"]),
    }
    if fc["html"] and importlib.util.find_spec("plotly"):
        stats["html_kb"] = round(terrain3d_html(cfg, T, G, f, out / fc["html_name"]))
    elif fc["html"]:
        print(f"skipped {fc['html_name']}: plotly is missing (uv run --extra viz ...)")
    stats["cairns"] = cairns(cfg, T, G, f, figs / fc["cairns"])
    return stats


def map2d(cfg, T, G, f, path) -> dict:
    """Plan view: summit flags (one label per group), stalled tents at their last new issue with an idle
    trail, cairns sized by weight (ids on the heavy ones), crevasses, camp bands, gear bars."""
    pc = cfg["map2d"] | {"max_kb": cfg["figures"]["max_kb"]}
    S, ST, K, X, L, R = (T[k] for k in ("summits", "stalled", "cairns", "crevasses", "lanes", "ranges"))
    labels = {g: group_label(sub) for g, sub in S.groupby("group")}
    tent = [
        pc["stalled_label"].format(name=short_name(r.project, r.milestone), idle=r.idle_days, open=r.open)
        for r in ST.itertuples()
    ]
    marks = pd.concat(
        [
            pd.DataFrame(dict(kind="flag", lane=S.lane, x=S.x, label=S.group.map(labels), group=S.group)),
            pd.DataFrame(dict(kind="tent", lane=ST.lane, x=ST.x_created, label=tent, trail=True)),
            pd.DataFrame(
                dict(
                    kind="cairn",
                    lane=K.lane,
                    x=K.x,
                    weight=K.weight,
                    label=K.id.where(K.weight >= f["ids"], ""),
                )
            ),
            pd.DataFrame(dict(kind="crack", lane=X.lane, x=X.x)),
        ],
        ignore_index=True,
    ).fillna({"label": "", "group": -1, "trail": False})
    blocks = pd.DataFrame(
        dict(
            header_lane=R.first_lane - 1,
            header=R.range,
            stats=[pc["block_stats"].format(done=r.done, open=r.open) for r in R.itertuples()],
            note=[pc["basecamp_note"].format(**f) if b else "" for b in R.basecamp],
        )
    )
    C = T["camps"]
    bands = pd.DataFrame(
        dict(x0=C.x_start, x1=C.x_end, tag=[pc["band_tag"].format(cycle=c) for c in C.cycle])
    )
    gear = T["gear"]
    weights = " · ".join(f"{w:g}" for w in pc["legend_weights"])
    text = {
        k: pc[k].format(**f, shift_days="{shift_days}", crack_days="{crack_days}")
        for k in ("title", "subtitle", "footnote", "edge")
    }
    text |= {k: pc[k].format(**f) for k in ("band_note", "side_title", "scale")} | {
        "side_note": pc["side_note"],
        "legend": {k: v.format(**f, weights=weights) for k, v in pc["legend"].items()},
    }
    edge = float(terrain.week_float(pd.Series([G["snapshot"]]), G["weeks"], cfg["data"]["timezone"])[0])
    return topo_map(
        G["height"],
        G["weeks"],
        L[["lane"]].assign(label=L.planet.where(L.kind == "ridge", "")),
        blocks,
        marks,
        bands,
        pd.DataFrame(dict(lane=gear.lane, value=gear.open)),
        edge,
        text,
        pc,
        path,
    )


def poles(S: pd.DataFrame) -> pd.DataFrame:
    """One pole per summit group: its mode lane, mean position, one pennant per summit."""
    return (
        S.sort_values("date")
        .groupby("group")
        .agg(
            lane=("lane", lambda s: int(s.mode().iloc[0])),
            x=("x", "mean"),
            n=("lane", "size"),
            range=("range", "first"),
        )
        .reset_index()
        .assign(kind="pole", key=lambda d: "pole " + d.group.astype(str))
    )


def glyph_table(T) -> pd.DataFrame:
    """Poles (summit groups), tents (stalled camps at their last new issue), cairns (stones by weight)."""
    ST, K = T["stalled"], T["cairns"]
    tents = pd.DataFrame(
        dict(kind="tent", key=ST.project + " · " + ST.milestone, lane=ST.lane, x=ST.x_created, n=1)
    )
    stones = pd.DataFrame(dict(kind="cairn", key=K.id, lane=K.lane, x=K.x, n=K.weight.map(stone_count)))
    return pd.concat([poles(T["summits"]), tents, stones], ignore_index=True)


def callouts(cfg, T, G) -> list[dict]:
    """The heaviest cairn, each range's summit pushes (poles with push_min pennants), the pole standing
    highest outside them, the longest-stalled camps, then the tallest peak of each range with no callout
    yet (the peak_callouts tallest of them). Everything else is in the html hover."""
    pc, S, K = (
        cfg["terrain3d"],
        T["summits"],
        T["cairns"].sort_values("weight", ascending=False, kind="stable"),
    )
    top = K.iloc[0]
    name = pc["names"].get(top.id) or textwrap.shorten(top.title, pc["name_chars"], placeholder="…")
    out = [
        {
            "range": top.range,
            "keys": [top.id],
            "text": pc["callout_cairn"].format(
                id=top.id,
                name=name,
                children=plural(top.children, "sub-issue"),
                mentions=plural(top.inbound, "mention"),
            ),
        }
    ]
    P = poles(S)
    push = P[P.n >= pc["push_min"]]
    for rng, sub in push.groupby("range"):
        items = [
            pc["push_item"].format(names=compress([head(m) for m in g.milestone], ", "), date=g.date.min())
            for _, g in S[S.group.isin(sub.group)].sort_values("date").groupby("group", sort=False)
        ]
        text = pc["callout_push"].format(
            range=rng, summits=int(sub.n.sum()), pushes=len(sub), list=" · ".join(items)
        )
        out.append({"range": rng, "keys": list(sub.key), "text": text})
    rest = P[P.n < pc["push_min"]].assign(z=lambda d: terrain.height_at(G["height"], d.lane, d.x.astype(int)))
    if len(rest):
        p = rest.loc[rest.z.idxmax()]
        s = S[S.group == p.group].iloc[0]
        rest_name = s.milestone.split(" — ", 1)[1] if " — " in s.milestone else ""
        name = short_name(s.project, s.milestone) + (f": {rest_name}" if rest_name else "")
        out.append({"range": p.range, "keys": [p.key], "text": pc["callout_summit"].format(name=name)})
    for r in T["stalled"].sort_values("idle_days", ascending=False).head(pc["stalled_callouts"]).itertuples():
        text = pc["callout_stalled"].format(project=r.project, milestone=r.milestone, idle=r.idle_days)
        out.append({"range": r.range, "keys": [f"{r.project} · {r.milestone}"], "text": text})
    L = T["lanes"]
    peaks = L[(L.kind == "ridge") & ~L.basecamp & ~L.range.isin([c["range"] for c in out])]
    peaks = peaks.sort_values("peak", ascending=False).drop_duplicates("range").head(pc["peak_callouts"])
    for r in peaks.itertuples():  # a point on the surface: the peak at its week's centre
        date = G["weeks"][r.peak_week] + pd.Timedelta(days=3.5)
        text = pc["callout_peak"].format(range=r.range, planet=r.planet, peak=r.peak, date=date)
        out.append({"range": r.range, "at": (r.lane, r.peak_week + 0.5), "text": text})
    return out


def relief_blocks(cfg, R) -> pd.DataFrame:
    label = R.range.where(~R.basecamp, cfg["terrain3d"]["basecamp"])
    return R[["first_lane", "last_lane", "mid_lane"]].assign(label=label, flat=R.basecamp.astype(bool))


def terrain3d(cfg, T, G, f, path) -> dict:
    pc = cfg["terrain3d"] | {"max_kb": cfg["figures"]["max_kb"]}
    text = {k: pc[k].format(**f) for k in ("title", "subtitle", "scale")} | {"shift": pc["shift"]}
    text["legend"] = {k: [line.format(**f) for line in v] for k, v in pc["legend"].items()}
    return relief(
        G["height"],
        G["weeks"],
        relief_blocks(cfg, T["ranges"]),
        glyph_table(T),
        callouts(cfg, T, G),
        text,
        pc,
        path,
    )


def terrain3d_html(cfg, T, G, f, path) -> float:
    """The same terrain in plotly, with every summit, cairn, stalled camp and camp in the hover."""
    pc, S, ST, K, C, L = cfg["terrain3d"], *(T[k] for k in ("summits", "stalled", "cairns", "camps", "lanes"))
    P = poles(S)
    hv = {k: pc[f"hover_{k}"] for k in ("summit", "cairn", "tent", "camp")}
    P["hover"] = [
        [hv["summit"].format(**r._asdict()) for r in S[S.group == g].sort_values("date").itertuples()]
        for g in P.group
    ]
    G3 = glyph_table(T).set_index("key")
    names = cfg["cairn_chart"]["kind_names"]
    G3.loc[K.id, "hover"] = [
        hv["cairn"].format(
            **r._asdict() | {"kinds": kind_label(r.kind, r.kinds, names)},
            mentions=plural(r.inbound, "mention"),
            subs=plural(r.children, "sub-issue"),
            retro_part=", retro" if r.retro else "",
        )
        for r in K.itertuples()
    ]
    G3.loc[ST.project + " · " + ST.milestone, "hover"] = [
        hv["tent"].format(**r._asdict()) for r in ST.itertuples()
    ]
    camp = [
        hv["camp"].format(
            **r._asdict() | {"retro": r.retro if isinstance(r.retro, str) else "none yet"}, team=f["team"]
        )
        for r in C.itertuples()
    ]
    route = pd.DataFrame(dict(kind="route", lane=C.lane, x=C.x, n=1, hover=camp))
    glyphs = pd.concat([P, G3[G3.kind != "pole"].reset_index(), route], ignore_index=True)
    names = (L.range.fillna("") + " · " + L.planet.fillna("")).where(L.kind == "ridge", "").to_numpy()
    text = {k: pc[k].format(**f) for k in ("title", "gap")} | {
        "subtitle": pc["html_subtitle"].format(**f),
        "scale": pc["html_scale_label"],
        "names": {k: v.format(**f) for k, v in pc["names_html"].items()},
    }
    blocks = relief_blocks(cfg, T["ranges"]).assign(label=lambda d: d.label.where(~d.flat, "Basecamp"))
    return relief_html(G["height"], G["count"], G["weeks"], names, blocks, glyphs, text, pc, path)


def index_rows(IP, K, ic, cc) -> pd.DataFrame:
    """Rows of the cairn index: the show_routes busiest routes, one fold for the smaller routes, then the
    rows that are never routes (no project, then one fold for projects under min_traffic)."""
    rows = []
    routes = IP[IP.route].sort_values("traffic", ascending=False)
    for r in routes.head(cc["show_routes"]).itertuples():
        rows.append(
            dict(key=[r.project], name=r.project, sub=r.range, route=True, fold=False, flag=r.unmarked_busy)
        )
    rest = routes.iloc[cc["show_routes"] :]
    if len(rest):
        name = cc["fold_routes"].format(n=len(rest), lo=rest.traffic.min(), hi=rest.traffic.max())
        rows.append(
            dict(key=rest.project.tolist(), name=name, sub=", ".join(rest.project), route=True, fold=True)
        )
    other = IP[~IP.route]
    for r in other[other.project.isin(ic["not_routes"])].itertuples():
        sub = cc["no_project_sub"].format(range=r.range) if r.project == ic["no_project"] else r.range
        rows.append(dict(key=[r.project], name=r.project, sub=sub, route=False, fold=False))
    small = other[~other.project.isin(ic["not_routes"])].sort_values(["cairns", "traffic"], ascending=False)
    if len(small):
        shown = small.project.head(cc["small_named"]).tolist()
        more = f" + {len(small) - len(shown)} more" if len(small) > len(shown) else ""
        name = cc["fold_small"].format(n=len(small), min_traffic=ic["min_traffic"])
        rows.append(
            dict(key=small.project.tolist(), name=name, sub=", ".join(shown) + more, route=False, fold=True)
        )
    R = pd.DataFrame(rows).fillna({"flag": False}).astype({"flag": bool})
    t = IP.set_index("project")
    R["traffic"] = [int(t.traffic.reindex(k).sum()) for k in R.key]
    R["cairns"] = [int(t.cairns.reindex(k).sum()) for k in R.key]
    R["ratio"] = R.cairns / R.traffic
    R["stones"] = [K[K.project.isin(k)].sort_values("completedAt") for k in R.key]
    return R


def cairns(cfg, T, G, f, path) -> dict:
    """Cairn index per project (folded tail) and the trail register of the heaviest cairns."""
    cc, ic = cfg["cairn_chart"] | {"max_kb": cfg["figures"]["max_kb"]}, cfg["index"]
    cc |= {"label_weight": f["ids"]}
    K, IP, IR = T["cairns"], T["index_project"], T["index_range"]
    busy, barely = G["cuts"]["busy_min"], G["cuts"]["barely_max"]
    top = K.sort_values(["weight", "inbound", "completedAt"], ascending=[False, False, True])
    reg = top.head(cc["register_rows"]).assign(bonus=lambda d: d.retro * cfg["cairns"]["retro_bonus"])
    rest = top.iloc[len(reg) :]
    by = rest.kind.value_counts().reindex(["parent", "mention", "retro"], fill_value=0)
    most, targets = int(K.inbound.max()), int((K.inbound > 0).sum())  # over cairns (completed) only
    flagged = IR[IR.unmarked_busy]
    note = "".join(
        cc["note_range"].format(
            range=r.range,
            traffic=r.traffic,
            cairns=r.cairns,
            ratio=pct(r.ratio),
            tie=cc["note_tie"].get(f"{int(r.on_busy_cut)}{int(r.on_barely_cut)}", ""),
        )
        for r in flagged.head(1).itertuples()
    )
    part = PART.get(ic["parts"], f"1/{ic['parts']}")
    fmt = f | dict(n=len(K), weight=int(K.weight.sum()), min_traffic=ic["min_traffic"], part=part)
    fmt |= dict(busy=math.ceil(busy), barely=pct(barely))
    text = {
        k: cc[k]
        for k in ("panel_a", "legend_size", "legend_kind", "section", "none", "frac", "flag", "flag_note")
    }
    text |= {k: cc[k] for k in ("legend_maps", "more")}
    text |= {k: cc[k] for k in ("no_cairn", "heads", "kind_names", "register_heads")}
    text |= {
        k: cc[k].format(**fmt)
        for k in ("title", "subtitle", "rule", "busy", "barely", "register_sub", "source")
    }
    text |= {
        "legend_ids": cc["legend_ids"].format(weight=cc["label_weight"]),
        "kinds_legend": {k: v.format(**fmt) for k, v in cc["kinds_legend"].items()},
        "clip": cc["clip"].format(max=f"{100 * cc['ratio_max']:.0f}%"),
        "note_a": note + cc["note_a"],
        "panel_b": cc["panel_b"].format(n=len(reg)),
        "register_foot": cc["register_foot"].format(
            n=len(rest), lo=rest.weight.min(), hi=rest.weight.max(), **by
        )
        + (
            cc["single_mention"].format(targets=targets)
            if most == 1
            else cc["most_mention"].format(most=most)
        ),
    }
    rows = index_rows(IP, K, ic, cc)
    return cairn_chart(rows, reg, (busy, barely), text, cc, path)
