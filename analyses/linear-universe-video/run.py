"""linear-universe-video: a ~4 minute ManimCE walkthrough of linear-universe.

export(): everything the scenes need, precomputed, in outputs/video_data.json. Positions are already
in manim frame units (16:9, height 8, origin at the centre, y up): each layout maps by
x_frame = a_x * x + b_x and y_frame = a_y * y + b_y (meta.transforms holds a and b).
Layouts: random and t-SNE (linear-issue-map), universe (linear-universe's chart), graph (a spring
layout of the human-issue graph, hub issues set aside in a belt). Also galaxy and planet circles,
delegate arrows, the 25 text clusters, the S4 subset with its edge layers, single-seed vs consensus
galaxies (S5), planet naming sources (S6), tour stops (S8), the ablation (S9) and the numbers
allowed on screen.

render(): each scene class of scenes.py with manim's Python API into outputs/scenes/<Scene>.mp4;
ffmpeg joins them into outputs/linear-universe.mp4 and grabs posters into outputs/posters/.
Needs: uv run --extra graph --extra video, system Pango headers and ffmpeg.
"""

import importlib.util
import json
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import linear_sum_assignment
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial import ConvexHull

from analytics.config import load_config, save_run_config
from analytics.features import graph

GOLDEN = np.pi * (3 - np.sqrt(5))  # the universe chart's sunflower angle step


def sibling(path, name):
    """Import a module by file path (the CLI loads run.py by path, not as a package)."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---- geometry -----------------------------------------------------------------------------------
def fit(lo, hi, box, aspect=True):
    """a, b per axis mapping the data extent [lo, hi] onto box [cx, cy, w, h] (frame = a * xy + b)."""
    a = np.asarray(box[2:], float) / (np.asarray(hi) - np.asarray(lo))
    a = np.full(2, a.min()) if aspect else a
    return a, np.asarray(box[:2], float) - a * (np.asarray(lo) + np.asarray(hi)) / 2


def enclosing(c, r, iters=400):
    """Approximate smallest circle around circles (centres c, radii r): Badoiu-Clarkson steps."""
    m = (np.min(c - r[:, None], 0) + np.max(c + r[:, None], 0)) / 2
    for i in range(iters):
        f = np.argmax(np.linalg.norm(c - m, axis=1) + r)
        u = (c[f] - m) / max(np.linalg.norm(c[f] - m), 1e-9)
        m = m + (c[f] + u * r[f] - m) / (i + 2)
    return m, float(np.max(np.linalg.norm(c - m, axis=1) + r))


def belt(k, n, r0, r1):
    """Sunflower annulus (the chart's template belt): offsets of items k = 0..n-1."""
    rad = np.sqrt(r0**2 + (r1**2 - r0**2) * (k + 0.5) / n)
    return np.c_[rad * np.cos(k * GOLDEN), rad * np.sin(k * GOLDEN)]


def planet_circles(a, pc, belt_name):
    """Planet centre and radius in chart units, recovered exactly from the sunflower moon layout
    (moon k of a planet, oldest first, sits at s*sqrt(k+0.5) from the centre at angle k*GOLDEN;
    the template belt is an annulus around the hub centre)."""
    s, rows = pc["moon_spacing"], []
    for p, sub in a[a.planet_id >= 0].sort_values("createdAt").groupby("planet_id"):
        xy, k = sub[["x", "y"]].to_numpy(), np.arange(len(sub))
        if sub.planet.iloc[0] == belt_name:
            r_in = float(np.sqrt(max(np.median(np.sum(xy**2, 1) - s**2 * (k + 0.5)), 0)))
            c, r = np.zeros(2), float(np.sqrt(r_in**2 + len(sub) * s**2))
        else:
            unit = np.c_[np.cos(k * GOLDEN), np.sin(k * GOLDEN)]
            c, r_in = np.median(xy - s * np.sqrt(k + 0.5)[:, None] * unit, axis=0), 0.0
            r = s * np.sqrt(len(sub) + 0.5) + pc["planet_rim"] * s
        rows.append(dict(planet_id=int(p), cx=c[0], cy=c[1], r=r, r_in=r_in))
    return pd.DataFrame(rows).set_index("planet_id")


def galaxy_circles(P, hub_id, pad):
    """Galaxy centre and radius (chart units): the hub is centred at the origin; the others are
    the smallest circle around their planets. Both get the chart's galaxy_pad."""
    out = {}
    for g, sub in P.groupby("galaxy_id"):
        c, r = sub[["cx", "cy"]].to_numpy(), sub.r.to_numpy()
        m, R = (np.zeros(2), float(np.max(np.linalg.norm(c, axis=1) + r))) if g == hub_id else enclosing(c, r)
        out[g] = (m, R + pad)
    return out


# ---- export pieces ------------------------------------------------------------------------------
def numbers_in(text):
    """Canonical numbers in a string ('1,143' -> '1143', '0.990' -> '0.99', '42%' -> '42')."""
    found = re.findall(r"\d[\d,]*(?:\.\d+)?", text)
    return {f"{float(x.replace(',', '')):.4f}".rstrip("0").rstrip(".") for x in found}


def leaves(obj):
    if isinstance(obj, dict):
        return [x for k, v in obj.items() for x in [str(k), *leaves(v)]]
    return [x for v in obj for x in leaves(v)] if isinstance(obj, list) else [str(obj)]


def name_source(P, iss, ucfg):
    """How each planet got its name (linear-universe's cascade): hub rule, milestone, ritual label,
    project, override (of distinctive terms) or distinctive terms; plus the evidence string."""
    nc, hc = ucfg["names"], ucfg["hub"]
    inv = {v: k for k, v in nc["override"].items()}
    out = []
    for p, row in P.iterrows():
        sub, base = iss[iss.planet_id == p], row["name"].removesuffix(nc["mixed_suffix"])
        ms = sub.milestone.dropna()
        named_by = lambda m: base in (m, m.split(" — ", 1)[-1]) or m.startswith(base.rstrip("…"))  # noqa: E731, B023
        labels = [lb for ls in sub.labels for lb in ls if nc["label_phrase"].get(lb) == base]
        if row.galaxy_id == iss.galaxy_id.max():
            out.append(("hub rule", hc["belt_pattern"] if base == hc["belt"] else ""))
        elif any(named_by(m) for m in ms.unique()):
            out.append(("milestone", ms.mode()[0]))
        elif labels:
            out.append(("ritual label", labels[0]))
        elif base in set(iss.project.dropna()):
            out.append(("project", base))
        else:
            out.append(("override", inv[base]) if base in inv else ("distinctive terms", base))
    return out


def universe_tables(a, iss, P, gal, metrics, ucfg, cfg, ua, ub):
    """Galaxies, planets and delegate arrows as records, circles in frame units."""
    hub_id, hc, st = int(a.galaxy_id.max()), ucfg["hub"], cfg["style"]
    to_frame = lambda xy: (np.asarray(xy) * ua + ub).round(4).tolist()  # noqa: E731
    G = pd.DataFrame({"id": sorted(gal)})
    G["name"] = G.id.map(a.groupby("galaxy_id").galaxy.first())
    gm = G.name.map(metrics["galaxies"])
    G["hue"], G["color"] = G.id, G.id.map(dict(enumerate(st["hues"])))
    G["n"], G["planets"], G["lead"] = gm.str["n"], gm.str["planets"], gm.str["lead"]
    opened = a.lifecycle.eq("open").groupby(a.galaxy_id)
    G["open"], G["open_share"] = G.id.map(opened.sum()), G.id.map(opened.mean()).round(3)
    G["centre"] = [to_frame(gal[g][0]) for g in G.id]
    G["radius"] = [round(gal[g][1] * ua[0], 4) for g in G.id]
    G["stats"] = [f"{n:,} issues · {k} planets" for n, k in zip(G.n, G.planets, strict=True)]

    Q = P.reset_index().rename(columns={"planet_id": "id"})
    deleg = hc["delegated"].format("")
    kinds = [Q.name.eq(hc["belt"]), Q.name.str.startswith(deleg), Q.galaxy_id.eq(hub_id)]
    Q["role"] = np.select(kinds, ["belt", "orbit", "core"], "")
    Q["mixed"] = Q.name.str.endswith(ucfg["names"]["mixed_suffix"])
    Q["centre"] = [to_frame(xy) for xy in Q[["cx", "cy"]].to_numpy()]
    Q["radius"], Q["r_in"] = (Q.r * ua[0]).round(4), (Q.r_in * ua[0]).round(4)
    Q[["name_source", "name_evidence"]] = name_source(P, iss, ucfg)
    cols = ["id", "galaxy_id", "name", "n", "open", "new_share", "cohesion", "idle_days", "pinned", "mixed"]
    planets = Q[[*cols, "role", "centre", "radius", "r_in", "name_source", "name_evidence"]]

    gid = dict(zip(G.name, G.id, strict=True))
    arrows = []
    for _, q in Q[Q.role.eq("orbit") & Q.name.str.removeprefix(deleg).isin(gid)].iterrows():
        target = gid[q["name"].removeprefix(deleg)]
        m, R = gal[target]
        c = np.array([q.cx, q.cy])
        u = (m - c) / np.linalg.norm(m - c)
        arrows.append(
            dict(planet_id=q.id, target=target, target_name=G.name[target], count=q.n)
            | dict(start=to_frame(c + u * q.r), end=to_frame(m - u * R))
        )
    return G.to_dict("records"), planets.to_dict("records"), arrows


def text_clusters(iss, tsne, belt_pattern, titles):
    """The 25 text clusters: terms, size, members, centroid and convex hull (frame units), plus the
    debrief ring (the cluster holding most template titles): its centre and median radius."""
    lab = iss.tfidf_cluster.to_numpy()
    items = []
    for c in sorted(set(lab) - {-1, -2}):
        m = np.flatnonzero(lab == c)
        pts = tsne[m]
        hull = pts[ConvexHull(pts).vertices].round(4).tolist()
        items.append(
            dict(id=int(c), terms=iss.tfidf_cluster_terms[m[0]], n=len(m), idx=m.tolist())
            | dict(centroid=np.median(pts, 0).round(4).tolist(), hull=hull)
        )
    is_belt = titles.str.match(belt_pattern).to_numpy()
    ring = max(items, key=lambda c: is_belt[c["idx"]].sum())
    rp = tsne[ring["idx"]]
    centre = np.median(rp, 0)
    ring = dict(cluster=ring["id"], centre=centre.round(4).tolist())
    ring["radius"] = round(float(np.median(np.linalg.norm(rp - centre, axis=1))), 4)
    out = dict(n=len(items), unclustered=int((lab == -1).sum()), unplaced=int((lab == -2).sum()))
    return out | dict(items=items, ring=ring)


def title_terms(a, M, s2):
    """S2: example titles with their top TF-IDF terms, weights and character spans in the title."""
    X = sparse.load_npz(M / "tfidf.npz").tocsr()
    vocab = np.array(json.loads((M / "tfidf_meta.json").read_text())["vocab"])
    out = []
    for i in [int(np.flatnonzero(a.id == x)[0]) for x in s2["example_ids"]]:
        row, terms = X[i], []
        for j in row.indices[np.argsort(-row.data)[: s2["terms_n"]]]:
            pattern = r"\b" + r"\W+".join(map(re.escape, vocab[j].split())) + r"\b"
            hit = re.search(pattern, a.title[i], re.I)
            span = [hit.start(), hit.end()] if hit else None
            terms.append(dict(term=vocab[j], weight=round(float(row[0, j]), 3), span=span))
        out.append(dict(i=i, id=a.id[i], title=a.title[i], team=a.team[i], terms=terms))
    return out


def seed_galaxies(lu, ucfg, df, X, layers, cons, human, seeds):
    """Galaxy labels of single Louvain seeds (full build, satellites attached), renumbered to the
    consensus galaxy they overlap most (Hungarian on the human issues). Also the human graph."""
    h = np.flatnonzero(~lu.is_hub(df, ucfg["hub"]))
    G = graph.to_graph(graph.combine(layers, ucfg["edges"]["weights"])[h][:, h])
    out = {}
    for s in seeds:
        run = graph.louvain(G, ucfg["galaxies"]["resolution"], s)
        lab = lu.build(df, X, layers, ucfg, runs=[run])[0].galaxy_id.to_numpy()
        ok = human & (lab >= 0) & (cons >= 0)
        C = pd.crosstab(lab[ok], cons[ok])
        r, c = linear_sum_assignment(-C.to_numpy())
        m = dict(zip(C.index[r], C.columns[c], strict=True))
        out[s] = np.array([m.get(g, g) if hm else g for g, hm in zip(lab, human, strict=True)])
    return out, G


def graph_layout(G, a, human, fr, s5):
    """S5 positions: spring layout of the human-issue graph fitted into graph_box (the 1-99% extent;
    outliers, i.e. issues with no links, are clipped to it); hub issues in a belt at hub_ring,
    oldest inside. Returns positions (frame units), the transform and the hub order."""
    import networkx as nx

    sp = nx.spring_layout(G, weight="weight", seed=s5["graph_seed"], iterations=s5["graph_iterations"])
    q = np.array([sp[i] for i in range(len(sp))])
    lo, hi = np.quantile(q, fr["graph_clip"], axis=0), np.quantile(q, 1 - fr["graph_clip"], axis=0)
    tf = fit(lo, hi, fr["graph_box"])
    pos = np.full((len(a), 2), np.nan)
    pos[human] = np.clip(q, lo, hi) * tf[0] + tf[1]
    hx, hy, r0, r1 = fr["hub_ring"]
    hub_idx = np.flatnonzero(~human)
    hub_idx = hub_idx[np.argsort(a.createdAt.to_numpy()[hub_idx], kind="stable")]
    pos[hub_idx] = [hx, hy] + belt(np.arange(len(hub_idx)), len(hub_idx), r0, r1)
    return pos, tf, hub_idx


def subset_graph(df, layers, ucfg, s4, box):
    """S4: issues of a few projects, a Kamada-Kawai layout of their combined graph (edge length
    1 / weight**length_power), and each edge layer as all pairs [i, j, weight] (local indices), a
    display set (text: top-k per issue; clique layers: a spanning tree per group by layout
    distance; parent: every link) and, for clique layers, the groups."""
    import networkx as nx

    idx = np.flatnonzero(df.project.isin(s4["projects"]).to_numpy())
    G = graph.to_graph(graph.combine(layers, ucfg["edges"]["weights"])[idx][:, idx])
    for u, v, w in G.edges(data="weight"):
        G[u][v]["length"] = 1 / max(w, 1e-6) ** s4["length_power"]
    p = nx.kamada_kawai_layout(G, weight="length")
    xy = np.array([p[i] for i in range(len(idx))])
    a, b = fit(xy.min(0), xy.max(0), box, aspect=False)
    xy = xy * a + b
    sub = df.iloc[idx]
    keys = {"project": sub.project, "milestone": sub.milestone_key, "repo": sub.repo, "sibling": sub.parentId}
    out = {}
    for name in s4["layers"]:
        M = sparse.triu(layers[name][idx][:, idx], 1).tocoo()
        pairs = [[int(i), int(j), round(float(w), 4)] for i, j, w in zip(M.row, M.col, M.data, strict=True)]
        layer = {"label": s4["layer_names"][name], "pairs": pairs}
        if name in keys:
            B = graph.incidence(keys[name].tolist()).tocsc()
            values = pd.Series([v for x in keys[name] for v in np.atleast_1d(x) if not pd.isna(v)]).unique()
            members = [B[:, j].nonzero()[0].tolist() for j in range(B.shape[1])]
            layer["groups"] = {str(k): m for k, m in zip(values, members, strict=True) if len(m) > 1}
            layer["display"] = []
            for m in layer["groups"].values():
                T = minimum_spanning_tree(np.linalg.norm(xy[m][:, None] - xy[m][None], axis=2)).tocoo()
                layer["display"] += [[m[i], m[j]] for i, j in zip(T.row, T.col, strict=True)]
        elif name == "text_knn":
            W = (M + M.T).tocsr()
            k = s4["text_top_k"]
            top = {
                tuple(sorted((i, int(j))))
                for i in range(W.shape[0])
                for j in W[i].indices[np.argsort(-W[i].data)[:k]]
            }
            layer["display"] = [list(t) for t in sorted(top)]
        else:
            layer["display"] = [q[:2] for q in pairs]
        out[name] = layer
    n = s4["title_chars"]
    short = sub.title.map(lambda t: t if len(t) <= n else t[: n - 1].rstrip() + "…")
    res = dict(projects=s4["projects"], idx=idx.tolist(), pos=xy.round(4).tolist(), ids=sub.id.tolist())
    res |= dict(
        titles=short.tolist(), project=sub.project.tolist(), milestone=sub.milestone.fillna("").tolist()
    )
    return res | dict(layers=out, order=s4["layers"], a=a.tolist(), b=b.tolist())


def tour_stops(galaxies, planets, iss, fr, tc):
    """S8: per galaxy in tour order, a camera (centre, width) framing it above the caption band,
    its stats line and story, and the planets / issues to highlight."""
    gal = {g["name"]: g for g in galaxies}
    pid = {p["name"]: p["id"] for p in planets}
    stops = []
    for name in tc["order"]:
        g = gal[name]
        view = 2 * g["radius"] * tc["margin"] / (1 - tc["caption_room"])  # camera frame height
        w = max(tc["min_width"], view * fr["width"] / fr["height"])
        cy = g["centre"][1] - (w * fr["height"] / fr["width"]) * tc["caption_room"] / 2
        fp, fi = [], []
        for f in tc["focus"][name]:
            if f.startswith("children:"):
                root = f.split(":", 1)[1]
                fi += np.flatnonzero((iss.parentId == root) | (iss.id == root)).tolist()
            else:
                fp.append(pid[f])
                fi += np.flatnonzero(iss.planet_id == pid[f]).tolist()
        camera = dict(centre=[g["centre"][0], round(cy, 4)], width=round(w, 4))
        stops.append(
            dict(galaxy_id=g["id"], name=name, stats=g["stats"], story=tc["story"][name], camera=camera)
        )
        stops[-1] |= dict(focus_planets=fp, focus_idx=fi)
    return stops


def allowed_numbers(cfg, metrics, imap, derived):
    """Numbers that may appear on screen: metrics.json, the READMEs and lessons, counts from the
    data. Raises if a tour story uses any other."""
    texts = [Path(cfg["paths"][k]).read_text() for k in ("universe_readme", "issue_map_readme", "lessons")]
    allowed = set().union(*map(numbers_in, leaves(metrics) + leaves(imap) + leaves(derived) + texts))
    bad = {
        k: sorted(numbers_in(v) - allowed) for k, v in cfg["tour"]["story"].items() if numbers_in(v) - allowed
    }
    if bad:
        raise ValueError(f"tour stories use numbers not in metrics.json, the READMEs or the data: {bad}")
    return sorted(allowed, key=float)


def export(cfg, path):
    pth, fr, st = cfg["paths"], cfg["frame"], cfg["style"]
    U, M = Path(pth["universe"]), Path(pth["issue_map"])
    a = pd.read_parquet(U / "assignments.parquet")
    iss = pd.read_parquet(M / "issues.parquet")
    metrics = json.loads((U / "metrics.json").read_text())
    pc = json.loads((U / "run_config.json").read_text())["plot"]
    imap = json.loads((M / "run_config.json").read_text())["stats"]
    if not (a.id.to_numpy() == iss.id.to_numpy()).all():
        raise ValueError("linear-universe and linear-issue-map outputs are not row-aligned: rerun both")
    ucfg = load_config(pth["universe_config"])
    lu = sibling(Path(pth["universe_config"]).with_name("run.py"), "linear_universe_run")
    df, X, _ = lu.load(ucfg)
    if not (df.id.to_numpy() == a.id.to_numpy()).all():
        raise ValueError("the snapshot changed since linear-universe ran: rerun it")
    layers = lu.edge_layers(df, X, ucfg["edges"])
    iss = iss.assign(galaxy_id=a.galaxy_id, planet_id=a.planet_id)
    n, gid, hub_id = len(a), a.galaxy_id.to_numpy(), int(a.galaxy_id.max())
    human, lab = (a.route != "hub").to_numpy(), iss.tfidf_cluster.to_numpy()

    # positions in frame units: random (stretched to its box), t-SNE, universe, graph (S5)
    tsne, rnd = iss[["x_tfidf", "y_tfidf"]].to_numpy(), iss[["x_random", "y_random"]].to_numpy()
    tf = {"random": fit([0, 0], [1, 1], fr["random_box"], aspect=False)}
    tf["tsne"] = fit(np.nanmin(tsne, 0), np.nanmax(tsne, 0), fr["tsne_box"])
    P = planet_circles(a, pc, ucfg["hub"]["belt"])
    names = (
        a[a.planet_id >= 0]
        .groupby("planet_id")
        .agg(name=("planet", "first"), galaxy_id=("galaxy_id", "first"))
    )
    pm = pd.DataFrame(metrics["planets"]).drop(columns="galaxy")
    P = P.join(names).reset_index().merge(pm, on="name", how="left").set_index("planet_id")
    gal = galaxy_circles(P, hub_id, pc["galaxy_pad"])
    ext = np.array([[m[0] - R, m[1] - R, m[0] + R, m[1] + R] for m, R in gal.values()])
    tf["universe"] = ua, ub = fit(ext[:, :2].min(0), ext[:, 2:].max(0), fr["universe_box"])
    seeds, G = seed_galaxies(lu, ucfg, df, X, layers, gid, human, cfg["s5"]["seeds"])
    gpos, tf["graph"], hub_idx = graph_layout(G, a, human, fr, cfg["s5"])
    unplaced = (a.planet_id.to_numpy() < 0)[:, None]  # the two issues with no links are not drawn
    pos = {"random": rnd * tf["random"][0] + tf["random"][1], "tsne": tsne * tf["tsne"][0] + tf["tsne"][1]}
    pos |= {"universe": np.where(unplaced, np.nan, a[["x", "y"]].to_numpy() * ua + ub), "graph": gpos}

    galaxies, planets, arrows = universe_tables(a, iss, P, gal, metrics, ucfg, cfg, ua, ub)
    s3 = np.flatnonzero((iss.project == cfg["s3"]["project"]).to_numpy())
    ck = metrics["checks"]
    seed_rows = [
        dict(seed=s, galaxy_id=lb.tolist(), ari=ck["seed_galaxy"][str(s)]["galaxy_ari"])
        | dict(moved_idx=np.flatnonzero(human & (lb != gid)).tolist())
        for s, lb in seeds.items()
    ]
    teams = [dict(name=t, n=int((a.team == t).sum()), hue=h) for h, t in enumerate(st["team_order"])]
    derived = {"teams": [t["n"] for t in teams], "issues": n, "s3": [len(s3), len(set(lab[s3]) - {-1, -2})]}

    xy = lambda v: [None if np.isnan(p).any() else [round(p[0], 4), round(p[1], 4)] for p in v]  # noqa: E731
    issues = {c: a[c].tolist() for c in ("id", "title", "team")}
    issues["team_idx"] = (
        a.team.map({t: i for i, t in enumerate(st["team_order"])}).fillna(-1).astype(int).tolist()
    )
    issues["project"], issues["milestone"] = a.project.fillna("").tolist(), iss.milestone.fillna("").tolist()
    issues["parent_idx"] = (
        iss.parentId.map(pd.Series(np.arange(n), index=a.id)).fillna(-1).astype(int).tolist()
    )
    issues |= {c: a[c].tolist() for c in ("galaxy_id", "planet_id", "lifecycle", "route")}
    issues |= {"cluster": lab.tolist(), "createdAt": a.createdAt.astype(str).tolist()}
    issues |= {k: xy(v) for k, v in pos.items()}
    moon = pc["moon_size"] * pc["moon_spacing"] / 2 * ua[0] * cfg["universe"]["moon_scale"]
    meta = dict(snapshot=str(a.createdAt.max().date()), n=n, frame=[fr["width"], fr["height"]], hub_id=hub_id)
    meta["transforms"] = {k: dict(a=v[0].round(6).tolist(), b=v[1].round(6).tolist()) for k, v in tf.items()}
    meta |= dict(moon_radius=round(moon, 4), issue_map=imap, rerun=cfg["text"]["rerun"])
    clusters = text_clusters(iss, pos["tsne"], ucfg["hub"]["belt_pattern"], a.title)
    summary = ck["seed_galaxy_summary"]["galaxy_ari"]
    s6 = dict(
        galaxy=cfg["s6"]["galaxy"],
        galaxy_id=next(g["id"] for g in galaxies if g["name"] == cfg["s6"]["galaxy"]),
    )
    data = dict(meta=meta, issues=issues, teams=teams, galaxies=galaxies, planets=planets, arrows=arrows)
    data |= dict(clusters=clusters, s2={"examples": title_terms(a, M, cfg["s2"])})
    data["s3"] = dict(project=cfg["s3"]["project"], n=len(s3), clusters_spanned=len(set(lab[s3]) - {-1, -2}))
    data["s3"] |= dict(of=clusters["n"], idx=s3.tolist())
    data["s4"] = subset_graph(df, layers, ucfg, cfg["s4"], fr["s4_box"])
    data["s5"] = dict(hub_idx=hub_idx.tolist(), runs=len(ucfg["galaxies"]["seeds"]), seeds=seed_rows)
    data["s5"] |= dict(seed_ari_mean=summary[0], seed_ari_min=summary[1])
    data["s6"] = s6 | {
        "cascade": [ucfg["names"][k] for k in ("milestone_share", "label_share", "project_share")]
    }
    data["tour"] = tour_stops(galaxies, planets, iss, fr, cfg["tour"])
    data["ablation"] = [
        dict(key=k, label=lb) | {m: ck["ablation"][k][m] for m in ("galaxy_ari", "planet_ari", "planets")}
        for k, lb in cfg["ablation"]["rows"]
    ]
    data["allowed_numbers"] = allowed_numbers(cfg, metrics, imap, derived)
    native = lambda o: o.item() if hasattr(o, "item") else str(o)  # noqa: E731
    path.write_text(json.dumps(data, ensure_ascii=False, default=native))
    print(f"exported {path.name}: {path.stat().st_size / 1024:.0f} KB")
    return data


# ---- render -------------------------------------------------------------------------------------
def stale(cfg, path):
    """Whether video_data.json needs rebuilding ([export] mode)."""
    p, mode = cfg["paths"], cfg["export"]["mode"]
    inputs = [Path(p["universe"]) / "assignments.parquet", Path(p["universe"]) / "metrics.json"]
    inputs += [Path(p["issue_map"]) / "issues.parquet", Path(cfg["_meta"]["config_path"]), Path(__file__)]
    newer = not path.exists() or max(f.stat().st_mtime for f in inputs) > path.stat().st_mtime
    return mode == "always" or (mode == "auto" and newer)


def render(cfg, scenes, names, out):
    """Render scene classes with manim's Python API; returns {name: mp4 path}."""
    from manim import tempconfig

    r, media = cfg["render"], out / "media"
    opts = {"quality": r["quality"]}  # first the preset, then the overrides below
    opts |= {"frame_rate": r["fps"]} if r["fps"] else {}
    opts |= {"pixel_width": r["pixel_width"], "pixel_height": r["pixel_height"]} if r["pixel_width"] else {}
    opts |= dict(media_dir=str(media), video_dir=str(out / "scenes"), images_dir=str(media / "images"))
    opts |= dict(partial_movie_dir=str(media / "partial" / "{scene_name}"), disable_caching=True)
    opts |= dict(
        progress_bar=r["progress_bar"], verbosity=r["verbosity"], background_color=cfg["style"]["background"]
    )
    paths = {}
    for name in names:
        with tempconfig(opts | {"output_file": name}):
            getattr(scenes, name)().render()
        paths[name] = out / "scenes" / f"{name}.mp4"
        print(f"rendered {paths[name]}")
    return paths


def ffmpeg(*args):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *map(str, args)], check=True)


def probe(path, entries="format=duration"):
    cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", entries, "-of", "csv=p=0"]
    return subprocess.run([*cmd, str(path)], capture_output=True, text=True, check=True).stdout.strip()


def duration(path):
    return float(probe(path))


def main(cfg: dict) -> None:
    out, r = Path(cfg["paths"]["outputs"]), cfg["render"]
    out.mkdir(parents=True, exist_ok=True)
    if stale(cfg, out / "video_data.json"):
        export(cfg, out / "video_data.json")
    scenes = sibling(Path(__file__).with_name("scenes.py"), "linear_universe_video_scenes")
    scenes.use(cfg)
    names = r["scenes"] or cfg["scenes"]["order"]
    if missing := [s for s in names if not hasattr(scenes, s)]:
        raise SystemExit(f"no scene class {missing} in scenes.py")
    paths = render(cfg, scenes, names, out)
    if r["posters"]:
        (out / "posters").mkdir(exist_ok=True)
        for name, p in paths.items():
            at = f"{duration(p) * cfg['posters']['at']:.3f}"
            ffmpeg("-ss", at, "-i", p, "-frames:v", 1, out / "posters" / f"{name}.png")
        if (main_poster := out / "posters" / f"{cfg['posters']['main']}.png").exists():
            (out / "poster.png").write_bytes(main_poster.read_bytes())
    stats = {"scenes": {k: round(duration(p), 2) for k, p in paths.items()}}
    every = [out / "scenes" / f"{s}.mp4" for s in cfg["scenes"]["order"]]
    kinds = {probe(p, "stream=width,height,r_frame_rate") for p in every if p.exists()}
    if r["concat"] and all(p.exists() for p in every) and len(kinds) > 1:
        print(f"not joining: scene renders differ in size or frame rate {kinds}; render all at one quality")
    elif r["concat"] and all(p.exists() for p in every):  # joins the latest render of every scene
        (lst := out / "media" / "concat.txt").write_text("".join(f"file '{p.resolve()}'\n" for p in every))
        ffmpeg(
            "-f", "concat", "-safe", 0, "-i", lst, "-c", "copy", "-movflags", "+faststart", out / r["output"]
        )
        stats["video_s"] = round(duration(out / r["output"]), 2)
    save_run_config(cfg, out / "run_config.json", stats=stats)
    print(f"{stats} -> {out}")
