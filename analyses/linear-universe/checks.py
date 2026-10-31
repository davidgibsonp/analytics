"""Validation suite for linear-universe (run when checks.enabled = true) and routing diagnostics.

- run(): stability (single Louvain seeds vs the consensus, planet seeds, 90% subsamples), the
  modularity of planets left whole, and one-field ablation, each as ARI against the base run.
- held_out(): containment of links that are NOT graph edges (same-team creation sessions and title
  id-mentions) against random-pair and same-project baselines.
- routing(): how confident the text routing of delegated hub items is (always reported).
"""

import copy
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score as ARI

from analytics.features import graph


def session_pairs(df, minutes):
    """Pairs of same-team issues created in one session (consecutive gaps <= minutes)."""
    d = df.sort_values(["team", "createdAt"])
    step = d.groupby("team", observed=True).createdAt.diff()
    sess = (step.isna() | (step > pd.Timedelta(minutes=minutes))).cumsum()
    return [p for m in d.groupby(sess).groups.values() for p in combinations(m, 2)]


def idref_pairs(df, pattern):
    idx = pd.Series(df.index, index=df.id)
    refs = df.title.str.findall(pattern).explode().dropna()
    return [(i, idx[r]) for i, r in refs.items() if r in idx and idx[r] != i]


def same_share(pairs, lab):
    a, b = np.array(pairs).T
    ok = (lab[a] >= 0) & (lab[b] >= 0)
    return float((lab[a] == lab[b])[ok].mean())


def random_share(lab, team=None):
    """Share of random issue pairs (optionally within one team) that share a label."""
    d = pd.DataFrame({"lab": lab, "team": 0 if team is None else team})[lab >= 0]
    n = d.groupby(["team", "lab"]).size()
    t = d.groupby("team").size()
    return float((n * (n - 1)).sum() / (t * (t - 1)).sum())


def random_cross_project(lab, proj):
    """Share of random pairs that do NOT share a project yet share a label (baseline for the
    cross-project rows)."""
    d = pd.DataFrame({"lab": lab, "proj": proj})[lab >= 0]
    pairs = lambda n: (n * (n - 1) / 2).sum()  # noqa: E731
    same_proj = pairs(d.groupby("proj").size())  # groupby drops issues without a project
    same_both = pairs(d.groupby("lab").size()) - pairs(d.groupby(["lab", "proj"]).size())
    return float(same_both / (len(d) * (len(d) - 1) / 2 - same_proj))


def held_out(df, cfg):
    """Containment of links that are NOT in the graph: creation sessions and title id-mentions.
    Baselines: random pairs, and pairs sharing a project (a graph edge), since containment of a
    galaxy is mostly containment of a project; 'cross_project' rows keep only pairs that do not."""
    idref = cfg["mentions"]["pattern"]
    pairs = {
        "session": session_pairs(df, cfg["checks"]["session_minutes"]),
        "id_mention": idref_pairs(df, idref),
    }
    hub = df.route.eq("hub").to_numpy()
    proj = df.project.to_numpy(object)
    res = {}
    for name, pr in pairs.items():
        a, b = np.array(pr).T
        same_proj = np.array([isinstance(x, str) and x == y for x, y in zip(proj[a], proj[b], strict=True)])
        hm = ~(hub[a] | hub[b])
        human, cross = [pr[i] for i in np.flatnonzero(hm)], [pr[i] for i in np.flatnonzero(hm & ~same_proj)]
        res[f"{name}_pairs"] = {
            "pairs": len(pr),
            "same_project": round(float(same_proj.mean()), 3),
            "human_pairs": len(human),
            "human_same_project": round(float(same_proj[hm].mean()), 3),
            "human_cross_project_pairs": len(cross),
        }
        for lvl in ("galaxy", "planet"):
            lab = df[f"{lvl}_id"].to_numpy()
            hl = np.where(hub, -1, lab)
            res[f"{name}_{lvl}"] = {
                "same": round(same_share(pr, lab), 3),
                "random": round(random_share(lab), 3),
                "random_same_team": round(random_share(lab, df.team.astype(str).to_numpy()), 3),
                "human_same": round(same_share(human, hl), 3),
                "human_random": round(random_share(hl), 3),
                "human_same_and_same_project": round(float(((hl[a] == hl[b]) & same_proj)[hm].mean()), 3),
                "human_cross_project_same": round(same_share(cross, hl), 3) if cross else None,
                "human_cross_project_random": round(random_cross_project(hl, proj), 3),
            }
    a, b = np.array(pairs["id_mention"]).T
    side = np.where(hub, "hub", "human")
    res["id_mention_direction"] = (
        pd.Series(np.char.add(np.char.add(side[a], "→"), side[b])).value_counts().to_dict()
    )
    d = df[df.hub_kind.str.startswith("to") & df.title.str.contains(idref)]
    gal = df.set_index("id").galaxy_id
    tgt = d.title.str.findall(idref).map(
        lambda r: [gal[x] for x in r if x in gal.index and gal[x] < df.galaxy_id.max()]
    )
    for col, key in (
        ("about_id", "delegated_route_matches_mentioned_issue"),
        ("about_raw", "text_route_alone"),
    ):
        hit = [x in t for x, t in zip(d[col], tgt, strict=True) if t and (col == "about_id" or x >= 0)]
        res[key] = {"items": len(hit), "hits": int(np.sum(hit)), "share": round(float(np.mean(hit)), 3)}
    return res


def routing(df, gnames, cfg):
    """Delegated hub items: targets before the min_delegated merge, and how weak the text routes are."""
    dl, ck, other = df[df.about_raw >= 0], cfg["checks"], cfg["hub"]["other"]
    weak = (dl.route_cos < ck["weak_cos"]) | (dl.route_margin < ck["weak_margin"])
    gname = lambda s: s.map(gnames).fillna(other)  # noqa: E731
    return {
        "routed_by_text": len(dl),
        "no_text": int((df.hub_kind.str.startswith("to") & (df.about_raw < 0)).sum()),
        "text_target_before_merge": gname(dl.about_raw).value_counts().to_dict(),
        "merged_into_other": dl[dl.about_id < 0][["id", "title"]]
        .assign(text_target=gname(dl.about_raw))
        .to_dict("records"),
        "best_cos_below": [ck["weak_cos"], round(float((dl.route_cos < ck["weak_cos"]).mean()), 3)],
        "margin_below": [ck["weak_margin"], round(float((dl.route_margin < ck["weak_margin"]).mean()), 3)],
        "weak_either": round(float(weak.mean()), 3),
    }


def compare(base, o, df, big):
    """ARI on human issues (the hub is the same by rule), plus shape numbers of run o."""
    h = (base.route != "hub").to_numpy()
    par = df.parentId.map(pd.Series(df.index, index=df.id)).dropna().astype(int)
    pl = o.planet_id.to_numpy()
    n = pd.Series(pl[pl >= 0]).value_counts()
    return {
        "galaxy_ari": round(ARI(base.galaxy_id[h], o.galaxy_id[h]), 3),
        "planet_ari": round(ARI(base.planet_id[h], o.planet_id[h]), 3),
        "galaxies": int(o.galaxy_id[o.galaxy_id >= 0].nunique()),
        "planets": int(len(n)),
        f"planets_over_{big}": int((n > big).sum()),
        "parent_pairs_same_planet": round(float((pl[par.index] == pl[par.to_numpy()]).mean()), 3),
    }


def subsample(df, share, seed):
    """Row positions and issues of a random `share` of df; parent links to dropped issues are cut."""
    keep = np.sort(np.random.default_rng(seed).choice(len(df), int(share * len(df)), replace=False))
    sub = df.iloc[keep].reset_index(drop=True)
    sub["parentId"] = sub.parentId.where(sub.parentId.isin(sub.id))
    return keep, sub


def run(df, X, layers, cfg, base, runs, *, build, edge_layers):
    """Stability (single seeds vs consensus, planet seeds, subsamples), split test and field ablation.
    `build` and `edge_layers` are run.py's, passed in so this module never imports run.py."""
    ck, pc = cfg["checks"], cfg["planets"]
    res = {"seed_galaxy": {}, "seed_planet": {}, "subsample": {}, "ablation": {}}
    cmp = lambda b, o, d: compare(b, o, d, pc["max_size"])  # noqa: E731
    for s, r in zip(cfg["galaxies"]["seeds"], runs, strict=True):
        res["seed_galaxy"][s] = cmp(base, build(df, X, layers, cfg, runs=[r])[0], df)
        res["seed_planet"][s] = cmp(base, build(df, X, layers, cfg, runs=runs, planet_seed=s)[0], df)
    for s in ck["subsample_draws"]:
        keep, sub = subsample(df, ck["subsample"], s)
        o = build(sub, X[keep], edge_layers(sub, X[keep], cfg["edges"]), cfg)[0]
        res["subsample"][s] = cmp(base.iloc[keep].reset_index(drop=True), o, sub)
    A = graph.combine(layers, cfg["edges"]["weights"])
    human = (base.route != "hub").to_numpy()
    for p, nodes in pd.Series(np.arange(len(df))[human]).groupby(base.planet_id[human].to_numpy()):
        if len(nodes) > pc["max_size"]:  # planets left whole: modularity of their split, every seed
            sub = A[nodes][:, nodes]
            q = [
                graph.modularity(sub, graph.split(sub, **pc | {"seed": s})) for s in cfg["galaxies"]["seeds"]
            ]
            res["split_test"] = res.get("split_test", {}) | {
                int(p): {"n": len(nodes), "seed0_q": round(q[0], 3), "best_q": round(max(q), 3)}
            }
    res["ablation"]["full"] = cmp(base, base, df)
    for f in ck["ablate"]:
        c = copy.deepcopy(cfg)
        c["edges"]["weights"][f] = 0.0
        res["ablation"][f"drop_{f}"] = cmp(base, build(df, X, layers, c)[0], df)
    for k in ("seed_galaxy", "seed_planet", "subsample"):
        v = pd.DataFrame(res[k]).T
        res[k + "_summary"] = {
            c: [round(v[c].mean(), 3), round(v[c].min(), 3)] for c in ("galaxy_ari", "planet_ari")
        }
    return res
