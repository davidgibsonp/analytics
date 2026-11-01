"""linear-universe: Linear issues as galaxies > planets > moons (issues), and the chart of them.

Galaxies: consensus of Louvain over several seeds on one weighted graph of human issues (title
TF-IDF kNN, parent, sibling, milestone, project and repo-label edges). Communities under min_size
join the galaxy they are most tied to, or the galaxy a config pin names (near-ties only). The hub
galaxy is a rule applied before the graph: one project plus every project-less issue of one team.
Its planets are the template belt, daily debriefs, the rest of the project, and 'Delegated -> X'
for items whose text is nearest human galaxy X. Planets of human galaxies: Louvain inside each.

Outputs (in outputs/): assignments.parquet, metrics.json, universe.png, run_config.json.
checks.enabled = true adds the validation suite (checks.py) to metrics.json; it takes minutes.
"""

import importlib.util
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import normalize

from analytics.config import save_run_config
from analytics.features import graph, text
from analytics.sources.linear import load_issues
from analytics.viz.universe import universe

ROLE = {"belt": "belt", "daily": "core", "pipeline": "core"}  # hub planet kind -> chart role; to<X> = orbit


def load(cfg):
    """Raw issues, title TF-IDF (linear-issue-map's settings), repo labels and milestone keys."""
    df = load_issues(cfg["data"]["snapshot"])
    X, terms = text.tfidf(text.document_text(df, cfg["tfidf"]["text_fields"]), **cfg["tfidf"])
    pre = cfg["labels"]["repo_prefix"]
    df["repo"] = df.labels.map(lambda ls: [x[len(pre) :] for x in ls if x.startswith(pre)])
    df["milestone_key"] = (df.project.fillna("?") + "::" + df.milestone).where(df.milestone.notna())
    return df, X, terms


def edge_layers(df, X, e):
    """Unit-strength layers; graph.combine() applies the weights."""
    idx = pd.Series(range(len(df)), index=df.id)
    child = np.flatnonzero(df.parentId.notna())
    P = sparse.csr_matrix(
        (np.ones(len(child)), (child, idx[df.parentId.iloc[child]].to_numpy())), shape=(len(df),) * 2
    )
    layers = {
        "parent": P + P.T,
        "sibling": graph.clique(df.parentId, 1),
        "milestone": graph.clique(df.milestone_key, 1),
    }
    layers |= {"project": graph.clique(df.project, 1), "repo": graph.clique(df.repo, 1)}
    layers["text_knn"] = graph.text_knn(X, e["knn"], e["min_cos"])
    return layers


def top_projects(proj, generic=()):
    s = pd.Series(proj).dropna()
    return s[~s.isin(generic)].value_counts()


def attach(A, nat, proj, gc, no_project):
    """Communities under min_size join the large one with the highest normalized affinity. A config
    pin overrides that only on a near-tie (best < pin_ratio x runner-up) or when required_pins
    names it. Unlinked ones stay. Returns galaxy labels, route per issue and satellite evidence."""
    aff, n = graph.affinity(graph.group_weights(A, nat)), np.bincount(nat)
    big = np.flatnonzero(n >= gc["min_size"])
    lead = [next(iter(top_projects(proj[nat == c]).index), "-") for c in range(len(n))]
    target, kind, rows = np.arange(len(n)), np.where(n >= gc["min_size"], "core", "unassigned"), []
    for c in np.flatnonzero(n < gc["min_size"]):
        a = aff[c, big]
        if a.max() <= 0 or len(big) < 2:
            continue
        best, second = big[np.argsort(-a)[:2]]
        near_tie = bool(aff[c, best] < gc["pin_ratio"] * aff[c, second])
        held = [(proj[nat == b] == gc["pins"].get(lead[c])).sum() for b in big]
        pinned = max(held) > 0 and (near_tie or lead[c] in gc["required_pins"])
        target[c], kind[c] = (big[np.argmax(held)], "pinned") if pinned else (best, "satellite")
        rows.append(
            dict(satellite=lead[c], n=int(n[c]), best=lead[best], best_affinity=round(aff[c, best], 4))
            | dict(runner_up=lead[second], runner_up_affinity=round(aff[c, second], 4))
            | dict(ratio=round(aff[c, best] / aff[c, second], 2), near_tie=near_tie)
            | dict(joined=lead[target[c]], route=kind[c])
            | dict(projects=pd.Series(proj[nat == c]).fillna(no_project).value_counts().to_dict())
        )
    return target[nat], kind[nat], rows


def is_hub(df, hc):
    return ((df.project == hc["project"]) | ((df.team == hc["team"]) & df.project.isna())).to_numpy()


def hub_planets(df, X, gal, hub, hc):
    """Hub planets by rule: template belt and daily debriefs by title, the rest of the hub project,
    and 'delegated' items routed to the human galaxy whose text centroid is nearest. Targets with
    fewer than min_delegated items, or items without text, get -1."""
    t = df.title
    kind = np.select(
        [t.str.match(hc["belt_pattern"]), t.str.match(hc["daily_pattern"]), df.project.eq(hc["project"])],
        ["belt", "daily", "pipeline"],
        "delegated",
    )
    human = np.flatnonzero((gal >= 0) & ~hub)
    cent = normalize(graph.onehot(gal[human]).T @ X[human])
    S = (X @ cent.T).toarray()
    top2 = -np.sort(-S, axis=1)[:, :2]
    routed = hub & (kind == "delegated") & (np.asarray(X.getnnz(axis=1)) > 0)
    raw = np.where(routed, S.argmax(axis=1), -1)  # text routing alone, before the merge rule
    about = raw.copy()
    n = pd.Series(about[about >= 0]).value_counts()
    about[np.isin(about, n.index[n < hc["min_delegated"]])] = -1
    conf = pd.DataFrame(
        {"route_cos": top2[:, 0], "route_margin": top2[:, 0] / np.maximum(top2[:, 1], 1e-12)}
    ).where(pd.Series(routed), axis=0)
    key = np.where(kind == "delegated", np.char.add("to", about.astype(str)), kind)
    return key, about, conf.assign(about_raw=raw)


def build(df, X, layers, cfg, runs=None, planet_seed=None):
    """Galaxy id, planet id, route and hub kind per issue. Human galaxies are 0..k-1 by size, the
    hub is k, unassigned -1. Also returns the Louvain runs and the satellite evidence."""
    gc, hub = cfg["galaxies"], is_hub(df, cfg["hub"])
    h = np.flatnonzero(~hub)
    A = graph.combine(layers, cfg["edges"]["weights"])[h][:, h]
    if runs is None:
        G = graph.to_graph(A)
        runs = [graph.louvain(G, gc["resolution"], s) for s in gc["seeds"]]
    nat = graph.consensus(runs, gc["share"])
    lab, kind, sats = attach(A, nat, df.project.to_numpy()[h], gc, cfg["names"]["no_project"])
    lab = graph.relabel_by_size(lab)
    kind = np.where(np.bincount(lab)[lab] < gc["min_named"], "unassigned", kind)
    lab = np.where(kind == "unassigned", -1, lab)
    gal, route = np.full(len(df), lab.max() + 1), np.full(len(df), "hub", dtype=object)
    gal[h], route[h] = lab, kind
    planet = np.full(len(df), -1)
    pc = cfg["planets"] | ({} if planet_seed is None else {"seed": planet_seed})
    planet[h] = graph.split_within(A, lab, **pc)
    key, about, conf = hub_planets(df, X, gal, hub, cfg["hub"])
    hk = pd.Series(key[hub])
    planet[hub] = hk.map({k: i for i, k in enumerate(hk.value_counts().index)}) + planet.max() + 1
    out = pd.DataFrame(
        {"galaxy_id": gal, "planet_id": planet, "route": route, "hub_kind": np.where(hub, key, "")}
    )
    return pd.concat([out.assign(about_id=about), conf], axis=1), runs, sats


def contrast_terms(X, inside, ref, terms, skip, n, pool):
    """Top-n terms by share of issues using them inside minus in ref. Skips words in `skip`
    (prefix match); a later bigram replaces its own word."""
    B = (X > 0).astype(float)
    score = np.asarray(B[inside].mean(0)).ravel() - (np.asarray(B[ref].mean(0)).ravel() if ref.any() else 0)
    seen = lambda w: any(w.startswith(k) or k.startswith(w) for k in skip)  # noqa: E731
    cand = [str(terms[j]) for j in np.argsort(-score)[: pool * 4] if score[j] > 0]
    cand = [t for t in cand if not any(seen(w) for w in t.split())][:pool]
    out = []
    for t in cand:
        words = set(t.split())
        if any(words <= set(o.split()) for o in out):
            continue
        out = [o for o in out if not set(o.split()) < words] + [t]
        if len(out) == n:
            break
    return out


def galaxy_names(df, nc):
    """Theme from config keyed by the lead project, used only while that project holds lead_share
    of the galaxy's issues that have a specific project; else '<project> · <project> +N'."""
    names, check = {}, {}
    for g, sub in df[df.galaxy_id >= 0].groupby("galaxy_id"):
        ps = top_projects(sub.project, nc["generic_projects"])
        share = ps.iloc[0] / ps.sum()
        theme = nc["galaxy_theme"].get(ps.index[0])
        fallback = " · ".join(ps.index[:2]) + (f" +{len(ps) - 2}" if len(ps) > 2 else "")
        names[g] = theme if theme and share >= nc["lead_share"] else fallback
        check[names[g]] = {"lead": ps.index[0], "lead_share": round(share, 3), "themed": names[g] == theme}
    return names, check


def planet_names(df, X, terms, gnames, cfg):
    """Milestone > ritual label > project > contrast terms, then the override map."""
    nc, hc = cfg["names"], cfg["hub"]
    proj = df.project.where(~df.project.isin(nc["generic_projects"]))
    names = {}
    for p, sub in df[df.planet_id >= 0].groupby("planet_id"):
        g, k = sub.galaxy_id.iloc[0], sub.hub_kind.iloc[0]
        if k:
            a = sub.about_id.iloc[0]
            names[p] = hc.get(k) or hc["delegated"].format(gnames[a] if a >= 0 else hc["other"])
            continue
        ms = sub.milestone.value_counts()
        labs = pd.Series([lb for ls in sub.labels for lb in ls if lb in nc["label_phrase"]]).value_counts()
        tp = proj[sub.index].value_counts()
        pure = len(tp) > 0 and tp.iloc[0] >= nc["project_share"] * len(sub)
        if len(ms) and ms.iloc[0] >= nc["milestone_share"] * len(sub):
            m = ms.index[0] if len(ms.index[0]) <= nc["width"] else ms.index[0].split(" — ", 1)[-1]
            names[p] = m if len(m) <= nc["width"] else m[: nc["width"] - 1].rstrip() + "…"
        elif len(labs) and labs.iloc[0] >= nc["label_share"] * len(sub):
            names[p] = nc["label_phrase"][labs.index[0]]
        elif pure and tp.iloc[0] >= nc["project_share"] * (proj == tp.index[0]).sum():
            names[p] = tp.index[0]
        else:
            pm, gm = (df.planet_id == p).to_numpy(), (df.galaxy_id == g).to_numpy()
            lead = proj[gm].value_counts().index[0]
            skip = (
                set(nc["stop_terms"])
                | set(lead.split("-"))
                | (set(tp.index[0].split("-")) if len(tp) else set())
            )
            topic = " · ".join(contrast_terms(X, pm, gm & ~pm, terms, skip, nc["terms_n"], nc["term_pool"]))
            names[p] = f"{tp.index[0]}: {topic}" if pure and tp.index[0] != lead else topic
    for nm in set(names.values()):  # make names unique with one more contrast term
        dup = [q for q, m in names.items() if m == nm]
        for q in dup if len(dup) > 1 else []:
            qm = (df.planet_id == q).to_numpy()
            gm = (df.galaxy_id == df.galaxy_id[qm].iloc[0]).to_numpy()
            skip = set(nc["stop_terms"]) | set(re.findall(r"[a-z]+", nm.lower()))
            extra = contrast_terms(X, qm, gm & ~qm, terms, skip, 1, nc["term_pool"])
            names[q] = f"{nm} · {extra[0]}" if extra else nm
    return {p: nc["override"].get(nm, nm) for p, nm in names.items()}


def cohesion(X, planet):
    """Mean title cosine of a planet's issues to its normalized TF-IDF centroid."""
    out = {}
    for p in np.unique(planet[planet >= 0]):
        m = X[planet == p]
        out[p] = float((m @ normalize(np.asarray(m.mean(axis=0))).T).mean())
    return pd.Series(out)


def sibling(name):
    """Import a module next to this file (the CLI loads run.py by path, not as a package)."""
    spec = importlib.util.spec_from_file_location(
        f"linear_universe_{name}", Path(__file__).with_name(f"{name}.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(cfg: dict) -> None:
    out_dir = Path(cfg["paths"]["outputs"])
    out_dir.mkdir(parents=True, exist_ok=True)
    checks = sibling("checks")
    df, X, terms = load(cfg)
    layers = edge_layers(df, X, cfg["edges"])
    base, runs, sats = build(df, X, layers, cfg)
    df = pd.concat([df, base], axis=1)
    rc, nc, hc = cfg["recency"], cfg["names"], cfg["hub"]
    df["lifecycle"] = df.statusType.astype(str).map(cfg["lifecycle"]).fillna("open")
    snap = df.createdAt.max()
    df["new"] = df.createdAt >= snap - pd.Timedelta(days=rc["days"])
    gnames, gcheck = galaxy_names(df, nc)
    coh = cohesion(X, df.planet_id.to_numpy())
    mixed = coh.index[coh < nc["mixed_cohesion"]]
    pnames = {
        p: nm + nc["mixed_suffix"] * (p in mixed) for p, nm in planet_names(df, X, terms, gnames, cfg).items()
    }
    df["galaxy"] = df.galaxy_id.map(gnames).fillna(nc["unassigned"])
    df["planet"], df["about"] = df.planet_id.map(pnames), df.about_id.map(gnames)
    pl = df[df.planet_id >= 0].groupby("planet_id")
    P = pl.agg(galaxy_id=("galaxy_id", "first"), n=("id", "size"), kind=("hub_kind", "first"))
    P["about_id"], P["open"], P["new_share"] = (
        pl.about_id.first(),
        pl.lifecycle.apply(lambda s: (s == "open").sum()),
        pl.new.mean(),
    )
    P["pinned"] = pl.route.apply(lambda s: (s == "pinned").mean() >= 0.5)
    P["glow"] = (P.new_share >= rc["glow"]) & (P.n >= rc["glow_min_n"])
    P["cohesion"], P["idle_days"] = coh, (snap - pl.createdAt.max()).dt.days
    P["name"], P["galaxy"] = P.index.map(pnames), P.galaxy_id.map(gnames)
    P["role"] = P.kind.map(ROLE).where(~P.kind.str.startswith("to"), "orbit").fillna("")
    df["cohesion"], df["new_share"] = df.planet_id.map(P.cohesion), df.planet_id.map(P.new_share)

    # galaxy affinity (orders the ring) and cross-galaxy sub-issue + id-mention links (human issues)
    k, idref = df.galaxy_id.max(), cfg["mentions"]["pattern"]
    gal = df.galaxy_id.where(df.galaxy_id >= 0, k + 1).to_numpy()
    aff = pd.DataFrame(
        graph.affinity(graph.group_weights(graph.combine(layers, cfg["edges"]["weights"]), gal)[:k, :k])
    )
    human = df[(df.route != "hub") & (df.galaxy_id >= 0)]
    gal_of = human.set_index("id").galaxy_id
    ends = [(gal_of[c], gal_of.get(p)) for c, p in zip(human.id, human.parentId, strict=True)]
    ends += [
        (gal_of[i], gal_of.get(r))
        for i, rs in zip(human.id, human.title.str.findall(idref), strict=True)
        for r in rs
    ]
    cross = pd.Series([tuple(sorted((a, b))) for a, b in ends if b is not None and a != b]).value_counts()
    pc = cfg["plot"]
    links = [(a, b, int(n)) for (a, b), n in cross.items() if n >= pc["link_min"]]
    un = df[df.galaxy_id < 0]
    foot = pc["footnote"].format(
        runs=len(cfg["galaxies"]["seeds"]),
        unassigned=pc["footnote_unassigned"].format(n=len(un), ids=", ".join(un.id)) * bool(len(un)),
        links=pc["footnote_no_links"].format(min=pc["link_min"], most=cross.max()) * (not links),
    )
    details = {}  # the stats line under each galaxy name ends with its top projects
    for g, sub in df[df.galaxy_id >= 0].groupby("galaxy_id"):
        top = top_projects(sub.project).index[:3].tolist() + [nc["no_project"]] * bool(
            sub.project.isna().mean() > nc["no_project_share"]
        )
        details[g] = ", ".join(top[:3])
    chart = universe(
        df[["galaxy_id", "planet_id", "createdAt"]].assign(state=df.lifecycle),
        P.assign(target=P.about_id)[["galaxy_id", "n", "open", "role", "target", "pinned", "glow"]],
        hub=k,
        galaxy_names=gnames,
        planet_names=pnames,
        details=details,
        affinity=aff,
        links=links,
        legend=cfg["legend"] | {"glow": cfg["legend"]["glow"].format(glow=rc["glow"], days=rc["days"])},
        pc=pc | {"footnote": foot},
        path=out_dir / "universe.png",
    )
    df["x"], df["y"] = chart["xy"][:, 0], chart["xy"][:, 1]

    cols = ["id", "title", "project", "team", "galaxy", "planet", "route", "about", "galaxy_id", "planet_id"]
    cols += ["lifecycle", "createdAt", "cohesion", "new_share", "route_cos", "route_margin", "x", "y"]
    df[cols].to_parquet(out_dir / "assignments.parquet", index=False)
    table = ["galaxy", "name", "n", "open", "new_share", "cohesion", "idle_days", "pinned"]
    stalled = P[(P.idle_days >= rc["idle_days"]) & (P.open > 0)].sort_values("idle_days", ascending=False)
    delegated = df[df.hub_kind.str.startswith("to")]
    metrics = {
        "galaxies": {
            gnames[g]: {"n": int((df.galaxy_id == g).sum()), "planets": int((P.galaxy_id == g).sum())}
            | gcheck[gnames[g]]
            for g in sorted(gnames)
        },
        "planets": P[table].round(3).to_dict("records"),
        "routes": df.route.value_counts().to_dict(),
        "satellites": sats,
        "mixed": P.loc[mixed, ["galaxy", "name", "n", "cohesion"]].round(3).to_dict("records"),
        "galaxy_affinity": aff.rename(index=gnames, columns=gnames).round(4).to_dict(),
        "cross_links": {f"{gnames[a]} – {gnames[b]}": int(n) for (a, b), n in cross.items()},
        "delegated": delegated.about.fillna(hc["other"]).value_counts().to_dict(),
        "delegated_status": delegated.statusType.astype(str).value_counts().to_dict(),
        "delegated_routing": checks.routing(df, gnames, cfg),
        "stories": {
            "new": P[P.glow][table[:5]].round(2).to_dict("records"),
            "stalled": stalled[["galaxy", "name", "n", "open", "idle_days"]].to_dict("records"),
        },
        "label_collisions": chart["label_collisions"],
        "layout": {
            "pt_per_unit": round(chart["pt_per_unit"], 2),
            "repair_rounds_used": chart["repair_rounds_used"],
        },
        "labels_outside_own_galaxy_px": {pnames[p]: v for p, v in chart["spill"].items()},
    }
    if cfg["checks"]["enabled"]:
        metrics["held_out"] = checks.held_out(df, cfg)
        metrics["checks"] = checks.run(df, X, layers, cfg, base, runs, build=build, edge_layers=edge_layers)
        metrics["checks"]["split_test"] = {
            pnames[p]: v for p, v in metrics["checks"].get("split_test", {}).items()
        }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=1, default=str, ensure_ascii=False))
    stats = {"issues": len(df), "galaxies": len(gnames), "planets": len(P), "png_kb": chart["kb"]}
    save_run_config(cfg, out_dir / "run_config.json", stats=stats)
    print(f"{stats} -> {out_dir}")
