# %% [markdown]
# Exploration for linear-universe. Run cell by cell (VS Code / Cursor) in the graph extra's env,
# after `uv run --extra graph analytics run linear-universe --set checks.enabled=true`.
# Evidence behind two findings: hub items make Louvain seed-unstable; what moves in weak subsamples.

# %%
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score as ARI

from analytics.config import load_config
from analytics.features import graph

HERE = Path(__file__).parent
spec = importlib.util.spec_from_file_location("linear_universe_run", HERE / "run.py")
R = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R)
C = R.sibling("checks")
cfg = load_config(HERE / "config.toml")
OUT, gc = Path(cfg["paths"]["outputs"]), cfg["galaxies"]
df, X, terms = R.load(cfg)
layers = R.edge_layers(df, X, cfg["edges"])

# %% 1. Louvain is far more seed-stable once the hub items are taken out of the graph
A, hub = graph.combine(layers, cfg["edges"]["weights"]), R.is_hub(df, cfg["hub"])
for name, keep in [("full graph", np.ones(len(df), bool)), ("human issues only", ~hub)]:
    G = graph.to_graph(A[keep][:, keep])
    runs = [graph.louvain(G, gc["resolution"], s) for s in gc["seeds"]]
    cons = graph.consensus(runs, gc["share"])
    ari = [ARI(cons, r) for r in runs]
    print(f"{name}: natural communities, seed vs consensus ARI mean {np.mean(ari):.3f} min {min(ari):.3f}")

# %% 2. What moves in the two weakest subsample draws (galaxy ARI in metrics.json checks)
a = pd.read_parquet(OUT / "assignments.parquet")
draws = json.loads((OUT / "metrics.json").read_text())["checks"]["subsample"]
for s in sorted(draws, key=lambda s: draws[s]["galaxy_ari"])[:2]:
    keep, sub = C.subsample(df, cfg["checks"]["subsample"], int(s))
    o, _, sats = R.build(sub, X[keep], R.edge_layers(sub, X[keep], cfg["edges"]), cfg)
    print(f"draw {s}\n", pd.crosstab(a.galaxy.iloc[keep].to_numpy(), o.galaxy_id.to_numpy()).to_string())
    print([(r["satellite"], r["joined"], r["route"], r["ratio"]) for r in sats])
