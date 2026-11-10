"""Dyadic homophily-in-ties test controlling for space and time.
Dyads: pairs of named-culture individuals in the same 500-yr bin at different localities.
Y = IBD link (any segment >=12 cM with total >= TH cM). X = same culture label.
Statistic: Mantel-Haenszel OR of Y on X, stratified by (bin x distance-decile).
Spatial null: X replaced by 'same cell of a random contiguous (Voronoi) partition' with K_t = #cultures in bin t.
Also DC-SBM description length of the culture@bin partition vs Voronoi partitions (same K) on the same subgraph.
usage: gt-python -I dyad.py <results_dir> <region> <TH> <n_null>"""
import sys, json, numpy as np, pandas as pd, graph_tool.all as gt
gt.openmp_set_num_threads(1)
sys.path.insert(0, sys.argv[0].rsplit("/",1)[0]); from common import *
RES, REG, TH, NN = sys.argv[1], sys.argv[2], float(sys.argv[3]), int(sys.argv[4])
rng = np.random.default_rng(2)
s = pd.read_csv(f"{RES}/{REG}_individuals.tsv", sep="\t")
s = s[s.is_culture].reset_index(drop=True)
e = pd.read_csv(f"{RES}/{REG}_ibd12_pairs.tsv", sep="\t"); e = e[e.sum12 >= TH]
E = set(map(tuple, np.sort(e[["iid1","iid2"]].values, axis=1)))
rows = []
for t, g in s.groupby("tbin"):
    ix = g.index.values
    if len(ix) < 2: continue
    i, j = np.triu_indices(len(ix), 1); a, b = ix[i], ix[j]
    rows.append(pd.DataFrame({"a": a, "b": b, "tbin": t}))
P = pd.concat(rows, ignore_index=True)
P = P[s.loc[P.a, "loc"].values != s.loc[P.b, "loc"].values].reset_index(drop=True)
P["d"] = hav(s.lat.values[P.a], s.lon.values[P.a], s.lat.values[P.b], s.lon.values[P.b])
ids = s.iid.values
P["y"] = [tuple(sorted((x, y))) in E for x, y in zip(ids[P.a], ids[P.b])]
P["x"] = s.culture.values[P.a] == s.culture.values[P.b]
P["dq"] = pd.qcut(P.d, 10, labels=False, duplicates="drop")
P["dt"] = np.abs(s.age.values[P.a] - s.age.values[P.b])
P["dtq"] = pd.cut(P.dt, [-1, 100, 250, 1e9], labels=False)
P["samestudy"] = s.study.values[P.a] == s.study.values[P.b]
STRAT = sys.argv[5] if len(sys.argv) > 5 else "bin_d"
P["stratum"] = P.tbin.astype(str) + "_" + P.dq.astype(str)
if "dt" in STRAT: P["stratum"] += "_" + P.dtq.astype(str)
if "study" in STRAT: P["stratum"] += "_" + P.samestudy.astype(str)
def mh(x, y, st):
    df = pd.DataFrame({"x": x, "y": y, "s": st}); num = den = 0.0
    for _, g in df.groupby("s"):
        n = len(g); A = ((g.x) & (g.y)).sum(); B = ((g.x) & (~g.y)).sum(); C = ((~g.x) & (g.y)).sum(); Dd = ((~g.x) & (~g.y)).sum()
        num += A*Dd/n; den += B*C/n
    return num/den if den > 0 else np.nan
obs = mh(P.x.values, P.y.values, P.stratum.values)
Kt = s.groupby("tbin").culture.nunique().to_dict()
nul = []
for _ in range(NN):
    lab = voronoi_labels(s, Kt, rng)
    nul.append(mh(lab[P.a] == lab[P.b], P.y.values, P.stratum.values))
nul = np.array(nul)
# per-distance-band raw rates
band = pd.cut(P.d, [0, 50, 100, 200, 400, 800, 5000])
rates = P.groupby([band, "x"], observed=True).y.agg(["mean", "size"]).unstack()
# DL comparison on subgraph of named individuals (all bins, all localities)
idx = {v: k for k, v in enumerate(ids)}
ee = e[e.iid1.isin(idx) & e.iid2.isin(idx)]
G = gt.Graph(directed=False); G.add_vertex(len(s)); G.add_edge_list(np.c_[ee.iid1.map(idx), ee.iid2.map(idx)])
def dl(lab):
    st = gt.BlockState(G, b=G.new_vp("int", vals=codes(lab)), deg_corr=True); return st.entropy()
cb = (s.culture + "@" + s.tbin.astype(str)).values
dl_c = dl(cb); dl_loc = dl(s["loc"].values); dl_cl = dl((s.culture + "@" + s["loc"]).values); dl_v = np.array([dl(voronoi_labels(s, Kt, rng)) for _ in range(min(NN, 200))])
gt.seed_rng(0); st = gt.minimize_blockmodel_dl(G, state_args=dict(deg_corr=True)); dl_opt = st.entropy()
out = {"region": REG, "TH": TH, "n_dyads": int(len(P)), "n_links": int(P.y.sum()), "frac_same_culture": float(P.x.mean()),
       "MH_OR_same_culture": obs, "null_MH_OR_mean": float(np.nanmean(nul)), "null_q95": float(np.nanquantile(nul, .95)),
       "p_spatial_null": float(((nul >= obs).sum()+1)/(NN+1)),
       "strata": STRAT, "DL_culture@bin": dl_c, "DL_locality": dl_loc, "DL_culture@locality": dl_cl, "B_locality": int(s["loc"].nunique()), "DL_voronoi_mean": float(dl_v.mean()), "DL_voronoi_min": float(dl_v.min()),
       "DL_inferred_flat_DCSBM": dl_opt, "B_inferred": int(st.get_nonempty_B()), "B_culture@bin": int(len(set(cb)))}
print(json.dumps(out, indent=1, default=float)); print(rates.to_string())
json.dump(out, open(f"{RES}/{REG}_dyad_th{int(TH)}_{STRAT}.json", "w"), indent=1, default=float)
rates.to_csv(f"{RES}/{REG}_dyad_rates_th{int(TH)}_{STRAT}.tsv", sep="\t")
