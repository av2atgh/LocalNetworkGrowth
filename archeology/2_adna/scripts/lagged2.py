"""Robustness version of lagged.py (cross-lagged dyadic test of temporal order).
Same dyads/variables as lagged.py, plus options:
  --noself        exclude self-pairs (x,x) from G_lag/C_lag when past neighbourhoods overlap
                  (in lagged.py a shared past individual adds 1 to C_lag and 0 to G_lag)
  --labels arch   drop individuals whose AADR group carries a genetic qualifier
                  (_o*, *EEF, numbered genetic sub-clusters like CordedWare_1)
  --drop T ...    drop dyads whose present bin t' is in the list (transition bins)
  --tie 12-20     ties = pairs whose total >=12cM sharing is in [12,20] cM (weak, mostly between-site)
  --nulls ...     nf nulls: decay (lagged.py), strat (G_lag permuted within strata of
                  bin x log-d decile x C_lag quartile x overlap), swap (per-bin degree- and
                  edge-length-class-preserving double-edge swaps of the past-bin graph);
                  hf nulls: voronoi (lagged.py), strat (C_lag permuted within bin x log-d x G_lag x overlap)
usage: gt-python -I lagged2.py <results_dir> <tag> [options]"""
import sys, json, argparse, numpy as np, pandas as pd
sys.path.insert(0, sys.argv[0].rsplit("/", 1)[0]); from common import *
ap = argparse.ArgumentParser()
ap.add_argument("res"); ap.add_argument("tag")
ap.add_argument("--region", default="CE"); ap.add_argument("--R", type=float, default=100)
ap.add_argument("--nn", type=int, default=100); ap.add_argument("--noself", action="store_true")
ap.add_argument("--labels", default="all"); ap.add_argument("--drop", type=int, nargs="*", default=[])
ap.add_argument("--tie", default="12"); ap.add_argument("--nulls", nargs="*", default=["decay", "strat", "swap", "voronoi", "hstrat"])
o = ap.parse_args()
rng = np.random.default_rng(3)
s = pd.read_csv(f"{o.res}/{o.region}_individuals.tsv", sep="\t"); s = s[s.is_culture]
if o.labels == "arch":
    gen = s.clst.str.contains(r"_o[A-Za-z0-9]*(?:\.|_|$)|EEF", regex=True) | s.label.str.contains(r"_\d+$", regex=True)
    s = s[~gen]
s = s.reset_index(drop=True)
e = pd.read_csv(f"{o.res}/{o.region}_ibd12_pairs.tsv", sep="\t")
if o.tie == "12-20": e = e[e.sum12 <= 20]
elif o.tie != "12": e = e[e.sum12 >= float(o.tie)]
idx = dict(zip(s.iid, s.index)); e = e[e.iid1.isin(idx) & e.iid2.isin(idx)]
n = len(s); A = np.zeros((n, n), bool); A[e.iid1.map(idx), e.iid2.map(idx)] = True; A |= A.T
D = hav(s.lat.values[:, None], s.lon.values[:, None], s.lat.values[None, :], s.lon.values[None, :])
sameloc = s["loc"].values[:, None] == s["loc"].values[None, :]
cult = codes(s.culture.values); tb = s.tbin.values
bins = sorted(s.tbin.unique())
def logit(X, y, it=60):
    X = np.c_[np.ones(len(y)), X]; b = np.zeros(X.shape[1])
    for _ in range(it):
        p = 1/(1+np.exp(-X@b)); W = p*(1-p)+1e-9
        H = X.T@(X*W[:, None]) + 1e-6*np.eye(X.shape[1]); db = np.linalg.solve(H, X.T@(y-p)); b += db
        if np.abs(db).max() < 1e-8: break
    return b
def decay_graph():
    B = np.zeros((n, n), bool)
    for t in bins:
        ix = np.where(tb == t)[0]
        if len(ix) < 2: continue
        i, j = np.triu_indices(len(ix), 1); a, b = ix[i], ix[j]
        y = A[a, b].astype(float); X = np.c_[np.log1p(D[a, b]), sameloc[a, b]]
        if y.sum() == 0: continue
        be = logit(X, y); p = 1/(1+np.exp(-(be[0] + X@be[1:])))
        r = rng.random(len(p)) < p; B[a[r], b[r]] = True
    return B | B.T
LCLS = np.array([0, 50, 100, 200, 400, 800, 1e9])
def lclass(a, b): return np.where(sameloc[a, b], -1, np.searchsorted(LCLS, D[a, b]))
def swap_graph(nsw=10):
    """per-bin double-edge swaps preserving degrees and each edge's length class (same locality, 0-50, 50-100, ... km)"""
    B = np.zeros((n, n), bool)
    for t in bins:
        ix = np.where(tb == t)[0]
        sub = A[np.ix_(ix, ix)]; i, j = np.where(np.triu(sub, 1))
        E = np.c_[ix[i], ix[j]]
        if len(E) < 2: B[E[:, 0], E[:, 1]] = True; continue
        Es = set(map(tuple, np.sort(E, 1))); cl = lclass(E[:, 0], E[:, 1])
        for _ in range(nsw*len(E)):
            k, l = rng.integers(len(E), size=2)
            if k == l: continue
            a, b = E[k]; c, d = E[l]
            if rng.random() < .5: c, d = d, c
            if len({a, b, c, d}) < 4: continue
            n1, n2 = tuple(sorted((a, d))), tuple(sorted((c, b)))
            if n1 in Es or n2 in Es: continue
            if lclass(a, d) != cl[k] or lclass(c, b) != cl[l]: continue
            Es.discard(tuple(sorted((a, b)))); Es.discard(tuple(sorted((c, d)))); Es.add(n1); Es.add(n2)
            E[k] = (a, d); E[l] = (c, b)
        B[E[:, 0], E[:, 1]] = True
    return B | B.T
Kt = s.groupby("tbin").culture.nunique().to_dict()
rows, nb = [], {}
for t_old, t_new in zip(bins[1:], bins[:-1]):
    if t_old - t_new != 500 or t_new in o.drop: continue
    new = np.where(tb == t_new)[0]; old = np.where(tb == t_old)[0]
    if len(new) < 2 or len(old) < 2: continue
    for a in new: nb[a] = old[D[a, old] <= o.R]
    i, j = np.triu_indices(len(new), 1); a, b = new[i], new[j]
    ok = (~sameloc[a, b]) & np.array([len(nb[x]) > 0 for x in a]) & np.array([len(nb[x]) > 0 for x in b])
    rows += list(zip(a[ok], b[ok], [t_new]*ok.sum()))
P = pd.DataFrame(rows, columns=["a", "b", "tbin"])
def masks():
    out = []
    for x, y in zip(P.a, P.b):
        u, v = nb[x], nb[y]; m = np.ones((len(u), len(v)), bool)
        if o.noself: m &= u[:, None] != v[None, :]
        out.append(m)
    return out
MK = masks()
def lag_G(M): return np.array([M[np.ix_(nb[x], nb[y])][m].mean() if m.any() else 0. for x, y, m in zip(P.a, P.b, MK)])
def lag_C(c): return np.array([(c[nb[x]][:, None] == c[nb[y]][None, :])[m].mean() if m.any() else 0. for x, y, m in zip(P.a, P.b, MK)])
keep = np.array([m.any() for m in MK]); P = P[keep].reset_index(drop=True); MK = [m for m, k in zip(MK, keep) if k]
overlap = np.array([len(np.intersect1d(nb[x], nb[y])) > 0 for x, y in zip(P.a, P.b)])
Gl, Cl = lag_G(A), lag_C(cult)
yC = (cult[P.a] == cult[P.b]).astype(float); yG = A[P.a, P.b].astype(float)
logd = np.log1p(D[P.a, P.b]); FE = pd.get_dummies(P.tbin, drop_first=True).values.astype(float)
z = lambda v: (v - v.mean())/(v.std() + 1e-12)
def fit(y, xc, xo): return logit(np.c_[z(xc), z(xo), z(logd), FE], y)[1]
def strata(other):
    q = lambda v, k: pd.qcut(pd.Series(v).rank(method="first"), k, labels=False).values
    return pd.Series(P.tbin.astype(str).values + "_" + q(logd, 10).astype(str) + "_" + q(other, 4).astype(str) + "_" + overlap.astype(str))
def strat_perm(v, st):
    w = v.copy()
    for _, g in pd.Series(range(len(v))).groupby(st.values): w[g.values] = rng.permutation(v[g.values])
    return w
obs_nf, obs_hf = fit(yC, Gl, Cl), fit(yG, Cl, Gl)
diag = {"corr(G_lag,logd)": float(np.corrcoef(Gl, logd)[0, 1]), "corr(G_lag,C_lag)": float(np.corrcoef(Gl, Cl)[0, 1]),
        "corr(C_lag,logd)": float(np.corrcoef(Cl, logd)[0, 1]), "frac_overlap": float(overlap.mean()),
        "nf_beta_G_only(+FE)": float(logit(np.c_[z(Gl), FE], yC)[1]),
        "nf_beta_G+logd(+FE)": float(logit(np.c_[z(Gl), z(logd), FE], yC)[1]),
        "nf_beta_full+overlap": float(logit(np.c_[z(Gl), z(Cl), z(logd), overlap, FE], yC)[1]),
        "corr(G_lag,C_lag|overlap)": float(np.corrcoef(Gl[overlap], Cl[overlap])[0, 1]),
        "corr(G_lag,C_lag|no overlap)": float(np.corrcoef(Gl[~overlap], Cl[~overlap])[0, 1]) if (~overlap).sum() > 2 else None}
res = {}
for nm in o.nulls:
    vals, cors, b1, b2 = [], [], [], []
    for k in range(o.nn):
        if nm in ("decay", "swap"):
            g = lag_G(decay_graph() if nm == "decay" else swap_graph()); vals.append(fit(yC, g, Cl)); cors.append(np.corrcoef(g, logd)[0, 1])
            b1.append(logit(np.c_[z(g), FE], yC)[1]); b2.append(logit(np.c_[z(g), z(logd), FE], yC)[1])
        elif nm == "strat": vals.append(fit(yC, strat_perm(Gl, strata(Cl)), Cl))
        elif nm == "voronoi": vals.append(fit(yG, lag_C(codes(voronoi_labels(s, Kt, rng))), Gl))
        elif nm == "hstrat": vals.append(fit(yG, strat_perm(Cl, strata(Gl)), Gl))
    vals = np.array(vals); ob = obs_hf if nm in ("voronoi", "hstrat") else obs_nf
    res[nm] = {"dir": "hf" if nm in ("voronoi", "hstrat") else "nf", "null_mean": vals.mean(), "null_sd": vals.std(),
               "z": (ob - vals.mean())/vals.std(), "p": float(((vals >= ob).sum()+1)/(len(vals)+1))}
    if cors: res[nm].update({"null_corr(G_lag,logd)": float(np.mean(cors)), "null_beta_G_only": float(np.mean(b1)), "null_beta_G+logd": float(np.mean(b2))})
out = {"tag": o.tag, "args": vars(o), "n_dyads": int(len(P)), "n_ind": int(n), "n_edges": int(A.sum()//2),
       "bins_t'": sorted(map(int, P.tbin.unique())), "rate_sameCulture": yC.mean(), "rate_IBD": yG.mean(),
       "beta_nf(G_lag->sameCulture)": obs_nf, "beta_hf(C_lag->IBD)": obs_hf, "diag": diag, "nulls": res}
print(json.dumps(out, indent=1, default=lambda x: round(float(x), 4)))
json.dump(out, open(f"{o.res}/lagged2_{o.tag}.json", "w"), indent=1, default=float)
