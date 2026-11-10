"""Cross-lagged dyadic test of temporal order (network-first vs homophily-first).
Dyads (a,b): named-culture individuals in bin t' at different localities. Their 'past' = named
individuals in the previous (older) 500-yr bin t within R km of a's / b's locality.
  G_lag = density of IBD links (>=12cM) between past(a) and past(b)
  C_lag = fraction of pairs in past(a) x past(b) sharing a culture label
Models (logistic, bin fixed effects, log distance):
  network-first   : sameCulture(a,b)  ~ G_lag + C_lag + log d
  homophily-first : IBD(a,b)          ~ C_lag + G_lag + log d
Spatial nulls: G_lag recomputed from distance-decay random graphs fitted per bin;
C_lag recomputed from random contiguous (Voronoi) partitions with K_t = #cultures in bin t.
usage: gt-python -I lagged.py <results_dir> <region> <R_km> <n_null>"""
import os, sys, json, numpy as np, pandas as pd
sys.path.insert(0, sys.argv[0].rsplit("/",1)[0]); from common import *
RES, REG, RKM, NN = sys.argv[1], sys.argv[2], float(sys.argv[3]), int(sys.argv[4])
rng = np.random.default_rng(3)
s = pd.read_csv(f"{RES}/{REG}_individuals.tsv", sep="\t"); s = s[s.is_culture].reset_index(drop=True)
e = pd.read_csv(f"{RES}/{REG}_ibd12_pairs.tsv", sep="\t")
idx = dict(zip(s.iid, s.index)); e = e[e.iid1.isin(idx) & e.iid2.isin(idx)]
n = len(s); A = np.zeros((n, n), bool); A[e.iid1.map(idx), e.iid2.map(idx)] = True; A |= A.T
D = hav(s.lat.values[:,None], s.lon.values[:,None], s.lat.values[None,:], s.lon.values[None,:])
sameloc = s["loc"].values[:,None] == s["loc"].values[None,:]
cult = codes(s.culture.values)
bins = sorted(s.tbin.unique())
def logit(X, y, it=50):
    X = np.c_[np.ones(len(y)), X]; b = np.zeros(X.shape[1])
    for _ in range(it):
        p = 1/(1+np.exp(-X@b)); W = p*(1-p)+1e-9
        H = X.T@(X*W[:,None]) + 1e-6*np.eye(X.shape[1]); g = X.T@(y-p); db = np.linalg.solve(H, g); b += db
        if np.abs(db).max() < 1e-8: break
    return b
def null_graph():
    """distance-decay random graph per bin (within-bin pairs only; cross-bin entries irrelevant here)"""
    B = np.zeros((n, n), bool)
    for t in bins:
        ix = np.where(s.tbin.values == t)[0]
        if len(ix) < 2: continue
        i, j = np.triu_indices(len(ix), 1); a, b = ix[i], ix[j]
        y = A[a, b].astype(float); X = np.c_[np.log1p(D[a, b]), sameloc[a, b]]
        if y.sum() == 0: continue
        beta = logit(X, y); p = 1/(1+np.exp(-(beta[0] + X@beta[1:])))
        r = rng.random(len(p)) < p; B[a[r], b[r]] = True
    return B | B.T
Kt = s.groupby("tbin").culture.nunique().to_dict()
# build dyad table and neighbourhood index lists
rows, nb = [], {}
for t_old, t_new in zip(bins[1:], bins[:-1]):   # bins ascending in BP: older = larger
    if t_old - t_new != 500: continue
    new = np.where(s.tbin.values == t_new)[0]; old = np.where(s.tbin.values == t_old)[0]
    if len(new) < 2 or len(old) < 2: continue
    for a in new: nb[a] = old[D[a, old] <= RKM]
    i, j = np.triu_indices(len(new), 1); a, b = new[i], new[j]
    ok = (~sameloc[a, b]) & np.array([len(nb[x]) > 0 for x in a]) & np.array([len(nb[x]) > 0 for x in b])
    for x, y in zip(a[ok], b[ok]): rows.append((x, y, t_new))
P = pd.DataFrame(rows, columns=["a","b","tbin"])
def lag_G(M): return np.array([M[np.ix_(nb[x], nb[y])].mean() for x, y in zip(P.a, P.b)])
def lag_C(c): return np.array([(c[nb[x]][:,None] == c[nb[y]][None,:]).mean() for x, y in zip(P.a, P.b)])
Gl, Cl = lag_G(A), lag_C(cult)
yC = (cult[P.a] == cult[P.b]).astype(float); yG = A[P.a, P.b].astype(float)
logd = np.log1p(D[P.a, P.b]);  # log1p: distinct localities can share coordinates (D=0)
FE = pd.get_dummies(P.tbin, drop_first=True).values.astype(float)
z = lambda v: (v - v.mean())/(v.std() + 1e-12)
def fit(y, x_cross, x_own): return logit(np.c_[z(x_cross), z(x_own), z(logd), FE], y)[1]
obs_nf, obs_hf = fit(yC, Gl, Cl), fit(yG, Cl, Gl)
nf_null, hf_null = [], []
for k in range(NN):
    nf_null.append(fit(yC, lag_G(null_graph()), Cl))
    hf_null.append(fit(yG, lag_C(codes(voronoi_labels(s, Kt, rng))), Gl))
nf_null, hf_null = np.array(nf_null), np.array(hf_null)
out = {"region": REG, "R_km": RKM, "n_dyads": int(len(P)), "bins_used": sorted(map(int, P.tbin.unique())),
       "n_individuals_t'": int(len(set(P.a) | set(P.b))), "rate_sameCulture": yC.mean(), "rate_IBD": yG.mean(),
       "frac_G_lag>0": float((Gl > 0).mean()), "mean_C_lag": float(Cl.mean()),
       "network_first_beta(G_lag->sameCulture)": obs_nf, "nf_null_mean": nf_null.mean(), "nf_null_sd": nf_null.std(),
       "nf_p": float(((nf_null >= obs_nf).sum()+1)/(NN+1)),
       "homophily_first_beta(C_lag->IBD)": obs_hf, "hf_null_mean": hf_null.mean(), "hf_null_sd": hf_null.std(),
       "hf_p": float(((hf_null >= obs_hf).sum()+1)/(NN+1)),
       "nf_z": (obs_nf - nf_null.mean())/nf_null.std(), "hf_z": (obs_hf - hf_null.mean())/hf_null.std()}
print(json.dumps(out, indent=1, default=float))
json.dump(out, open(f"{RES}/{REG}_lagged_R{int(RKM)}.json", "w"), indent=1, default=float)
