import numpy as np, pandas as pd
R = 6371.0
def hav(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    return 2*R*np.arcsin(np.sqrt(np.sin((lat2-lat1)/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin((lon2-lon1)/2)**2))
def entropy(c):
    p = np.bincount(c)/len(c); p = p[p > 0]; return -(p*np.log(p)).sum()
def codes(x): return pd.factorize(pd.Series(x).astype(str))[0]
def mi(a, b):
    a, b = codes(a), codes(b); n = len(a)
    M = np.zeros((a.max()+1, b.max()+1)); np.add.at(M, (a, b), 1); P = M/n
    pa, pb = P.sum(1, keepdims=True), P.sum(0, keepdims=True); nz = P > 0
    return (P[nz]*np.log(P[nz]/(pa@pb)[nz])).sum(), entropy(a), entropy(b)
def nmi(a, b):
    I, Ha, Hb = mi(a, b); return 0.0 if Ha+Hb == 0 else 2*I/(Ha+Hb)
def ari(a, b):
    a, b = codes(a), codes(b)
    M = np.zeros((a.max()+1, b.max()+1)); np.add.at(M, (a, b), 1)
    c2 = lambda x: (x*(x-1)/2).sum(); n = len(a)
    s, sa, sb = c2(M), c2(M.sum(1)), c2(M.sum(0)); e = sa*sb/(n*(n-1)/2); mx = (sa+sb)/2
    return (s-e)/(mx-e) if mx != e else 0.0
def voronoi_labels(df, kcol_counts, rng):
    """Random spatially contiguous partition within each time bin: K_t random seed localities,
    every locality assigned to nearest seed. df needs tbin, loc, lat, lon. kcol_counts: dict tbin->K."""
    out = pd.Series(index=df.index, dtype=object)
    for t, g in df.groupby("tbin"):
        locs = g.groupby("loc")[["lat","lon"]].first()
        K = min(kcol_counts.get(t, 1), len(locs))
        seeds = locs.iloc[rng.choice(len(locs), K, replace=False)]
        d = hav(locs.lat.values[:,None], locs.lon.values[:,None], seeds.lat.values[None,:], seeds.lon.values[None,:])
        lab = dict(zip(locs.index, [f"{t}_{k}" for k in d.argmin(1)]))
        out[g.index] = g["loc"].map(lab).values
    return out.values
