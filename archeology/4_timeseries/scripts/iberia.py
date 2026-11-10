"""Re-analysis of Rosas-Casals et al. 2027 (JAMT 34:30) ESM: 100 sites x 12 economic categories,
chronology EBA / MBA / EMBA (persisting), settlement form, lat/lon.
Edges = number of shared categories (as in the paper). Tests:
 1. densities (paper: ~0.93) and weight distribution;
 2. structure beyond a presence/absence null: curveball null preserving each site's number of categories
    and each category's prevalence; statistic = DC-SBM B and Sigma on the graph 'shared >= k' (k chosen so
    density ~0.3), and the variance of the weighted degree-corrected similarity;
 3. geography and form: Mantel-type permutation correlation between shared categories and distance; mean
    shared within vs between forms, with category-curveball and label-permutation nulls.
"""
import os, sys, numpy as np, pandas as pd
os.environ.setdefault('OMP_NUM_THREADS', '1')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

f = os.path.join(DATA, 'rosascasals2027', 'MOESM1_ESM.csv')
t = pd.read_csv(f, sep=';', decimal=',', encoding='latin-1')
cats = [c for c in t.columns if '(' in c and c != 'Size (ha)']
X = t[cats].values.astype(int)
rng = np.random.default_rng(1)

def curveball(M, rng, nsw=None):
    M = M.copy(); rows = [set(np.nonzero(r)[0]) for r in M]; n = len(rows)
    for _ in range(nsw or 5 * n):
        a, b = rng.choice(n, 2, replace=False)
        A, B = rows[a], rows[b]; ua = list(A - B); ub = list(B - A)
        if not ua and not ub: continue
        pool = ua + ub; rng.shuffle(pool); k = len(ua)
        rows[a] = (A & B) | set(pool[:k]); rows[b] = (A & B) | set(pool[k:])
    out = np.zeros_like(M)
    for i, r in enumerate(rows): out[i, list(r)] = 1
    return out

def haversine(lat, lon):
    la, lo = np.radians(lat), np.radians(lon)
    dla = la[:, None] - la[None]; dlo = lo[:, None] - lo[None]
    h = np.sin(dla / 2) ** 2 + np.cos(la[:, None]) * np.cos(la[None]) * np.sin(dlo / 2) ** 2
    return 2 * 6371 * np.arcsin(np.sqrt(h))

def sbm_stat(W, k):
    A = (W >= k).astype(np.int8); np.fill_diagonal(A, 0)
    B, dl, dl1, b, m = sbm_fit(A, nrep=3, seed=3)
    return B, (dl1 - dl) / max(A.sum() / 2, 1), A[np.triu_indices(len(A), 1)].mean()

sets = {'EBA (paper n=20)': t.Cronology == 'EBA', 'MBA (paper n=45)': t.Cronology == 'MBA',
        'EMBA (n=35)': t.Cronology == 'EMBA', 'EBA+EMBA (n=55)': t.Cronology.isin(['EBA', 'EMBA']),
        'MBA+EMBA (n=80)': t.Cronology.isin(['MBA', 'EMBA']), 'all (n=100)': t.Cronology.notna()}
lines = []
for name, msk in sets.items():
    M = X[msk.values]; n = len(M)
    W = M @ M.T; np.fill_diagonal(W, 0)
    iu = np.triu_indices(n, 1); w = W[iu]
    dens = (w > 0).mean()
    # choose k with density closest to 0.3
    ks = range(1, 13); k = min(ks, key=lambda k: abs((w >= k).mean() - 0.3))
    LIGHT = os.environ.get('LIGHT') == '1'
    B, sig, dk = sbm_stat(W, k) if not LIGHT else (-1, np.nan, (w >= k).mean())
    nullB, nullsig = [], []
    for r in range(0 if LIGHT else 50):
        Mn = curveball(M, rng); Wn = Mn @ Mn.T; np.fill_diagonal(Wn, 0)
        b_, s_, _ = sbm_stat(Wn, k); nullB.append(b_); nullsig.append(s_)
    D = haversine(t.Latitude.values[msk.values], t.Longitude.values[msk.values])
    ok = np.isfinite(D[iu])
    r_obs = np.corrcoef(w[ok], D[iu][ok])[0, 1]
    r_perm = []
    for _ in range(999):
        p = rng.permutation(n); r_perm.append(np.corrcoef(W[np.ix_(p, p)][iu][ok], D[iu][ok])[0, 1])
    p_mantel = (np.sum(np.array(r_perm) <= r_obs) + 1) / 1000
    form = t.Form.values[msk.values]; same = (form[iu[0]] == form[iu[1]])
    dform = w[same].mean() - w[~same].mean()
    dperm = []
    for _ in range(999):
        fp = rng.permutation(form); s2 = fp[iu[0]] == fp[iu[1]]; dperm.append(w[s2].mean() - w[~s2].mean())
    p_form = (np.sum(np.array(dperm) >= dform) + 1) / 1000
    cb = [curveball(M, rng) for _ in range(200)]
    wv = [np.var((m @ m.T)[iu]) for m in cb]
    lines.append(f"  var(w) obs={w.var():.3f} curveball={np.mean(wv):.3f}+-{np.std(wv):.3f} (p_high={(np.sum(np.array(wv)>=w.var())+1)/201:.3f})")
    lines.append(f"{name:18s} n={n:3d} density(w>0)={dens:.3f} mean_w={w.mean():.2f} | k={k} dens_k={dk:.2f} "
                 f"SBM B={B} Sigma={sig:.3f}; curveball null: P(B>=2)={np.mean(np.array(nullB)>=2):.2f} "
                 f"Sigma={np.mean(nullsig):.3f}+-{np.std(nullsig):.3f} | Mantel r(w,dist)={r_obs:+.3f} p_low={p_mantel:.3f} "
                 f"| within-minus-between form={dform:+.2f} p={p_form:.3f}")
    print(lines[-1], flush=True)
open(os.path.join(RES, 'iberia_light.txt' if os.environ.get('LIGHT') == '1' else 'iberia.txt'), 'w').write('\n'.join(lines) + '\n')
