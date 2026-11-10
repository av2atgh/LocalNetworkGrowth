"""(ii) persistence / path dependence and (iii) temporal order, pair level, consecutive slices.

For slices t, t+lag with common sites (both in the LCC of the thresholded graph):
  P  : logit same_comm(t+1) ~ same_comm(t) + distance-decile dummies + same_region      -> OR_persist
  Cf : s(t+1)-s(t)   ~ s(t)   + same_comm(t)   + controls                              -> beta_fwd
  Cb : s(t)-s(t+1)   ~ s(t+1) + same_comm(t+1) + controls   (time-reversed)            -> beta_bwd
  R  : logit same_comm(t+1) ~ same_comm(t) + z[s(t)] + controls                        -> gamma (sim -> later comm)
CIs: node bootstrap (sites resampled, pair weight w_i w_j), partitions held fixed.
Nulls:
  geo  : independent dist_attr null graphs at t and t+1 (no memory, same geography/regions) -> OR, beta
  stat : (Chaco only, counts available) stationary-truth resampling: both slices redrawn
         multinomially from the pooled composition with each slice's sample sizes; whole pipeline rerun.
"""
import os, sys
for _v in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:
    os.environ.setdefault(_v, '1')
import pickle, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from scale_detect import dmat
from multiprocessing import Pool

THR = float(os.environ.get('THR', 0.75))
NBOOT, NNULL = 200, int(os.environ.get('NNULL', 20))

def partition(S, seed, nrep=5):
    A = threshold(S, THR)
    B, dl, dl1, b, m = sbm_fit(A, nrep=nrep, seed=seed)
    lab = -np.ones(len(S), int); lab[m] = b
    return lab

def wls(X, y, w):
    W = np.sqrt(w)[:, None]
    beta, *_ = np.linalg.lstsq(X * W, y * W[:, 0], rcond=None)
    return beta

def logit(X, y, w, it=50):
    b = np.zeros(X.shape[1])
    for _ in range(it):
        eta = np.clip(X @ b, -30, 30); p = 1 / (1 + np.exp(-eta))
        H = (X * (w * p * (1 - p) + 1e-9)[:, None]).T @ X + 1e-6 * np.eye(X.shape[1])
        g = X.T @ (w * (y - p))
        step = np.linalg.solve(H, g); b += step
        if np.abs(step).max() < 1e-8: break
    return b

def design(d, same_reg, extra):
    qs = np.quantile(d, np.linspace(0, 1, 11)[1:-1])
    bins = np.searchsorted(qs, d)
    Db = np.eye(10)[bins][:, 1:]
    return np.column_stack([np.ones(len(d))] + extra + [Db, same_reg])

def models(I, J, s0, s1, c0, c1, d, sr, w):
    sc0 = (c0[I] == c0[J]).astype(float); sc1 = (c1[I] == c1[J]).astype(float)
    a, b = s0[I, J], s1[I, J]
    out = {}
    X = design(d, sr, [sc0]); out['logOR_persist'] = logit(X, sc1, w)[1]
    X = design(d, sr, [a, sc0]); out['beta_fwd'] = wls(X, b - a, w)[2]
    X = design(d, sr, [b, sc1]); out['beta_bwd'] = wls(X, a - b, w)[2]
    z = (a - a.mean()) / a.std()
    X = design(d, sr, [sc0, z]); out['gamma_sim'] = logit(X, sc1, w)[2]
    out['npairs'] = len(I)
    return out

def analyse_pair(v0, v1, c0_full, c1_full, rng, boot=True):
    common = np.intersect1d(v0['sites'], v1['sites'])
    i0 = np.array([np.where(v0['sites'] == s)[0][0] for s in common])
    i1 = np.array([np.where(v1['sites'] == s)[0][0] for s in common])
    c0, c1 = c0_full[i0], c1_full[i1]
    keep = (c0 >= 0) & (c1 >= 0)
    i0, i1, c0, c1 = i0[keep], i1[keep], c0[keep], c1[keep]
    n = len(i0)
    if n < 10: return None
    s0 = v0['S'][np.ix_(i0, i0)]; s1 = v1['S'][np.ix_(i1, i1)]
    D = dmat(v0['xy'][i0]); reg = v0['attr'][i0]
    I, J = np.triu_indices(n, 1)
    d = D[I, J]; sr = (reg[I] == reg[J]).astype(float)
    res = models(I, J, s0, s1, c0, c1, d, sr, np.ones(len(I)))
    res['n'] = n; res['ARI'] = ari(c0, c1); res['NMI'] = nmi(c0, c1)
    res['ARI_reg_t'] = ari(c0, reg)
    if boot:
        bs = []
        for _ in range(NBOOT):
            cnt = np.bincount(rng.integers(n, size=n), minlength=n).astype(float)
            w = cnt[I] * cnt[J]; k = w > 0
            bs.append(models(I[k], J[k], s0, s1, c0, c1, d[k], sr[k], w[k]))
        res['boot'] = {key: np.percentile([b[key] for b in bs], [2.5, 97.5]) for key in bs[0] if key != 'npairs'}
    return res

def geo_null(v, rng):
    A = threshold(v['S'], THR); D = dmat(v['xy'])
    return null_probs(A, D, v['attr'], 'dist_attr')

def stationary_pair(v0, v1, rng):
    """Redraw both slices from pooled composition of common sites (stationary truth)."""
    common = np.intersect1d(v0['sites'], v1['sites'])
    i0 = np.array([np.where(v0['sites'] == s)[0][0] for s in common])
    i1 = np.array([np.where(v1['sites'] == s)[0][0] for s in common])
    # align ware columns: counts arrays have slice-specific ware sets -> use stored ware names
    X0 = v0['wcounts'].loc[common]; X1 = v1['wcounts'].loc[common]
    cols = X0.columns.union(X1.columns)
    X0 = X0.reindex(columns=cols, fill_value=0).values; X1 = X1.reindex(columns=cols, fill_value=0).values
    pool = X0 + X1; p = pool / pool.sum(1, keepdims=True)
    n0 = np.maximum(np.rint(X0.sum(1)), 10).astype(int); n1 = np.maximum(np.rint(X1.sum(1)), 10).astype(int)
    Y0 = np.array([rng.multinomial(k, q) for k, q in zip(n0, p)]); Y1 = np.array([rng.multinomial(k, q) for k, q in zip(n1, p)])
    w0 = dict(sites=common, S=br_similarity(Y0), xy=v0['xy'][i0], attr=v0['attr'][i0])
    w1 = dict(sites=common, S=br_similarity(Y1), xy=v1['xy'][i1], attr=v1['attr'][i1])
    return w0, w1

def job(args):
    ds, p0, p1, v0, v1 = args
    rng = np.random.default_rng(abs(hash((ds, p0, p1))) % 2**32)
    c0 = partition(v0['S'], 7); c1 = partition(v1['S'], 11)
    obs = analyse_pair(v0, v1, c0, c1, rng)
    if obs is None: return None
    # self-consistency of the partition (two independent fits of the same slice)
    obs['ARI_selfcons_t'] = ari(c0[c0 >= 0], partition(v0['S'], 99)[c0 >= 0])
    P0, P1 = geo_null(v0, rng), geo_null(v1, rng)
    geo = []
    for k in range(NNULL):
        A0 = sample_from_probs(P0, len(v0['S']), rng); A1 = sample_from_probs(P1, len(v1['S']), rng)
        # null 'similarity' = adjacency (binary); partitions from SBM on the null graphs
        u0 = dict(v0, S=A0.astype(float) + np.eye(len(A0))); u1 = dict(v1, S=A1.astype(float) + np.eye(len(A1)))
        q0 = partition(u0['S'], 200 + k, nrep=2); q1 = partition(u1['S'], 300 + k, nrep=2)
        r = analyse_pair(dict(v0, S=u0['S']), dict(v1, S=u1['S']), q0, q1, rng, boot=False)
        if r: geo.append(r)
    statn = []
    if 'wcounts' in v0:
        for k in range(NNULL):
            w0, w1 = stationary_pair(v0, v1, rng)
            q0 = partition(w0['S'], 400 + k, nrep=2); q1 = partition(w1['S'], 500 + k, nrep=2)
            r = analyse_pair(w0, w1, q0, q1, rng, boot=False)
            if r: statn.append(r)
    print(ds, p0, p1, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in obs.items() if k != 'boot'}, flush=True)
    return dict(ds=ds, p0=p0, p1=p1, obs=obs, geo=geo, stat=statn)

if __name__ == '__main__':
    import pandas as pd
    d = pickle.load(open(os.path.join(RES, 'slices.pkl'), 'rb'))
    # attach ware-count tables for Chaco (stationary null)
    c = pd.read_csv(os.path.join(DATA, 'chaco_timeavg', 'AllCeramics.csv'), index_col=0)
    P = [x for x in c.columns if x.startswith('P')]
    for per, v in d['chaco'].items():
        k = P.index(f'P{per}')
        tab = c.assign(val=c[P[k]] + c[P[k + 1]]).pivot_table(index='SWSN_ID', columns='SWSN_Ware', values='val', aggfunc='sum', fill_value=0)
        v['wcounts'] = tab.loc[v['sites']]
    jobs = []
    for ds in d:
        pers = sorted(p for p in d[ds] if len(d[ds][p]['S']) >= 20)
        for lag in (1, 2):
            for a, b in zip(pers, pers[lag:]):
                jobs.append((ds, a, b, d[ds][a], d[ds][b]))
    jobs = jobs[:int(os.environ.get('NJOBS', len(jobs)))]  # NJOBS=1 for a smoke test
    part = os.path.join(RES, f'dyn_parts_thr{THR}' + ('' if NNULL == 20 else f'_null{NNULL}')); os.makedirs(part, exist_ok=True)
    out = []
    for j in jobs:  # single process, checkpointed
        f = os.path.join(part, f'{j[0]}_{j[1]}_{j[2]}.pkl')
        if os.path.exists(f):
            out.append(pickle.load(open(f, 'rb'))); continue
        r = job(j); pickle.dump(r, open(f, 'wb')); out.append(r)
    pickle.dump([o for o in out if o], open(os.path.join(RES, f'dynamics_thr{THR}' + ('' if NNULL == 20 else f'_null{NNULL}') + '.pkl'), 'wb'))
