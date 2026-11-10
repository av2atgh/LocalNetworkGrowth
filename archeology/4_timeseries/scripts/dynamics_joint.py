"""Chaco: (1) joint stationary null for the forward/backward asymmetry pooled over slice pairs;
(2) contact test: does within-community convergence decay with distance?

(1) Each draw redraws EVERY Chaco slice from the site's composition pooled over all slices in which
    the site appears (a stationary truth for the whole series), with the observed sample sizes,
    re-partitions every slice once, and evaluates all pairs on the same draw. The null distribution of
    the pooled statistic (mean of fwd-bwd over the 8 pairs with n>=30; lag-1 and lag-2 means
    separately) therefore carries the dependence between overlapping pairs.
(2) Forward model with interaction: ds ~ s_t + same_t + same_t * z(log d) + controls. Contact
    predicts a negative interaction (convergence strongest between nearby community members); a
    region-wide tradition predicts none. The same interaction is computed in the time-reversed model
    and under the joint null.
Usage: NNULL=200 python -I dynamics_joint.py
"""
import os, sys
for _v in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS']:
    os.environ.setdefault(_v, '1')
import pickle, json, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from scale_detect import dmat
import dynamics as dy

NNULL = int(os.environ.get('NNULL', 200))
PAIRS = [(950, 1000), (1000, 1050), (1050, 1100), (1100, 1150),
         (950, 1050), (1000, 1100), (1050, 1150), (1100, 1200)]

def pair_stats(v0, v1, c0f, c1f):
    common = np.intersect1d(v0['sites'], v1['sites'])
    i0 = np.array([np.where(v0['sites'] == s)[0][0] for s in common])
    i1 = np.array([np.where(v1['sites'] == s)[0][0] for s in common])
    c0, c1 = c0f[i0], c1f[i1]; k = (c0 >= 0) & (c1 >= 0)
    i0, i1, c0, c1 = i0[k], i1[k], c0[k], c1[k]; n = len(i0)
    if n < 10:
        return dict(n=n, fwd=np.nan, bwd=np.nan, asym=np.nan, fwd_int=np.nan, bwd_int=np.nan)
    s0 = v0['S'][np.ix_(i0, i0)]; s1 = v1['S'][np.ix_(i1, i1)]
    D = dmat(v0['xy'][i0]); reg = v0['attr'][i0]
    I, J = np.triu_indices(n, 1); d = D[I, J]; sr = (reg[I] == reg[J]).astype(float)
    a, b = s0[I, J], s1[I, J]
    sc0 = (c0[I] == c0[J]).astype(float); sc1 = (c1[I] == c1[J]).astype(float)
    w = np.ones(len(I))
    X = dy.design(d, sr, [a, sc0]); fwd = dy.wls(X, b - a, w)[2]
    X = dy.design(d, sr, [b, sc1]); bwd = dy.wls(X, a - b, w)[2]
    zl = np.log1p(d); zl = (zl - zl.mean()) / zl.std()
    X = dy.design(d, sr, [a, sc0, sc0 * zl, zl]); fi = dy.wls(X, b - a, w)[3]
    X = dy.design(d, sr, [b, sc1, sc1 * zl, zl]); bi = dy.wls(X, a - b, w)[3]
    return dict(n=n, fwd=fwd, bwd=bwd, asym=fwd - bwd, fwd_int=fi, bwd_int=bi)

def summarise(rows):
    a = np.array([r['asym'] for r in rows]); fi = np.array([r['fwd_int'] for r in rows]); bi = np.array([r['bwd_int'] for r in rows])
    m = np.nanmean
    return dict(asym_all=m(a), asym_lag1=m(a[:4]), asym_lag2=m(a[4:]),
                int_fwd=m(fi), int_bwd=m(bi), int_diff=m(fi - bi), n_missing=float(np.isnan(a).sum()))

def run_all(sl, seeds):
    parts = {p: dy.partition(sl[p]['S'], seeds[p], nrep=seeds.get('nrep', 5)) for p in sl}
    return [pair_stats(sl[p0], sl[p1], parts[p0], parts[p1]) for p0, p1 in PAIRS]

if __name__ == '__main__':
    d = pickle.load(open(os.path.join(RES, 'slices.pkl'), 'rb'))['chaco']
    c = pd.read_csv(os.path.join(DATA, 'chaco_timeavg', 'AllCeramics.csv'), index_col=0)
    P = [x for x in c.columns if x.startswith('P')]
    pers = sorted({p for pr in PAIRS for p in pr})
    tabs = {}
    for per in pers:
        k = P.index(f'P{per}')
        tabs[per] = c.assign(val=c[P[k]] + c[P[k + 1]]).pivot_table(index='SWSN_ID', columns='SWSN_Ware', values='val', aggfunc='sum', fill_value=0).loc[d[per]['sites']]
    cols = sorted(set().union(*[t.columns for t in tabs.values()]))
    tabs = {p: t.reindex(columns=cols, fill_value=0) for p, t in tabs.items()}
    pooled = pd.concat(tabs.values()).groupby(level=0).sum()
    prob = pooled.div(pooled.sum(1), axis=0)
    sl = {p: d[p] for p in pers}
    obs_rows = run_all(sl, {p: 7 for p in pers})
    obs = summarise(obs_rows)
    rng = np.random.default_rng(2026)
    null = []
    for k in range(NNULL):
        sk = {}
        for p in pers:
            n_s = np.maximum(np.rint(tabs[p].values.sum(1)), 10).astype(int)
            q = prob.loc[d[p]['sites']].values
            Y = np.array([rng.multinomial(m, qq) for m, qq in zip(n_s, q)])
            sk[p] = dict(d[p], S=br_similarity(Y))
        seeds = {p: 1000 + 10 * k + i for i, p in enumerate(pers)}; seeds['nrep'] = 2
        null.append(summarise(run_all(sk, seeds)))
        if k % 20 == 0: print('null', k, flush=True)
    out = dict(pairs=[f'{a}-{b}' for a, b in PAIRS], obs_rows=[{kk: float(v) for kk, v in r.items()} for r in obs_rows], obs={kk: float(v) for kk, v in obs.items()})
    out['null_pairs_missing_mean'] = float(np.mean([r['n_missing'] for r in null]))
    for key in [k for k in obs if k != 'n_missing']:
        nv = np.array([r[key] for r in null]); nv = nv[np.isfinite(nv)]
        if key.startswith('int'):
            p = (1 + np.sum(nv <= obs[key])) / (1 + len(nv))  # contact predicts negative
        else:
            p = (1 + np.sum(nv >= obs[key])) / (1 + len(nv))
        out[key] = dict(obs=round(float(obs[key]), 4), null_mean=round(float(nv.mean()), 4),
                        null_q=[round(float(x), 4) for x in np.quantile(nv, [.025, .975])], p=float(p))
    json.dump(out, open(os.path.join(RES, f'dynamics_joint_null{NNULL}.json'), 'w'), indent=1)
    np.save(os.path.join(RES, f'dynamics_joint_null{NNULL}_draws.npy'), np.array([[r[k] for k in ['asym_all', 'asym_lag1', 'asym_lag2', 'int_fwd']] for r in null]))
    print(json.dumps({k: out[k] for k in obs if k != 'n_missing'}, indent=1)); print('null missing pairs/draw', out['null_pairs_missing_mean'])
    print([{k: round(v, 3) for k, v in r.items()} for r in out['obs_rows']])
