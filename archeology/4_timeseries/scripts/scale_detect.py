"""(i) Community detectability vs number of sites, observed vs nulls.
Part A: each time slice as is (n changes with time).
Part B: within large slices, nested subsets (spatial kNN ball around a random site, or random sites).
Detection = DC-SBM (graph-tool) best partition on the LCC has B>=2; Sigma = (DL_B1 - DL_best)/E.
Nulls: dist (empirical edge prob per distance-quantile bin), attr (block densities by region),
dist_attr (distance bin x same-region), config (degree-preserving rewiring)."""
import os, sys
for _v in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:
    os.environ.setdefault(_v, '1')
import pickle, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from multiprocessing import Pool

THR = float(os.environ.get('THR', 0.75))
NNULL = 20
SIZES = [10, 15, 20, 30, 45, 65, 90, 130]
NREP = 10
NULLS = ['dist', 'attr', 'dist_attr', 'config']

def dmat(xy):
    return np.hypot(xy[:, None, 0] - xy[None, :, 0], xy[:, None, 1] - xy[None, :, 1])

def stat(A, seed, nrep=2):
    if A.sum() == 0: return dict(B=1, sig=0.0, n=0)
    B, dl, dl1, b, m = sbm_fit(A, nrep=nrep, seed=seed)
    E = A[np.ix_(m, m)].sum() / 2
    return dict(B=B, sig=(dl1 - dl) / max(E, 1), n=int(m.sum()))

def null_graph(kind, A, D, attr, P, rng):
    if kind == 'config': return rewire(A, rng)
    return sample_from_probs(P[kind], A.shape[0], rng)

def job(args):
    ds, per, v = args
    rng = np.random.default_rng(abs(hash((ds, per))) % 2**32)
    A = threshold(v['S'], THR); D = dmat(v['xy']); attr = v['attr']; n = len(A)
    P = {k: null_probs(A, D, attr, k) for k in ['dist', 'attr', 'dist_attr']}
    res = dict(ds=ds, per=per, n=n, density=float(A[np.triu_indices(n, 1)].mean()))
    res['obs'] = stat(A, 1, nrep=3)
    for k in NULLS:
        res[k] = [stat(null_graph(k, A, D, attr, P, rng), 100 + i, nrep=2) for i in range(NNULL)]
    # Part B subsets
    sub = []
    if n >= 140:
        for mode in ['knn', 'rand']:
            for s in [x for x in SIZES if x < n]:
                for r in range(NREP):
                    c = rng.integers(n)
                    idx = np.argsort(D[c])[:s] if mode == 'knn' else rng.choice(n, s, replace=False)
                    ix = np.ix_(idx, idx)
                    row = dict(mode=mode, size=s, rep=r, obs=stat(A[ix], r))
                    for k in ['dist', 'dist_attr', 'config']:
                        G = null_graph(k, A, D, attr, P, rng) if k != 'config' else rewire(A[ix], rng)
                        row[k] = stat(G[ix] if k != 'config' else G, 1000 + r)
                    sub.append(row)
    res['sub'] = sub
    print(ds, per, n, 'obs', res['obs'], flush=True)
    return res

if __name__ == '__main__':
    d = pickle.load(open(os.path.join(RES, 'slices.pkl'), 'rb'))
    jobs = [(ds, p, v) for ds in d for p, v in d[ds].items() if len(v['S']) >= 20]
    jobs.sort(key=lambda j: len(j[2]['S']))
    # single process, per-slice checkpoints (shared-machine limit; resumable)
    part = os.path.join(RES, f'scale_parts_thr{THR}'); os.makedirs(part, exist_ok=True)
    out = []
    for j in jobs:
        f = os.path.join(part, f'{j[0]}_{j[1]}.pkl')
        if os.path.exists(f):
            out.append(pickle.load(open(f, 'rb'))); continue
        r = job(j); pickle.dump(r, open(f, 'wb')); out.append(r)
    pickle.dump(out, open(os.path.join(RES, f'scale_thr{THR}.pkl'), 'wb'))
