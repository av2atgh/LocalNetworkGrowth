"""Shared machinery for workstream 4 (archaeological time series).

Similarity -> thresholded graph -> DC-SBM detection, plus dyad-independent nulls
(distance decay, attribute blocks, distance x attribute) and degree-preserving rewiring.
Run with ~/anaconda3/envs/gt/bin/python -I.
"""
import os
for _v in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:
    os.environ.setdefault(_v, '1')
import numpy as np, pandas as pd
import graph_tool.all as gt
gt.openmp_set_num_threads(1)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'data')
RES = os.path.join(HERE, '..', 'results')
os.makedirs(RES, exist_ok=True)

def br_similarity(counts):
    """Brainerd-Robinson similarity rescaled to [0,1] from a sites x categories count matrix."""
    P = counts / counts.sum(1, keepdims=True)
    return 1.0 - 0.5 * np.abs(P[:, None, :] - P[None, :, :]).sum(-1)

def threshold(S, thr):
    A = (S >= thr).astype(np.int8)
    np.fill_diagonal(A, 0)
    return A

def lcc_mask(A):
    g = adj_to_graph(A)
    comp, hist = gt.label_components(g)
    big = np.argmax(hist)
    return comp.a == big

def adj_to_graph(A):
    n = A.shape[0]
    g = gt.Graph(directed=False)
    g.add_vertex(n)
    i, j = np.nonzero(np.triu(A, 1))
    g.add_edge_list(np.column_stack([i, j]))
    return g

def sbm_fit(A, deg_corr=True, nrep=3, seed=None):
    """Best-of-nrep DC-SBM fit on the LCC. Returns (B, dl, dl_B1, partition over LCC nodes, lcc mask)."""
    if seed is not None:
        gt.seed_rng(int(seed)); np.random.seed(int(seed) % 2**32)
    m = lcc_mask(A)
    g = adj_to_graph(A[np.ix_(m, m)])
    if g.num_edges() == 0:
        return 1, 0.0, 0.0, np.zeros(m.sum(), int), m
    best = None
    for _ in range(nrep):
        st = gt.minimize_blockmodel_dl(g, state_args=dict(deg_corr=deg_corr))
        if best is None or st.entropy() < best.entropy():
            best = st
    b = best.get_blocks().a.copy()
    _, b = np.unique(b, return_inverse=True)
    st1 = gt.BlockState(g, b=np.zeros(g.num_vertices(), int), deg_corr=deg_corr)
    return int(b.max() + 1), float(best.entropy()), float(st1.entropy()), b, m

# ---------- null models (dyad-independent) ----------
def pair_index(n):
    return np.triu_indices(n, 1)

def binned_probs(A, keys):
    """Empirical edge probability per dyad class; keys: int array over upper-triangle pairs."""
    iu = pair_index(A.shape[0])
    y = A[iu].astype(float)
    k = keys
    nk = k.max() + 1
    num = np.bincount(k, weights=y, minlength=nk)
    den = np.bincount(k, minlength=nk)
    p = np.where(den > 0, num / np.maximum(den, 1), 0.0)
    return p[k]

def dist_bins(D, nbins=20):
    iu = pair_index(D.shape[0])
    d = D[iu]
    qs = np.unique(np.quantile(d, np.linspace(0, 1, nbins + 1)))
    return np.clip(np.searchsorted(qs, d, side='right') - 1, 0, len(qs) - 2)

def null_probs(A, D=None, attr=None, kind='dist', nbins=20):
    n = A.shape[0]; iu = pair_index(n)
    if kind == 'dist':
        k = dist_bins(D, nbins)
    elif kind == 'attr':
        a = pd.factorize(np.asarray(attr))[0]
        lo = np.minimum(a[iu[0]], a[iu[1]]); hi = np.maximum(a[iu[0]], a[iu[1]])
        k = pd.factorize(lo * 10000 + hi)[0]
    elif kind == 'dist_attr':
        a = pd.factorize(np.asarray(attr))[0]
        same = (a[iu[0]] == a[iu[1]]).astype(int)
        k = dist_bins(D, nbins) * 2 + same
    else:
        raise ValueError(kind)
    return binned_probs(A, k)

def sample_from_probs(p, n, rng):
    iu = pair_index(n)
    A = np.zeros((n, n), np.int8)
    e = rng.random(len(p)) < p
    A[iu[0][e], iu[1][e]] = 1
    return A + A.T

def rewire(A, rng, niter=10):
    g = adj_to_graph(A)
    gt.random_rewire(g, model='configuration', n_iter=niter, edge_sweep=True)
    B = gt.adjacency(g).toarray().astype(np.int8)
    B = ((B + B.T) > 0).astype(np.int8)
    np.fill_diagonal(B, 0)
    return B

def _contingency(a, b):
    a = pd.factorize(np.asarray(a))[0]; b = pd.factorize(np.asarray(b))[0]
    C = np.zeros((a.max() + 1, b.max() + 1))
    np.add.at(C, (a, b), 1)
    return C

def ari(a, b):
    C = _contingency(a, b); n = C.sum()
    c2 = lambda x: x * (x - 1) / 2
    sij = c2(C).sum(); sa = c2(C.sum(1)).sum(); sb = c2(C.sum(0)).sum()
    exp = sa * sb / c2(n); mx = 0.5 * (sa + sb)
    return 1.0 if mx == exp else (sij - exp) / (mx - exp)

def nmi(a, b):
    C = _contingency(a, b); n = C.sum(); P = C / n
    pa = P.sum(1); pb = P.sum(0)
    nz = P > 0
    I = (P[nz] * np.log(P[nz] / np.outer(pa, pb)[nz])).sum()
    H = lambda p: -(p[p > 0] * np.log(p[p > 0])).sum()
    ha, hb = H(pa), H(pb)
    return 1.0 if ha + hb == 0 else 2 * I / (ha + hb)
