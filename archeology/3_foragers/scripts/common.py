"""Shared helpers: logistic regression, AUC, NMI/ARI, DC-SBM scoring.

Imported by the analysis scripts via explicit sys.path insertion (scripts run with -I).
No dependencies beyond numpy/scipy/graph-tool.
"""
import os
for _v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit


# ---------- logistic regression (L2, standardised features) ----------
def fit_logit(X, y, lam=1.0):
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1.0
    Z = (X - mu) / sd
    Z1 = np.column_stack([np.ones(len(Z)), Z])

    def f(w):
        eta = Z1 @ w
        ll = np.sum(y * eta - np.logaddexp(0, eta))
        pen = 0.5 * lam * np.sum(w[1:] ** 2)
        g = Z1.T @ (y - expit(eta))
        g[1:] -= lam * w[1:]
        return -(ll - pen), -g

    w0 = np.zeros(Z1.shape[1])
    w0[0] = np.log((y.mean() + 1e-9) / (1 - y.mean() + 1e-9))
    r = minimize(f, w0, jac=True, method="L-BFGS-B")
    return (mu, sd, r.x)


def predict_logit(model, X):
    mu, sd, w = model
    Z = (X - mu) / sd
    return expit(w[0] + Z @ w[1:])


def auc(y, s):
    y = np.asarray(y).astype(bool)
    s = np.asarray(s, float)
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s))
    # average ranks for ties
    ss = s[order]
    i = 0
    r = np.arange(1, len(s) + 1, dtype=float)
    while i < len(s):
        j = i
        while j + 1 < len(s) and ss[j + 1] == ss[i]:
            j += 1
        r[i:j + 1] = 0.5 * (i + 1 + j + 1)
        i = j + 1
    ranks[order] = r
    n1, n0 = y.sum(), (~y).sum()
    return (ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def logloss(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


# ---------- partition similarity ----------
def _contingency(a, b):
    _, ai = np.unique(a, return_inverse=True)
    _, bi = np.unique(b, return_inverse=True)
    C = np.zeros((ai.max() + 1, bi.max() + 1))
    np.add.at(C, (ai, bi), 1)
    return C


def nmi(a, b):
    C = _contingency(a, b)
    n = C.sum()
    pa, pb = C.sum(1) / n, C.sum(0) / n
    P = C / n
    nz = P > 0
    I = np.sum(P[nz] * np.log(P[nz] / np.outer(pa, pb)[nz]))
    Ha = -np.sum(pa[pa > 0] * np.log(pa[pa > 0]))
    Hb = -np.sum(pb[pb > 0] * np.log(pb[pb > 0]))
    if Ha + Hb == 0:
        return 1.0
    return 2 * I / (Ha + Hb)


def ari(a, b):
    from scipy.special import comb
    C = _contingency(a, b)
    n = C.sum()
    sij = comb(C, 2).sum()
    sa = comb(C.sum(1), 2).sum()
    sb = comb(C.sum(0), 2).sum()
    exp = sa * sb / comb(n, 2)
    mx = 0.5 * (sa + sb)
    return (sij - exp) / (mx - exp) if mx != exp else 0.0


def perm_test(a, b, stat, nperm=2000, rng=None):
    rng = rng or np.random.default_rng(0)
    obs = stat(a, b)
    null = np.array([stat(rng.permutation(a), b) for _ in range(nperm)])
    return obs, null.mean(), (1 + np.sum(null >= obs)) / (nperm + 1)


# ---------- graph-tool helpers ----------
def build_graph(n, edges):
    import graph_tool.all as gt
    gt.openmp_set_num_threads(1)
    g = gt.Graph(directed=False)
    g.add_vertex(n)
    if len(edges):
        g.add_edge_list(np.asarray(edges))
    return g


def fit_sbm(g, nrep=10, seed=0, nested=False):
    import graph_tool.all as gt
    gt.seed_rng(seed)
    np.random.seed(seed)
    best = None
    for r in range(nrep):
        if nested:
            s = gt.minimize_nested_blockmodel_dl(g, state_args=dict(deg_corr=True))
        else:
            s = gt.minimize_blockmodel_dl(g, state_args=dict(deg_corr=True))
        if best is None or s.entropy() < best.entropy():
            best = s
    return best


def dl_fixed(g, labels):
    import graph_tool.all as gt
    _, b = np.unique(labels, return_inverse=True)
    bp = g.new_vp("int", vals=b)
    return gt.BlockState(g, b=bp, deg_corr=True).entropy()


def dcsbm_scores(n, edges, labels, pairs):
    """Expected DC-SBM edge weight lambda_ij = theta_i theta_j e_rs / (e_r e_s) * ...;
    with ML estimates theta_i = k_i, omega_rs = e_rs/(e_r e_s): lambda = k_i k_j e_rs/(e_r e_s)."""
    _, b = np.unique(labels, return_inverse=True)
    B = b.max() + 1
    k = np.zeros(n)
    E = np.zeros((B, B))
    for u, v in edges:
        k[u] += 1
        k[v] += 1
        E[b[u], b[v]] += 1
        E[b[v], b[u]] += 1
    er = E.sum(1)
    er[er == 0] = 1
    pairs = np.asarray(pairs)
    i, j = pairs[:, 0], pairs[:, 1]
    lam = k[i] * k[j] * E[b[i], b[j]] / (er[b[i]] * er[b[j]])
    return lam


def neighbour_features(n, edges, pairs):
    A = np.zeros((n, n))
    for u, v in edges:
        A[u, v] = A[v, u] = 1
    k = A.sum(1)
    CN = A @ A
    w = np.where(k > 1, 1 / np.log(np.maximum(k, 2)), 0)
    AA = (A * w) @ A
    pairs = np.asarray(pairs)
    i, j = pairs[:, 0], pairs[:, 1]
    cn = CN[i, j]
    un = k[i] + k[j] - cn - 2 * A[i, j]
    jac = np.where(un > 0, cn / np.maximum(un, 1), 0)
    # leave-own-edge-out degrees: a training dyad's own edge must not enter its features
    ki, kj = k[i] - A[i, j], k[j] - A[i, j]
    return np.column_stack([cn, AA[i, j], jac, np.log1p(ki), np.log1p(kj),
                            np.log1p(ki * kj)])
