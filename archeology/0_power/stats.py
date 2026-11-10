"""Discriminating statistics (README table) for one network.  Run in the gt env.

a  align_nmi, align_ari    NMI / ARI between the DC-SBM partition and the attribute
b  beyond_dc, beyond_ndc   (S_attr - S_free)/E: description length of the attribute
                           block model (partition given as data, partition_dl off) minus
                           that of the freely fitted SBM (partition paid for); > 0 means
                           structure beyond attributes.  expl_dc = (S_1 - S_attr)/E is
                           what the attribute explains over one block; dcgain = (S_ndc -
                           S_dc)/E for the free fits.
c  logit_ratio             dyadic logistic tie ~ same + CN + log-degrees + distance,
                           standardized; ratio b_same/(|b_same|+|b_CN|)
d  hist_ari                mean pairwise ARI between the DC-SBM partitions of H histories
                           (same population, different growth history); NaN for pairs
                           where both partitions are trivial
e  fis_attr, fis_topo,     fission cut from a two-leader majority cascade; ARI of the cut
   fis_diff                with the attribute, with the forced B = 2 DC-SBM bipartition
                           of the pre-fission network, and their difference
f  conv, sel               temporal order (contagion variant): conv = P(same at n | tied
                           at t1, different at t1) - same for untied pairs; sel = same-type
                           fraction of ties formed after t1, at formation, minus the
                           newcomer's same-type baseline
"""
import numpy as np
import graph_tool.all as gt


# ------------------------------------------------------------ partition scores

def _contingency(a, b):
    _, ai = np.unique(a, return_inverse=True)
    _, bi = np.unique(b, return_inverse=True)
    C = np.zeros((ai.max() + 1, bi.max() + 1))
    np.add.at(C, (ai, bi), 1)
    return C


def ari(a, b):
    C = _contingency(a, b)
    n = C.sum()
    c2 = lambda x: x * (x - 1) / 2
    sij = c2(C).sum(); sa = c2(C.sum(1)).sum(); sb = c2(C.sum(0)).sum()
    exp = sa * sb / c2(n)
    mx = (sa + sb) / 2
    if mx == exp:
        return 1.0 if sij == mx else 0.0
    return (sij - exp) / (mx - exp)


def nmi(a, b):
    C = _contingency(a, b)
    n = C.sum(); P = C / n
    pa, pb = P.sum(1), P.sum(0)
    nz = P > 0
    I = (P[nz] * np.log(P[nz] / np.outer(pa, pb)[nz])).sum()
    Ha = -(pa[pa > 0] * np.log(pa[pa > 0])).sum()
    Hb = -(pb[pb > 0] * np.log(pb[pb > 0])).sum()
    if Ha == 0 and Hb == 0:
        return np.nan
    if Ha == 0 or Hb == 0:
        return 0.0
    return 2 * I / (Ha + Hb)


# ------------------------------------------------------------ SBM fits

def graph(edges, n):
    g = gt.Graph(directed=False)
    g.add_vertex(n)
    g.add_edge_list(edges)
    return g


def fit(g, deg_corr=True, nrep=1, B=None):
    args = dict(B_min=B, B_max=B) if B else {}
    best = None
    for _ in range(nrep):
        s = gt.minimize_blockmodel_dl(g, state_args=dict(deg_corr=deg_corr),
                                      multilevel_mcmc_args=args)
        if best is None or s.entropy() < best.entropy():
            best = s
    b = np.array(best.get_blocks().a)
    return b, best.entropy(), best.get_nonempty_B()


def fixed_entropy(g, b, deg_corr, partition_dl):
    bp = g.new_vp("int", vals=np.asarray(b, dtype=int))
    return gt.BlockState(g, b=bp, deg_corr=deg_corr).entropy(partition_dl=partition_dl)


# ------------------------------------------------------------ dyadic logit

def logit(X, y, ridge=1e-2, it=50):
    X = np.column_stack([np.ones(len(y)), X])
    beta = np.zeros(X.shape[1])
    R = ridge * np.eye(X.shape[1]); R[0, 0] = 0
    for _ in range(it):
        eta = np.clip(X @ beta, -30, 30)
        p = 1 / (1 + np.exp(-eta))
        W = p * (1 - p) + 1e-9
        H = X.T @ (X * W[:, None]) + R
        step = np.linalg.solve(H, X.T @ (y - p) - R @ beta)
        beta += step
        if np.abs(step).max() < 1e-8:
            break
    return beta[1:]


def dyadic(edges, n, types, pos):
    A = np.zeros((n, n))
    A[edges[:, 0], edges[:, 1]] = 1; A[edges[:, 1], edges[:, 0]] = 1
    CN = A @ A
    k = A.sum(1)
    iu, ju = np.triu_indices(n, 1)
    y = A[iu, ju]
    same = (types[iu] == types[ju]).astype(float)
    cn = CN[iu, ju]
    ld = np.log1p(k[iu] - y) + np.log1p(k[ju] - y)
    dist = np.sqrt(((pos[iu] - pos[ju]) ** 2).sum(1))
    X = np.column_stack([same, cn, ld, dist])
    sd = X.std(0); sd[sd == 0] = 1
    X = (X - X.mean(0)) / sd
    b = logit(X, y)
    bs, bc = b[0], b[1]
    return dict(b_same=bs, b_cn=bc, b_deg=b[2], b_dist=b[3],
                logit_ratio=bs / (abs(bs) + abs(bc) + 1e-12))


# ------------------------------------------------------------ fission

def fission(edges, n, seed):
    rng = np.random.default_rng(seed)
    nb = [[] for _ in range(n)]
    for a, b in edges:
        nb[a].append(b); nb[b].append(a)
    k = np.array([len(x) for x in nb], float) + rng.random(n) * 0.1
    l1, l2 = np.argsort(-k)[:2]
    side = -np.ones(n, int); side[l1] = 0; side[l2] = 1
    frontier = list(nb[l1]) + list(nb[l2])
    while (side < 0).any():
        cand = [v for v in set(frontier) if side[v] < 0]
        if not cand:                                  # disconnected rest
            cand = list(np.where(side < 0)[0])
        v = cand[rng.integers(len(cand))]
        s = side[nb[v]]; s = s[s >= 0]
        side[v] = rng.integers(2) if len(s) == 0 or s.mean() == 0.5 else int(s.mean() > 0.5)
        frontier = [u for u in frontier if side[u] < 0] + [u for u in nb[v] if side[u] < 0]
    for _ in range(5):                               # majority relaxation
        for v in rng.permutation(n):
            if v in (l1, l2) or not nb[v]:
                continue
            m = side[nb[v]].mean()
            if m != 0.5:
                side[v] = int(m > 0.5)
    return side


# ------------------------------------------------------------ temporal

def temporal(net, n, t1):
    ex = net["extra"]
    T1, Tn = ex["types_t1"], net["types"][:t1]
    e1 = ex["edges_t1"]
    tied = np.zeros((t1, t1), bool)
    tied[e1[:, 0], e1[:, 1]] = True; tied |= tied.T
    iu, ju = np.triu_indices(t1, 1)
    diff1 = T1[iu] != T1[ju]
    same_n = Tn[iu] == Tn[ju]
    t = tied[iu, ju]
    a = diff1 & t; b = diff1 & ~t
    conv = same_n[a].mean() - same_n[b].mean() if a.sum() and b.sum() else np.nan
    sr = ex["sel_rows"]
    sel = (sr[:, 0] - sr[:, 1]).mean() if len(sr) else np.nan
    return dict(conv=conv, sel=sel, n_conv_pairs=int(a.sum()))


# ------------------------------------------------------------ all of it

def snapshot_stats(edges, n, types, pos, seed, nrep=2):
    g = graph(edges, n)
    E = max(g.num_edges(), 1)
    out = dict(E=g.num_edges(), density=2 * g.num_edges() / (n * (n - 1)))
    b_dc, S_dc, B_dc = fit(g, True, nrep)
    b_ndc, S_ndc, B_ndc = fit(g, False, 1)
    out.update(B_dc=B_dc, B_ndc=B_ndc, detect=float(B_dc > 1))
    out["align_nmi"] = nmi(b_dc, types) if B_dc > 1 else 0.0
    out["align_ari"] = ari(b_dc, types)
    out["align_nmi_ndc"] = nmi(b_ndc, types) if B_ndc > 1 else 0.0
    one = np.zeros(n, int)
    for dc, S_free in ((True, S_dc), (False, S_ndc)):
        tag = "dc" if dc else "ndc"
        S_att = fixed_entropy(g, types, dc, partition_dl=False)
        S_1 = fixed_entropy(g, one, dc, partition_dl=True)
        out[f"beyond_{tag}"] = (S_att - S_free) / E
        out[f"expl_{tag}"] = (S_1 - S_att) / E
    out["dcgain"] = (S_ndc - S_dc) / E
    out.update(dyadic(edges, n, types, pos))
    cut = fission(edges, n, seed)
    b2, _, _ = fit(g, True, 1, B=2)
    out["fis_attr"] = ari(cut, types)
    out["fis_topo"] = ari(cut, b2)
    out["fis_diff"] = out["fis_attr"] - out["fis_topo"]
    return out, b_dc


def hist_ari(parts):
    vals = []
    for i in range(len(parts)):
        for j in range(i + 1, len(parts)):
            a, b = parts[i], parts[j]
            if len(np.unique(a)) == 1 and len(np.unique(b)) == 1:
                continue
            vals.append(ari(a, b))
    return np.mean(vals) if vals else np.nan
