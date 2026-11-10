"""Generative family for the power study: local growth rules mixed with attribute homophily.

Every node carries a categorical attribute (K types).  A growing network adds one
node per step; the newcomer makes its links either by the book's local rule
(probability 1 - lam) or by attribute homophily (probability lam), link by link.

  LS   local search LS(n, l=1) (= triadic closure): link 1 to a uniformly random
       anchor, link 2 to a random neighbour of the anchor.  With prob. lam each of
       the two links is instead a homophily draw.
  DS   duplication-split DS(n, q=0.5): copy all links of a random node (q) or split
       one of its links.  With prob. lam the newcomer is instead a homophily node
       with m = 2 homophily draws.
  SP   spatial: nodes uniform in the unit square, attribute spatially clustered
       (type = side of x = 1/2, flipped with prob. 0.1); each of m = 2 links is a
       distance-decay draw, weight exp(-d/ell), ell = 0.1, or with prob. lam a
       homophily draw.
  ASBM static attribute SBM, <k> = 4, fraction of within-type edges matched to
       the homophily draw (h + (1 - h)/K).
  BA   Barabasi-Albert m = 2 (type-blind, global): the no-locality control.

Homophily draw (Karimi et al. 2018 style, PA x homophily): with prob. h the target
is restricted to nodes of the newcomer's type, otherwise to all nodes; within the
restricted set the target is chosen proportionally to (k + 1).  The PA factor keeps
degree heterogeneity comparable to LS/DS, so that the knob lam changes locality,
not the degree distribution.

Contagion variants (LS/DS with contagion=True): after every growth step U random
nodes update their attribute; with prob. 1 - lam they copy a random neighbour
(network-driven similarity: tie -> similarity), with prob. lam they redraw it
uniformly (exogenous drift).  Ties then select on the *current* types.  The
newcomer's type at birth is exogenous (uniform).

All generators take the exogenous attribute/position sequence (types0, pos) from the
caller, so a 'history replicate' re-grows the same population, arriving in the same
order with the same attributes and positions, with a different structural RNG.
"""
import numpy as np

K_DEFAULT = 2
H_DEFAULT = 0.8
ELL = 0.1
U_CONTAGION = 2


def exogenous(n, K, family, seed):
    """Attribute sequence and positions (birth order)."""
    rng = np.random.default_rng(seed)
    pos = rng.random((n, 2))
    if family == "SP":
        t = (pos[:, 0] > 0.5).astype(int) % K
        flip = rng.random(n) < 0.1
        t[flip] = rng.integers(0, K, flip.sum())
    else:
        t = rng.integers(0, K, n)
    return t, pos


class Net:
    """Adjacency sets plus bookkeeping for formation times."""

    def __init__(self, n):
        self.nb = [set() for _ in range(n)]
        self.deg = np.zeros(n)
        self.tform = {}           # edge (i<j) -> step formed
        self.sform = {}           # edge -> same type at formation

    def add(self, i, j, step, same):
        if i == j or j in self.nb[i]:
            return False
        self.nb[i].add(j); self.nb[j].add(i)
        self.deg[i] += 1; self.deg[j] += 1
        e = (min(i, j), max(i, j))
        self.tform[e] = step; self.sform[e] = same
        return True

    def remove(self, i, j):
        self.nb[i].discard(j); self.nb[j].discard(i)
        self.deg[i] -= 1; self.deg[j] -= 1
        e = (min(i, j), max(i, j))
        self.tform.pop(e, None); self.sform.pop(e, None)

    def edges(self):
        return np.array(sorted(self.tform), dtype=np.int64).reshape(-1, 2)


def homophily_draw(rng, i, ti, types, deg, h, exclude):
    """PA x homophily target among nodes 0..i-1."""
    cand = np.arange(i)
    if rng.random() < h:
        same = cand[types[:i] == ti]
        if len(same):
            cand = same
    w = deg[cand] + 1.0
    if exclude:
        mask = ~np.isin(cand, list(exclude))
        if mask.any():
            cand, w = cand[mask], w[mask]
    return int(rng.choice(cand, p=w / w.sum()))


def spatial_draw(rng, i, pos, exclude):
    d = np.sqrt(((pos[:i] - pos[i]) ** 2).sum(1))
    w = np.exp(-d / ELL)
    if exclude:
        w[list(exclude)] = 0
    if w.sum() <= 0:
        w = np.ones(i)
    return int(rng.choice(i, p=w / w.sum()))


def grow(family, n, lam, types0, pos, seed, h=H_DEFAULT, K=K_DEFAULT,
         contagion=False, t1=None):
    """Grow one network.  Returns dict(edges, types, pos, extra)."""
    rng = np.random.default_rng(seed)
    types = types0.copy()
    if family == "ASBM":
        return _asbm(n, types, pos, rng, h, K)
    net = Net(n)
    extra = {}
    if family == "DS":
        for a, b in ((0, 1), (1, 2), (2, 3), (3, 0)):
            net.add(a, b, 0, types[a] == types[b])
        start = 4
    else:
        net.add(0, 1, 1, types[0] == types[1])
        start = 2
    counts = np.bincount(types[:start], minlength=K).astype(float)
    sel_rows = []                 # (same at formation, baseline same fraction), ties after t1
    for i in range(start, n):
        ti = types[i]
        made = []
        if family == "BA":
            for _ in range(2):
                j = homophily_draw(rng, i, ti, types, net.deg, 0.0, set(made))
                if net.add(i, j, i, types[j] == ti): made.append(j)
        elif family in ("LS", "SP"):
            for link in range(2):
                if rng.random() < lam:
                    j = homophily_draw(rng, i, ti, types, net.deg, h, set(made))
                elif family == "SP":
                    j = spatial_draw(rng, i, pos, set(made))
                elif link == 0 or not made or not net.nb[made[0]]:
                    j = int(rng.integers(i))
                else:
                    j = int(rng.choice(sorted(net.nb[made[0]])))
                if net.add(i, j, i, types[j] == ti): made.append(j)
        elif family == "DS":
            if rng.random() < lam:
                for _ in range(2):
                    j = homophily_draw(rng, i, ti, types, net.deg, h, set(made))
                    if net.add(i, j, i, types[j] == ti): made.append(j)
            else:
                j = int(rng.integers(i))
                while not net.nb[j]:
                    j = int(rng.integers(i))
                if rng.random() < 0.5:                     # duplicate
                    for k in list(net.nb[j]):
                        if net.add(i, k, i, types[k] == ti): made.append(k)
                else:                                      # split j-k into j-i-k
                    k = int(rng.choice(sorted(net.nb[j])))
                    net.remove(j, k)
                    for x in (j, k):
                        if net.add(i, x, i, types[x] == ti): made.append(x)
        else:
            raise ValueError(family)
        if t1 is not None and i >= t1:
            base = (counts[ti]) / i
            for j in made:
                sel_rows.append((float(types[j] == ti), base))
        counts[ti] += 1
        if contagion:
            for _ in range(U_CONTAGION):
                v = int(rng.integers(i + 1))
                old = types[v]
                if rng.random() < lam:
                    types[v] = rng.integers(K)
                elif net.nb[v]:
                    types[v] = types[int(rng.choice(sorted(net.nb[v])))]
                counts[old] -= 1; counts[types[v]] += 1
        if t1 is not None and i + 1 == t1:
            extra["types_t1"] = types[:t1].copy()
            extra["edges_t1"] = net.edges()
    extra["sel_rows"] = np.array(sel_rows).reshape(-1, 2)
    return dict(edges=net.edges(), types=types, pos=pos, extra=extra)


def _asbm(n, types, pos, rng, h, K, kmean=4.0):
    E = kmean * n / 2
    nt = np.bincount(types, minlength=K)
    W = sum(c * (c - 1) / 2 for c in nt)
    X = n * (n - 1) / 2 - W
    psame = h + (1 - h) / K
    p_in = min(1.0, psame * E / max(W, 1))
    p_out = min(1.0, (1 - psame) * E / max(X, 1))
    iu, ju = np.triu_indices(n, 1)
    p = np.where(types[iu] == types[ju], p_in, p_out)
    keep = rng.random(len(p)) < p
    return dict(edges=np.stack([iu[keep], ju[keep]], 1), types=types, pos=pos, extra={})


def densify(edges, n, density, seed):
    """Similarity-network transform: keep the `density` fraction of pairs with the
    smallest graph distance in the latent sparse network (ties broken at random).
    Stands in for material-culture similarity networks thresholded at high density."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import shortest_path
    rng = np.random.default_rng(seed)
    A = csr_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(n, n))
    D = shortest_path(A, directed=False, unweighted=True)
    iu, ju = np.triu_indices(n, 1)
    d = D[iu, ju]
    d[~np.isfinite(d)] = n + 1
    d = d + rng.random(len(d)) * 0.5
    m = int(round(density * len(d)))
    order = np.argsort(d)[:m]
    return np.stack([iu[order], ju[order]], 1)
