"""Spectral (normalized-Laplacian Fiedler) cut computed on the sex-specific subgraph only, 2011-2014.
Usage: python -I ngogo_spectral_bysex.py <D> <nodes_csv>"""
import sys, numpy as np, pandas as pd, networkx as nx
D, NC = sys.argv[1:3]
rng = np.random.default_rng(3)
names = list(np.load(f'{D}/learned/L/data/chimps_names.npy')); N = len(names)
lab = np.load(f'{D}/learned/L/src/combined_membership_labels.npy').astype(float)
nd = pd.read_csv(NC)
A = np.zeros((N, N))
for y in range(2011, 2015):
    for m in range(1, 13):
        A += np.nan_to_num(np.load(f'{D}/learned/L/data/learned_graphs/full_array_{y}_{m}.npy'))
A = (A + A.T) / 2; np.fill_diagonal(A, 0)
def bal(p, y): return 0.5 * (np.mean(p[y == 1] == 1) + np.mean(p[y == 0] == 0))
for nm, sel in [('males', nd[nd.sex == 1].node.tolist()), ('females', nd[nd.sex == 0].node.tolist())]:
    ix = np.array([names.index(s) for s in sel])
    W = np.log1p(A[np.ix_(ix, ix)])
    G = nx.from_numpy_array(W); cc = sorted(max(nx.connected_components(G), key=len)); W = W[np.ix_(cc, cc)]
    d = W.sum(1); Dm = 1 / np.sqrt(d)
    v = np.linalg.eigh(np.eye(len(W)) - Dm[:, None] * W * Dm[None, :])[1][:, 1] * Dm
    y = lab[ix[cc]]; p = (v > 0).astype(float)
    o = max(bal(p, y), bal(1 - p, y))
    null = np.array([max(bal(p, q), bal(1 - p, q)) for q in (rng.permutation(y) for _ in range(2000))])
    print(nm, 'n_gcc', len(cc), 'density', round((W > 0).sum() / len(W) / (len(W) - 1), 3),
          'fiedler bal_acc', round(o, 3), 'perm p', round((1 + (null >= o).sum()) / 2001, 4))
