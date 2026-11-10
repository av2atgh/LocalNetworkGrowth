"""Calibration: Zachary karate club (n=34, fission into Mr Hi / Officer factions; networkx built-in).
Same predictors as Ngogo: Fiedler cut, leading modularity eigvec, DC-SBM B=2 and B free."""
import sys, os, numpy as np, networkx as nx
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sklearn_free import ari
import graph_tool.all as gt; gt.openmp_set_num_threads(1)
G = nx.karate_club_graph(); n = G.number_of_nodes()
y = np.array([0 if G.nodes[i]['club'] == 'Mr. Hi' else 1 for i in range(n)])
W = nx.to_numpy_array(G, weight=None)
d = W.sum(1); Dm = 1 / np.sqrt(d)
f = np.linalg.eigh(np.eye(n) - Dm[:, None] * W * Dm[None, :])[1][:, 1] * Dm
B = W - np.outer(d, d) / d.sum(); mv = np.linalg.eigh(B)[1][:, -1]
for nm, v in [('fiedler', f), ('modularity', mv)]:
    p = (v > 0).astype(int); a = max(np.mean(p == y), 1 - np.mean(p == y))
    print(nm, 'acc', round(a, 3), 'errors', int(round((1 - a) * n)), 'ARI', round(ari(p, y), 3))
g = gt.Graph(directed=False); g.add_vertex(n); g.add_edge_list(list(G.edges()))
for Bf in [2, None]:
    best = None
    for s in range(30):
        gt.seed_rng(s); np.random.seed(s)
        kw = dict(state_args=dict(deg_corr=True))
        if Bf: kw['multilevel_mcmc_args'] = dict(B_min=Bf, B_max=Bf)
        st = gt.minimize_blockmodel_dl(g, **kw)
        if best is None or st.entropy() < best.entropy(): best = st
    b = np.array(best.get_blocks().a)
    print('DC-SBM B=%s' % Bf, 'found B', len(set(b)), 'ARI', round(ari(b, y), 3), 'DL', round(best.entropy(), 1))
