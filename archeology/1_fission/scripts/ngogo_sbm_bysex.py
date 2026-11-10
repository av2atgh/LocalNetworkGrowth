"""DC-SBM (B=2 and nested) within sex strata, 2011-2014 window. Usage: python -I ngogo_sbm_bysex.py <D> <nodes_csv>"""
import sys, os, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sklearn_free import ari
import graph_tool.all as gt; gt.openmp_set_num_threads(1)
D, NC = sys.argv[1:3]
names = list(np.load(f'{D}/learned/L/data/chimps_names.npy')); N = len(names)
lab = np.load(f'{D}/learned/L/src/combined_membership_labels.npy')
nd = pd.read_csv(NC)
A = np.zeros((N, N))
for y in range(2011, 2015):
    for m in range(1, 13):
        A += np.nan_to_num(np.load(f'{D}/learned/L/data/learned_graphs/full_array_{y}_{m}.npy'))
A = (A + A.T) / 2; np.fill_diagonal(A, 0)
for nm, sel in [('all', nd.node.tolist()), ('males', nd[nd.sex == 1].node.tolist()), ('females', nd[nd.sex == 0].node.tolist())]:
    ix = [names.index(s) for s in sel]
    W = np.log1p(A[np.ix_(ix, ix)])
    g = gt.Graph(directed=False); g.add_vertex(len(ix)); w = g.new_ep('double')
    for a in range(len(ix)):
        for b in range(a + 1, len(ix)):
            if W[a, b] > 0: e = g.add_edge(a, b); w[e] = W[a, b]
    y = lab[ix]
    for B in [2, None]:
        best = None
        for s in range(20):
            gt.seed_rng(s); np.random.seed(s)
            kw = dict(state_args=dict(deg_corr=True))
            if B: kw['multilevel_mcmc_args'] = dict(B_min=B, B_max=B)
            st = gt.minimize_blockmodel_dl(g, **kw)
            if best is None or st.entropy() < best.entropy(): best = st
        b = np.array(best.get_blocks().a)
        print(nm, 'n', len(ix), 'B', len(set(b)), 'ARI', round(ari(b, y), 3), pd.crosstab(b, y).values.tolist())
    best = None
    for s in range(10):
        gt.seed_rng(s); np.random.seed(s)
        st = gt.minimize_nested_blockmodel_dl(g, state_args=dict(deg_corr=True))
        if best is None or st.entropy() < best.entropy(): best = st
    for l in range(3):
        b = np.array(best.project_level(l).get_blocks().a)
        print(nm, 'nested level', l, 'B', len(set(b)), 'ARI', round(ari(b, y), 3))
