"""Ngogo chimpanzee fission (Sandel et al. 2026, Science): does the pre-fission network or
individual attributes (sex, age) or space (GPS home-range centroid) predict post-fission membership?

Inputs (untrusted, read-only) from Zenodo 10.5281/zenodo.19210783 (learned graphs, names, labels)
and 10.5281/zenodo.18603419 (GPS subsample with Sex/Age).  Code->name mapping for males from
map_codes.py (heuristic; only pairs with year-Jaccard >= 0.5 are trusted).

Usage: python -I ngogo_fission.py <ngogo_data_dir> <codemap.csv> <out_dir> [y0 y1]
"""
import sys, json, numpy as np, pandas as pd, networkx as nx
from scipy.optimize import minimize
from scipy.sparse.linalg import eigsh
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sklearn_free import ari  # local helper (same dir)

D, CM, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
Y0, Y1 = (int(sys.argv[4]), int(sys.argv[5])) if len(sys.argv) > 5 else (2011, 2014)
rng = np.random.default_rng(1)
NPERM = 1000

names = np.load(f'{D}/learned/L/data/chimps_names.npy')
lab = np.load(f'{D}/learned/L/src/combined_membership_labels.npy')
N = len(names)

def window_graph(y0, y1):
    A = np.zeros((N, N))
    for y in range(y0, y1 + 1):
        for m in range(1, 13):
            try:
                A += np.nan_to_num(np.load(f'{D}/learned/L/data/learned_graphs/full_array_{y}_{m}.npy'))
            except FileNotFoundError:
                pass
    A = (A + A.T) / 2
    np.fill_diagonal(A, 0)
    return A

# ---------------- attributes ----------------
g = pd.read_csv(f'{D}/space/space-use-archive/data/gps_qualified.csv')
g['key'] = g.Individual.str.lower().str.replace(' ', '')
cm = pd.read_csv(CM)
cm = cm[cm.jaccard >= 0.5]
code2name = {r.code: r.name.lower().replace(' ', '') for r in cm.itertuples()}
key = [code2name.get(n, None) if i < 97 else n for i, n in enumerate(names)]
g['birth'] = g.Year + (g.Month - 0.5) / 12 - g.Age_at_sampling
info = g.groupby('key').agg(sex=('Sex', 'first'), birth=('birth', 'median'))
gw = g[(g.Year >= Y0) & (g.Year <= Y1)]
lat0, lon0 = g.Lat.mean(), g.Lon.mean()
gw = gw.assign(x=(gw.Lon - lon0) * 111.32 * np.cos(np.radians(lat0)), yk=(gw.Lat - lat0) * 110.57)
cent = gw.groupby('key').agg(cx=('x', 'median'), cy=('yk', 'median'), npts=('x', 'size'))
cent = cent[cent.npts >= 30]

sex = np.full(N, np.nan); birth = np.full(N, np.nan); cx = np.full(N, np.nan); cy = np.full(N, np.nan)
for i in range(N):
    if i < 97:
        sex[i] = 1.0  # all coded nodes are males
    k = key[i]
    if k is not None and k in info.index:
        sex[i] = 1.0 if info.loc[k, 'sex'] == 'M' else 0.0
        birth[i] = info.loc[k, 'birth']
    if k is not None and k in cent.index:
        cx[i], cy[i] = cent.loc[k, 'cx'], cent.loc[k, 'cy']
mid = (Y0 + Y1 + 1) / 2
age = mid - birth

# ---------------- network ----------------
A = window_graph(Y0, Y1)
act = A.sum(1) > 0
G = nx.from_numpy_array(A)
G.remove_nodes_from([i for i in range(N) if not act[i]])
gcc = max(nx.connected_components(G), key=len)
G = G.subgraph(gcc).copy()
nodes = np.array(sorted(G.nodes()))
Wl = np.log1p(A[np.ix_(nodes, nodes)])  # log weights: learned weights are heavy-tailed

def fiedler(W):
    d = W.sum(1); Dm = 1 / np.sqrt(d)
    L = np.eye(len(W)) - Dm[:, None] * W * Dm[None, :]
    vals, vecs = np.linalg.eigh(L)
    return Dm * vecs[:, 1]

def mod_leading(W):
    k = W.sum(1); m2 = k.sum()
    B = W - np.outer(k, k) / m2
    vals, vecs = np.linalg.eigh(B)
    return vecs[:, -1]

spec = dict(zip(nodes, fiedler(Wl)))
modv = dict(zip(nodes, mod_leading(Wl)))

# Louvain (best of 20 seeds) on log weights
Gl = nx.Graph()
for a in range(len(nodes)):
    for b in range(a + 1, len(nodes)):
        if Wl[a, b] > 0: Gl.add_edge(nodes[a], nodes[b], weight=Wl[a, b])
best = None
for s in range(20):
    P = nx.community.louvain_communities(Gl, weight='weight', seed=s)
    q = nx.community.modularity(Gl, P, weight='weight')
    if best is None or q > best[0]: best = (q, P)
louv = {v: ci for ci, c in enumerate(best[1]) for v in c}

# DC-SBM via graph-tool (binary, B=2 fixed and B free), best of 20 runs
import graph_tool.all as gt; gt.openmp_set_num_threads(1)
gtg = gt.Graph(directed=False)
idx = {v: i for i, v in enumerate(nodes)}
gtg.add_vertex(len(nodes))
ew = gtg.new_ep('double')
for a, b, d in Gl.edges(data=True):
    e = gtg.add_edge(idx[a], idx[b]); ew[e] = d['weight']
def sbm(Bfix, weighted):
    bestS = None
    for s in range(20):
        gt.seed_rng(s); np.random.seed(s)
        sa = dict(deg_corr=True)
        if weighted: sa.update(recs=[ew], rec_types=['real-exponential'])
        kw = dict(state_args=sa)
        if Bfix: kw['multilevel_mcmc_args'] = dict(B_min=Bfix, B_max=Bfix)
        st = gt.minimize_blockmodel_dl(gtg, **kw)
        if bestS is None or st.entropy() < bestS.entropy(): bestS = st
    b = np.array(bestS.get_blocks().a)
    return dict(zip(nodes, b)), bestS.entropy(), len(set(b))
sbm2, dl2, _ = sbm(2, False)
sbmF, dlF, BF = sbm(None, False)
sbm2w, dl2w, _ = sbm(2, True)

# ---------------- evaluation set ----------------
ev = np.array([v for v in nodes if lab[v] >= 0])
y = lab[ev].astype(float)
n = len(ev)

def acc_bip(pred, yy):
    a = np.mean(pred == yy); return max(a, 1 - a)

def perm_p(stat_fn, pred, yy, nperm=NPERM):
    obs = stat_fn(pred, yy)
    null = np.array([stat_fn(pred, rng.permutation(yy)) for _ in range(nperm)])
    return obs, float((1 + np.sum(null >= obs - 1e-12)) / (nperm + 1)), float(null.mean())

res = dict(window=[Y0, Y1], n_active_gcc=int(len(nodes)), n_eval=int(n),
           n_label1=int(y.sum()), n_label0=int(n - y.sum()), n_unlabeled_in_gcc=int(len(nodes) - n))
unsup = {}
for nm, part, cont in [('spectral_fiedler', None, spec), ('modularity_leading_eig', None, modv),
                       ('louvain', louv, None), ('dcsbm_B2_binary', sbm2, None),
                       ('dcsbm_B2_weighted', sbm2w, None), (f'dcsbm_Bfree_binary', sbmF, None)]:
    if cont is not None:
        pred = np.array([1.0 if cont[v] > 0 else 0.0 for v in ev])
        o, p, nl = perm_p(acc_bip, pred, y)
        a2, p2, n2 = perm_p(lambda a, b: ari(a, b), pred, y)
        unsup[nm] = dict(acc=round(o, 3), acc_p=p, acc_null=round(nl, 3), ari=round(a2, 3), ari_p=p2)
    else:
        pred = np.array([part[v] for v in ev])
        a2, p2, n2 = perm_p(lambda a, b: ari(a, b), pred, y)
        d = dict(ari=round(a2, 3), ari_p=p2, n_blocks=int(len(set(part.values()))))
        if len(set(part.values())) == 2:
            pb = (pred == pred.max()).astype(float)
            o, p, nl = perm_p(acc_bip, pb, y); d.update(acc=round(o, 3), acc_p=p, acc_null=round(nl, 3))
        unsup[nm] = d
res['unsupervised'] = unsup
res['louvain_Q'] = round(best[0], 3)
res['dl'] = dict(B2=round(dl2, 1), Bfree=round(dlF, 1), Bfree_B=int(BF))

# ---------------- supervised LOO ----------------
def logit_fit(X, yy, lam=1.0):
    Xb = np.c_[np.ones(len(X)), X]
    def f(w):
        z = Xb @ w; p = 1 / (1 + np.exp(-z))
        ll = -np.sum(yy * z - np.logaddexp(0, z)) + lam * np.sum(w[1:] ** 2) / 2
        gr = Xb.T @ (p - yy); gr[1:] += lam * w[1:]
        return ll, gr
    return minimize(f, np.zeros(Xb.shape[1]), jac=True, method='L-BFGS-B').x

def loo(X, yy):
    pr = np.zeros(len(yy))
    for i in range(len(yy)):
        m = np.ones(len(yy), bool); m[i] = False
        mu, sd = X[m].mean(0), X[m].std(0) + 1e-9
        w = logit_fit((X[m] - mu) / sd, yy[m])
        pr[i] = 1 / (1 + np.exp(-(w[0] + ((X[i] - mu) / sd) @ w[1:])))
    return pr

def auc(p, yy):
    r = pd.Series(p).rank().values; n1 = yy.sum(); n0 = len(yy) - n1
    return (r[yy == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)

def metrics(p, yy):
    pred = (p > 0.5).astype(float)
    bal = 0.5 * (np.mean(pred[yy == 1] == 1) + np.mean(pred[yy == 0] == 0))
    ll = -np.mean(yy * np.log(np.clip(p, 1e-9, 1)) + (1 - yy) * np.log(np.clip(1 - p, 1e-9, 1)))
    return dict(acc=round(float(np.mean(pred == yy)), 3), bal_acc=round(float(bal), 3),
                auc=round(float(auc(p, yy)), 3), logloss=round(float(ll), 3))

# label-neighbour feature computed inside each LOO fold (uses only other nodes' labels)
pos = {v: i for i, v in enumerate(nodes)}
Wev = Wl[np.ix_([pos[v] for v in ev], [pos[v] for v in ev])]
def neigh_frac(yy):
    s1 = Wev @ yy; s0 = Wev @ (1 - yy)
    n1, n0 = yy.sum(), len(yy) - yy.sum()
    return (s1 / n1) / (s1 / n1 + s0 / n0 + 1e-12)  # self excluded since diag=0
F = dict(
    net_fiedler=np.array([[spec[v]] for v in ev]),
    net_neigh=None,  # built per fold
    sex=sex[ev][:, None], age=np.c_[age[ev], age[ev] ** 2],
    attr=np.c_[sex[ev], age[ev], age[ev] ** 2, sex[ev] * age[ev]],
    space=np.c_[cx[ev], cy[ev]])

def run_sets(mask, yy):
    out = {}
    combos = {'net_fiedler': ['net_fiedler'], 'net_neigh': ['net_neigh'], 'attr(sex,age)': ['attr'],
              'space': ['space'], 'attr+net_fiedler': ['attr', 'net_fiedler'],
              'space+net_fiedler': ['space', 'net_fiedler'], 'attr+space': ['attr', 'space'],
              'attr+space+net_fiedler': ['attr', 'space', 'net_fiedler']}
    def build(keys, ylab):
        cols = []
        for k in keys:
            if k == 'net_neigh':
                cols.append(neigh_frac(ylab)[mask][:, None])
            else:
                cols.append(F[k][mask])
        return np.hstack(cols)
    yfull = y.copy()
    for nm, keys in combos.items():
        if 'net_neigh' in keys:
            # LOO: for node i, neighbour fraction recomputed with label of i hidden (set to prior)
            idxs = np.where(mask)[0]; pr = np.zeros(len(idxs))
            for t, i in enumerate(idxs):
                yh = yfull.copy(); yh[i] = yfull[np.arange(len(yfull)) != i].mean()
                f = neigh_frac(yh)[mask][:, None]
                m = np.ones(len(idxs), bool); m[t] = False
                mu, sd = f[m].mean(0), f[m].std(0) + 1e-9
                w = logit_fit((f[m] - mu) / sd, yy[m])
                pr[t] = 1 / (1 + np.exp(-(w[0] + ((f[t] - mu) / sd) @ w[1:])))
        else:
            X = build(keys, yfull); pr = loo(X, yy)
        mt = metrics(pr, yy)
        # permutation null on AUC (labels permuted, features fixed); neighbour feature is
        # label-dependent so its null is rebuilt from permuted labels
        null = []
        for _ in range(200):
            yp = rng.permutation(yy)
            if 'net_neigh' in keys:
                yfp = yfull.copy(); yfp[mask] = yp
                f = neigh_frac(yfp)[mask][:, None]
                null.append(auc(f[:, 0], yp))  # in-sample AUC of raw feature (fast proxy)
            else:
                null.append(auc(loo(build(keys, yfull), yp), yp))
        null = np.array(null)
        stat = auc(neigh_frac(yfull)[mask], yy) if 'net_neigh' in keys else mt['auc']
        mt['auc_perm_p'] = float((1 + np.sum(null >= stat)) / (len(null) + 1))
        mt['auc_null_mean'] = round(float(null.mean()), 3)
        out[nm] = mt
    return out

full = np.isfinite(sex[ev]) & np.isfinite(age[ev]) & np.isfinite(cx[ev])
res['common_subset'] = dict(n=int(full.sum()), n_label1=int(y[full].sum()),
                            n_male=int(np.nansum(sex[ev][full])),
                            majority_baseline=round(float(max(y[full].mean(), 1 - y[full].mean())), 3))
res['supervised_LOO_common_subset'] = run_sets(full, y[full])
# network-only on all evaluated nodes
res['majority_baseline_all'] = round(float(max(y.mean(), 1 - y.mean())), 3)

# sex composition and age by label (descriptive)
desc = pd.DataFrame(dict(lab=y, sex=sex[ev], age=age[ev], cx=cx[ev], cy=cy[ev]))
res['descriptive'] = json.loads(desc.groupby('lab').agg(n=('sex', 'size'), male_frac=('sex', 'mean'),
                                age_mean=('age', 'mean'), cx=('cx', 'mean'), cy=('cy', 'mean')).round(3).to_json())

# sex-stratified network test: is the split visible within females and within males separately?
strat = {}
for sname, sv in [('males', 1.0), ('females', 0.0)]:
    m = sex[ev] == sv
    if m.sum() > 10:
        pred = np.array([1.0 if spec[v] > 0 else 0.0 for v in ev])[m]
        o, p, nl = perm_p(acc_bip, pred, y[m])
        strat[sname] = dict(n=int(m.sum()), fiedler_acc=round(o, 3), p=p)
res['fiedler_by_sex'] = strat

tag = f'{Y0}_{Y1}'
with open(f'{OUT}/ngogo_results_{tag}.json', 'w') as fh:
    json.dump(res, fh, indent=1, default=float)
pd.DataFrame(dict(node=names[ev], label=y, fiedler=[spec[v] for v in ev], sbm2=[sbm2[v] for v in ev],
                  louvain=[louv[v] for v in ev], sex=sex[ev], age=age[ev], cx=cx[ev], cy=cy[ev])
             ).to_csv(f'{OUT}/ngogo_nodes_{tag}.csv', index=False)
print(json.dumps(res, indent=1, default=float))
