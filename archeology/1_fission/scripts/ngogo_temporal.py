"""Year-by-year: how well do (i) the unsupervised spectral cut of the year's network,
(ii) the network neighbour vote, (iii) the GPS centroid (space), (iv) sex+age
predict eventual (2018-2022) membership?  Same inputs as ngogo_fission.py.
Usage: python -I ngogo_temporal.py <ngogo_data_dir> <codemap.csv> <out_csv>"""
import sys, os, numpy as np, pandas as pd, networkx as nx
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sklearn_free import ari
from scipy.optimize import minimize

D, CM, OUT = sys.argv[1:4]
rng = np.random.default_rng(2)
names = np.load(f'{D}/learned/L/data/chimps_names.npy')
lab = np.load(f'{D}/learned/L/src/combined_membership_labels.npy').astype(float)
N = len(names)
g = pd.read_csv(f'{D}/space/space-use-archive/data/gps_qualified.csv')
g['key'] = g.Individual.str.lower().str.replace(' ', '')
g['birth'] = g.Year + (g.Month - 0.5) / 12 - g.Age_at_sampling
lat0, lon0 = g.Lat.mean(), g.Lon.mean()
g['x'] = (g.Lon - lon0) * 111.32 * np.cos(np.radians(lat0)); g['yk'] = (g.Lat - lat0) * 110.57
cm = pd.read_csv(CM); cm = cm[cm.jaccard >= 0.5]
c2n = {r.code: r.name.lower().replace(' ', '') for r in cm.itertuples()}
key = [c2n.get(n) if i < 97 else n for i, n in enumerate(names)]
info = g.groupby('key').agg(sex=('Sex', 'first'), birth=('birth', 'median'))
sex = np.array([1.0 if i < 97 else (1.0 if (key[i] in info.index and info.loc[key[i], 'sex'] == 'M')
               else (0.0 if key[i] in info.index else np.nan)) for i in range(N)])
birth = np.array([info.loc[k, 'birth'] if k in info.index else np.nan for k in key])

def logit_loo(X, y, lam=1.0):
    pr = np.zeros(len(y))
    for i in range(len(y)):
        m = np.ones(len(y), bool); m[i] = False
        mu, sd = X[m].mean(0), X[m].std(0) + 1e-9
        Xb = np.c_[np.ones(m.sum()), (X[m] - mu) / sd]; yy = y[m]
        def f(w):
            z = Xb @ w; p = 1 / (1 + np.exp(-z))
            return (-np.sum(yy * z - np.logaddexp(0, z)) + lam * np.sum(w[1:] ** 2) / 2,
                    Xb.T @ (p - yy) + np.r_[0, lam * w[1:]])
        w = minimize(f, np.zeros(Xb.shape[1]), jac=True, method='L-BFGS-B').x
        pr[i] = 1 / (1 + np.exp(-(w[0] + ((X[i] - mu) / sd) @ w[1:])))
    return pr

def bal(pred, y):
    return 0.5 * (np.mean(pred[y == 1] == 1) + np.mean(pred[y == 0] == 0))

rows = []
for yr in list(range(1998, 2020)):
    A = np.zeros((N, N))
    for m in range(1, 13):
        f = f'{D}/learned/L/data/learned_graphs/full_array_{yr}_{m}.npy'
        if os.path.exists(f): A += np.nan_to_num(np.load(f))
    A = (A + A.T) / 2; np.fill_diagonal(A, 0)
    G = nx.from_numpy_array(A); G.remove_nodes_from([i for i in range(N) if A[i].sum() == 0])
    if G.number_of_nodes() < 10: continue
    nodes = np.array(sorted(max(nx.connected_components(G), key=len)))
    W = np.log1p(A[np.ix_(nodes, nodes)]); d = W.sum(1); Dm = 1 / np.sqrt(d)
    vals, vecs = np.linalg.eigh(np.eye(len(W)) - Dm[:, None] * W * Dm[None, :])
    fv = Dm * vecs[:, 1]
    evm = lab[nodes] >= 0
    ev = nodes[evm]; y = lab[ev]
    pred = (fv[evm] > 0).astype(float)
    acc = max(np.mean(pred == y), 1 - np.mean(pred == y))
    bacc = max(bal(pred, y), bal(1 - pred, y))
    null = [max(bal(pred, p), bal(1 - pred, p)) for p in (rng.permutation(y) for _ in range(500))]
    # neighbour vote (LOO-safe: diag zero, own label never used)
    We = W[np.ix_(evm, evm)]
    s1 = We @ y; s0 = We @ (1 - y); n1, n0 = y.sum(), len(y) - y.sum()
    nv = ((s1 / n1) > (s0 / n0)).astype(float)
    row = dict(year=yr, n_gcc=len(nodes), n_eval=len(ev), n_lab1=int(y.sum()),
               fiedler_acc=round(acc, 3), fiedler_balacc=round(bacc, 3),
               fiedler_balacc_null95=round(float(np.quantile(null, .95)), 3),
               fiedler_ari=round(ari(pred, y), 3), neighvote_balacc=round(bal(nv, y), 3))
    # space and attributes, same year, labeled individuals with >=10 GPS points that year
    gy = g[g.Year == yr].groupby('key').agg(cx=('x', 'median'), cy=('yk', 'median'), n=('x', 'size'))
    gy = gy[gy.n >= 10]
    idx = [i for i in range(N) if lab[i] >= 0 and key[i] in gy.index]
    if len(idx) >= 20:
        ys = lab[idx]
        Xs = np.array([[gy.loc[key[i], 'cx'], gy.loc[key[i], 'cy']] for i in idx])
        ps = logit_loo(Xs, ys)
        row.update(space_n=len(idx), space_balacc=round(bal((ps > .5).astype(float), ys), 3))
        ia = [t for t, i in enumerate(idx) if np.isfinite(birth[idx[t]])]
        Xa = np.array([[sex[idx[t]], yr - birth[idx[t]], (yr - birth[idx[t]]) ** 2] for t in ia])
        pa = logit_loo(Xa, ys[ia])
        row.update(attr_n=len(ia), attr_balacc=round(bal((pa > .5).astype(float), ys[ia]), 3))
        # network restricted to the same individuals (those in this year's GCC)
        common = [t for t, i in enumerate(idx) if i in set(nodes)]
        if len(common) >= 20:
            pos = {v: k for k, v in enumerate(nodes)}
            fc = np.array([fv[pos[idx[t]]] for t in common]); yc = ys[common]
            pc = (fc > 0).astype(float)
            row.update(net_on_space_set_n=len(common),
                       net_on_space_set_balacc=round(max(bal(pc, yc), bal(1 - pc, yc)), 3))
    rows.append(row); print(row, flush=True)
pd.DataFrame(rows).to_csv(OUT, index=False)
