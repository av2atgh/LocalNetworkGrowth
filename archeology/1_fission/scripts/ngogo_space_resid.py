"""Ngogo: does the social network predict the fission beyond spatial overlap?

For individuals with >= 30 GPS fixes in the window: pairwise home-range overlap (Bhattacharyya
affinity of Gaussian KDEs on a 100 m grid) and centroid distance. Edge weights log(1+w) are
regressed on spline-like bins of overlap and distance; the residual network carries association
not explained by shared space.

Tests (labelled individuals in GPS set):
  - residual neighbour vote: score_i = sum_j r_ij (2 y_j - 1), AUC for y_i.
  - overlap-only vote (same formula with the fitted spatial component) for comparison.
  - null: residuals permuted among pairs within joint overlap x distance bins (keeps the spatial
    structure of every pair, scrambles who is socially tied beyond it), 1000 draws.
  - conditional: LOO logistic of y on [space centroid] vs [space centroid + residual vote].
Usage: python -I ngogo_space_resid.py <ngogo_dir> <codemap.csv> <out.json> [y0 y1]
"""
import sys, json, os, numpy as np, pandas as pd
from scipy.optimize import minimize
D, CM, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
Y0, Y1 = (int(sys.argv[4]), int(sys.argv[5])) if len(sys.argv) > 5 else (2011, 2014)
rng = np.random.default_rng(3)

names = np.load(f'{D}/learned/L/data/chimps_names.npy')
lab = np.load(f'{D}/learned/L/src/combined_membership_labels.npy')
N = len(names)
A = np.zeros((N, N))
for yy in range(Y0, Y1 + 1):
    for m in range(1, 13):
        f = f'{D}/learned/L/data/learned_graphs/{os.environ.get("LAYER", "full")}_array_{yy}_{m}.npy'
        if os.path.exists(f): A += np.nan_to_num(np.load(f))
A = (A + A.T) / 2; np.fill_diagonal(A, 0)

g = pd.read_csv(f'{D}/space/space-use-archive/data/gps_qualified.csv')
g['key'] = g.Individual.str.lower().str.replace(' ', '')
cm = pd.read_csv(CM); cm = cm[cm.jaccard >= 0.5]
code2name = {r.code: r.name.lower().replace(' ', '') for r in cm.itertuples()}
key = [code2name.get(n, None) if i < 97 else n for i, n in enumerate(names)]
gw = g[(g.Year >= Y0) & (g.Year <= Y1)].copy()
lat0, lon0 = g.Lat.mean(), g.Lon.mean()
gw['x'] = (gw.Lon - lon0) * 111.32 * np.cos(np.radians(lat0)); gw['y'] = (gw.Lat - lat0) * 110.57
cnt = gw.groupby('key').size()

idx = [i for i in range(N) if key[i] is not None and key[i] in cnt.index and cnt[key[i]] >= 30 and A[i].sum() > 0]
n = len(idx)
# KDE on grid
gx = np.arange(gw.x.min() - 1, gw.x.max() + 1, 0.1); gy = np.arange(gw.y.min() - 1, gw.y.max() + 1, 0.1)
X, Y = np.meshgrid(gx, gy); H = 0.3
P = []; C = []
for i in idx:
    pts = gw[gw.key == key[i]][['x', 'y']].values
    dens = np.zeros_like(X)
    for px, py in pts:  # fixes are few hundred per individual
        dens += np.exp(-((X - px) ** 2 + (Y - py) ** 2) / (2 * H * H))
    P.append((dens / dens.sum()).ravel()); C.append(np.median(pts, 0))
P = np.array(P); C = np.array(C)
BA = np.sqrt(P) @ np.sqrt(P).T
Dc = np.sqrt(((C[:, None] - C[None]) ** 2).sum(-1))
W = np.log1p(A[np.ix_(idx, idx)])
I, J = np.triu_indices(n, 1)
ov, dd, w = BA[I, J], Dc[I, J], W[I, J]
qo = np.quantile(ov, np.linspace(0, 1, 11)[1:-1]); qd = np.quantile(dd, np.linspace(0, 1, 11)[1:-1])
bo, bd = np.searchsorted(qo, ov), np.searchsorted(qd, dd)
Xs = np.column_stack([np.eye(10)[bo], np.eye(10)[bd][:, 1:], ov, ov ** 2, np.log1p(dd)])
beta, *_ = np.linalg.lstsq(Xs, w, rcond=None)
fit = Xs @ beta; res = w - fit
r2 = 1 - res.var() / w.var()
R = np.zeros((n, n)); R[I, J] = res; R += R.T
Fm = np.zeros((n, n)); Fm[I, J] = fit; Fm += Fm.T
yl = lab[idx]; L = yl >= 0
y = yl[L].astype(float); RL, FL, WL = R[np.ix_(L, L)], Fm[np.ix_(L, L)], W[np.ix_(L, L)]

def auc(s, yy):
    r = pd.Series(s).rank().values; n1 = yy.sum(); n0 = len(yy) - n1
    return (r[yy == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
def vote(M, yy):
    return M @ (2 * yy - 1)  # diag 0 -> own label excluded
def bal(s, yy):
    p = (s > 0).astype(float); return 0.5 * (np.mean(p[yy == 1] == 1) + np.mean(p[yy == 0] == 0))

out = dict(window=[Y0, Y1], n_gps=n, n_labelled=int(L.sum()), n_label1=int(y.sum()),
           R2_weights_on_space=round(float(r2), 3))
for nm, M in [('raw_network', WL), ('spatial_fit', FL), ('residual_network', RL)]:
    s = vote(M, y); out[nm] = dict(auc=round(float(auc(s, y)), 3), bal_acc=round(float(bal(s, y)), 3))
# stratified permutation null for the residual vote
cell = bo * 10 + bd
Lidx = np.where(L)[0]; obs = auc(vote(RL, y), y); null = []
for _ in range(1000):
    rp = res.copy()
    for c in np.unique(cell):
        k = np.where(cell == c)[0]; rp[k] = rp[rng.permutation(k)]
    Rp = np.zeros((n, n)); Rp[I, J] = rp; Rp += Rp.T
    null.append(auc(vote(Rp[np.ix_(L, L)], y), y))
null = np.array(null)
out['residual_network']['null_mean'] = round(float(null.mean()), 3)
out['residual_network']['null_q95'] = round(float(np.quantile(null, .95)), 3)
out['residual_network']['p'] = float((1 + np.sum(null >= obs)) / 1001)
# within-group-boundary: individuals whose centroid is ambiguous (space vote margin small)
sf = vote(FL, y); amb = np.abs(sf) <= np.quantile(np.abs(sf), 0.33)
out['ambiguous_space_third'] = dict(n=int(amb.sum()),
    space_bal=round(float(bal(sf[amb], y[amb])), 3), resid_bal=round(float(bal(vote(RL, y)[amb], y[amb])), 3),
    resid_auc=round(float(auc(vote(RL, y)[amb], y[amb])), 3) if 0 < y[amb].sum() < amb.sum() else None)
json.dump(out, open(OUT, 'w'), indent=1); print(json.dumps(out, indent=1))

# ---- matched-space test: among labelled pairs, are ties to future groupmates stronger than ties
# across the future line at the same overlap x distance cell?  Node bootstrap CI.
lab_full = np.full(n, -1); lab_full[Lidx] = y
pi, pj = lab_full[I], lab_full[J]; keep = (pi >= 0) & (pj >= 0)
same = (pi == pj).astype(float)
def matched(wt, kk):
    tot, den = 0.0, 0.0
    for c in np.unique(cell[kk]):
        k = kk & (cell == c)
        s, d = k & (same == 1), k & (same == 0)
        ws, wd = wt[s].sum(), wt[d].sum()
        if ws > 0 and wd > 0:
            m = min(ws, wd)
            tot += m * ((wt[s] * w[s]).sum() / ws - (wt[d] * w[d]).sum() / wd); den += m
    return tot / den if den else np.nan
base = np.ones(len(w))
eff = matched(base, keep)
# regression version: w ~ spatial design + same
Xr = np.column_stack([Xs[keep], same[keep]]); br, *_ = np.linalg.lstsq(Xr, w[keep], rcond=None)
boots, bootr = [], []
for _ in range(1000):
    c = np.bincount(rng.integers(n, size=n), minlength=n).astype(float)
    wt = c[I] * c[J]; kk = keep & (wt > 0)
    boots.append(matched(wt, kk))
    sw = np.sqrt(wt[kk]); b2, *_ = np.linalg.lstsq(Xr[(wt > 0)[keep]] * sw[:, None], w[kk] * sw, rcond=None)
    bootr.append(b2[-1])
nc = [int(((cell == c) & keep & (same == 1)).sum() > 0 and ((cell == c) & keep & (same == 0)).sum() > 0) for c in np.unique(cell[keep])]
out['matched_same_group'] = dict(
    n_pairs=int(keep.sum()), n_cells_with_both=int(sum(nc)),
    pairs_in_mixed_cells=int(sum(((cell == c) & keep).sum() for c in np.unique(cell[keep]) if ((cell == c) & keep & (same == 1)).any() and ((cell == c) & keep & (same == 0)).any())),
    mean_logw_same=round(float(w[keep & (same == 1)].mean()), 3), mean_logw_diff=round(float(w[keep & (same == 0)].mean()), 3),
    matched_diff=round(float(eff), 3), matched_ci=[round(float(x), 3) for x in np.nanpercentile(boots, [2.5, 97.5])],
    regression_same_coef=round(float(br[-1]), 3), regression_ci=[round(float(x), 3) for x in np.percentile(bootr, [2.5, 97.5])],
    sd_logw=round(float(w[keep].std()), 3))
json.dump(out, open(OUT, 'w'), indent=1); print(json.dumps(out['matched_same_group'], indent=1))

# ---- same test, also matched on sex pair (MM/MF/FF): males are focal and dominate weights
sx = gw.groupby('key').Sex.first()
sexv = np.array([1 if sx.get(key[i], 'M') == 'M' else 0 for i in idx])
sp = sexv[I] + sexv[J]  # 0 FF, 1 MF, 2 MM
cell_s = cell * 3 + sp
Xs2 = np.column_stack([Xs, np.eye(3)[sp][:, 1:]])
cell_backup = cell; cell = cell_s
eff2 = matched(base, keep)
Xr2 = np.column_stack([Xs2[keep], same[keep]]); b2r, *_ = np.linalg.lstsq(Xr2, w[keep], rcond=None)
bs = []
for _ in range(500):
    c = np.bincount(rng.integers(n, size=n), minlength=n).astype(float)
    wt = c[I] * c[J]; bs.append(matched(wt, keep & (wt > 0)))
bysex = {}
for k_, nm in [(2, 'MM'), (1, 'MF'), (0, 'FF')]:
    kk = keep & (sp == k_)
    bysex[nm] = dict(n=int(kk.sum()), same=round(float(w[kk & (same == 1)].mean()), 3), diff=round(float(w[kk & (same == 0)].mean()), 3),
                     matched=round(float(matched(base, kk)), 3))
out['matched_same_group_sexpair'] = dict(matched_diff=round(float(eff2), 3), ci=[round(float(x), 3) for x in np.nanpercentile(bs, [2.5, 97.5])],
                                         regression_coef=round(float(b2r[-1]), 3), by_sexpair=bysex)
cell = cell_backup
json.dump(out, open(OUT, 'w'), indent=1); print(json.dumps(out['matched_same_group_sexpair'], indent=1))
