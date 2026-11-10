"""Arang Dak: does support-network closure survive once kinship is modelled as a NETWORK?

Kinship enters three ways:
  attribute : dyadic r, affinal r, godparent, household relatedness  (as before)
  kin network: common kin (r>=0.125), kin-weighted 2-paths sum_k r_ik r_kj (consanguineal and
               affinal), kin-graph path length (capped at 4)
Support closure is split by the intermediary k (support ties to both i and j, training ties only):
  CN_kin    : k is kin (r>=0.0625 or affinal>=0.0625) of i or j, or co-resident with i or j
  CN_nonkin : k is unrelated to both and lives in neither household
CN_nonkin is closure that genealogy and co-residence cannot produce.
Held-out AUC, 5-fold x 5 reps over dyads; features recomputed from training ties.
usage: python -I koster_kinnet.py <csv> <out.json>
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import common as C
from scipy.sparse.csgraph import shortest_path

csv, out = sys.argv[1], sys.argv[2]
d = pd.read_csv(csv); d["i"] = d.i_ID - 1; d["j"] = d.j_ID - 1; n = 108
node = {}
for _, r in d.iterrows():
    node[r.i] = (r.i_sex, r.i_age, r.k_ID, r.k_wealth); node[r.j] = (r.j_sex, r.j_age, r.l_ID, r.l_wealth)
sex = np.array([1 if node[v][0] == "M" else 0 for v in range(n)]); age = np.array([node[v][1] for v in range(n)], float)
hh = np.array([node[v][2] for v in range(n)]); wealth = np.array([node[v][3] for v in range(n)], float)
d["a"] = d[["i", "j"]].min(1); d["b"] = d[["i", "j"]].max(1)
d["tie"] = ((d.y == 1) | (d["y.donor.oriented"] == 1)).astype(int)
g = d.groupby(["a", "b"]).agg(tie=("tie", "max"), r=("ij_deg_r", "max"), raff=("ij_affinal_r", "max"),
                              god=("ij_god_relation", "max"), hr=("kl_avg_interhouse_R", "max"),
                              dist=("kl_distance", "max")).reset_index()
pairs = g[["a", "b"]].to_numpy(); ia, ib = pairs[:, 0], pairs[:, 1]; y = g.tie.to_numpy()

R = np.zeros((n, n)); Ra = np.zeros((n, n))
R[ia, ib] = R[ib, ia] = g.r; Ra[ia, ib] = Ra[ib, ia] = g.raff
same_hh = hh[:, None] == hh[None, :]; np.fill_diagonal(same_hh, False)
# kin network
K = (R >= 0.125).astype(float)
kin_cn = (K @ K)[ia, ib]
kin2 = (R @ R)[ia, ib]; aff2 = (Ra @ Ra + R @ Ra + Ra @ R)[ia, ib]
Kh = K.copy(); Kh[same_hh] = 1  # co-residents linked in the kin graph (genealogy unknown within hh)
sp = shortest_path(Kh, unweighted=True, directed=False); kin_path = np.minimum(sp[ia, ib], 5)
feat_kinnet = np.column_stack([kin_cn, kin2, aff2, kin_path]).astype(float)
related = (R >= 0.0625) | (Ra >= 0.0625) | same_hh  # i~k relation (incl. co-residence)

feat_demo = np.column_stack([sex[ia] == sex[ib], sex[ia] + sex[ib], np.abs(age[ia] - age[ib]),
    np.minimum(age[ia], age[ib]), np.maximum(age[ia], age[ib]),
    np.log(np.minimum(wealth[ia], wealth[ib])), np.log(np.maximum(wealth[ia], wealth[ib]))]).astype(float)
feat_kin = np.column_stack([g.r, g.raff, g.god, np.log(g.hr)]).astype(float)
feat_space = np.log(g.dist.to_numpy())[:, None]

rng = np.random.default_rng(11); res = {}
for rep in range(5):
    fold = rng.integers(0, 5, len(y))
    for f in range(5):
        tr, te = fold != f, fold == f
        A = np.zeros((n, n)); e = pairs[tr & (y == 1)]; A[e[:, 0], e[:, 1]] = A[e[:, 1], e[:, 0]] = 1
        k = A.sum(1); ki, kj = k[ia] - A[ia, ib], k[ib] - A[ia, ib]
        deg = np.column_stack([np.log1p(ki), np.log1p(kj), np.log1p(ki * kj)])
        # split common support partners by relation of intermediary to i or j
        rel_any = related[ia] | related[ib]          # (pairs x n): k related to i or j
        both = A[ia] * A[ib]                          # k supports-tied to both
        cn_kin = (both * rel_any).sum(1); cn_non = (both * ~rel_any).sum(1)
        cl_kin = np.column_stack([cn_kin]); cl_non = np.column_stack([cn_non])
        base = [feat_demo, feat_kin, feat_space, feat_kinnet, deg]
        sets = {"attr+kinnet+degree": base,
                "+CN via kin/co-residents": base + [cl_kin],
                "+CN via non-kin": base + [cl_non],
                "+both": base + [cl_kin, cl_non],
                "attr(no kinnet)+degree+both": [feat_demo, feat_kin, feat_space, deg, cl_kin, cl_non],
                "attr+degree (no kinnet)": [feat_demo, feat_kin, feat_space, deg], "kinnet only": [feat_kinnet], "attr only": [feat_demo, feat_kin, feat_space]}
        for nm, bl in sets.items():
            X = np.column_stack(bl); m = C.fit_logit(X[tr], y[tr])
            res.setdefault(nm, []).append(C.auc(y[te], C.predict_logit(m, X[te])))
        # coefficient of CN_nonkin in the full-data fit (standardised)
o = {k: dict(auc=round(float(np.mean(v)), 3), sd=round(float(np.std(v)), 3)) for k, v in res.items()}
b = np.array(res["attr+kinnet+degree"])
for k in res: o[k]["delta_vs_base"] = round(float(np.mean(np.array(res[k]) - b)), 4); o[k]["delta_sd"] = round(float(np.std(np.array(res[k]) - b)), 4); o[k]["folds_better"] = int(np.sum(np.array(res[k]) > b))
# descriptive: share of common partners that are non-kin, among tied vs untied pairs
A = np.zeros((n, n)); e = pairs[y == 1]; A[e[:, 0], e[:, 1]] = A[e[:, 1], e[:, 0]] = 1
both = A[ia] * A[ib]; rel_any = related[ia] | related[ib]
o["desc"] = dict(mean_cn_nonkin_tied=round(float((both * ~rel_any).sum(1)[y == 1].mean()), 3),
                 mean_cn_nonkin_untied=round(float((both * ~rel_any).sum(1)[y == 0].mean()), 3),
                 mean_cn_kin_tied=round(float((both * rel_any).sum(1)[y == 1].mean()), 3),
                 mean_cn_kin_untied=round(float((both * rel_any).sum(1)[y == 0].mean()), 3),
                 frac_pairs_unrelated=round(float(1 - related[ia, ib].mean()), 3))
json.dump(o, open(out, "w"), indent=1); print(json.dumps(o, indent=1))
