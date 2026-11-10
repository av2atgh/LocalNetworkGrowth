"""Arang Dak (Mayangna/Miskito, Nicaragua) support network, Koster 2018 RSOS.

Attribute vs network models of tie formation, by held-out dyad prediction,
plus SBM partitions vs attributes.

usage: python -I koster_analysis.py <csv> <outdir>
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
import common as C

csv, outdir = sys.argv[1], sys.argv[2]
os.makedirs(outdir, exist_ok=True)
d = pd.read_csv(csv)
d["i"] = d.i_ID - 1
d["j"] = d.j_ID - 1
n = 108

# ---- node attributes
node = {}
for _, r in d.iterrows():
    node[r.i] = (r.i_sex, r.i_age, r.k_ID, r.k_wealth)
    node[r.j] = (r.j_sex, r.j_age, r.l_ID, r.l_wealth)
sex = np.array([1 if node[v][0] == "M" else 0 for v in range(n)])
age = np.array([node[v][1] for v in range(n)], float)
hh = np.array([node[v][2] for v in range(n)])
wealth = np.array([node[v][3] for v in range(n)], float)
agegrp = np.digitize(age, np.quantile(age, [1 / 3, 2 / 3]))
sexage = sex * 3 + agegrp

# ---- undirected dyads (between households only)
d["a"] = d[["i", "j"]].min(1)
d["b"] = d[["i", "j"]].max(1)
d["tie"] = ((d.y == 1) | (d["y.donor.oriented"] == 1)).astype(int)
d["tie_recv"] = d.y.astype(int)
g = d.groupby(["a", "b"]).agg(tie=("tie", "max"), tie_recv=("tie_recv", "max"),
                              r=("ij_deg_r", "max"), raff=("ij_affinal_r", "max"),
                              god=("ij_god_relation", "max"),
                              hr=("kl_avg_interhouse_R", "max"),
                              dist=("kl_distance", "max")).reset_index()
pairs = g[["a", "b"]].to_numpy()
ia, ib = pairs[:, 0], pairs[:, 1]
print("individuals", n, "households", len(np.unique(hh)), "between-hh dyads", len(g),
      "ties(union)", g.tie.sum(), "density", round(g.tie.mean(), 3),
      "ties(recipient report only)", g.tie_recv.sum())

# household distance matrix -> spatial clusters
H = np.unique(hh)
hidx = {h: t for t, h in enumerate(H)}
Dh = np.zeros((len(H), len(H)))
for _, r in d.iterrows():
    Dh[hidx[r.k_ID], hidx[r.l_ID]] = Dh[hidx[r.l_ID], hidx[r.k_ID]] = np.log(r.kl_distance)
Lk = linkage(squareform(Dh, checks=False), "average")

within = [(u, v) for u in range(n) for v in range(u + 1, n) if hh[u] == hh[v]]
print("within-household pairs (not surveyed):", len(within))

feat_demo = np.column_stack([
    sex[ia] == sex[ib], sex[ia] + sex[ib], np.abs(age[ia] - age[ib]),
    np.minimum(age[ia], age[ib]), np.maximum(age[ia], age[ib]),
    np.log(np.minimum(wealth[ia], wealth[ib])), np.log(np.maximum(wealth[ia], wealth[ib]))
]).astype(float)
feat_kin = np.column_stack([g.r, g.raff, g.god, np.log(g.hr)]).astype(float)
feat_space = np.log(g.dist.to_numpy())[:, None]

y_all = {"union": g.tie.to_numpy(), "recipient": g.tie_recv.to_numpy()}


def run_cv(y, add_within, nfold=5, nrep=2, seed=1):
    rng = np.random.default_rng(seed)
    res = {}
    for rep in range(nrep):
        fold = rng.integers(0, nfold, len(y))
        for f in range(nfold):
            tr, te = fold != f, fold == f
            train_edges = [tuple(p) for p in pairs[tr & (y == 1)]]
            if add_within:
                train_edges += within
            nb = C.neighbour_features(n, train_edges, pairs)
            gtr = C.build_graph(n, train_edges)
            st = C.fit_sbm(gtr, nrep=3, seed=100 * rep + f)
            bl = np.array(st.get_blocks().a)
            lam_sbm = np.log(C.dcsbm_scores(n, train_edges, bl, pairs) + 1e-3)
            lam_hh = np.log(C.dcsbm_scores(n, train_edges, hh, pairs) + 1e-3)
            lam_sa = np.log(C.dcsbm_scores(n, train_edges, sexage, pairs) + 1e-3)
            sp = fcluster(Lk, t=max(2, len(np.unique(bl))), criterion="maxclust")
            sp_node = sp[[hidx[h] for h in hh]]
            lam_sp = np.log(C.dcsbm_scores(n, train_edges, sp_node, pairs) + 1e-3)
            deg = nb[:, 3:6]
            closure = nb[:, 0:3]
            sets = {
                "demo(sex,age,wealth)": [feat_demo],
                "kin": [feat_kin],
                "space(log dist)": [feat_space],
                "attr=demo+kin+space": [feat_demo, feat_kin, feat_space],
                "degree only": [deg],
                "attr+degree": [feat_demo, feat_kin, feat_space, deg],
                "net: degree+closure": [deg, closure],
                "net: degree+SBM": [deg, lam_sbm[:, None]],
                "net: degree+closure+SBM": [deg, closure, lam_sbm[:, None]],
                "attr-block DC-SBM: household": [lam_hh[:, None]],
                "attr-block DC-SBM: sex x age": [lam_sa[:, None]],
                "attr-block DC-SBM: spatial clusters": [lam_sp[:, None]],
                "attr+degree+closure": [feat_demo, feat_kin, feat_space, deg, closure],
                "attr+degree+SBM": [feat_demo, feat_kin, feat_space, deg, lam_sbm[:, None]],
                "all": [feat_demo, feat_kin, feat_space, deg, closure, lam_sbm[:, None]],
            }
            for name, blocks in sets.items():
                X = np.column_stack(blocks)
                m = C.fit_logit(X[tr], y[tr])
                p = C.predict_logit(m, X[te])
                res.setdefault(name, []).append((C.auc(y[te], p), C.logloss(y[te], p)))
    out = {k: dict(auc=float(np.mean([a for a, _ in v])), auc_sd=float(np.std([a for a, _ in v])),
                   logloss=float(np.mean([l for _, l in v]))) for k, v in res.items()}
    return out


results = {}
for yname in ["union", "recipient"]:
    for aw in [False, True]:
        key = f"tie={yname}, within-hh pairs as edges={aw}"
        print("\n==", key)
        r = run_cv(y_all[yname], aw)
        base = C.logloss(y_all[yname], np.full(len(g), y_all[yname].mean()))
        for k, v in r.items():
            print(f"  {k:40s} AUC {v['auc']:.3f} +- {v['auc_sd']:.3f}  logloss {v['logloss']:.4f}"
                  f"  (null {base:.4f})")
        results[key] = r

# ---- full-graph SBM vs attributes
print("\n== Full-graph SBM partitions vs attributes")
comm = {}
for aw in [False, True]:
    edges = [tuple(p) for p in pairs[y_all["union"] == 1]] + (within if aw else [])
    G = C.build_graph(n, edges)
    st = C.fit_sbm(G, nrep=20, seed=7)
    bl = np.array(st.get_blocks().a)
    nst = C.fit_sbm(G, nrep=5, seed=8, nested=True)
    lv = [np.array(s.get_blocks().a) for s in nst.get_levels()]
    # project nested level-1 labels to nodes
    l0 = np.array(nst.get_levels()[0].get_blocks().a)
    l1 = np.array(nst.get_levels()[1].get_blocks().a)[l0]
    B = len(np.unique(bl))
    sp = fcluster(Lk, t=max(2, B), criterion="maxclust")[[hidx[h] for h in hh]]
    # kinship partition: SBM on consanguineal r >= 0.125 graph (between-household pairs + within-hh)
    kin_edges = [tuple(p) for p in pairs[g.r.to_numpy() >= 0.125]] + within
    kst = C.fit_sbm(C.build_graph(n, kin_edges), nrep=10, seed=9)
    kb = np.array(kst.get_blocks().a)
    labels = {"household": hh, "sex": sex, "age tercile": agegrp, "sex x age": sexage,
              f"spatial clusters (k={B})": sp, "kin-graph SBM blocks": kb}
    tag = f"within-hh as edges={aw}"
    rows = []
    for part_name, part in [("flat DC-SBM", bl), ("nested level 0", l0), ("nested level 1", l1)]:
        for ln, lab in labels.items():
            o, mnull, p = C.perm_test(part, lab, C.nmi, nperm=1000)
            a_o = C.ari(part, lab)
            rows.append(dict(partition=part_name, B=int(len(np.unique(part))), attribute=ln,
                             NMI=round(o, 3), NMI_null=round(mnull, 3), p=round(p, 4),
                             ARI=round(a_o, 3)))
    df = pd.DataFrame(rows)
    print("\n--", tag, "| flat B =", B, "| nested B per level:",
          [len(np.unique(x)) for x in lv][:4])
    print(df.to_string(index=False))
    # description lengths
    dls = {"free DC-SBM (flat)": st.entropy(), "household blocks": C.dl_fixed(G, hh),
           "sex x age blocks": C.dl_fixed(G, sexage), "spatial clusters": C.dl_fixed(G, sp),
           "kin-graph blocks": C.dl_fixed(G, kb), "one block": C.dl_fixed(G, np.zeros(n, int))}
    print("  description length (nats):", {k: round(v, 1) for k, v in dls.items()})
    comm[tag] = dict(table=rows, DL={k: float(v) for k, v in dls.items()}, B=B)

json.dump(dict(cv=results, communities=comm), open(os.path.join(outdir, "koster_results.json"), "w"),
          indent=1)
