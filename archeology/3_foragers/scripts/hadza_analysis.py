"""Hadza camps (Fedurek et al. 2024, Hunter Gatherer Research): GPS proximity vs friendship nominations.

No individual attributes in the public files (kinship matrices are in the Dryad set, not
downloadable without a token). Tests: friendship ~ proximity (contact/space) vs closure
(common friends) vs degree, held-out; and SBM partitions of proximity vs friendship.

usage: python -I hadza_analysis.py <datadir> <outdir>
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import common as C
import graph_tool.all as gt
gt.openmp_set_num_threads(1)

dd, outdir = sys.argv[1], sys.argv[2]
out = {}
for camp in [1, 2, 3]:
    P = pd.read_csv(os.path.join(dd, f"Camp{camp}_proximity.csv"), index_col=0)
    F = pd.read_csv(os.path.join(dd, f"Camp{camp}_friendship.csv"), index_col=0)
    F = F.loc[P.index, P.index]
    n = len(P)
    Pm = P.to_numpy(float)
    Fm = F.to_numpy(float)
    Fu = ((Fm > 0) | (Fm.T > 0)).astype(int)
    iu = np.triu_indices(n, 1)
    pairs = np.column_stack(iu)
    yf = Fu[iu]
    prox = Pm[iu]
    print(f"\n== Camp {camp}: n={n}, friendship (undirected) ties={yf.sum()}, "
          f"density={yf.mean():.3f}, proximity>0 frac={np.mean(prox>0):.2f}, "
          f"nomination values={sorted(np.unique(Fm))}")
    if n < 15:
        print("  too small for CV/SBM; skipped")
        continue
    # correlation friendship vs proximity
    print("  AUC of raw proximity for friendship (in-sample, no fit):", round(C.auc(yf, prox), 3))
    # CV: friendship ~ proximity vs closure vs degree
    res = {}
    rng = np.random.default_rng(5)
    for rep in range(10):
        fold = rng.integers(0, 5, len(yf))
        for f in range(5):
            tr, te = fold != f, fold == f
            if yf[te].sum() == 0 or yf[te].sum() == te.sum():
                continue
            edges = [tuple(p) for p in pairs[tr & (yf == 1)]]
            nb = C.neighbour_features(n, edges, pairs)
            deg, clo = nb[:, 3:6], nb[:, 0:3]
            lp = np.log1p(prox)[:, None]
            # proximity-network closure: weighted common neighbours in proximity graph
            Pn = Pm / Pm.max()
            pcn = (Pn @ Pn)[iu][:, None]
            for name, bl in {"proximity (dyad)": [lp], "proximity + prox-closure": [lp, pcn],
                             "degree": [deg], "degree+friend-closure": [deg, clo],
                             "proximity+degree": [lp, deg],
                             "proximity+degree+friend-closure": [lp, deg, clo]}.items():
                X = np.column_stack(bl)
                m = C.fit_logit(X[tr], yf[tr])
                p = C.predict_logit(m, X[te])
                res.setdefault(name, []).append(C.auc(yf[te], p))
    for k, v in res.items():
        print(f"  {k:34s} held-out AUC {np.mean(v):.3f} +- {np.std(v):.3f}")
    # SBM partitions
    gF = C.build_graph(n, [tuple(p) for p in pairs[yf == 1]])
    sF = C.fit_sbm(gF, nrep=20, seed=1)
    bF = np.array(sF.get_blocks().a)
    # proximity graph: keep top edges to match friendship density (and 2x)
    stats = {}
    for mult in [1, 2]:
        m = int(mult * yf.sum())
        thr = np.sort(prox)[::-1][m - 1]
        gP = C.build_graph(n, [tuple(p) for p in pairs[prox >= thr]])
        sP = C.fit_sbm(gP, nrep=20, seed=2)
        bP = np.array(sP.get_blocks().a)
        o, nm, pv = C.perm_test(bF, bP, C.nmi, nperm=2000)
        stats[f"prox top {mult}x"] = dict(B_prox=int(len(np.unique(bP))), NMI=o, NMI_null=nm, p=pv)
    print("  friendship SBM B =", len(np.unique(bF)), "| vs proximity SBM:",
          {k: {kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in v.items()}
           for k, v in stats.items()})
    out[camp] = dict(n=n, cv={k: float(np.mean(v)) for k, v in res.items()},
                     B_friend=int(len(np.unique(bF))), sbm=stats)
json.dump(out, open(os.path.join(outdir, "hadza_results.json"), "w"), indent=1, default=float)
