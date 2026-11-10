"""Agta sharing-game gift network (Smith et al. 2018 EHB, 'A friend in need').

Directed ego->alter gift ~ attributes vs network (reciprocity, closure, degree),
held-out prediction with 5-fold x 5 CV over directed dyads.

usage: python -I agta_analysis.py <xlsx> <outdir>   (run with base python: needs openpyxl)
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import common as C

a = pd.read_excel(sys.argv[1])
outdir = sys.argv[2]
a["camp"] = a.camp.astype(str)
a["e"] = a["i.d"].astype(int)
a["l"] = a["alter_id"].astype(int)
print("rows", len(a), "egos", a.e.nunique(), "camps", a.camp.nunique(), "gifts", a.gift.sum())
key = {(e, l): t for t, (e, l) in enumerate(zip(a.e, a.l))}
rev = np.array([key.get((l, e), -1) for e, l in zip(a.e, a.l)])
print("rows whose reverse dyad is present:", (rev >= 0).mean().round(3))
both = rev >= 0
print("recip column vs reverse gift (where reverse present):")
print(pd.crosstab(a.recip[both], a.gift.to_numpy()[rev[both]]))
# completeness per camp
for c, s in a.groupby("camp"):
    k = s.e.nunique()
    print(f"  camp {c}: n={k}, rows={len(s)}, n(n-1)={k*(k-1)}, gifts={s.gift.sum()}")

kin = pd.get_dummies(a.kinship.astype(str), drop_first=False).astype(float)
kin = kin.drop(columns=["0"]) if "0" in kin else kin
X_demo = np.column_stack([a["age.ego"], a["age.alt"], a["age.gap"], a["sex.ego"], a["sex.alt"],
                          a["sexdiff"]]).astype(float)
X_kin = kin.to_numpy()
X_space = pd.get_dummies(a.prox.astype(int), drop_first=True).astype(float).to_numpy()
X_coop = np.column_stack([a["coop.ego"], a["coop.alt"], a["coop.sim"]]).astype(float)
X_need = np.column_stack([a["need"]]).astype(float)
y = a.gift.to_numpy().astype(int)


def net_feats(train):
    """Features from training ties only."""
    gifts = {(e, l) for e, l, t, g in zip(a.e, a.l, train, y) if t and g}
    obs = {(e, l) for e, l, t in zip(a.e, a.l, train) if t}
    outd, ind = {}, {}
    for e, l in gifts:
        outd[e] = outd.get(e, 0) + 1
        ind[l] = ind.get(l, 0) + 1
    out_nb, in_nb = {}, {}
    for e, l in gifts:
        out_nb.setdefault(e, set()).add(l)
        in_nb.setdefault(l, set()).add(e)
    F = []
    for e, l, t, g in zip(a.e, a.l, train, y):
        own = 1 if (t and g) else 0          # leave own tie out of its degree features
        r_obs = (l, e) in obs
        r = 1.0 if (l, e) in gifts else 0.0
        trans = len(out_nb.get(e, set()) & in_nb.get(l, set()))      # e->w->l
        cyc = len(in_nb.get(e, set()) & out_nb.get(l, set()))        # l->w->e
        co_out = len(out_nb.get(e, set()) & out_nb.get(l, set()))    # shared recipients
        co_in = len(in_nb.get(e, set()) & in_nb.get(l, set()))       # shared donors
        F.append([r, float(r_obs), trans, cyc, co_out, co_in,
                  np.log1p(outd.get(e, 0) - own), np.log1p(ind.get(l, 0) - own)])
    F = np.array(F)
    return F[:, :2], F[:, 2:6], F[:, 6:8]


res = {}
rng = np.random.default_rng(3)
for rep in range(5):
    fold = rng.integers(0, 5, len(y))
    for f in range(5):
        tr, te = fold != f, fold == f
        recip, closure, deg = net_feats(tr)
        sets = {
            "demo(sex,age)": [X_demo],
            "kin category": [X_kin],
            "household proximity": [X_space],
            "coop (game) incl. similarity": [X_coop],
            "need": [X_need],
            "attr=demo+kin+prox+coop+need": [X_demo, X_kin, X_space, X_coop, X_need],
            "exog attr=demo+kin+prox+need (no coop)": [X_demo, X_kin, X_space, X_need],
            "exog attr+degree": [X_demo, X_kin, X_space, X_need, deg],
            "exog attr+degree+reciprocity+closure": [X_demo, X_kin, X_space, X_need, deg, recip, closure],
            "degree only": [deg],
            "attr+degree": [X_demo, X_kin, X_space, X_coop, X_need, deg],
            "net: degree+reciprocity": [deg, recip],
            "net: degree+reciprocity+closure": [deg, recip, closure],
            "attr+degree+reciprocity": [X_demo, X_kin, X_space, X_coop, X_need, deg, recip],
            "all": [X_demo, X_kin, X_space, X_coop, X_need, deg, recip, closure],
        }
        for name, bl in sets.items():
            X = np.column_stack(bl)
            m = C.fit_logit(X[tr], y[tr])
            p = C.predict_logit(m, X[te])
            res.setdefault(name, []).append((C.auc(y[te], p), C.logloss(y[te], p)))
base = C.logloss(y, np.full(len(y), y.mean()))
out = {}
print("\nheld-out (5x5-fold CV over directed dyads); null logloss", round(base, 4))
for k, v in res.items():
    A = np.array(v)
    out[k] = dict(auc=A[:, 0].mean(), auc_sd=A[:, 0].std(), logloss=A[:, 1].mean())
    print(f"  {k:36s} AUC {A[:,0].mean():.3f} +- {A[:,0].std():.3f}  logloss {A[:,1].mean():.4f}")

# leave-one-camp-out for attribute models (pure transfer to unseen camps)
print("\nleave-one-camp-out (attributes only; network features need in-camp ties)")
camps = a.camp.to_numpy()
for name, bl in {"attr": [X_demo, X_kin, X_space, X_coop, X_need],
                 "exog attr (no coop)": [X_demo, X_kin, X_space, X_need], "kin": [X_kin],
                 "household proximity": [X_space]}.items():
    X = np.column_stack(bl)
    ps = np.zeros(len(y))
    for c in np.unique(camps):
        te = camps == c
        m = C.fit_logit(X[~te], y[~te])
        ps[te] = C.predict_logit(m, X[te])
    print(f"  {name:20s} pooled AUC {C.auc(y, ps):.3f} logloss {C.logloss(y, ps):.4f}")
json.dump(out, open(os.path.join(outdir, "agta_results.json"), "w"), indent=1)
