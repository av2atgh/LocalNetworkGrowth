#!/usr/bin/env python3
"""Does the density of communities set the Ramsey community number?

Chapter 6 measures r_kappa, the size past which communities are certain, and
chapter 7 measures <n_B> = n/B, the mean number of nodes per community, which
stops depending on size.  This asks whether the two are the same statement:

    kappa(n) ~ n/<n_B>   =>   <kappa> reaches two at  n_2 = 2 <n_B>
    r_kappa  = 2 <n_B> * tau,   tau the gap between the mean crossing and the
                                ninety-five per cent one

Both quantities are read off the *nested* degree-corrected block model, which is
the method chapter 7 measures the density with, so the comparison stays inside
one instrument.

Only kappa_dcsbm is needed, so this fits the block model and skips the motif
alphabet of ramsey_motif.py, which is much the more expensive half.  The rules
and the seeding are ramsey_motif.py's, and the overlap check in --stage report
confirms that the two pipelines return identical kappa on identical seeds.
Four rules the book names but never ran a Ramsey grid for are added here --
LS(n,2), DS(n,0.7), DD(n,0.3) and the longer bubbles -- to widen the range of
<n_B> the relation is tested over.

Usage:
    python3 ramsey_density.py --stage sweep --model "BB(n,3)" \
        --sizes 200,240,280,320,400 --reps 100 --out data/scratch.csv
    python3 ramsey_density.py --stage report      # writes tab-onset.tex
"""
import argparse
import csv
import glob
import os
import sys
import time

import numpy as np
import pandas as pd

from ramsey_motif import (MODELS, to_edges, generator_local_search,
                          generator_dup_split, generator_dup_divergence,
                          generator_bubble)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
ROOT = os.path.dirname(HERE)
KEY = "kappa_dcsbm"
EPS = 0.05

MODELS.update({
    "LS(n,2)": (generator_local_search, {"d": 2}),
    "DS(n,0.7)": (generator_dup_split, {"q": 0.7}),
    "DD(n,0.3)": (generator_dup_divergence, {"q": 0.3}),
    "BB(n,3)": (generator_bubble, {"L": 3}),
    "BB(n,4)": (generator_bubble, {"L": 4}),
    "BB(n,5)": (generator_bubble, {"L": 5}),
    "BB(n,6)": (generator_bubble, {"L": 6}),
})

# the order the table prints in, by <n_B>; BB(n,6) is measured and excluded,
# see the report
ORDER = ["DS(n,0.7)", "DD(n,0.3)", "DS(n,0.5)", "LS(n,2)", "DD(n,0.5)",
         "LS(n,1)", "BB(n,1)", "DS(n,0.3)", "BB(n,2)", "BB(n,3)", "BB(n,5)",
         "BB(n,4)"]
LABEL = {"DS(n,0.3)": r"$DS(n,1/3)$"}


# ------------------------------------------------------------------- sweep

def kappa_nested(n_nodes, edges, seed):
    """Exactly motif_sbm.MotifSBM.dcsbm_blocks(nested=True), counting groups."""
    import graph_tool.all as gt
    g = gt.Graph(directed=False)
    g.add_vertex(n_nodes)
    g.add_edge_list(edges)
    gt.seed_rng(seed)
    st = gt.minimize_nested_blockmodel_dl(g, state_args=dict(deg_corr=True))
    return len({int(x) for x in list(st.get_bs()[0])[:n_nodes]})


def one(model, n, seed):
    gen, params = MODELS[model]
    nb = gen(n, params, seed)
    n_actual = len(nb)                    # the bubble rules overshoot n
    edges = to_edges(nb)
    return dict(model=model, n=n, n_nodes=n_actual, seed=seed, m=len(edges),
                kappa_dcsbm=kappa_nested(n_actual, edges, seed))


def sweep(model, sizes, reps, out):
    rows = []
    for n in sizes:
        t0 = time.time()
        for r in range(reps):
            rows.append(one(model, n, 1000 * n + r))   # ramsey_motif.py's seeds
        k = [x[KEY] for x in rows[-reps:]]
        print(f"  {model:<10} n={n:<6} P={sum(1 for v in k if v >= 2)/reps:.3f}"
              f"  <kappa>={np.mean(k):6.2f}  n/<kappa>={n/np.mean(k):7.1f}"
              f"  [{time.time()-t0:.0f}s]", flush=True)
        with open(out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


# ------------------------------------------------------------------ report

def load():
    """This script's runs, plus every kappa_dcsbm ramsey_motif.py has measured."""
    d = [pd.read_csv(f"{DATA}/ramsey_density.csv")]
    d[0]["src"] = "density"
    for f in sorted(glob.glob(f"{DATA}/ramsey_motif_*.csv")):
        x = pd.read_csv(f)
        if KEY in x:
            x["src"] = "motif"
            d.append(x[["model", "n", "seed", "m", KEY, "src"]])
    d = pd.concat(d, ignore_index=True).dropna(subset=[KEY])
    d[KEY] = d[KEY].astype(int)
    shared = d.groupby(["model", "n", "seed"])
    n_both = int((shared.src.nunique() > 1).sum())
    n_bad = int((shared[KEY].nunique() > 1).sum())
    print(f"overlap with ramsey_motif.py: {n_both} (model,n,seed) in both, "
          f"{n_bad} disagree")
    return d.drop_duplicates(subset=["model", "n", "seed"])


def r_kappa(curve):
    """Smallest measured size from which P_kappa >= 1-eps and never falls back."""
    ns = sorted(curve)
    for i, n in enumerate(ns):
        if all(curve[m] >= 1 - EPS for m in ns[i:]):
            return n
    return None


def n_two(kk):
    """Size at which <kappa> first reaches two, interpolated in log-log."""
    ns = np.array(sorted(kk.index), float)
    ks = np.array([kk[n] for n in ns], float)
    for i in range(1, len(ns)):
        if ks[i] >= 2 > ks[i - 1]:
            x0, x1 = np.log(ns[i - 1]), np.log(ns[i])
            y0, y1 = np.log(max(ks[i - 1], 1e-9)), np.log(ks[i])
            return float(np.exp(x0 + (np.log(2) - y0) * (x1 - x0) / (y1 - y0)))
    return np.nan


def plateau(g):
    """<n_B> over the sizes where the density has set in, found self
    consistently as n >= 3<n_B>, with the exponent fitted over the same sizes."""
    kk = g.groupby("n")[KEY].mean()
    nB = kk.index.max() / kk[kk.index.max()]
    for _ in range(20):
        sel = kk[kk.index >= 3 * nB]
        if len(sel) < 2:
            sel = kk[kk.index >= kk.index.max() / 4]
        new = 1 / (sel.values / sel.index.values).mean()
        if abs(new - nB) < 1e-6 * nB:
            break
        nB = new
    beta = (np.polyfit(np.log(sel.index.values), np.log(sel.values), 1)[0]
            if len(sel) > 2 else np.nan)
    return nB, beta, kk


def boot(g, reps=500, seed=11):
    rng = np.random.default_rng(seed)
    by = {n: s[KEY].values for n, s in g.groupby("n") if len(s) >= 20}
    out = [r for _ in range(reps)
           if (r := r_kappa({n: (rng.choice(v, len(v)) >= 2).mean()
                             for n, v in by.items()})) is not None]
    return tuple(np.percentile(out, [2.5, 97.5])) if out else (np.nan, np.nan)


def report():
    d = load()
    rows = []
    for m in ORDER + ["BB(n,6)"]:
        g = d[d.model == m]
        nB, beta, kk = plateau(g)
        rk = r_kappa({int(n): (s[KEY] >= 2).mean() for n, s in g.groupby("n")})
        lo, hi = boot(g)
        rows.append(dict(model=m, nB=nB, beta=beta, rk=rk, lo=lo, hi=hi,
                         n2=n_two(kk), nmax=int(g.n.max())))
    r = pd.DataFrame(rows).set_index("model")
    r["r/nB"], r["n2/nB"], r["tau"] = r.rk / r.nB, r.n2 / r.nB, r.rk / r.n2
    pd.set_option("display.width", 200)
    print(r.round(2).to_string())

    fin = r.loc[ORDER]
    x, y = np.log(fin.nB.values), np.log(fin.rk.values.astype(float))
    for lab, sub in (("all", fin), ("|beta-1|<=0.1", fin[abs(fin.beta - 1) <= 0.1])):
        u, v = np.log(sub.nB.values), np.log(sub.rk.values.astype(float))
        a, c = np.polyfit(u, v, 1)
        res = v - (a * u + c)
        se = np.sqrt((res ** 2).sum() / (len(u) - 2) / ((u - u.mean()) ** 2).sum())
        r2 = 1 - (res ** 2).sum() / ((v - v.mean()) ** 2).sum()
        cc = np.exp(np.mean(v - u))
        q = np.exp(v - u - np.log(cc))
        print(f"\n{lab} (N={len(u)}): slope {a:.2f} +- {se:.2f}, R^2={r2:.2f}; "
              f"forced slope one, r_kappa = {cc:.2f} <n_B>, "
              f"residual x{q.min():.2f}-{q.max():.2f}")
    print(f"\nn_2/<n_B> = {fin['n2/nB'].mean():.2f} +- {fin['n2/nB'].std():.2f}")
    print(f"tau = {fin.tau.min():.2f}-{fin.tau.max():.2f}")

    # the bubble family at a common size, where the parity sits
    print("\nBB(n,L) at n = 6400 unless noted")
    for L in range(1, 7):
        g = d[d.model == f"BB(n,{L})"]
        kk = g.groupby("n")[KEY].mean()
        n = 6400 if 6400 in kk.index else int(kk.index.max())
        rk = r_kappa({int(q): (s[KEY] >= 2).mean() for q, s in g.groupby("n")})
        nB = n / kk[n]
        print(f"  L={L} cycle {L+2:<2} "
              f"{'bipartite' if L % 2 == 0 else 'not      '}"
              f"  <n_B>(n={n}) = {nB:6.1f}   "
              + (f"r_kappa = {rk:<6} ratio = {rk/nB:.1f}" if rk
                 else "r_kappa = never within reach"))

    # ------------------------------------------------------------ the table
    L = [r"\begin{center}", r"\small", r"\begin{tabular}{lrrrrr}", r"\hline",
         r"rule & $\langle n_B\rangle$ & $r_\kappa$ & $n_2$ &"
         r" $n_2/\langle n_B\rangle$ & $r_\kappa/n_2$\\", r"\hline"]
    for m in ORDER:
        v = r.loc[m]
        L.append(f"{LABEL.get(m, '$' + m + '$')} & ${v.nB:.0f}$ & ${v.rk:.0f}$ & "
                 f"${v.n2:.0f}$ & ${v['n2/nB']:.2f}$ & ${v.tau:.2f}$\\\\")
    L += [r"\hline", r"\end{tabular}", r"\end{center}"]
    open(os.path.join(ROOT, "tab-onset.tex"), "w").write("\n".join(L) + "\n")
    print(f"\nwrote {os.path.join(ROOT, 'tab-onset.tex')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("sweep", "report"), default="report")
    ap.add_argument("--model")
    ap.add_argument("--sizes")
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.stage == "report":
        return report()
    sweep(a.model, [int(x) for x in a.sizes.split(",")], a.reps, a.out)


if __name__ == "__main__":
    sys.exit(main())
