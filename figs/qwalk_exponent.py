#!/usr/bin/env python3
r"""
qwalk_exponent.py -- how does the localization volume really grow with size?

Companion to qwalk_ipr.py.  That script establishes that ell = 1/IPR grows far
more slowly than n for every local rule; this one asks what it actually does.
Three candidates are indistinguishable over the three octaves of the main
sweep and have to be separated on a longer lever arm:

    (A) ell -> const            genuine localization
    (B) ell ~ a + b ln n        marginal, log-divergent
    (C) ell ~ n^lambda          a fresh exponent, 0 < lambda < 1

(C) with 0 < lambda < 1 would be the quantum counterpart of the "fractal" limit
class that chapter 8 defines and finds empty.

Two things are done differently here, both because they matter for the answer:

  * the averaging window scales as [2n, 4n] rather than sitting at a fixed
    [400, 800].  A ballistic walker on a ring of 3200 nodes has not crossed it
    by t = 800, so a fixed window truncates the delocalized reference and biases
    every exponent downward.  Scaling the window removes that bias, at the cost
    of making the large sizes no cheaper than they already are.

  * sizes run to 3200 for the stochastic rules and to a few thousand for the
    deterministic lattices, which are the cleanest evidence available because
    there is exactly one network per size and no ensemble scatter at all.

Usage:
    python3 qwalk_exponent.py [--force]
"""

import os
import sys

import numpy as np
import networkx as nx
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
CSV = os.path.join(HERE, "data", "qwalk-exponent.csv")

from qwalk_ipr import (ipr, RULES, pseudofractal, rec_triangle, diamond,
                       MK, panel, save, W, plt)

# (size, repetitions).  The largest size is the expensive one -- cost grows like
# n^2 in the propagation and n^3 in the diagonalization -- so it gets fewer.
SWEEP = [(100, 6), (200, 6), (400, 6), (800, 5), (1600, 4), (3200, 2)]

STOCH = ["ring", "ER(n,4)", "BA(n,2)", "BB(n,1)", "BB(n,2)", "BB(n,3)",
         "BB(n,5)", "BB(n,6)", "LS(n,1)", "DS(n,0.5)", "DD(n,0.5)",
         "BB2+link", "BB2+diag"]

LATTICE = {
    "pseudofractal": [pseudofractal(t) for t in (3, 4, 5, 6)],
    "rec.triangle":  [rec_triangle(d) for d in (4, 5, 6, 7, 8, 9)],
    "diamond(2,2)":  [diamond(t, 2, 2) for t in (3, 4, 5, 6)],
    "diamond(2,3)":  [diamond(t, 2, 3) for t in (2, 3, 4)],
    "diamond(3,2)":  [diamond(t, 3, 2) for t in (2, 3, 4)],
}


# Chapter 9's exponent for the total meso-structure, kappa_0 + kappa ~ n^beta,
# from tab-meso.  The chapter's conjecture is that lambda = 1 - beta: if the
# walker holds a fixed number of meso-units, then ell ~ 1/b_0 ~ n^{1-beta}.
BETA = {"BB(n,1)": 1.04, "BB(n,2)": 1.01, "BB(n,3)": 0.99, "BB(n,4)": 0.99,
        "BB(n,5)": 0.98, "LS(n,1)": 1.01, "DS(n,0.5)": 1.21, "DD(n,0.5)": 1.11,
        "BB2+link": 1.10, "BB2+diag": 0.94, "pseudofractal": 1.00,
        "diamond(2,2)": 1.00, "diamond(2,3)": 1.00, "diamond(3,2)": 1.00}


def scaled(g, seed=0):
    """IPR averaged over [2n, 4n], so the window grows with the system."""
    n = g.number_of_nodes()
    return ipr(g, window=(2.0 * n, 4.0 * n), nt=240, nstart=24, seed=seed)


def measure():
    rows = []
    for rule in STOCH:
        build = RULES[rule][0]
        for n, reps in SWEEP:
            for seed in range(1 if rule == "ring" else reps):
                g = build(n, seed + 1)
                v = scaled(g, seed)
                rows.append(dict(rule=rule, n=n, n_real=g.number_of_nodes(),
                                 seed=seed, ipr=v.mean(), iprmax=v.max()))
            print(f"  {rule} n={n}", flush=True)
    for rule, graphs in LATTICE.items():
        for g in graphs:
            if g.number_of_nodes() > 3500:
                continue
            v = scaled(g)
            rows.append(dict(rule=rule, n=g.number_of_nodes(),
                             n_real=g.number_of_nodes(), seed=0,
                             ipr=v.mean(), iprmax=v.max()))
            print(f"  {rule} n={g.number_of_nodes()}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(CSV, index=False)
    print("wrote", os.path.relpath(CSV, HERE))
    return df


# ------------------------------------------------------------------- model fits

def fits(x, y):
    """Compare the three candidates on (n, ell) by corrected AIC.

    Returns a dict of model -> (AICc, parameters).  All three are fitted to ell
    directly by least squares, so the AICc values are comparable.
    """
    out = {}

    def aicc(resid, k):
        n = len(resid)
        rss = float((resid ** 2).sum())
        if rss <= 0:
            rss = 1e-300
        a = n * np.log(rss / n) + 2 * k
        return a + (2 * k * (k + 1)) / max(n - k - 1, 1e-9)

    out["const"] = (aicc(y - y.mean(), 1), (y.mean(),))
    c = np.polyfit(np.log(x), y, 1)
    out["log"] = (aicc(y - np.polyval(c, np.log(x)), 2), tuple(c))
    p = np.polyfit(np.log(x), np.log(y), 1)          # ell = a n^lambda
    out["power"] = (aicc(y - np.exp(p[1]) * x ** p[0], 2),
                    (float(p[0]), float(np.exp(p[1]))))
    return out


def slope_ci(x, y):
    """Log-log slope with a standard error, on log ell vs log n."""
    lx, ly = np.log(x), np.log(y)
    sx = lx - lx.mean()
    lam = float((sx * (ly - ly.mean())).sum() / (sx ** 2).sum())
    b = ly.mean() - lam * lx.mean()
    r = ly - (lam * lx + b)
    se = float(np.sqrt((r ** 2).sum() / max(len(x) - 2, 1) / (sx ** 2).sum()))
    return lam, se


def report(df):
    g = df.groupby(["rule", "n"]).ipr.mean().reset_index()
    g["L"] = 1.0 / g.ipr
    print(f"\n{'rule':16s} {'oct':>4s} {'lambda':>8s} {'se':>6s} {'1-beta':>7s} "
          f"{'best':>6s} {'d(log)':>7s} {'d(pow)':>7s}   ell across sizes")
    order = STOCH + list(LATTICE)
    for rule in order:
        d = g[g.rule == rule].sort_values("n")
        if len(d) < 4:
            continue
        x, y = d.n.values.astype(float), d.L.values
        lam, se = slope_ci(x, y)
        f = fits(x, y)
        best = min(f, key=lambda k: f[k][0])
        a0 = f["const"][0]
        seq = " ".join(f"{v:.1f}" for v in y)
        pred = BETA.get(rule)
        ps = f"{1 - pred:7.2f}" if pred is not None else "      -"
        print(f"{rule:16s} {np.log2(x[-1]/x[0]):4.1f} {lam:8.3f} {se:6.3f} {ps} "
              f"{best:>6s} {f['log'][0]-a0:7.1f} {f['power'][0]-a0:7.1f}   {seq}")
    return g


def fig_exponent(g):
    fig, axes = plt.subplots(1, 2, figsize=(W, W * 0.50))

    ax = axes[0]
    show = [("ring", "ring"), ("ER(n,4)", "$ER(n,4)$"), ("BB2+link", r"$BB(n,2)+$link"),
            ("BB(n,5)", "$BB(n,5)$"), ("BB(n,1)", "$BB(n,1)$"),
            ("DD(n,0.5)", "$DD(n,0.5)$")]
    for i, (r, lab) in enumerate(show):
        d = g[g.rule == r].sort_values("n")
        delocal = r in ("ring", "ER(n,4)")
        ax.plot(d.n, d.L, ls="-" if delocal else "none", lw=0.7, ms=3.0,
                color="0.45" if delocal else "none", label=lab, **MK[i % len(MK)])
    ax.plot([90, 3600], [30, 1200], ls=":", lw=0.7, c="0.25", zorder=1)
    ax.text(900, 130, r"$\propto n$", fontsize=6.5, color="0.25")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("$n$"); ax.set_ylabel(r"$\ell$")
    ax.set_xticks([100, 400, 1600]); ax.set_xticklabels(["100", "400", "1600"])
    ax.set_xticks([], minor=True)
    ax.set_yticks([5, 20, 100, 500]); ax.set_yticklabels(["5", "20", "100", "500"])
    ax.set_yticks([], minor=True)
    ax.legend(frameon=False, loc="upper left", fontsize=5.2,
              handletextpad=0.3, labelspacing=0.22, borderpad=0.1)
    panel(ax, "(a)")

    # (b) the lattices, where there is no ensemble scatter to hide behind
    ax = axes[1]
    for i, (r, lab) in enumerate([("diamond(2,2)", "diamond $(2,2)$"),
                                  ("diamond(3,2)", "diamond $(3,2)$"),
                                  ("pseudofractal", "pseudofractal"),
                                  ("rec.triangle", "rec. triangle")]):
        d = g[g.rule == r].sort_values("n")
        ax.plot(d.n, d.L, ls="none", ms=3.0, label=lab, **MK[i % len(MK)])
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("$n$"); ax.set_ylabel(r"$\ell$")
    ax.set_xticks([30, 100, 300, 1000, 3000])
    ax.set_xticklabels(["30", "100", "300", "1000", "3000"])
    ax.set_xticks([], minor=True)
    ax.set_yticks([2, 5, 10, 20]); ax.set_yticklabels(["2", "5", "10", "20"])
    ax.set_yticks([], minor=True)
    ax.set_ylim(1.5, 30)
    ax.legend(frameon=False, loc="upper left", fontsize=5.2,
              handletextpad=0.3, labelspacing=0.22, borderpad=0.1)
    panel(ax, "(b)")

    fig.tight_layout(pad=0.3)
    save(fig, "fig-qw-exponent.pdf")


def table(g):
    out = [r"\begin{center}", r"\footnotesize\setlength{\tabcolsep}{4pt}",
           r"\begin{tabular}{lrrrl}", r"\hline",
           r"rule & sizes & $\nu$ & $\Delta\mathrm{AIC}_c$ & best fit\\",
           r"\hline"]
    from qwalk_ipr import TEX
    names = dict(TEX)
    names.update({"pseudofractal": "pseudofractal", "rec.triangle": "rec.\\ triangle",
                  "diamond(2,2)": r"diamond $(2,2)$", "diamond(2,3)": r"diamond $(2,3)$",
                  "diamond(3,2)": r"diamond $(3,2)$"})
    label = {"const": "constant", "log": r"$\ln n$", "power": r"$n^{\nu}$"}
    for rule in ["ring", "ER(n,4)"] + [r for r in STOCH if r not in
                                       ("ring", "ER(n,4)")] + list(LATTICE):
        d = g[g.rule == rule].sort_values("n")
        if len(d) < 4:
            continue
        x, y = d.n.values.astype(float), d.L.values
        lam, se = slope_ci(x, y)
        f = fits(x, y)
        best = min(f, key=lambda k: f[k][0])
        # margin of the winner over the runner-up: how sure the choice is
        rest = sorted(v[0] for k, v in f.items() if k != best)
        out.append(f"{names.get(rule, rule)} & ${int(x[0])}$--${int(x[-1])}$ & "
                   f"${lam:.2f}\\pm{se:.2f}$ & ${rest[0]-f[best][0]:.1f}$ & "
                   f"{label[best]}" + r"\\")
    out += [r"\hline", r"\end{tabular}", r"\end{center}"]
    p = os.path.join(os.path.dirname(HERE), "tab-qw-exponent.tex")
    open(p, "w").write("\n".join(out) + "\n")
    print("wrote", os.path.relpath(p, os.path.dirname(HERE)))


def main():
    if "--force" not in sys.argv and os.path.exists(CSV):
        df = pd.read_csv(CSV)
        print("read", os.path.relpath(CSV, HERE))
    else:
        df = measure()
    g = report(df)
    fig_exponent(g)
    table(g)


if __name__ == "__main__":
    main()
