#!/usr/bin/env python3
"""The closing table of Chapters 6--9: every local rule, counted as meso-structure.

Chapter 8's master table reports r_kappa and three exponents for the community
count. This one drops both. A motif and a community are the same object at two
scales (Section 9.8), so the community count on its own is a statement about
which vocabulary the description happened to use, and the quantity that does not
depend on that choice is the total: kappa_0 + kappa, motifs plus communities.

Rules are grouped by the limit they converge to under the best description
available -- no communities, or a constant density of them -- and the one
exponent is there to show that the grouping does not survive into it. Every local
rule in this book carries meso-structure in proportion to its size, whichever
class it is in.

BA(n,m) is left out: it is the non-local control, and this is a table about what
local rules converge to.

Exponents are fitted over every size measured for each rule, which is why they
differ in the third decimal from Chapter 9's table, fitted from n = 400.

Writes tab-meso.tex.
"""
import glob
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
ROOT = os.path.dirname(HERE)
SKIP = ("bb3.csv", "bb4.csv", "bb5.csv")


def beta(n, y):
    n, y = np.asarray(n, float), np.asarray(y, float)
    k = y > 0
    return np.polyfit(np.log(n[k]), np.log(y[k]), 1)[0] if k.sum() > 2 else np.nan


def collect():
    out = {}
    old = pd.concat([pd.read_csv(f) for f in
                     glob.glob(f"{DATA}/ramsey_motif_scaling*.csv")
                     if not f.endswith(SKIP)], ignore_index=True)
    for key in ("LS(n,1)", "DS(n,0.3)", "DS(n,0.5)", "DD(n,0.5)",
                "BB(n,1)", "BB(n,2)"):
        d = old[old.model == key].copy()
        d["k0"] = d.n_tri + d.n_sq
        g = d.groupby("n").agg(k0=("k0", "mean"), k=("kappa", "mean"))
        top = d[d.n == d.n.max()]
        out[key] = (beta(g.index, g.k0 + g.k),
                    (3 * top.n_tri + 4 * top.n_sq).sum() / top.m.sum())
    for L in (3, 4, 5):
        d = pd.read_csv(f"{DATA}/ramsey_motif_scaling_bb{L}.csv").copy()
        t = "pent" if L == 3 else "cyc"
        d["k0"] = d[f"{t}_{t}"] + d[f"tri_{t}"] + d[f"sq_{t}"]
        g = d.groupby("n").agg(k0=("k0", "mean"), k=(f"kappa_{t}", "mean"))
        top = d[d.n == d.n.max()]
        out[f"BB(n,{L})"] = (beta(g.index, g.k0 + g.k),
                             ((L + 2) * top[f"{t}_{t}"]).sum() / top.m.sum())
    ch = pd.read_csv(f"{DATA}/bb2_chord.csv")
    for r in ("bb2par", "bb2chord"):
        d = ch[ch.rule == r].copy()
        d["k0"] = d.tri_kite + d.sq_kite + d.cs_kite
        g = d.groupby("n").agg(k0=("k0", "mean"), k=("kappa_kite", "mean"))
        out[r] = (beta(g.index, g.k0 + g.k),
                  d[d.n == d.n.max()].named_kite.mean())
    # The lattices are exact, not fitted: each tiles perfectly, so the best
    # description has m/(links per motif) motifs and one community, and m is
    # proportional to n, so the exponent is one by construction.
    for name, per, seq in (("pseudofractal", 3, [(3 ** (t + 1) + 3) // 2 for t in range(4, 9)]),
                           ("diamond", 4, [(2 * 4 ** t + 4) // 3 for t in range(3, 8)])):
        m = [3 ** (t + 1) for t in range(4, 9)] if per == 3 else [4 ** t for t in range(3, 8)]
        tot = np.array(m, float) / per + 1.0
        out[name] = (beta(np.array(seq, float), tot), 1.0)
    return out


ROWS = (("no communities",
         (("BB(n,2)", r"$BB(n,2)$", r"\ngon{4}"),
          ("BB(n,4)", r"$BB(n,4)$", r"\ngon{6}"),
          ("pseudofractal", "pseudofractal", r"\ngon{3}"),
          ("diamond", "diamond", r"\ngon{4}"))),
        ("constant density of communities",
         (("LS(n,1)", r"$LS(n,1)$", r"\ngon{3}"),
          ("BB(n,1)", r"$BB(n,1)$", r"\ngon{3}"),
          ("BB(n,3)", r"$BB(n,3)$", r"\ngon{5}"),
          ("BB(n,5)", r"$BB(n,5)$", r"\ngon{7}"),
          ("DS(n,0.3)", r"$DS(n,1/3)$", r"\ngon{4}"),
          ("DS(n,0.5)", r"$DS(n,0.5)$", r"\ngon{4}"),
          ("DD(n,0.5)", r"$DD(n,0.5)$", r"$K_{2,d}$"),
          ("bb2par", r"$BB(n,2)+$ a link", r"\sqstub"),
          ("bb2chord", r"$BB(n,2)+$ a diagonal", r"\sqdiag"))))


def main():
    d = collect()
    L = [r"\begin{center}", r"\footnotesize\setlength{\tabcolsep}{5pt}",
         r"\begin{tabular}{llrr}", r"\hline",
         r"rule & motif & links named & $\beta$ for $\kappa_0+\kappa$\\",
         r"\hline"]
    for cls, keys in ROWS:
        L.append(r"\multicolumn{4}{l}{\emph{" + cls + r"}}\\")
        for key, lab, glyph in keys:
            b, cov = d[key]
            L.append(f"\\quad {lab} & {glyph} & ${100*cov:.0f}\\%$ & "
                     f"$\\mathbf{{{b:.2f}}}$\\\\")
        L.append(r"\hline")
    L += [r"\end{tabular}", r"\end{center}"]
    open(os.path.join(ROOT, "tab-meso.tex"), "w").write("\n".join(L) + "\n")
    for cls, keys in ROWS:
        print(f" {cls}")
        for key, lab, _ in keys:
            b, cov = d[key]
            print(f"   {lab:26s} named {100*cov:3.0f}%   beta(k0+k) {b:.2f}")


if __name__ == "__main__":
    main()
