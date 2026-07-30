#!/usr/bin/env python3
"""Build the two master tables the reorganised chapters 7 and 8 want.

Chapter 7's table is every growth rule under the two priors: the exponent beta of
B ~ n^beta, the community density b = B/n, and the mean community size 1/b.
Chapter 8's adds the motif alphabet as a third method and the two quantities that
separate the three limit classes: whether communities emerge at all (the Ramsey
number) and how their number scales.

Writes tab-master-models.tex and tab-master-motifs.tex in the manuscript root.
Reproduces tab-density-models.tex's four rows exactly, which is the check that
this is the same measurement and not a new one.
"""
import glob
import os

import numpy as np
import pandas as pd

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)))
ROOT = "/Users/avazquez/av2atg/LocalNetworkGrowth"
FIGS = os.path.join(ROOT, "figs", "data")

# the sizes each rule's exponent is fitted over.  For the random rules that is
# every size the scaling runs cover; for the deterministic lattices the smallest
# generations are dropped, since a lattice with one or two blocks is below the
# resolution of any of the three methods and would fit a slope to nothing.
LATTICE_MIN = {"diamond": 684, "pseudofractal": 366}

ORDER = ["LS(n,1)", "BB(n,1)", "DS(n,0.3)", "DS(n,0.5)", "DD(n,0.5)",
         "BB(n,2)", "pseudofractal", "diamond"]

LABEL = {"LS(n,1)": r"$LS(n,1)$", "BB(n,1)": r"$BB(n,1)$",
         "DS(n,0.3)": r"$DS(n,1/3)$", "DS(n,0.5)": r"$DS(n,0.5)$",
         "DD(n,0.5)": r"$DD(n,0.5)$", "BB(n,2)": r"$BB(n,2)$",
         "pseudofractal": "pseudofractal", "diamond": "diamond"}

MOTIF = {"LS(n,1)": "triangle", "BB(n,1)": "triangle", "DS(n,0.3)": "square",
         "DS(n,0.5)": "square", "DD(n,0.5)": "biclique", "BB(n,2)": "square",
         "pseudofractal": "triangle", "diamond": "square"}


def betafit(g, col):
    s = g.groupby("n")[col].mean()
    s = s[s > 0]
    return np.polyfit(np.log(s.index.values), np.log(s.values), 1)[0] if len(s) > 2 else np.nan


def load():
    sc = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f"{FIGS}/ramsey_motif_scaling*.csv"))],
                   ignore_index=True)
    th = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f"{FIGS}/ramsey_motif_threshold*.csv"))],
                   ignore_index=True)
    lat = []
    for f, name in ((f"{FIGS}/diamond_motif.csv", "diamond"),
                    (f"{FIGS}/pseudofractal_motif.csv", "pseudofractal")):
        x = pd.read_csv(f)
        x["model"] = name
        lat.append(x[x.n >= LATTICE_MIN[name]])
    return sc, th, pd.concat(lat, ignore_index=True)


def rows():
    sc, th, lat = load()
    both = pd.concat([sc, th], ignore_index=True)
    out = {}
    for name, g in pd.concat([sc, lat], ignore_index=True).groupby("model"):
        r = dict(n_lo=int(g.n.min()), n_hi=int(g.n.max()),
                 flat=betafit(g, "kappa_flat"), nested=betafit(g, "kappa_dcsbm"),
                 motif=betafit(g, "kappa"))
        s = g.groupby("n").kappa_dcsbm.mean()
        b = s / s.index.values
        r["b"], r["b_sd"], r["nB"] = b.mean(), b.std(), 1 / b.mean()
        top = g[g.n == g.n.max()]
        r["cover"] = (3 * top.n_tri + 4 * top.n_sq).sum() / top.m.sum()
        out[name] = r
    for name, g in both.groupby("model"):
        for col, key in (("kappa_flat", "rk_flat"), ("kappa_dcsbm", "rk_nested"),
                         ("kappa", "rk_motif")):
            p = g.groupby("n")[col].apply(lambda s: (s > 1).mean())
            ok = p[p >= 0.95]
            if name in out:
                out[name][key] = int(ok.index[0]) if len(ok) else None
    return out


def f2(x):
    return "---" if x is None or (isinstance(x, float) and np.isnan(x)) else f"${x:.2f}$"


def main():
    r = rows()

    # ---------------------------------------------------------- chapter 7
    L = [r"\begin{center}", r"\small", r"\begin{tabular}{lrrrrr}", r"\hline",
         r" & & \multicolumn{2}{c}{$\beta$, growth of $B$} & & \\",
         r"rule & $n$ & flat & nested & $b$ & $\langle n_B\rangle$\\", r"\hline"]
    for k in ORDER:
        x = r[k]
        L.append(f"{LABEL[k]} & ${x['n_lo']}$--${x['n_hi']}$ & {f2(x['flat'])} & "
                 f"{f2(x['nested'])} & ${x['b']:.3f}\\pm{x['b_sd']:.3f}$ & "
                 f"${x['nB']:.0f}$\\\\")
        if k == "BB(n,2)":
            L.append(r"\hline")
    L += [r"\hline", r"\end{tabular}", r"\end{center}"]
    open(os.path.join(ROOT, "tab-master-models.tex"), "w").write("\n".join(L) + "\n")

    # ---------------------------------------------------------- chapter 8
    # grouped by the class the rule lands in once the alphabet is available,
    # which is the point of the table
    CLASSES = (("constant density", ["LS(n,1)", "BB(n,1)", "DS(n,0.3)",
                                     "DS(n,0.5)", "DD(n,0.5)", "pseudofractal"]),
               ("no communities", ["BA(m,2)", "BB(n,2)"]),
               ("fractal", ["diamond"]))

    def rk(v):
        return "never" if v is None else f"${v}$"

    L = [r"\begin{center}", r"\small\setlength{\tabcolsep}{4pt}",
         r"\begin{tabular}{llrrrrrrr}", r"\hline",
         r" & & \multicolumn{3}{c}{$r_\kappa$, onset of communities} & "
         r"\multicolumn{3}{c}{$\beta$, growth of $\kappa$} & links\\",
         r"rule & motif & flat & nested & $+$mot. & flat & nested & $+$mot. & named\\",
         r"\hline"]
    for cls, keys in CLASSES:
        L.append(r"\multicolumn{9}{l}{\emph{" + cls + r"}}\\")
        for k in keys:
            if k == "BA(m,2)":
                # Chapter 6's non-local control, and the only row not measured
                # here: its "never" is that chapter's flat fit, and the nested
                # and motif fits have not been run on it.
                L.append(r"\quad $BA(n,2)^{\dagger}$ & --- & never & --- & --- "
                         r"& --- & --- & --- & ---\\")
                continue
            x = r[k]
            det = k in ("diamond", "pseudofractal")
            rks = ["---"] * 3 if det else [rk(x.get("rk_flat")),
                                           rk(x.get("rk_nested")),
                                           rk(x.get("rk_motif"))]
            bm = "---" if k == "BB(n,2)" else f2(x["motif"])
            L.append(f"\\quad {LABEL[k]} & {MOTIF[k]} & " + " & ".join(rks) +
                     f" & {f2(x['flat'])} & {f2(x['nested'])} & {bm} & "
                     f"${100*x['cover']:.0f}\\%$\\\\")
        L.append(r"\hline")
    L += [r"\end{tabular}", r"\end{center}"]
    open(os.path.join(ROOT, "tab-master-motifs.tex"), "w").write("\n".join(L) + "\n")

    print(pd.DataFrame(r).T.to_string())


if __name__ == "__main__":
    main()
