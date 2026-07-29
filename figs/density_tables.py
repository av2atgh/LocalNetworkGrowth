#!/usr/bin/env python3
"""Tables: the density of communities, for the growth rules and for real networks.

Chapter 7 establishes that the number of communities grows as B ~ n^beta, and
that beta depends on the prior. This script asks the next question. If beta = 1
then B/n is a constant, and that constant --- equivalently the typical community
size n/B --- is a property of the network rather than of its size, and can be
tabulated like a clustering coefficient or a degree exponent.

The two tables answer it differently, which is the point of having both:

  models    B/n is flat to within a few per cent over one to two decades of n,
            and its value separates the rules by a factor of three. A constant.

  real      B/n falls as n^-0.20. Community size is not constant; it grows
            slowly. The fall is present inside every domain separately, so it
            is not an artefact of large networks happening to be of one kind.

Both use the nested degree-corrected model, since Section 7.9 showed the flat
one is answering a different question.

Reads data/ramsey_motif_scaling*.csv (from ramsey_motif.py) and
data/census_nested.csv (from census_nested.py). Writes the two LaTeX tables to
stdout and to tab-density-*.tex at the repository root.

Usage:  python3 density_tables.py
"""
import glob
import os
import re

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# The four rules of Figure 7.4, in its order, with the book's math styling.
RULES = (("LS(n,1)",   r"$LS(n,1)$"),
         ("DS(n,0.5)", r"$DS(n,0.5)$"),
         ("DD(n,0.5)", r"$DD(n,0.5)$"),
         ("BB(n,2)",   r"$BB(n,2)$"))

# Domain of each network, by name. The census is a convenience sample and these
# labels are coarse, but they are enough to ask whether the trend in B/n is a
# real one or an accident of which networks happen to be large.
DOMAIN = (
    ("biological", r"macaque|interactome|celegans|malaria|collins|drosophila"
                   r"|reactome|kegg|tree-of-life|mist|fullerene"),
    ("infrastructure", r"contiguous|london_transport|eu_airlines|euroroad"
                       r"|urban_streets|power|route_views|internet_as|topology"
                       r"|montreal"),
    ("information", r"adjnoun|netscience|webkb|us_congress|wiki_science"
                    r"|product_space|bible_nouns|wikiconflict|wordnet"),
    ("social", r".*"),          # the remainder, and much the largest group
)


def slope(x, y):
    return np.polyfit(np.log(x), np.log(y), 1)[0]


def models():
    d = pd.concat([pd.read_csv(f) for f in
                   sorted(glob.glob(os.path.join(HERE, "data",
                                                 "ramsey_motif_scaling*.csv")))],
                  ignore_index=True)
    rows = []
    for key, label in RULES:
        s = d[d.model == key]
        g = s.groupby("n").kappa_dcsbm.agg(["mean", "size"])
        dens = g["mean"] / g.index
        rows.append(dict(label=label, lo=g.index.min(), hi=g.index.max(),
                         reps=int(g["size"].min()),
                         beta=slope(g.index, g["mean"]),
                         dens=dens.mean(), spread=dens.std(),
                         size=1 / dens.mean()))
    return pd.DataFrame(rows)


def real():
    d = pd.read_csv(os.path.join(HERE, "data", "census_nested.csv"))
    d = d[d.B_nested > 0].copy()

    def dom(name):
        for label, pat in DOMAIN:
            if re.search(pat, name):
                return label
        return "other"

    d["domain"] = d.network.map(dom)
    d["dens"] = d.B_nested / d.n
    rows = []
    for label in ("social", "biological", "infrastructure", "information"):
        s = d[d.domain == label]
        rows.append(dict(label=label, nets=len(s), lo=s.n.min(), hi=s.n.max(),
                         beta=slope(s.n, s.B_nested),
                         size=(s.n / s.B_nested).median(),
                         drift=slope(s.n, s.dens)))
    rows.append(dict(label=r"\emph{all}", nets=len(d), lo=d.n.min(), hi=d.n.max(),
                     beta=slope(d.n, d.B_nested),
                     size=(d.n / d.B_nested).median(),
                     drift=slope(d.n, d.dens)))
    return pd.DataFrame(rows)


def write(path, body):
    with open(os.path.join(ROOT, path), "w") as fh:
        fh.write(body)
    print(f"wrote {path}")


def main():
    m = models()
    print(m.to_string(index=False), "\n")
    body = ["\\begin{center}", "\\small", "\\begin{tabular}{lrrrr}", "\\hline",
            "rule & $n$ & $\\beta$ & $B/n$ & nodes per\\\\",
            " & & & & community\\\\", "\\hline"]
    for r in m.itertuples():
        body.append(f"{r.label} & ${r.lo}$--${r.hi}$ & ${r.beta:.2f}$ & "
                    f"${r.dens:.3f}\\pm{r.spread:.3f}$ & ${r.size:.0f}$\\\\")
    body += ["\\hline", "\\end{tabular}", "\\end{center}"]
    write("tab-density-models.tex", "\n".join(body) + "\n")

    r = real()
    print(r.to_string(index=False))
    body = ["\\begin{center}", "\\small", "\\begin{tabular}{lrrrrr}", "\\hline",
            "networks & no. & $n$ & $\\beta$ & nodes per & $B/n$\\\\",
            " & & & & community & slope\\\\", "\\hline"]
    for x in r.itertuples():
        if x.label.startswith(r"\emph"):
            body.append("\\hline")
        body.append(f"{x.label} & ${x.nets}$ & ${x.lo}$--${x.hi}$ & "
                    f"${x.beta:.2f}$ & ${x.size:.0f}$ & ${x.drift:+.2f}$\\\\")
    body += ["\\hline", "\\end{tabular}", "\\end{center}"]
    write("tab-density-real.tex", "\n".join(body) + "\n")


if __name__ == "__main__":
    main()
