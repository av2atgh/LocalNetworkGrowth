#!/usr/bin/env python3
r"""
infomap_modules.py -- does compressing a walk find the same structure as
compressing an adjacency matrix?

Section 11.5 asks whether the modules the map equation prefers are chapter 9's
meso-structure or a different decomposition.  The test is a pair of networks that
the two currencies of chapter 9 treat oppositely:

  * the pseudofractal, DBB(n,1), whose footprint tiles it completely -- the motif
    alphabet names 100% of its links and the fit then reports NO communities,
    while the meso-structure total kappa_0 + kappa still grows in proportion to n
    (beta = 1.00);

  * BB(n,1), the stochastic triangle rule, where the alphabet names 54% and a
    genuine community residue is left over.

If the map equation buys the meso-structure total it must find modules in both,
since both have that total growing like n.  If it buys the community residue it
finds them only in the second.

The rings are the controls chapter 6 supplies: it proves the ring is split by a
block model exactly when its links outnumber its nodes, so C_n(1) sits at the
margin and C_n(1,2) is past it.

Infomap is run as libs.py runs it, regularized, with the regularization strength
swept because the answer depends on it -- which is itself part of the result.

Usage:  python3 infomap_modules.py
"""

import os
import sys

import numpy as np
import networkx as nx
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
CSV = os.path.join(HERE, "data", "infomap-modules.csv")

from infomap import Infomap
from qwalk_ipr import RULES, pseudofractal, diamond, giant

STRENGTHS = (0.0, 0.5, 1.0, 2.0)


def modules(g, strength):
    """Number of Infomap modules.  strength 0 means unregularized."""
    kw = dict(silent=True)
    if strength > 0:
        kw.update(regularized=True, regularization_strength=strength)
    im = Infomap(**kw)
    idx = {v: i for i, v in enumerate(g.nodes())}
    im.add_links([(idx[u], idx[v]) for u, v in g.edges()])
    im.run()
    return len({nd.module_id for nd in im.nodes}), im.codelength


def circulant(n, offs):
    g = nx.Graph()
    g.add_nodes_from(range(n))
    for i in range(n):
        for o in offs:
            g.add_edge(i, (i + o) % n)
    return g


CASES = [
    # label, builder, sizes, what chapter 9 says about it
    ("pseudofractal $DBB(n,1)$", lambda t: pseudofractal(t), [5, 6, 7],
     "alphabet names 100\\%, no communities"),
    ("diamond $(2,2)$", lambda t: giant(diamond(t, 2, 2)), [5, 6],
     "alphabet names 100\\%, no communities"),
    ("$BB(n,1)$", lambda n: giant(RULES["BB(n,1)"][0](n, 1)), [400, 1200, 3600],
     "alphabet names 54\\%, residue left"),
    ("$BB(n,2)$", lambda n: giant(RULES["BB(n,2)"][0](n, 1)), [1200, 3600],
     "alphabet names 57\\%, residue left"),
    ("ring $C_n(1)$", lambda n: nx.cycle_graph(n), [1200, 3600],
     "$m/n=1$, at the margin"),
    ("ring $C_n(1,2)$", lambda n: circulant(n, (1, 2)), [1200, 3600],
     "$m/n=2$, past it"),
    ("ring $C_n(1,2,3)$", lambda n: circulant(n, (1, 2, 3)), [1200],
     "$m/n=3$, past it"),
]


def measure():
    rows = []
    for lab, build, sizes, note in CASES:
        for a in sizes:
            g = build(a)
            n = g.number_of_nodes()
            rec = dict(label=lab, note=note, n=n,
                       m_over_n=g.number_of_edges() / n)
            for s in STRENGTHS:
                k, cl = modules(g, s)
                rec[f"mod_{s}"] = k
                rec[f"dens_{s}"] = k / n
            rows.append(rec)
            print(f"  {lab:26s} n={n:5d}  " +
                  "  ".join(f"s={s}:{rec[f'mod_{s}']:4d}" for s in STRENGTHS),
                  flush=True)
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(CSV), exist_ok=True)
    df.to_csv(CSV, index=False)
    print("wrote", os.path.relpath(CSV, HERE))
    return df


def table(df):
    out = [r"\begin{center}", r"\footnotesize\setlength{\tabcolsep}{4pt}",
           r"\begin{tabular}{lrrrrrr}", r"\hline",
           r" & & & \multicolumn{4}{c}{modules, by regularization strength}\\",
           r"network & $n$ & $m/n$ & none & $0.5$ & $1.0$ & $2.0$\\", r"\hline"]
    for _, r in df.iterrows():
        out.append(f"{r.label} & ${r.n:.0f}$ & ${r.m_over_n:.2f}$ & "
                   f"${r['mod_0.0']:.0f}$ & ${r['mod_0.5']:.0f}$ & "
                   f"$\\mathbf{{{r['mod_1.0']:.0f}}}$ & ${r['mod_2.0']:.0f}$"
                   + r"\\")
    out += [r"\hline", r"\end{tabular}", r"\end{center}"]
    p = os.path.join(os.path.dirname(HERE), "tab-hc-infomap.tex")
    open(p, "w").write("\n".join(out) + "\n")
    print("wrote", os.path.relpath(p, os.path.dirname(HERE)))


def main():
    if "--force" not in sys.argv and os.path.exists(CSV):
        df = pd.read_csv(CSV)
        print("read", os.path.relpath(CSV, HERE))
    else:
        df = measure()
    pd.set_option("display.width", 200)
    print(df.to_string(index=False))
    table(df)


if __name__ == "__main__":
    main()
