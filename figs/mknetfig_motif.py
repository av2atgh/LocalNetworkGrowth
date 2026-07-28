#!/usr/bin/env python3
"""Draw one instance each of LS(100,1) and DS(100,0.5) resolved into three groups.

Companion to the network pictures of Chapter 6, drawn the same way: graph-tool
lays the network out and colours the nodes by community, the SVG goes through
svg2tikz.py, and the chapter includes the resulting TikZ. Here the colouring is
the partition found by the motif-alphabet model of motif_sbm.py, unconstrained,
so both pictures show a division the method arrived at on its own rather than one
imposed on it.

Three groups rather than two, because two is a rarity for both rules and three is
what a reader is likely to meet. Out of a hundred realisations at n = 100 the
number of communities came out as

    kappa        1   2   3   4   5   6  ...
    LS(n,1)     48   2  24  16   9   1
    DS(n,0.5)    3   2   6  23  24  14  ...

so kappa = 2 is 2 in 100 for either rule while kappa = 3 is 24 in 100 for local
search. Local search is also stubborn about two at larger sizes: over 300
realisations of LS(200,1) the count came out as 1 six times and then 4 through
10, never 2 or 3, so when it splits it splits into several groups at once.

The fit is a stochastic search whose answer varies between runs on the same
graph, so a seed that gave three communities once may give five the next time.
The seeds below were chosen by refitting each candidate four times and keeping
only those that returned three every time.

Usage:  python3 mknetfig_motif.py     (writes the SVGs and netfig-*.tex at the
                                       repository root)
"""
import os
import subprocess
import sys

import numpy as np
import graph_tool.all as gt

from motif_sbm import MotifSBM
from ramsey_motif import MODELS, to_edges

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# model, n, seed, stem. Seeds verified to return two communities on every refit;
# see the note above on why that has to be checked.
CASES = (("LS(n,1)", 100, 100005, "model-ls_d1_n100_seed100005_blocks"),
         ("DS(n,0.5)", 100, 100066, "model-ds_q0.5_n100_seed100066_blocks"))
GROUPS = 3

# dark / white / mid-grey, matching the three node styles svg2tikz.py emits
SHADES = ([0.22, 0.22, 0.22, 1.0], [1.0, 1.0, 1.0, 1.0], [0.62, 0.62, 0.62, 1.0])


def draw(name, n, seed, stem):
    gen, params = MODELS[name]
    edges = to_edges(gen(n, params, seed))

    model = MotifSBM(n, edges).fit()
    _, b = np.unique(model.blocks, return_inverse=True)
    counts = np.bincount(b)
    if len(counts) != GROUPS:
        raise SystemExit(f"{name} seed {seed} returned {len(counts)} communities, "
                         f"not {GROUPS} -- pick another seed")
    motifs = model.motif_counts()

    g = gt.Graph(directed=False)
    g.add_vertex(n)
    g.add_edge_list(edges)
    # largest group dark, so the pictures read the same way as Chapter 6's
    rank = {g_: i for i, g_ in enumerate(np.argsort(-counts))}
    col = g.new_vertex_property("vector<double>")
    for v in g.vertices():
        col[v] = SHADES[rank[b[int(v)]]]

    gt.seed_rng(7)
    np.random.seed(7)
    pos = gt.sfdp_layout(g)
    svg = os.path.join(ROOT, stem + ".svg")
    gt.graph_draw(g, pos=pos, vertex_fill_color=col,
                  vertex_color=[0.1, 0.1, 0.1, 1.0], vertex_pen_width=0.8,
                  vertex_size=14, edge_pen_width=1.0,
                  edge_color=[0.4, 0.4, 0.4, 0.8],
                  output_size=(1200, 1200), output=svg)

    tag = stem.split("_")[0].split("-")[1]
    tikz = os.path.join(ROOT, f"netfig-{tag}{n}.tex")
    subprocess.run([sys.executable, os.path.join(ROOT, "svg2tikz.py"), svg, tikz],
                   check=True)
    print(f"  {name} n={n} seed={seed}: {len(edges)} links, split "
          f"{sorted(counts.tolist(), reverse=True)}, motifs {dict(motifs)} "
          f"-> {os.path.relpath(tikz, ROOT)}")


def main():
    for case in CASES:
        draw(*case)


if __name__ == "__main__":
    main()
