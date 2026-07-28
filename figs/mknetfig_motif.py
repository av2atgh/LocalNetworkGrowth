#!/usr/bin/env python3
"""Draw one instance each of LS(100,1) and DS(100,0.5) resolved into two groups.

Companion to the network pictures of Chapter 6, drawn the same way: graph-tool
lays the network out and colours the nodes by community, the SVG goes through
svg2tikz.py, and the chapter includes the resulting TikZ. Here the colouring is
the partition found by the motif-alphabet model of motif_sbm.py, unconstrained,
so both pictures show a division the method arrived at on its own rather than one
imposed on it.

Two things had to be worked around, and both say something about the models.

Local search does not split in two at its own Ramsey number. Over 300
realisations of LS(200,1) the number of communities came out as

    kappa   1   4   5   6   7   8   9  10
    count   6   5  21  81  96  70  19   2

with no realisation at two or three at all: when local search splits, it splits
into several at once. Two-way splits do occur at smaller sizes -- 8 realisations
in 100 at n = 90, 2 at n = 100, none at 120 or beyond -- so the picture is drawn
at n = 100, where a genuine two-community instance exists.

And the fit is a stochastic search whose answer varies between runs on the same
graph, so a seed that gave two communities once may give four the next time.
The seeds below were chosen by refitting each candidate several times and keeping
only those that returned two every time.

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
CASES = (("LS(n,1)", 100, 100054, "model-ls_d1_n100_seed100054_blocks"),
         ("DS(n,0.5)", 100, 100099, "model-ds_q0.5_n100_seed100099_blocks"))

DARK = [0.22, 0.22, 0.22, 1.0]
LIGHT = [1.0, 1.0, 1.0, 1.0]


def draw(name, n, seed, stem):
    gen, params = MODELS[name]
    edges = to_edges(gen(n, params, seed))

    model = MotifSBM(n, edges).fit()
    _, b = np.unique(model.blocks, return_inverse=True)
    counts = np.bincount(b)
    if len(counts) != 2:
        raise SystemExit(f"{name} seed {seed} returned {len(counts)} communities, "
                         f"not 2 -- pick another seed")
    motifs = model.motif_counts()

    g = gt.Graph(directed=False)
    g.add_vertex(n)
    g.add_edge_list(edges)
    # larger group dark, so the pictures read the same way as Chapter 6's
    big = int(np.argmax(counts))
    col = g.new_vertex_property("vector<double>")
    for v in g.vertices():
        col[v] = DARK if b[int(v)] == big else LIGHT

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
