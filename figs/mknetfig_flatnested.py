#!/usr/bin/env python3
"""One local-search network, divided by the flat and by the nested block model.

The two models express the same partitions and differ only in what it costs to
name one, so the difference between them is invisible in any statement about what
they *can* represent. It is very visible in what they *do* return, which is what
this figure is for: the same network, the same layout, coloured twice.

Both panels use graph-tool to fit and to lay out, and the SVG goes through
svg2tikz.py, exactly as the network pictures of Chapter 6 do. The layout is
computed once and shared, so the only thing that changes between the panels is
the colouring.

Usage:  python3 mknetfig_flatnested.py   (writes the SVGs and netfig-fn-*.tex at
                                          the repository root)
"""
import os
import subprocess
import sys

import numpy as np
import graph_tool.all as gt

from ramsey_motif import MODELS, to_edges

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# n = 200, where the flat ceiling has begun to bite: at n = 100 the two models
# return the same three groups and there is nothing to show.
MODEL, N, SEED = "LS(n,1)", 200, 111
SHADES = ([0.28, 0.28, 0.28, 1.0], [1.0, 1.0, 1.0, 1.0], [0.70, 0.70, 0.70, 1.0],
          [0.48, 0.48, 0.48, 1.0], [0.88, 0.88, 0.88, 1.0], [0.12, 0.12, 0.12, 1.0],
          [0.60, 0.60, 0.60, 1.0])


def fit(g, nested, seed=0):
    gt.seed_rng(seed)
    if nested:
        st = gt.minimize_nested_blockmodel_dl(g, state_args=dict(deg_corr=True))
        b = np.asarray(st.get_bs()[0], dtype=int)[:g.num_vertices()]
    else:
        st = gt.minimize_blockmodel_dl(g, state_args=dict(deg_corr=True))
        b = np.asarray(st.get_blocks().a, dtype=int)[:g.num_vertices()]
    _, b = np.unique(b, return_inverse=True)
    return b, st.entropy() / np.log(2)


def main():
    gen, params = MODELS[MODEL]
    edges = to_edges(gen(N, params, SEED))
    g = gt.Graph(directed=False)
    g.add_vertex(N)
    g.add_edge_list(edges)

    gt.seed_rng(7)
    np.random.seed(7)
    pos = gt.sfdp_layout(g)          # one layout, shared by both panels

    for nested, tag in ((False, "flat"), (True, "nested")):
        b, sigma = fit(g, nested)
        counts = np.bincount(b)
        order = {grp: i for i, grp in enumerate(np.argsort(-counts))}
        if len(counts) > len(SHADES):
            raise SystemExit(f"{len(counts)} groups; only {len(SHADES)} shades")
        col = g.new_vertex_property("vector<double>")
        for v in g.vertices():
            col[v] = SHADES[order[b[int(v)]]]
        svg = os.path.join(ROOT, f"model-ls_d1_n{N}_seed{SEED}_{tag}.svg")
        gt.graph_draw(g, pos=pos, vertex_fill_color=col,
                      vertex_color=[0.1, 0.1, 0.1, 1.0], vertex_pen_width=0.8,
                      vertex_size=14, edge_pen_width=1.0,
                      edge_color=[0.4, 0.4, 0.4, 0.8],
                      output_size=(1200, 1200), output=svg)
        tikz = os.path.join(ROOT, f"netfig-fn-{tag}.tex")
        subprocess.run([sys.executable, os.path.join(ROOT, "svg2tikz.py"),
                        svg, tikz], check=True)
        print(f"  {tag:<7} {len(counts)} groups, sizes "
              f"{sorted(counts.tolist(), reverse=True)}, Sigma = {sigma:.0f} bits")


if __name__ == "__main__":
    main()
