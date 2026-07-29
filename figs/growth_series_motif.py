#!/usr/bin/env python3
"""Do real growing networks keep their community density once motifs are named?

Section 7.13 measured beta along three real growth series and found it at one:
the number of communities grows in proportion to size, so the density b = B/n is
a constant. Those fits used the nested block model only. Section 8.6 then found
that naming motifs can destroy that density -- on the diamond lattice beta falls
from 1.05 to 0.31, and the motif description is the one minimum description
length prefers, by a margin that widens with size.

So the density measured in Chapter 7 was measured with a vocabulary that could
not name a triangle. This script asks whether it survives one that can.

Three outcomes, and they are not equally interesting:

  beta stays near 1   the density class is robust, and the diamond's fractal
                      behaviour needs a tiled lattice to appear -- real networks
                      are safe because their motifs overlap.

  beta falls to 1/2..1  genuine fractal scaling, surviving motif correction. This
                      would be the first instance in the book that is not an
                      instrument artefact: Chapter 6's 0.61 was the flat model's
                      ceiling and Chapter 7's 0.80 was a mixed sample.

  beta falls below 1/2  the diamond's class, in a real network.

Two series are small enough to fit with the full alphabet: the Internet at the
autonomous-system level (n up to 6474) and the yeast interactome accumulated by
publication year (n up to about 7000). Co-authorship reaches 60000 nodes and is
left out; a note in the manuscript says so.

Writes data/growth_series_motif.csv. Resumable.

Usage:  python3 growth_series_motif.py --series route_views [--restarts 2]
        python3 growth_series_motif.py --series yeast --bgzip /path/to/biogrid.zip
"""
import argparse
import os
import time

import numpy as np
import pandas as pd

from motif_sbm import MotifSBM

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data", "growth_series_motif.csv")

FULL = ("edge", "triangle", "square")


def fit_both(n, edges, restarts):
    """Link-only and full alphabet on the same network.

    With the alphabet reduced to the link, Sigma is exactly the description
    length of the ordinary degree-corrected model, so this is the strictly
    nested comparison of Section 8.4 and a negative dSigma means the motifs paid
    for themselves.
    """
    out = {}
    for alpha, tag in ((("edge",), "link"), (FULL, "motif")):
        best = None
        for s in range(restarts):
            m = MotifSBM(n, edges, alphabet=alpha)
            m.fit()
            sigma = m.description_length()
            if best is None or sigma < best[0]:
                c = m.motif_counts()
                best = (sigma, len(set(m.blocks)), c.get("triangle", 0),
                        c.get("square", 0))
        out[tag] = best
    return out


def route_views_snapshots(points):
    import graph_tool.all as gt
    import graph_tool.collection as coll
    ks = sorted(x for x in coll.ns.keys() if x.startswith("route_views/"))
    idx = np.unique(np.linspace(0, len(ks) - 1, points).round().astype(int))
    for k in (ks[i] for i in idx):
        g = gt.Graph(coll.ns[k], directed=False, prune=True)
        gt.remove_self_loops(g)          # a few AS snapshots carry them
        gt.remove_parallel_edges(g)
        g = gt.extract_largest_component(g, prune=True)
        n = g.num_vertices()
        if n < 2000:              # collector partly down that day, not a smaller Internet
            continue
        yield k, n, [(int(u), int(v)) for u, v in g.get_edges()]


def yeast_snapshots(bgzip, points):
    from growth_series_bio import yeast_snapshots as ys
    import graph_tool.all as gt
    snaps = [(name, g) for name, _, g in ys(bgzip)]
    idx = np.unique(np.linspace(0, len(snaps) - 1, points).round().astype(int))
    for name, g in (snaps[i] for i in idx):
        gt.remove_self_loops(g)
        gt.remove_parallel_edges(g)
        g = gt.extract_largest_component(g, prune=True)
        yield name, g.num_vertices(), [(int(u), int(v)) for u, v in g.get_edges()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--series", required=True, choices=("route_views", "yeast"))
    ap.add_argument("--bgzip", default=None)
    ap.add_argument("--points", type=int, default=8)
    ap.add_argument("--restarts", type=int, default=2)
    a = ap.parse_args()

    out = os.path.join(HERE, "data", f"growth_series_motif_{a.series}.csv")
    done = pd.read_csv(out) if os.path.exists(out) else pd.DataFrame(columns=["name"])
    rows, have = done.to_dict("records"), set(done.name)

    src = (route_views_snapshots(a.points) if a.series == "route_views"
           else yeast_snapshots(a.bgzip, a.points))

    for name, n, edges in src:
        if name in have:
            continue
        t = time.time()
        r = fit_both(n, edges, a.restarts)
        rows.append(dict(series=a.series, name=name, n=n, m=len(edges),
                         kappa_link=r["link"][1], kappa_motif=r["motif"][1],
                         sigma_link=round(r["link"][0], 1),
                         sigma_motif=round(r["motif"][0], 1),
                         d_sigma=round(r["motif"][0] - r["link"][0], 1),
                         n_tri=r["motif"][2], n_sq=r["motif"][3],
                         secs=round(time.time() - t, 1)))
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"  {name:<24} n={n:6d} m={len(edges):7d}  kappa {r['link'][1]:4d}"
              f" -> {r['motif'][1]:4d}   dSigma {rows[-1]['d_sigma']:+10.0f}"
              f"  ({time.time()-t:.0f}s)", flush=True)

    d = pd.DataFrame(rows)
    if len(d) > 3:
        for col, lab in (("kappa_link", "link only"), ("kappa_motif", "+ motifs")):
            x = d[d[col] > 1]
            if len(x) > 3:
                b = np.polyfit(np.log(x.n), np.log(x[col]), 1)[0]
                print(f"\n  {lab:10s} beta = {b:.3f}  from {list(x[col])}")


if __name__ == "__main__":
    main()
