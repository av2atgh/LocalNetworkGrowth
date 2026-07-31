#!/usr/bin/env python3
"""Two more growth series: co-authorship and the yeast interactome.

Companion to growth_series.py, which does the Internet. Same question: measure
beta along one network's own growth rather than across a census of different
networks, since a cross-sectional slope estimates how B varies *between* systems
and that need not be how it varies *within* one.

  coauthor   MAG history-of-science co-authorship, whose links carry the year
             they were made, accumulated: n = 2886 in 1900 to 242048 in 2017.
             A genuine growth series -- the collaboration graph really did grow.

  yeast      The BioGRID physical interactome of S. cerevisiae, accumulated by
             the publication year of the paper reporting each interaction.
             n runs from a few hundred in the early 1990s to about 6000.

The second is a growth series of a different kind and the difference matters.
Yeast's interactome is not growing; our catalogue of it is. So its beta measures
how the number of detectable communities scales as a fixed object is discovered,
not as a network is built. That is worth measuring precisely because it is the
confound Chapter 7 already warns about for the tree-of-life series -- there
across organisms, here across time -- and putting a number on it says how much
of a census beta could be discovery rather than growth.

The BioGRID release archive is a 160 MB download; point --bgzip at it.

Writes data/growth_series_bio.csv. Resumable.

Usage:  python3 growth_series_bio.py --bgzip /path/to/BIOGRID-ORGANISM-*.zip
"""
import argparse
import collections
import os
import time
import zipfile

import numpy as np
import pandas as pd
import graph_tool.all as gt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data", "growth_series_bio.csv")
# Two --only runs in parallel would clobber each other: each loads the file at
# start and rewrites it whole. Pass --out to give each its own.

MAG = "mag_history_coauthor/full-proj"
MAG_YEARS = (1900, 1910, 1920, 1930, 1940, 1948, 1955, 1960, 1965, 1970,
             1975, 1980, 1985, 1990, 1995, 2000, 2005, 2010)
YEAST = "BIOGRID-ORGANISM-Saccharomyces_cerevisiae_S288c"
YEAST_YEARS = tuple(range(1994, 2023, 2))


def reps_for(n):
    """Always the same number of restarts, however slow.

    An earlier version tapered this to 1 above n = 80000, to save time, and that
    quietly wrecked the measurement. A single run of the minimiser lands in a
    worse optimum than the best of five, and it does so *more* often on larger
    networks -- so tapering the restarts underfits the big end of a series and
    biases the slope downward. The tell was unmistakable once looked for: the
    2005 co-authorship snapshot came back with fewer communities than the 1995
    one, which strictly contains it. A cumulative series cannot lose groups as it
    grows. Any schedule that varies with n puts a trend into the fit that the
    network does not have."""
    return 5


def fit(g):
    g = gt.extract_largest_component(g, prune=True)
    n = g.num_vertices()
    best = None
    for r in range(reps_for(n)):
        gt.seed_rng(r)
        st = gt.minimize_nested_blockmodel_dl(g, state_args=dict(deg_corr=True))
        s = st.entropy() / np.log(2)
        if best is None or s < best[0]:
            b = np.asarray(st.get_bs()[0], dtype=int)[:n]
            best = (s, len(np.unique(b)))
    return n, g.num_edges(), best[1], best[0]


def mag_snapshots():
    import graph_tool.collection as coll
    g = coll.ns[MAG]
    t = g.ep.time
    for y in MAG_YEARS:
        v = gt.Graph(gt.GraphView(g, efilt=t.a <= y), prune=True)
        gt.remove_parallel_edges(v)
        gt.remove_self_loops(v)
        yield f"coauthor/{y}", "coauthor", v


def yeast_snapshots(bgzip):
    """Cumulative physical-interaction graphs, by publication year."""
    with zipfile.ZipFile(bgzip) as z:
        name = next(x for x in z.namelist() if YEAST in x)
        first = collections.defaultdict(lambda: 9999)   # link -> first year seen
        with z.open(name) as fh:
            head = fh.readline().decode().rstrip("\n").split("\t")
            iA, iB = head.index("Official Symbol Interactor A"), \
                     head.index("Official Symbol Interactor B")
            iT, iY = head.index("Experimental System Type"), head.index("Author")
            for raw in fh:
                f = raw.decode("utf8", "replace").rstrip("\n").split("\t")
                if len(f) <= iY or f[iT] != "physical":
                    continue
                a, b = f[iA], f[iB]
                if a == b or a == "-" or b == "-":
                    continue
                y = f[iY].rsplit("(", 1)[-1].rstrip(")")
                if not y.isdigit():
                    continue
                e = (a, b) if a < b else (b, a)
                first[e] = min(first[e], int(y))

    print(f"  yeast: {len(first)} distinct physical links", flush=True)
    for y in YEAST_YEARS:
        es = [e for e, fy in first.items() if fy <= y]
        if len(es) < 200:
            continue
        idx = {}
        g = gt.Graph(directed=False)
        for a, b in es:
            for x in (a, b):
                if x not in idx:
                    idx[x] = int(g.add_vertex())
            g.add_edge(idx[a], idx[b])
        yield f"yeast/{y}", "yeast", g


def main():
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--bgzip", default=None)
    ap.add_argument("--only", default=None, choices=("coauthor", "yeast"))
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()

    OUT = a.out
    done = pd.read_csv(OUT) if os.path.exists(OUT) else pd.DataFrame(columns=["name"])
    rows, have = done.to_dict("records"), set(done.name)

    sources = []
    if a.only != "yeast":
        sources.append(mag_snapshots())
    if a.only != "coauthor" and a.bgzip:
        sources.append(yeast_snapshots(a.bgzip))

    for src in sources:
        for name, series, g in src:
            if name in have:
                continue
            t = time.time()
            n, m, B, sigma = fit(g)
            rows.append(dict(series=series, name=name, n=n, m=m, B_nested=B,
                             sigma=round(sigma, 1), reps=reps_for(n),
                             secs=round(time.time() - t, 1)))
            pd.DataFrame(rows).to_csv(OUT, index=False)
            print(f"  {name:<18} n={n:7d} m={m:8d} B={B:5d} B/n={B/n:.4f}"
                  f"  ({time.time()-t:.0f}s)", flush=True)

    d = pd.DataFrame(rows)
    print()
    for s in d.series.unique():
        x = d[d.series == s]
        if len(x) > 3:
            b = np.polyfit(np.log(x.n), np.log(x.B_nested), 1)[0]
            print(f"  {s:<10} {len(x):3d} points, n = {x.n.min()}-{x.n.max()}, "
                  f"beta = {b:.3f}")


if __name__ == "__main__":
    main()
