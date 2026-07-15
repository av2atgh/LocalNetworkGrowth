#!/usr/bin/env python3
"""Fetch real networks from Netzschleuder and measure them for the Introduction.

Networks come from the Netzschleuder repository (networks.skewed.com) through the
graph-tool interface, `graph_tool.collection.ns`. The four chosen are exactly the
systems the Introduction names when it claims two families:

  family 1 (technological / biological, expected disassortative)
    internet_as                  Internet at the autonomous-system level
    gnutella/31                  Gnutella peer-to-peer file-sharing overlay
    interactome_yeast            yeast protein interaction network
  family 2 (social, expected assortative)
    arxiv_collab/cond-mat-2005   cond-mat co-authorship

This script only WRITES CSVs (data/intro_curves.csv, data/intro_summary.csv);
mkfigs_intro.py plots them. The split is deliberate: measuring needs graph-tool
and a network connection, plotting needs neither, so the figures stay
reproducible from the committed CSVs alone.

Running it needs graph-tool plus zstandard (Netzschleuder serves zstd-compressed
files). graph-tool is awkward to pip-install; on macOS the easiest route is

    brew install graph-tool
    python3 -m venv --system-site-packages /tmp/gtenv   # using brew's python3.13
    /tmp/gtenv/bin/pip install zstandard
    /tmp/gtenv/bin/python fetch_intro_data.py

Everything is measured on the largest connected component of the simple,
undirected version of each graph (Gnutella is directed as published; direction is
dropped, as the Introduction's measures are defined for undirected graphs).
"""

import os
import sys

import numpy as np

import graph_tool.all as gt

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

NETWORKS = [
    # (netzschleuder key, short label, family)
    ("internet_as", "Internet (AS)", "technological"),
    ("gnutella/31", "Gnutella", "technological"),
    ("interactome_yeast", "Yeast proteins", "biological"),
    ("arxiv_collab/cond-mat-2005", "cond-mat co-authorship", "social"),
]

NBINS = 22  # logarithmic bins in degree


def prepare(name):
    """Load, drop direction/loops/parallels, keep the largest component."""
    g = gt.collection.ns[name]
    g = gt.GraphView(g, directed=False)
    g = gt.Graph(g, prune=True, directed=False)
    gt.remove_self_loops(g)
    gt.remove_parallel_edges(g)
    g = gt.Graph(gt.GraphView(g, vfilt=gt.label_largest_component(g)), prune=True)
    return g


def logbin(k, y, nbins=NBINS):
    """Average y over logarithmic bins in k. Returns (centre, mean_y, count)."""
    k = np.asarray(k, float)
    y = np.asarray(y, float)
    lo, hi = max(k.min(), 1.0), k.max()
    edges = np.unique(np.floor(np.logspace(np.log10(lo), np.log10(hi + 1), nbins)))
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (k >= a) & (k < b)
        if m.sum() == 0:
            continue
        out.append((np.exp(np.log(k[m]).mean()), y[m].mean(), int(m.sum())))
    return np.array(out).T if out else (np.array([]),) * 3


def logbin_pdf(k, nbins=NBINS):
    """Log-binned degree distribution: counts per bin / (N * bin width).

    NOT the mean count per distinct degree -- that is a different quantity and a
    trap. In the far tail each distinct degree occurs exactly once, so a mean
    count is pinned at 1 and p(k) appears to go FLAT instead of continuing to
    fall. Dividing by the bin width is what makes it a density.
    """
    k = np.asarray(k, float)
    N = k.size
    edges = np.unique(
        np.floor(np.logspace(np.log10(max(k.min(), 1.0)), np.log10(k.max() + 1), nbins))
    )
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (k >= a) & (k < b)
        if m.sum() == 0:
            continue
        out.append((np.exp(np.log(k[m]).mean()), m.sum() / (N * (b - a))))
    return np.array(out).T


def omega_i(ki, nbr_degs):
    """Maximum number of edges realizable among the neighbours of a vertex.

    This is omega_i of Soffer and Vazquez, Phys. Rev. E 71, 057101 (2005). The
    standard clustering coefficient divides the triangles t_i by C(k_i,2), the
    number of neighbour PAIRS -- but most of those pairs may be impossible: a
    neighbour j of degree k_j has only k_j - 1 links to spend, so it can reach at
    most min(k_i, k_j) - 1 of i's other neighbours. Dividing by an unreachable
    maximum is what makes a hub surrounded by leaves look unclustered no matter
    how its neighbours are wired.

    Their greedy (Havel-Hakimi): build the list min(k_i,k_j)-1 over neighbours,
    sort descending; repeatedly take the head, spend its capacity on as many of
    the largest remaining entries as possible, decrementing each; drop the head
    and any zeros; re-sort. omega_i is the total number of edges drawn.

    Verified against their Fig. 2: k_i=5, neighbour degrees {8,7,2,2,2} gives
    omega_i=4, against Omega_i=5 and C(k_i,2)=10.
    """
    d = np.sort(np.minimum(ki, np.asarray(nbr_degs, dtype=np.int64)) - 1)[::-1]
    d = d[d > 0]
    edges = 0
    while d.size:
        h = int(d[0])
        rest = d[1:].copy()
        take = min(h, rest.size)
        if take > 0:
            rest[:take] -= 1  # rest is sorted descending: spend on the largest
            edges += take
        d = rest[rest > 0]
        d = np.sort(d)[::-1]
    return edges


def mean_distance(g, n_sources=300, seed=7):
    """Average shortest-path distance, sampled over sources (exact per source)."""
    rng = np.random.default_rng(seed)
    N = g.num_vertices()
    srcs = rng.choice(N, size=min(n_sources, N), replace=False)
    tot, cnt = 0.0, 0
    for s in srcs:
        d = gt.shortest_distance(g, source=g.vertex(s)).a
        d = d[(d > 0) & (d < 2**30)]  # drop self and unreachable
        tot += d.sum()
        cnt += d.size
    return tot / cnt


def measure(g, label, family):
    N = g.num_vertices()
    E = g.num_edges()
    k = g.get_total_degrees(g.get_vertices()).astype(float)
    k1 = k.mean()
    k2 = (k**2).mean()

    # local clustering, per node
    c = gt.local_clustering(g).a.astype(float)

    # triangles per node: t_i = c_i * C(k_i,2), recovered from graph-tool's c_i
    t = np.rint(c * k * (k - 1) / 2).astype(np.int64)

    # average neighbour degree, and the Soffer-Vazquez corrected clustering
    knn = np.zeros(N)
    ctilde = np.full(N, np.nan)
    for v in g.iter_vertices():
        nb = g.get_all_neighbors(v)
        if nb.size == 0:
            knn[v] = np.nan
            continue
        knn[v] = k[nb].mean()
        if nb.size >= 2:
            w = omega_i(int(k[v]), k[nb].astype(np.int64))
            # omega_i = 0 means the neighbours CANNOT be joined at all (e.g. a
            # hub of leaves): clustering is then undefined, not zero.
            if w > 0:
                ctilde[v] = t[v] / w

    # Two null expectations for the clustering.
    #
    # (1) Erdos-Renyi with the same n and <k>. This is the comparison Watts and
    #     Strogatz made, and the one the Introduction's text refers to.
    c_er = k1 / (N - 1)
    #
    # (2) A degree-preserving rewiring -- the null the rest of this book insists
    #     on. We MEASURE it rather than use the textbook formula
    #     (<k^2>-<k>)^2 / (n <k>^3): that formula assumes <k^2> converges, which
    #     fails for the Internet (gamma ~ 2.1), where it returns a value above
    #     the measured clustering and even approaching 1 -- nonsense. Rewiring
    #     and measuring costs a few seconds and cannot mislead.
    gr = gt.Graph(g, prune=True)
    gt.random_rewire(gr, model="configuration", n_iter=100)
    c_rewired = gt.local_clustering(gr).a.astype(float).mean()

    summary = dict(
        label=label,
        family=family,
        n=N,
        E=E,
        mean_k=k1,
        mean_c=c.mean(),
        mean_ctilde=np.nanmean(ctilde),
        c_er=c_er,
        c_rewired=c_rewired,
        c_conf_formula=(k2 - k1) ** 2 / (N * k1**3),
        mean_d=mean_distance(g),
        r=gt.assortativity(g, "total")[0],
    )

    kb, pk = logbin_pdf(k)

    # <c>(k) is only DEFINED for k >= 2: a degree-1 node has no pairs of
    # neighbours, so c_i = 2 e_i / [k_i (k_i - 1)] is 0/0. graph-tool reports 0
    # for those, and since degree-1 nodes dominate the small-k bins, including
    # them drags the low-k end to zero and makes <c>(k) appear to RISE with
    # degree -- the exact opposite of the hierarchy. Restrict to k >= 2.
    sel = k >= 2
    kc, ck, _ = logbin(k[sel], c[sel])

    kn, knnb, _ = logbin(k[~np.isnan(knn)], knn[~np.isnan(knn)])

    # corrected clustering, over the nodes where it is defined
    ok = ~np.isnan(ctilde)
    kt, ctk, _ = logbin(k[ok], ctilde[ok])

    return summary, (kb, pk), (kc, ck), (kn, knnb), (kt, ctk)


def main():
    os.makedirs(DATA, exist_ok=True)
    rows, curves = [], []
    for name, label, family in NETWORKS:
        sys.stderr.write(f"fetching {name} ...\n")
        g = prepare(name)
        s, (kb, pk), (kc, ck), (kn, knn), (kt, ctk) = measure(g, label, family)
        s["netzschleuder"] = name
        rows.append(s)
        sys.stderr.write(
            f"  {label}: n={s['n']} E={s['E']} <k>={s['mean_k']:.2f} "
            f"<c>={s['mean_c']:.4f} (rewired {s['c_rewired']:.4f}, "
            f"corrected {s['mean_ctilde']:.4f}) "
            f"<d>={s['mean_d']:.2f} r={s['r']:+.3f}\n"
        )
        for q, (x, y) in [("pk", (kb, pk)), ("ck", (kc, ck)), ("knn", (kn, knn)),
                          ("ctilde", (kt, ctk))]:
            for xi, yi in zip(x, y):
                curves.append((label, q, xi, yi))

    with open(os.path.join(DATA, "intro_summary.csv"), "w") as f:
        cols = ["label", "family", "netzschleuder", "n", "E", "mean_k", "mean_c",
                "mean_ctilde", "c_er", "c_rewired", "c_conf_formula", "mean_d", "r"]
        f.write(",".join(cols) + "\n")
        for s in rows:
            f.write(",".join(str(s[c]) for c in cols) + "\n")

    with open(os.path.join(DATA, "intro_curves.csv"), "w") as f:
        f.write("label,quantity,k,value\n")
        for label, q, x, y in curves:
            f.write(f"{label},{q},{x:.6g},{y:.6g}\n")

    sys.stderr.write("wrote data/intro_summary.csv and data/intro_curves.csv\n")


if __name__ == "__main__":
    main()
