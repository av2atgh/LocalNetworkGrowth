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

    # average neighbour degree, per node (nodes of degree 0 excluded by the LCC)
    knn = np.zeros(N)
    for v in g.iter_vertices():
        nb = g.get_all_neighbors(v)
        knn[v] = k[nb].mean() if nb.size else np.nan

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

    return summary, (kb, pk), (kc, ck), (kn, knnb)


def main():
    os.makedirs(DATA, exist_ok=True)
    rows, curves = [], []
    for name, label, family in NETWORKS:
        sys.stderr.write(f"fetching {name} ...\n")
        g = prepare(name)
        s, (kb, pk), (kc, ck), (kn, knn) = measure(g, label, family)
        s["netzschleuder"] = name
        rows.append(s)
        sys.stderr.write(
            f"  {label}: n={s['n']} E={s['E']} <k>={s['mean_k']:.2f} "
            f"<c>={s['mean_c']:.4f} (rewired {s['c_rewired']:.4f}) "
            f"<d>={s['mean_d']:.2f} r={s['r']:+.3f}\n"
        )
        for q, (x, y) in [("pk", (kb, pk)), ("ck", (kc, ck)), ("knn", (kn, knn))]:
            for xi, yi in zip(x, y):
                curves.append((label, q, xi, yi))

    with open(os.path.join(DATA, "intro_summary.csv"), "w") as f:
        cols = ["label", "family", "netzschleuder", "n", "E", "mean_k", "mean_c",
                "c_er", "c_rewired", "c_conf_formula", "mean_d", "r"]
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
