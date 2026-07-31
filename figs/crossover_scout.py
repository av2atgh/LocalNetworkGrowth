#!/usr/bin/env python3
"""Scouting pass for the L* crossover experiment: which networks can be tested?

Chapter 7 predicts that a block-model fit undercounts cycles of length L while
lambda^L < n and overcounts after, so the crossover sits at L* = ln n / ln lambda.
Two things have to hold before a network can say anything about that:

  visibility   the crossover has to fall inside the range of L we can count
               exactly, 3..8, i.e.  kappa^3 < n < kappa^8;
  tractability enumeration cost is bounded below by the answer, and the answer
               at L=8 is of order kappa^8/16, so large kappa is unreachable
               however good the pruning is.

The branching factor here is kappa = <k^2 - k>/<k>, not the <k^2>/<k> of chapter
7 as written: a cycle leaves a node by a different edge than it entered by, and
the fitted model holds degrees fixed, so a node offers k(k-1) ordered pairs of
edges rather than k^2.  See branching_verification.py, which settles it against
exact counts on synthetic ensembles.

Both criteria are decided by (n, kappa) alone and kappa needs no fit, so this
pass is cheap: download, reduce, measure two numbers, bin.  It writes
data/crossover_candidates.csv and caches edge lists so the measurement pass
never re-downloads.

Usage:  python3 crossover_scout.py [--names name1,name2,...]
"""
import argparse
import csv
import os
import sys

import numpy as np

from crossover_measure import fit_gamma

CACHE = os.environ.get("CROSSOVER_CACHE",
                       os.path.join(os.path.dirname(__file__), "_netcache"))
DATA = os.path.join(os.path.dirname(__file__), "data")

# The ~90 networks behind data/community_communities_vs_n.csv -- the natural
# set, per the chapter's own specification, and already vetted as real systems
# rather than toys.
NAMES_CSV = os.path.join(DATA, "community_communities_vs_n.csv")

# Enumeration budget.  kappa^8/16 is the order of the eight-cycle count; past
# ~2e9 the C enumerator cannot finish whatever we do.  (Networks between the
# depth cap of crossover_measure.py and this bound are still usable, just to
# shallower L.)
MAX_C8 = 2e9

# Beyond the census set, networks chosen from the Netzschleuder catalogue by its
# published degree statistics -- n and the degree standard deviation give kappa
# with no download -- to fill the range of L* that the census does not cover.
# Of 163 catalogue families with n >= 200, only 20 have any subnetwork with L*
# in [3, 8] at all; these are those, minus the ones already in the census and
# minus us_patents at 3.8M nodes.
EXTRA = ["celegans_interactomes/WI8", "tree-of-life/1148", "faa_routes",
         "add_health/comm50", "mist/genetic_mouse", "arxiv_collab/cond-mat-2003",
         "gnutella/31", "chicago_road", "urban_streets/richmond",
         "fullerene_structures/C240", "amazon_copurchases/302",
         "soc_net_comms/amazon", "dblp_coauthor_snap"]


def load_names():
    with open(NAMES_CSV) as fh:
        names = [row["network"] for row in csv.DictReader(fh)]
    return names + [e for e in EXTRA if e not in names]


def edgelist_path(name):
    return os.path.join(CACHE, name.replace("/", "__") + ".txt")


def fetch(name):
    """Largest connected component of the simple, undirected version, as a
    0-indexed edge list.  Cached on disk; the cache is the only thing the
    measurement pass reads, so it never needs the network."""
    path = edgelist_path(name)
    if os.path.exists(path):
        with open(path) as fh:
            n, m = (int(x) for x in fh.readline().split())
            E = [tuple(int(x) for x in line.split()) for line in fh]
        return n, E

    from graph_tool.all import collection, extract_largest_component
    from graph_tool import GraphView
    g = collection.ns[name]
    g = GraphView(g, directed=False)
    g = extract_largest_component(g, prune=True)
    g = g.copy()
    from graph_tool.generation import remove_parallel_edges, remove_self_loops
    remove_self_loops(g)
    remove_parallel_edges(g)
    E = sorted({(min(int(u), int(v)), max(int(u), int(v)))
                for u, v in g.get_edges()})
    n = g.num_vertices()
    os.makedirs(CACHE, exist_ok=True)
    with open(path, "w") as fh:
        fh.write(f"{n} {len(E)}\n")
        fh.writelines(f"{u} {v}\n" for u, v in E)
    return n, E


def degrees(n, E):
    k = np.zeros(n)
    for u, v in E:
        k[u] += 1
        k[v] += 1
    return k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--names")
    args = ap.parse_args()
    names = args.names.split(",") if args.names else load_names()

    rows = []
    for name in names:
        try:
            n, E = fetch(name)
        except Exception as exc:                      # network or format trouble
            print(f"  skip {name}: {type(exc).__name__}: {exc}", file=sys.stderr)
            continue
        if not E:
            continue
        k = degrees(n, E)
        lam = (k ** 2).sum() / k.sum()
        kap = (k * (k - 1)).sum() / k.sum()
        # gamma costs nothing here and the lambda-gamma scaling relation is
        # testable on far more networks than the cycle counting is, since it
        # needs only degrees
        gam = fit_gamma(k)
        lstar = np.log(n) / np.log(kap) if kap > 1 else np.inf
        visible = 3.0 <= lstar <= 8.0
        tractable = kap ** 8 / 16 <= MAX_C8
        rows.append(dict(network=name, n=n, m=len(E),
                         mean_k=round(k.mean(), 4), max_k=int(k.max()),
                         lam=round(float(lam), 4), kap=round(float(kap), 4),
                         lstar=round(float(lstar), 4),
                         visible=int(visible), tractable=int(tractable),
                         est_c8=f"{kap ** 8 / 16:.3e}", **gam))
        print(f"  {name:<45} n={n:<8} m={len(E):<9} kap={kap:<9.2f} "
              f"L*={lstar:<6.2f} {'vis' if visible else '   '} "
              f"{'tract' if tractable else ''}")

    os.makedirs(DATA, exist_ok=True)
    out = os.path.join(DATA, "crossover_candidates.csv")
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    usable = [r for r in rows if r["visible"] and r["tractable"]]
    print(f"\n{len(rows)} networks measured, {len(usable)} both visible and "
          f"tractable -> {out}")
    if usable:
        ls = [r["lstar"] for r in usable]
        print(f"L* range among usable: {min(ls):.2f} .. {max(ls):.2f}")


if __name__ == "__main__":
    main()
