#!/usr/bin/env python3
"""Two descriptions of one network, and the region where both survive.

Section 8.5 found that BB(n,2) has two nearly degenerate descriptions -- plant
many squares and report one community, or plant fewer and report forty -- and
that which one comes back is settled by where the greedy search starts. That is
metastability described as a caveat. This script measures it as one.

The rule is BB(n, L) with L drawn per step: L=1 with probability p1, L=2
otherwise. The two endpoints are in different limit classes. At p1=1 the
triangles overlap, the alphabet cannot take up the network, and the community
density survives (beta = 0.892 with the alphabet against 0.886 without). At p1=0
the squares are nearly disjoint, the alphabet takes them up, and P_kappa collapses
with size -- 0.65 at n=1600, 0.20 at 3200, 0.00 at 6400. So somewhere in between
the description that minimizes Sigma changes, and the question is whether that
happens at an interior p1 and how it happens.

Rather than average kappa over realisations, which mixes the two outcomes into a
number that is neither, each network is fitted twice under the full alphabet:

    from disorder   every node in one block, so motifs are promoted first and
                    the partition has to be discovered against them
    from order      the nested block model's own partition, so the communities
                    are there first and the motifs have to pay to displace them

Where the two agree there is one description. Where they disagree both are
locally stable and the network is in the coexistence region -- the width of that
region in p1 is the hysteresis loop, and its edges are the spinodals. This is
sharper than <kappa> and costs twice a fit rather than a hundred times.

Two order parameters are recorded for each fit: b = kappa/n, which is zero in the
motif phase and positive in the community phase, and phi, the fraction of links
absorbed into planted motifs, which is the average of a per-link binary variable
and so is the quantity that behaves like a magnetisation.

Writes data/bb_mix_hysteresis.csv. Resumable: rows already present are skipped.

Usage:
    python3 bb_mix_hysteresis.py --n 1600 --reps 10 --jobs 7
    python3 bb_mix_hysteresis.py --calibrate        # time one fit and stop
"""
import argparse
import os
import random
import time
from multiprocessing import Pool

import pandas as pd

from motif_sbm import MotifSBM

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data", "bb_mix_hysteresis.csv")

P_GRID = (0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0)


def bb_mixed(n, p1, seed):
    """BB(n, L) with L=1 at probability p1 and L=2 otherwise, per step.

    Same construction as arity_models.bb -- a chain of L new nodes laid across a
    randomly chosen existing link, closing a cycle of length L+2, with the
    original link kept -- but L is redrawn at every step. The seed ring is the
    square, so that p1=0 reproduces BB(n,2) exactly; at p1=1 the ring is one node
    larger than arity_models.bb would use, which is irrelevant at these sizes.

    Note the two rules have different densities: L=1 adds one node and two links,
    L=2 adds two nodes and three links, so the mean degree runs from 3 at p1=0 to
    4 at p1=1 as 2(3-p1)/(2-p1). Any transition found here is therefore confounded
    with a density change, and cannot be attributed to the motif until a control
    that matches the mean degree without changing the rule has been run.
    """
    rng = random.Random(seed)
    adj = {v: {(v - 1) % 4, (v + 1) % 4} for v in range(4)}
    edges = [(v, (v + 1) % 4) for v in range(4)]
    n1 = 0
    while len(adj) < n:
        L = 1 if rng.random() < p1 else 2
        n1 += (L == 1)
        u, v = edges[rng.randrange(len(edges))]
        path = [u]
        for _ in range(L):
            w = len(adj)
            adj[w] = set()
            path.append(w)
        path.append(v)
        for a, b in zip(path, path[1:]):
            adj[a].add(b)
            adj[b].add(a)
            edges.append((a, b))
    return adj, n1


def to_edges(adj):
    idx = {v: i for i, v in enumerate(sorted(adj))}
    return sorted({(min(idx[u], idx[w]), max(idx[u], idx[w]))
                   for u in adj for w in adj[u] if u != w})


def one(job):
    """Both starts on the same network."""
    p1, n, seed = job
    adj, n1 = bb_mixed(n, p1, seed)
    nn = len(adj)
    edges = to_edges(adj)
    m = len(edges)

    base = MotifSBM(nn, edges)
    try:
        order_blocks = base.dcsbm_blocks(seed=seed)
    except ImportError:
        order_blocks = None
    k_nested = len(set(order_blocks)) if order_blocks else float("nan")

    row = dict(p1=p1, n=n, n_nodes=nn, m=m, seed=seed,
               mean_k=round(2 * m / nn, 3), kappa_nested=k_nested)

    for tag, init in (("dis", None), ("ord", order_blocks)):
        t = time.time()
        f = MotifSBM(nn, edges)
        f.fit(init_blocks=init)
        c = f.motif_counts()
        kap = len(set(f.blocks))
        named = 3 * c.get("triangle", 0) + 4 * c.get("square", 0)
        row.update({
            f"kappa_{tag}": kap,
            f"sigma_{tag}": round(f.description_length(), 2),
            f"b_{tag}": round(kap / nn, 6),
            f"phi_{tag}": round(named / m, 4),
            f"tri_{tag}": c.get("triangle", 0),
            f"sq_{tag}": c.get("square", 0),
            f"secs_{tag}": round(time.time() - t, 1),
        })
    # the description the criterion actually prefers, and by how much
    row["d_sigma"] = round(row["sigma_dis"] - row["sigma_ord"], 2)
    row["prefers"] = "ord" if row["d_sigma"] > 0 else "dis"
    row["split"] = int((row["kappa_dis"] > 1) != (row["kappa_ord"] > 1))
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1600)
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--jobs", type=int, default=7)
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--pgrid", default=None,
                    help="comma-separated p1 values; default is the wide scan. "
                         "Once the crossing is bracketed the larger sizes are "
                         "run on a narrow grid around it instead.")
    a = ap.parse_args()
    grid = (tuple(float(x) for x in a.pgrid.split(",")) if a.pgrid else P_GRID)

    if a.calibrate:
        t = time.time()
        r = one((0.0, a.n, 12345))
        print(f"one network, both starts, n={a.n}: {time.time()-t:.0f}s")
        print(f"  kappa {r['kappa_dis']} (dis) / {r['kappa_ord']} (ord),"
              f"  dSigma {r['d_sigma']:+.1f},  phi {r['phi_dis']}/{r['phi_ord']}")
        return

    done = pd.read_csv(OUT) if os.path.exists(OUT) else pd.DataFrame()
    have = set() if done.empty else set(zip(done.p1, done.n, done.seed))
    jobs = [(p, a.n, 900000 + 1000 * int(round(p * 100)) + r)
            for p in grid for r in range(a.reps)]
    jobs = [j for j in jobs if (j[0], j[1], j[2]) not in have]
    print(f"{len(jobs)} fits to run at n={a.n} on {a.jobs} workers", flush=True)

    rows = [] if done.empty else done.to_dict("records")
    with Pool(a.jobs) as pool:
        for r in pool.imap_unordered(one, jobs):
            rows.append(r)
            print(f"  p1={r['p1']:.2f} seed={r['seed']}  "
                  f"kappa {r['kappa_dis']:4d}/{r['kappa_ord']:4d}  "
                  f"phi {r['phi_dis']:.3f}/{r['phi_ord']:.3f}  "
                  f"dSigma {r['d_sigma']:+9.1f}  "
                  f"{'SPLIT' if r['split'] else '     '}  "
                  f"({r['secs_dis']+r['secs_ord']:.0f}s)", flush=True)
            pd.DataFrame(rows).to_csv(OUT, index=False)

    d = pd.DataFrame(rows)
    d = d[d.n == a.n]
    print("\n  p1    <k>   kappa dis   kappa ord   phi dis  phi ord   "
          "split   prefers ord")
    for p, g in d.groupby("p1"):
        print(f"  {p:.2f}  {g.mean_k.mean():.2f}  "
              f"{g.kappa_dis.mean():9.1f}   {g.kappa_ord.mean():9.1f}   "
              f"{g.phi_dis.mean():7.3f}  {g.phi_ord.mean():7.3f}   "
              f"{g.split.mean():5.2f}   {(g.prefers=='ord').mean():8.2f}")


if __name__ == "__main__":
    main()
