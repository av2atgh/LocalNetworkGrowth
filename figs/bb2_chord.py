#!/usr/bin/env python3
"""Is it the parity, or is it the tiling? Three bubble rules that separate them.

Across BB(n,L) for L = 1..5 the outcome alternates: naming the rule's own cycle
dissolves the communities of L = 2 and L = 4 and leaves L = 1, 3 and 5 alone.
Coverage sits at 53-58% for all five and separates nothing; the number of
communities to lose separates nothing either, since L = 4 and L = 5 have almost
the same and go opposite ways. What tracks the outcome is parity: the two rules
that dissolve are the two whose cycles are all even, and so the two whose
networks are bipartite.

Two accounts survive that, and they disagree about why.

    parity   a square and a hexagon are what a bipartite network is made of, so
             on those rules the alphabet and the partition are describing the
             same structure and one can take the other's place. An odd cycle
             cannot express bipartiteness, so there is nothing to take over.

    tiling   what matters is only that the rule lays down a fixed footprint that
             an edge-disjoint decomposition can take up whole, one fresh copy per
             step. The even rules do; the odd ones happen not to.

They are separated by adding a diagonal to BB(n,2). The chain is still two nodes,
the reach is unchanged, and one fresh footprint is still laid down per step -- but
either diagonal of the square joins two nodes on the same side of the bipartition,
so the network stops being bipartite. Parity predicts the communities survive;
tiling predicts they dissolve, since K_4 minus an edge tiles exactly as the
squares did.

Three rules, because the chord also adds a link and densification is the
alternative explanation for anything that moves:

    bb2       BB(n,2)                          <k> = 3, bipartite
    bb2par    plus a link to a random node on the opposite side of the
              bipartition                      <k> = 4, bipartite
    bb2chord  plus the diagonal                 <k> = 4, NOT bipartite

If the first two dissolve and the third does not, parity is established with
density held fixed. If all three dissolve, parity is dead and it is tiling or
density.

Each is fitted twice on the same networks, with and without the K_4-minus-an-edge
symbol, since without it the chorded rule would be described as a triangle and
two loose links -- the pieces of its footprint rather than the footprint.

Writes data/bb2_chord.csv.

Usage:
    python3 bb2_chord.py --sizes 100,200,400,800,1600,3200 --reps 20 --jobs 7
    python3 bb2_chord.py --calibrate --n 1600
"""
import argparse
import os
import random
import time
from multiprocessing import Pool

import pandas as pd

from motif_sbm import MotifSBM

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data", "bb2_chord.csv")

CONTROL = ("edge", "triangle", "square")
TEST = ("edge", "triangle", "square", "chorded_square")
RULES = ("bb2", "bb2par", "bb2chord")


def grow(n, rule, seed):
    """BB(n,2), optionally with one extra link per step.

    The colour array is the bipartition BB(n,2) has by construction: the chain
    u-a-b-v alternates, so a takes the colour opposite u and b takes u's own.
    bb2par joins a to a random node of the opposite colour, which adds a link and
    keeps the network bipartite. bb2chord joins a to v, which are the same
    colour -- that is what a diagonal of the square is -- and so closes two
    triangles and destroys the bipartition.
    """
    rng = random.Random(seed)
    adj = {v: {(v - 1) % 4, (v + 1) % 4} for v in range(4)}
    col = {0: 0, 1: 1, 2: 0, 3: 1}
    edges = [(v, (v + 1) % 4) for v in range(4)]
    by_col = {0: [0, 2], 1: [1, 3]}

    def link(x, y):
        adj[x].add(y)
        adj[y].add(x)
        edges.append((x, y))

    while len(adj) < n:
        u, v = edges[rng.randrange(len(edges))]
        a, b = len(adj), len(adj) + 1
        adj[a], adj[b] = set(), set()
        col[a], col[b] = 1 - col[u], col[u]
        by_col[col[a]].append(a)
        by_col[col[b]].append(b)
        link(u, a)
        link(a, b)
        link(b, v)
        if rule == "bb2par":
            # A link that keeps the two sides apart, so that only the density
            # moves. It has to stay local as well: joining a to a random node of
            # the opposite colour adds a long-range link at every step, and a
            # rule with long-range links has no communities to begin with --
            # tried first, and it drove the block model to two groups at
            # n = 1600, which would have made the control vacuous. So the
            # partner is drawn from v's own neighbours, which are the opposite
            # colour to a and three steps away, the reach BB(n,2) already has.
            opts = [w for w in adj[v] if w != b and w not in adj[a]]
            if opts:
                link(a, rng.choice(opts))
        elif rule == "bb2chord":
            # the diagonal: a and v are the same colour, so this is the odd link
            if v not in adj[a]:
                link(a, v)
    return adj


def to_edges(adj):
    idx = {v: i for i, v in enumerate(sorted(adj))}
    return sorted({(min(idx[u], idx[w]), max(idx[u], idx[w]))
                   for u in adj for w in adj[u] if u != w})


def is_bipartite(adj):
    from collections import deque
    col = {}
    for s in adj:
        if s in col:
            continue
        col[s] = 0
        q = deque([s])
        while q:
            u = q.popleft()
            for w in adj[u]:
                if w not in col:
                    col[w] = 1 - col[u]
                    q.append(w)
                elif col[w] == col[u]:
                    return False
    return True


def one(job):
    rule, n, seed = job
    adj = grow(n, rule, seed)
    nn = len(adj)
    edges = to_edges(adj)
    m = len(edges)

    base = MotifSBM(nn, edges)
    try:
        k_dc = len(set(base.dcsbm_blocks(seed=seed)))
        k_flat = len(set(base.dcsbm_blocks(nested=False, seed=seed)))
    except ImportError:
        k_dc = k_flat = float("nan")

    row = dict(rule=rule, n=n, n_nodes=nn, seed=seed, m=m,
               mean_k=round(2 * m / nn, 3), bipartite=int(is_bipartite(adj)),
               kappa_dcsbm=k_dc, kappa_flat=k_flat)

    for tag, alpha in (("ctl", CONTROL), ("kite", TEST)):
        t = time.time()
        f = MotifSBM(nn, edges, alphabet=alpha)
        f.fit()
        c = f.motif_counts()
        named = (3 * c.get("triangle", 0) + 4 * c.get("square", 0)
                 + 5 * c.get("chorded_square", 0))
        row.update({
            f"kappa_{tag}": len(set(f.blocks)),
            f"sigma_{tag}": round(f.description_length(), 2),
            f"tri_{tag}": c.get("triangle", 0),
            f"sq_{tag}": c.get("square", 0),
            f"cs_{tag}": c.get("chorded_square", 0),
            f"named_{tag}": round(named / m, 4),
            f"secs_{tag}": round(time.time() - t, 1),
        })
    row["d_sigma"] = round(row["sigma_kite"] - row["sigma_ctl"], 2)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="100,200,400,800,1600,3200")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--jobs", type=int, default=7)
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--n", type=int, default=1600)
    a = ap.parse_args()

    if a.calibrate:
        for rule in RULES:
            t = time.time()
            r = one((rule, a.n, 12345))
            print(f"  {rule:9s} <k>={r['mean_k']:.2f} bip={r['bipartite']}  "
                  f"kappa nested {r['kappa_dcsbm']:3d}  ctl {r['kappa_ctl']:3d} "
                  f"+kite {r['kappa_kite']:3d}   {r['cs_kite']:4d} kites, "
                  f"{r['named_kite']:.0%} named  dSigma {r['d_sigma']:+7.0f}  "
                  f"({time.time()-t:.0f}s)", flush=True)
        return

    sizes = [int(x) for x in a.sizes.split(",")]
    done = pd.read_csv(OUT) if os.path.exists(OUT) else pd.DataFrame()
    have = set() if done.empty else set(zip(done.rule, done.n, done.seed))
    jobs = [(u, n, 1000 * n + r) for u in RULES for n in sizes
            for r in range(a.reps) if (u, n, 1000 * n + r) not in have]
    print(f"{len(jobs)} fits on {a.jobs} workers", flush=True)

    rows = [] if done.empty else done.to_dict("records")
    with Pool(a.jobs) as pool:
        for r in pool.imap_unordered(one, jobs):
            rows.append(r)
            print(f"  {r['rule']:9s} n={r['n']:<5} nested {r['kappa_dcsbm']:3d} "
                  f"ctl {r['kappa_ctl']:3d} +kite {r['kappa_kite']:3d}  "
                  f"{r['cs_kite']:4d} kites {r['named_kite']:.0%}  "
                  f"dS {r['d_sigma']:+8.0f}", flush=True)
            pd.DataFrame(rows).to_csv(OUT, index=False)

    d = pd.DataFrame(rows)
    for u in RULES:
        x = d[d.rule == u]
        if x.empty:
            continue
        print(f"\n{u}   <k>={x.mean_k.mean():.2f}  bipartite={x.bipartite.min()}")
        print(f"{'n':>6} {'nested':>7} {'ctl':>6} {'+kite':>6} {'P_nest':>7} "
              f"{'P_kite':>7} {'named':>6}")
        for n, g in x.groupby("n"):
            print(f"{n:6d} {g.kappa_dcsbm.mean():7.2f} {g.kappa_ctl.mean():6.2f} "
                  f"{g.kappa_kite.mean():6.2f} {(g.kappa_dcsbm>1).mean():7.2f} "
                  f"{(g.kappa_kite>1).mean():7.2f} {g.named_kite.mean():6.0%}")


if __name__ == "__main__":
    main()
