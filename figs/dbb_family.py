#!/usr/bin/env python3
"""DBB(n,L): the bubble rule made deterministic, and the same study run on it.

BB(n,L) lays a chain of L new nodes across *one* randomly chosen link per step.
Do it across *every* link at once and the rule becomes deterministic, one network
per generation, in the way the pseudofractal web and the diamond lattice are
deterministic. Call the family DBB(n,L).

    seed        a single link
    generation  every link of the current network receives a chain of L new
                nodes joining its two ends, closing a cycle of length L+2. The
                original link is kept, so a link laid down at generation t
                receives a fresh chain at every generation after it.

Two members are already in this book under other names. DBB(n,1) hangs one node
on both ends of every link and is exactly the pseudofractal web of
Dorogovtsev and Goltsev. DBB(n,2) is a cousin of the diamond lattice rather than
the lattice itself: both quadruple their link count each generation and agree on
n and m at every size, but the diamond *replaces* each link by two parallel paths
and so leaves its endpoints non-adjacent, while DBB keeps the link and runs the
new chain alongside it.

The counts follow from the rule. A generation multiplies the links by L+2 and
adds L new nodes per existing link, so

    m_t = (L+2)^t,    n_t = 2 + L((L+2)^t - 1)/(L+1),    <k> -> 2(L+1)/L,

which is the same limiting mean degree as the random family -- the two differ in
how the motifs are laid down, not in how dense they end up.

The study is the one Chapter 9 runs on BB(n,L): fit each network with the flat
and nested block models, with the control alphabet of link, triangle and square,
and with the control plus the rule's own L+2 cycle, then ask whether naming the
motif moves the community count. The deterministic family is worth the trouble
because it removes the sampling: there is one network per size, no realisation to
average over, and the odd-even alternation, if it is there, has to show up in a
single object at each size rather than in a mean.

Writes data/dbb_family.csv.

Usage:
    python3 dbb_family.py --sizes            # print the size ladder and stop
    python3 dbb_family.py --L 2 --tmax 6 --restarts 3
"""
import argparse
import os
import time
from collections import deque

import pandas as pd

from motif_sbm import MotifSBM

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data", "dbb_family.csv")

CONTROL = ("edge", "triangle", "square")
CYCLE_OF = {1: "triangle", 2: "square", 3: "pentagon", 4: "hexagon",
            5: "heptagon"}


def dbb(L, t):
    """Every link of the current network gets a chain of L new nodes."""
    adj = {0: {1}, 1: {0}}
    edges = [(0, 1)]
    for _ in range(t):
        added = []
        for (u, v) in edges:
            prev = u
            for _ in range(L):
                w = len(adj)
                adj[w] = set()
                adj[prev].add(w)
                adj[w].add(prev)
                added.append((prev, w))
                prev = w
            adj[prev].add(v)
            adj[v].add(prev)
            added.append((prev, v))
        edges = edges + added
    return adj


def to_edges(adj):
    return sorted({(min(u, w), max(u, w)) for u in adj for w in adj[u] if u != w})


def is_bipartite(adj):
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


def ladder(Lmax=5, tmax=8, cap=4000):
    for L in range(1, Lmax + 1):
        row = []
        for t in range(1, tmax + 1):
            m = (L + 2) ** t
            n = 2 + L * ((L + 2) ** t - 1) // (L + 1)
            if n > cap:
                break
            row.append((t, n, m))
        yield L, row


def one(L, t, restarts):
    adj = dbb(L, t)
    n = len(adj)
    edges = to_edges(adj)
    row = dict(model=f"DBB(n,{L})", L=L, t=t, n=n, m=len(edges),
               bipartite=int(is_bipartite(adj)), cycle=L + 2)

    # Restarts of the block model are scored by description length, not by the
    # number of groups they happen to return. Taking the smallest kappa over
    # restarts would bias the column downward and taking the largest upward;
    # neither is "the best fit". Sigma is the criterion everywhere else in this
    # book, so each candidate partition is priced under the link-only alphabet,
    # where Sigma is exactly the ordinary degree-corrected description length,
    # and the count is read off whichever partition is cheapest.
    scorer = MotifSBM(n, edges, alphabet=("edge",))
    base = MotifSBM(n, edges)

    def best_partition(nested):
        best = None
        for s in range(restarts):
            b = base.dcsbm_blocks(nested=nested, seed=s)
            sig = scorer.description_length(
                blocks=b, instances=[("edge", e) for e in scorer.edges])
            if best is None or sig < best[0]:
                best = (sig, len(set(b)))
        return best[1]

    row["kappa_dcsbm"] = best_partition(True)
    row["kappa_flat"] = best_partition(False)

    cyc = CYCLE_OF[L]
    test = CONTROL if cyc in CONTROL else CONTROL + (cyc,)
    for tag, alpha in (("ctl", CONTROL), ("cyc", test)):
        best = None
        for s in range(restarts):
            f = MotifSBM(n, edges, alphabet=alpha)
            f.fit()
            sig = f.description_length()
            if best is None or sig < best[0]:
                c = f.motif_counts()
                best = (sig, len(set(f.blocks)), c.get("triangle", 0),
                        c.get("square", 0), c.get(cyc, 0))
        row.update({f"sigma_{tag}": round(best[0], 2), f"kappa_{tag}": best[1],
                    f"tri_{tag}": best[2], f"sq_{tag}": best[3],
                    f"cyc_{tag}": best[4]})
    row["named"] = round((L + 2) * row["cyc_cyc"] / row["m"], 4)
    row["d_sigma"] = round(row["sigma_cyc"] - row["sigma_ctl"], 2)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int)
    ap.add_argument("--tmax", type=int, default=8)
    ap.add_argument("--cap", type=int, default=4000)
    ap.add_argument("--restarts", type=int, default=3)
    ap.add_argument("--sizes", action="store_true")
    a = ap.parse_args()

    if a.sizes:
        for L, row in ladder(tmax=a.tmax, cap=a.cap):
            got = "  ".join(f"t={t}:{n}" for t, n, _ in row)
            print(f"  DBB(n,{L})  <k>->{2*(L+1)/L:.2f}  {got}")
        return

    # one file per L, so the five can be run as separate processes without
    # racing each other on a shared csv; make_dbb_table.py globs them back
    out = (OUT if a.L is None else
           os.path.join(HERE, "data", f"dbb_family_L{a.L}.csv"))
    done = pd.read_csv(out) if os.path.exists(out) else pd.DataFrame()
    have = set() if done.empty else set(zip(done.L, done.t))
    rows = [] if done.empty else done.to_dict("records")

    Ls = [a.L] if a.L else [1, 2, 3, 4, 5]
    for L, lad in ladder(tmax=a.tmax, cap=a.cap):
        if L not in Ls:
            continue
        for t, n, _ in lad:
            if (L, t) in have:
                continue
            s = time.time()
            r = one(L, t, a.restarts)
            rows.append(r)
            print(f"  DBB(n,{L}) t={t} n={r['n']:<6} bip={r['bipartite']}  "
                  f"flat {r['kappa_flat']:4d} nested {r['kappa_dcsbm']:4d}  "
                  f"ctl {r['kappa_ctl']:4d} +cyc {r['kappa_cyc']:4d}   "
                  f"{r['cyc_cyc']:5d} {CYCLE_OF[L]}s ({r['named']:.0%})  "
                  f"dS {r['d_sigma']:+9.0f}  ({time.time()-s:.0f}s)", flush=True)
            pd.DataFrame(rows).to_csv(out, index=False)


if __name__ == "__main__":
    main()
