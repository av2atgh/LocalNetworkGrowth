#!/usr/bin/env python3
"""Does the motif fit's answer survive being started from the right place?

Section 8.6 reports that the pseudofractal web keeps its communities under the
motif alphabet while the diamond lattice loses its, and Section 8.7 explains the
difference by tiling: the diamond's squares cover ninety-two per cent of its
links, the pseudofractal's triangles only fifty-six.

The second half of that is not a property of the network. Both lattices give
every existing link exactly one new motif per generation, so the motifs closed at
the LAST generation have distinct old links and disjoint new chains and cover
every link exactly once: a perfect edge-disjoint tiling exists by construction,
at every size, for both. Fifty-six per cent is what the greedy search finds, not
what is there.

So the fit is run three ways on the same network:

  fit          the ordinary fit of Chapter 8, from one block and from the nested
               partition, best of both
  tiling       the known tiling, priced as it stands with a single block
  seeded       the same tiling handed to the optimiser as a starting state, and
               allowed to move

If seeded beats fit, then Chapter 8's number for that lattice is a statement
about the search rather than about the network, and the contrast the section
draws between the two lattices has to be re-examined.

Usage:  python3 seeded_tiling.py [--tmax 8]
"""
import argparse
import time

from motif_sbm import MotifSBM


def dbb_with_last(L, t):
    """DBB(n,L) and the motifs closed at its last generation, in cyclic order."""
    adj = {0: {1}, 1: {0}}
    edges = [(0, 1)]
    for _ in range(t):
        added, last = [], []
        for (u, v) in edges:
            prev, chain = u, []
            for _ in range(L):
                w = len(adj)
                adj[w] = set()
                chain.append(w)
                adj[prev].add(w)
                adj[w].add(prev)
                added.append((prev, w))
                prev = w
            adj[prev].add(v)
            adj[v].add(prev)
            added.append((prev, v))
            last.append((u,) + tuple(chain) + (v,))
        edges = edges + added
    E = sorted({(min(u, w), max(u, w)) for u in adj for w in adj[u] if u != w})
    return len(adj), E, last


def diamond_with_last(t):
    """The diamond lattice: each link becomes two parallel paths, original gone.

    The 4-cycle closed at the last generation is u-a-v-b, so in cyclic order the
    square is (u, a, v, b).
    """
    adj = {0: {1}, 1: {0}}
    edges = [(0, 1)]
    for _ in range(t):
        nxt, last = [], []
        for (u, v) in edges:
            adj[u].discard(v)
            adj[v].discard(u)
            mids = []
            for _ in range(2):
                w = len(adj)
                adj[w] = {u, v}
                adj[u].add(w)
                adj[v].add(w)
                mids.append(w)
                nxt += [(u, w), (w, v)]
            last.append((u, mids[0], v, mids[1]))
        edges = nxt
    E = sorted({(min(u, w), max(u, w)) for u in adj for w in adj[u] if u != w})
    return len(adj), E, last


CASES = (
    ("pseudofractal", "triangle", lambda t: dbb_with_last(1, t), (6, 7, 8)),
    ("DBB(n,2)", "square", lambda t: dbb_with_last(2, t), (5, 6)),
    ("diamond", "square", diamond_with_last, (5, 6)),
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tmax", type=int, default=8)
    a = ap.parse_args()

    print(f"{'lattice':14s} {'t':>2} {'n':>6} {'m':>6} | "
          f"{'fit Sigma':>10} {'k':>4} {'named':>6} | "
          f"{'tiling':>10} | {'seeded':>10} {'k':>4} {'named':>6} | winner")
    for name, motif, build, ts in CASES:
        for t in ts:
            if t > a.tmax:
                continue
            n, E, last = build(t)
            k = len(last[0])
            s = time.time()

            f = MotifSBM(n, E)
            f.fit()
            s_fit, k_fit = f.description_length(), len(set(f.blocks))
            c_fit = k * f.motif_counts().get(motif, 0) / len(E)

            raw = MotifSBM(n, E)
            raw.blocks = [0] * n
            raw.instances = [(motif, q) for q in last]
            s_raw = raw.description_length()

            sd = MotifSBM(n, E)
            s_seed = sd.fit_from_state([0] * n, [(motif, q) for q in last])
            k_seed = len(set(sd.blocks))
            c_seed = k * sd.motif_counts().get(motif, 0) / len(E)

            win = "seeded" if s_seed < s_fit - 1e-6 else "fit"
            print(f"{name:14s} {t:2d} {n:6d} {len(E):6d} | "
                  f"{s_fit:10.1f} {k_fit:4d} {c_fit:6.0%} | {s_raw:10.1f} | "
                  f"{s_seed:10.1f} {k_seed:4d} {c_seed:6.0%} | {win} "
                  f"({s_fit - s_seed:+.0f} bits, {time.time()-s:.0f}s)",
                  flush=True)


if __name__ == "__main__":
    main()
