#!/usr/bin/env python3
"""Shortest-path (geodesic) multiplicity on the (b,s) diamond hierarchical
lattice.  Reproduces the two tables of multiplicity.tex from scratch:

  - Table I  : pole-to-pole distance D_t = s^t and multiplicity
               P_t = b^{(s^t-1)/(s-1)}  (= 2^{2^t-1} for the diamond),
               plus a general-(b,s) check of the closed form.
  - Table II : total geodesic count S_t = sum_{u<v} sigma(u,v), the average
               multiplicity avg = S_t / C(n,2), and the asymptotic ratios
               kappa = avg * 4^t / 2^{2^t}  and  avg * n_t / P_t.

Geodesic counts are exact: BFS with arbitrary-precision integer accumulation
of sigma (number of shortest paths), so the doubly-exponential values are
computed without overflow or rounding.
"""

from collections import deque
from math import comb


def diamond(t, b=2, s=2):
    """Build G_t by edge replacement: each edge -> b parallel paths of s edges.
    Poles are vertices 0 and 1.  Returns an adjacency dict of sets."""
    adj = {0: {1}, 1: {0}}
    edges = [(0, 1)]
    nxt = 2
    for _ in range(t):
        newadj, new_edges = {}, []

        def add(u, v):
            newadj.setdefault(u, set()).add(v)
            newadj.setdefault(v, set()).add(u)

        for (u, v) in edges:
            for _p in range(b):                 # b parallel paths
                prev = u
                for _k in range(s - 1):         # s-1 interior vertices
                    x = nxt
                    nxt += 1
                    add(prev, x)
                    new_edges.append((prev, x))
                    prev = x
                add(prev, v)
                new_edges.append((prev, v))
        edges, adj = new_edges, newadj
    return adj


def bfs_sigma(adj, src):
    """BFS from src; returns (dist, sigma) dicts, sigma = #shortest paths."""
    dist = {src: 0}
    sigma = {src: 1}
    q = deque([src])
    while q:
        v = q.popleft()
        for w in adj[v]:
            if w not in dist:
                dist[w] = dist[v] + 1
                sigma[w] = sigma[v]
                q.append(w)
            elif dist[w] == dist[v] + 1:
                sigma[w] += sigma[v]
    return dist, sigma


def pole_stats(t, b=2, s=2):
    adj = diamond(t, b, s)
    dist, sigma = bfs_sigma(adj, 0)
    return dist[1], sigma[1]                     # D_t, P_t


def total_and_avg(t, b=2, s=2):
    adj = diamond(t, b, s)
    nodes = sorted(adj)
    n = len(nodes)
    S = 0
    for src in nodes:
        _, sig = bfs_sigma(adj, src)
        for v in nodes:
            if v > src:                          # unordered pairs
                S += sig[v]
    return n, S, S / comb(n, 2)


def main():
    print("=" * 72)
    print("Table I: pole distance D_t = s^t and multiplicity "
          "P_t = b^((s^t-1)/(s-1))")
    print("=" * 72)
    print(f"{'t':>2} {'D_t=2^t':>9} {'P_t=2^(2^t-1)':>26}")
    for t in range(1, 8):
        D, P = pole_stats(t)
        assert D == 2 ** t and P == 2 ** (2 ** t - 1)
        print(f"{t:>2} {D:>9} {P:>26}")

    print("\nGeneral-(b,s) check of P_t = b^((s^t-1)/(s-1)):")
    for (b, s) in [(2, 2), (3, 2), (2, 3), (4, 2), (3, 3)]:
        ok = []
        for t in range(1, 4):
            D, P = pole_stats(t, b, s)
            pred = b ** ((s ** t - 1) // (s - 1))
            ok.append(D == s ** t and P == pred)
        print(f"  (b,s)=({b},{s}): D_t=s^t and P_t=b^((s^t-1)/(s-1)) for "
              f"t=1..3 -> {'OK' if all(ok) else 'FAIL'}")

    print("\n" + "=" * 72)
    print("Table II: total S_t and average multiplicity avg = S_t / C(n,2)")
    print("=" * 72)
    print(f"{'t':>2} {'n_t':>6} {'S_t':>26} {'avg':>16} "
          f"{'kappa':>8} {'avg*n/P':>9}")
    for t in range(1, 7):
        n, S, avg = total_and_avg(t)
        _, P = pole_stats(t)
        kappa = avg * (4 ** t) / (2 ** (2 ** t))
        print(f"{t:>2} {n:>6} {S:>26} {avg:>16.6g} "
              f"{kappa:>8.4f} {avg * n / P:>9.4f}")
    print("\nkappa -> 1.576 and avg*n/P -> 2.10, i.e. "
          "avg ~ 1.576 * 2^(2^t)/4^t ~ 2.1 * P_t/n_t.")


if __name__ == "__main__":
    main()
