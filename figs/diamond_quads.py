#!/usr/bin/env python3
"""How much of the diamond lattice's 4-point structure survives a two-variable
(block-model) description?

Companion to pseudofractal_triples.py. The pseudofractal is built from triangles,
so its natural motif is the triangle; the diamond lattice (b=2, s=2) is bipartite
and holds no triangle at all, so its natural motif is the four-cycle -- the arity
of the growth rule showing up in the shape of what a two-variable model misses.

Structure after t generations (every link replaced by 2 parallel paths of 2):
  - m = 4^t links and n = (2*4^t+4)/3 nodes, mean degree -> 3;
  - 2 poles of degree 2^t; 2*4^(j-1) nodes born at step j with degree 2^(t-j+1),
    which is a power law with gamma = 3 -- far milder than the pseudofractal's
    2.585, and that difference is the whole story below;
  - exactly 4^(t-1) four-cycles: only the diamonds closed at the final generation
    are still four-cycles, since earlier ones have had their links subdivided;
  - chapter 6's cut: the two top-level parallel branches, poles assigned to the
    first, giving a seam of e_01 = 2^t = sqrt(m).

Expected four-cycle counts follow from tr(P^4)/8 for the expected adjacency
matrix P, which for block-structured P collapses to a sum over block quadruples.
Corrections for repeated indices are O(1/n) here and are dropped.

Usage:  python3 diamond_quads.py
"""
from fractions import Fraction as F
import itertools
import math


def blocks(t):
    """Block sizes, link counts, degree sums S_r, squared-degree sums Q_r."""
    n0 = 3 + (4**t - 4) // 3
    n1 = 1 + (4**t - 4) // 3
    e00 = 4**t // 2
    e11 = 4**t // 2 - 2**t
    e01 = 2**t
    S = [F(4**t + 2**t), F(4**t - 2**t)]
    Q = [F((t + 2) * 4**t), F(t * 4**t)]
    return (n0, n1), (e00, e11, e01), S, Q


def quad_counts(t):
    """(n, true, plain-SBM, degree-corrected) four-cycle counts."""
    (n0, n1), (e00, e11, e01), S, Q = blocks(t)
    n = n0 + n1
    true = 4 ** (t - 1)

    p = [[F(2 * e00, n0 * (n0 - 1)), F(e01, n0 * n1)],
         [F(e01, n0 * n1), F(2 * e11, n1 * (n1 - 1))]]
    N = [F(n0), F(n1)]
    plain = sum(N[r] * N[s] * N[u] * N[v] *
                p[r][s] * p[s][u] * p[u][v] * p[v][r]
                for r, s, u, v in itertools.product((0, 1), repeat=4)) / 8

    w = [[F(2 * e00, S[0] * S[0]), F(e01, S[0] * S[1])],
         [F(e01, S[0] * S[1]), F(2 * e11, S[1] * S[1])]]
    dc = sum(Q[r] * Q[s] * Q[u] * Q[v] *
             w[r][s] * w[s][u] * w[u][v] * w[v][r]
             for r, s, u, v in itertools.product((0, 1), repeat=4)) / 8
    return n, true, float(plain), float(dc)


if __name__ == "__main__":
    hdr = (f"{'t':>3} {'n':>14} {'true':>14} {'plain':>8} {'true/plain':>12} "
           f"{'dc':>10} {'true/dc':>12} {'dc/(t+1)^4':>11}")
    print(hdr)
    print("-" * len(hdr))
    for t in [2, 3, 4, 5, 6, 8, 10, 14, 18, 22, 26]:
        n, true, plain, dc = quad_counts(t)
        print(f"{t:>3} {n:>14} {true:>14} {plain:>8.2f} {true/plain:>12.1f} "
              f"{dc:>10.4g} {true/dc:>12.4g} {dc/(t+1)**4:>11.4f}")

    print()
    print("  Q/S = t+1 exactly, so the degree-corrected count grows as (t+1)^4:")
    print("  polylogarithmic in n, against a truth growing as n/6. The plain")
    print("  model saturates at a constant, as it did for the pseudofractal.")
