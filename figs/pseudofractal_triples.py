#!/usr/bin/env python3
"""How much of the pseudofractal web's 3-point structure survives a two-variable
(block-model) description?

Backs the "two questions" section of chapter 7. Everything is closed form and
exact in integer arithmetic; nothing is simulated.

Structure of the DGM pseudofractal web after t generations:
  - 3 seed nodes of degree 2^(t+1); 3^s nodes born at step s with degree
    2^(t-s+1); n = (3^(t+1)+3)/2 nodes and 3^(t+1) links.
  - every added node closes exactly one triangle, so the true triangle count is
    exactly n-2.
  - chapter 6's branch cut: block 1 = one branch plus the three seed hubs,
    block 2 = the other two branches, with the link counts of Eq. (com-pfcounts).

For a model whose expected adjacency matrix is P, the expected triangle count is
tr(P^3)/6, which for block-structured P collapses to a sum over block triples.

Usage:  python3 pseudofractal_triples.py
"""
from fractions import Fraction as F
import math


def blocks(t):
    """Block sizes, link counts, degree sums S_r and squared-degree sums Q_r."""
    n1 = (3**t + 5) // 2
    n2 = 3**t - 1
    E1, E2, E12 = 3**t + 2, 2 * 3**t + 2 - 2 ** (t + 2), 2 ** (t + 2) - 4

    S = [F(3 * 2 ** (t + 1)), F(0)]
    Q = [F(3 * (2 ** (t + 1)) ** 2), F(0)]
    for s in range(1, t + 1):
        d = 2 ** (t - s + 1)
        S[0] += F(3 ** (s - 1) * d)
        Q[0] += F(3 ** (s - 1) * d * d)
        S[1] += F(2 * 3 ** (s - 1) * d)
        Q[1] += F(2 * 3 ** (s - 1) * d * d)
    return (n1, n2), (E1, E2, E12), S, Q


def triangle_counts(t):
    """(n, true, plain-SBM, degree-corrected-SBM) triangle counts."""
    (n1, n2), (E1, E2, E12), S, Q = blocks(t)
    n = n1 + n2
    true = n - 2

    # Plain block model: one constant density per block pair.
    p = [[F(2 * E1, n1 * (n1 - 1)), F(E12, n1 * n2)],
         [F(E12, n1 * n2), F(2 * E2, n2 * (n2 - 1))]]
    N = [n1, n2]
    plain = sum(F(N[r] * N[s] * N[u]) * p[r][s] * p[s][u] * p[u][r]
                for r in (0, 1) for s in (0, 1) for u in (0, 1)) / 6

    # Degree-corrected: P_ij = k_i k_j omega_{g_i g_j}, so tr(P^3) weights each
    # block by a degree sum rather than by its size. The weight is sum k(k-1)
    # and not sum k^2: a triangle leaves a node by a different edge than it
    # entered by, and the fit holds degrees fixed, so each node offers k(k-1)
    # ordered pairs of edges. See branching_verification.py, which settles this
    # against exact counts on synthetic ensembles.
    Qe = [Q[r] - S[r] for r in (0, 1)]
    w = [[F(2 * E1, S[0] * S[0]), F(E12, S[0] * S[1])],
         [F(E12, S[0] * S[1]), F(2 * E2, S[1] * S[1])]]
    dc = sum(Qe[r] * Qe[s] * Qe[u] * w[r][s] * w[s][u] * w[u][r]
             for r in (0, 1) for s in (0, 1) for u in (0, 1)) / 6
    return n, true, float(plain), float(dc)


if __name__ == "__main__":
    hdr = f"{'t':>3} {'n':>12} {'true':>12} {'plain':>10} {'true/plain':>11} {'dc':>11} {'true/dc':>9}"
    print(hdr)
    print("-" * len(hdr))
    for t in [5, 6, 7, 8, 10, 12, 14, 16, 18, 20, 24, 28]:
        n, true, plain, dc = triangle_counts(t)
        print(f"{t:>3} {n:>12} {true:>12} {plain:>10.2f} {true/plain:>11.1f} "
              f"{dc:>11.4g} {true/dc:>9.4f}")

    # Asymptotics. Q ~ 4^t and S ~ 3^t, so the degree-corrected count grows as
    # (Q/S)^3 ~ (4^3/3^3)^t = (64/27)^t against a true count ~ 3^t.
    print()
    print(f"  plain model saturates at a constant, so its deficit grows like n")
    print(f"  degree-corrected grows as (64/27)^t = {64/27:.4f}^t vs true 3^t,")
    print(f"  so its deficit grows as (81/64)^t = n^{math.log(81/64)/math.log(3):.4f}")
