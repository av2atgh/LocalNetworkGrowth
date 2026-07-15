#!/usr/bin/env python3
r"""
diamond_rg.py  --  exact renormalization of the Ramsey community number
on the diamond hierarchical lattice (b=2, s=2 == the (2,2)-flower / Migdal-Kadanoff cell).

Importable: exposes diamond(), closed(), the sufficient-statistic RG map M,
and the closed-form / direct plain & degree-corrected (DC) evidence.
Run as a script to reproduce every number, table, and slope in the manuscript.

Requires: networkx, numpy, mpmath.
"""
import math
import numpy as np
import networkx as nx
from mpmath import mp, mpf, loggamma, exp, log as mlog

mp.dps = 80  # 80-digit arithmetic


# ----------------------------------------------------------- construction
def diamond(t, b=2, s=2):
    """(b,s) diamond hierarchical lattice after t generations by edge-replacement.
    Each edge -> b parallel paths of s edges.  Returns G, pole set, and the
    self-similar bundle partition (block 0 = bundle 0 + poles; block j = bundle j)."""
    G = nx.Graph(); A, B = 0, 1; G.add_edge(A, B); poles = {A, B}
    branch = {}; tag = {(A, B): None, (B, A): None}; nxt = 2
    for gen in range(1, t + 1):
        new = {}
        for (u, v) in list(G.edges()):
            tg = tag[(u, v)]; G.remove_edge(u, v)
            for k in range(b):                      # b parallel paths
                prev = u
                for step in range(s):               # path of s edges (s-1 internal nodes)
                    nxt_node = v if step == s - 1 else nxt
                    if step != s - 1:
                        nxt += 1
                    bt = k if gen == 1 else tg       # generation 1 defines the bundles
                    if nxt_node not in (A, B):
                        branch[nxt_node] = bt
                    G.add_edge(prev, nxt_node)
                    new[(prev, nxt_node)] = bt; new[(nxt_node, prev)] = bt
                    prev = nxt_node
        tag = new
    part = {n: (0 if n in poles else branch[n]) for n in G}
    return G, poles, part


def closed(t):
    """Closed-form (n1,n2,E11,E22,E12,kappa1,kappa2) for the b=s=2 bundle cut."""
    a = 4 ** t; x = 2 ** t
    return (2 + (a - 1) // 3, (a - 1) // 3,
            a // 2, a // 2 - x, x, a + x, a - x)


# exact linear RG map on v=(E11,E22,E12,kappa1,kappa2):  v(t+1) = M v(t)
M = np.array([[4, 0, 0, 0, 0],
              [0, 4, 2, 0, 0],
              [0, 0, 2, 0, 0],
              [0, 0, -2, 4, 0],
              [0, 0, 2, 0, 4]])


# --------------------------------------------------------- generic evidence
def _stats(G, part):
    blocks = sorted(set(part.values())); idx = {b: i for i, b in enumerate(blocks)}
    K = len(blocks); n = [0] * K; kap = [0] * K; deg = dict(G.degree())
    for v in G: n[idx[part[v]]] += 1; kap[idx[part[v]]] += deg[v]
    e = {(i, j): 0 for i in range(K) for j in range(i, K)}
    for u, v in G.edges():
        r, s = idx[part[u]], idx[part[v]]
        if r > s: r, s = s, r
        e[(r, s)] += 1
    return K, n, e, kap

def _logB(a, b): return loggamma(a) + loggamma(b) - loggamma(a + b)

def logR_dc(G, part, alpha):
    """Degree-corrected (Poisson, configuration-null) log evidence ratio on any graph."""
    a = mpf(alpha); K, n, e, kap = _stats(G, part); twom = sum(kap); m = G.number_of_edges()
    lZ = lambda e_, O: loggamma(e_ + a) - (e_ + a) * mlog(O + a) + a * mlog(a) - loggamma(a)
    tot = mpf(0)
    for i in range(K):
        for j in range(i, K):
            O = mpf(kap[i] * kap[i]) / (2 * twom) if i == j else mpf(kap[i] * kap[j]) / twom
            tot += lZ(e[(i, j)], O)
    return float((tot - lZ(m, mpf(m))).real)

def logR_plain(G, part, alpha):
    """Plain Bernoulli-SBM log evidence ratio for a two-block partition, using the
    companion papers' label-prior and (fully merged) null convention."""
    a = mpf(alpha); K, n, e, kap = _stats(G, part); assert K == 2
    N = sum(n); Mtot = N * (N - 1) // 2; E = G.number_of_edges()
    n1, n2 = n
    m11 = n1 * (n1 - 1) // 2; m22 = n2 * (n2 - 1) // 2; m12 = n1 * n2
    num = (_logB(e[(0, 0)] + a, m11 - e[(0, 0)] + a)
           + _logB(e[(1, 1)] + a, m22 - e[(1, 1)] + a)
           + _logB(e[(0, 1)] + a, m12 - e[(0, 1)] + a) + _logB(n1 + a, n2 + a))
    den = 2 * _logB(a, a) + _logB(E + a, Mtot - E + a) + _logB(a, N + a)
    return float((num - den).real)


# ------------------------------------------------ closed-form evidence (any t)
def logR_dc_cf(t, alpha):
    a = mpf(alpha); _, _, E11, E22, E12, k1, k2 = closed(t); m = 4 ** t; twom = 2 * m
    O11 = mpf(k1 * k1) / (2 * twom); O22 = mpf(k2 * k2) / (2 * twom); O12 = mpf(k1 * k2) / twom
    lZ = lambda e, O: loggamma(e + a) - (e + a) * mlog(O + a) + a * mlog(a) - loggamma(a)
    return float((lZ(E11, O11) + lZ(E22, O22) + lZ(E12, O12) - lZ(m, mpf(m))).real)

def logR_plain_cf(t, alpha):
    a = mpf(alpha); n1, n2, E11, E22, E12, k1, k2 = closed(t)
    N = n1 + n2; Mtot = N * (N - 1) // 2; E = E11 + E22 + E12
    m11 = n1 * (n1 - 1) // 2; m22 = n2 * (n2 - 1) // 2; m12 = n1 * n2
    num = (_logB(E11 + a, m11 - E11 + a) + _logB(E22 + a, m22 - E22 + a)
           + _logB(E12 + a, m12 - E12 + a) + _logB(n1 + a, n2 + a))
    den = 2 * _logB(a, a) + _logB(E + a, Mtot - E + a) + _logB(a, N + a)
    return float((num - den).real)

def Psplit(lr): return float(1 / (1 + exp(-mpf(lr))))
def Nof(t): return (2 * 4 ** t + 4) // 3
def densities(t):
    _, _, E11, E22, E12, k1, k2 = closed(t); m = 4 ** t; twom = 2 * m
    O11 = (k1 * k1) / (2 * twom); O22 = (k2 * k2) / (2 * twom); O12 = (k1 * k2) / twom
    return E11 / O11, E22 / O22, E12 / O12


# ================================================================= __main__
if __name__ == "__main__":
    print("== closed-form counts, N, E, and logR_dc vs direct construction (t<=7) ==")
    for t in range(1, 8):
        G, poles, part = diamond(t); K, n, e, kap = _stats(G, part)
        assert (n[0], n[1], e[(0, 0)], e[(1, 1)], e[(0, 1)], kap[0], kap[1]) == closed(t)
        assert (G.number_of_nodes(), G.number_of_edges()) == (Nof(t), 4 ** t)
        assert abs(logR_dc(G, part, 1.0) - logR_dc_cf(t, 1.0)) < 1e-6
        assert abs(logR_plain(G, part, 1.0) - logR_plain_cf(t, 1.0)) < 1e-6
    print("   OK\n")

    print("== exact linear RG map v(t+1)=M v(t); eigenvalues {bs,b}={4,2} ==")
    for t in range(1, 8):
        assert np.all(M @ np.array(closed(t)[2:]) == np.array(closed(t + 1)[2:]))
    print("   OK. eigenvalues =", sorted(set(int(round(z.real)) for z in np.linalg.eigvals(M))), "\n")

    print("== Table II data: evidence and density (alpha=1) ==")
    print("   t     N        logR_dc     logR_plain    logR_dc/m")
    for t in range(1, 11):
        print(f"   {t:<3}{Nof(t):>8}{logR_dc_cf(t,1.):>13.2f}{logR_plain_cf(t,1.):>13.2f}{logR_dc_cf(t,1.)/4**t:>12.5f}")
    print(f"\n   fixed-point density  logR_dc/m -> ln K = ln 2 = {math.log(2):.5f}")
    sp = (logR_plain_cf(13, 1.) - logR_plain_cf(12, 1.)) / (4 ** 13 - 4 ** 12)
    print(f"   plain slope ~ {sp:.5f}   (DC/plain ~ {math.log(2)/sp:.2f})\n")

    print("== RG flow of densities -> (u_in, u_cross) = (K, 0) = (2, 0) ==")
    for t in range(1, 9):
        u11, u22, u12 = densities(t)
        print(f"   t={t}:  u_in={u22:.5f}  u_cross={u12:.5f}")

    print("\n== Table III: r_kappa = N_{t*}  (alpha=1) ==")
    for q in (0.5, 0.9, 0.99, 0.999):
        thr = math.log(q / (1 - q))
        td = next(t for t in range(1, 40) if logR_dc_cf(t, 1.) >= thr)
        tp = next(t for t in range(1, 40) if logR_plain_cf(t, 1.) >= thr)
        print(f"   q={q:<6}  DC: r_kappa=N_{td}={Nof(td):<6}   plain: r_kappa=N_{tp}={Nof(tp)}")
