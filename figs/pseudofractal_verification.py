#!/usr/bin/env python3
"""
Verification script for "The pseudofractal web wants to be broken".

Reproduces, from scratch:
  * the pseudofractal (DGM) web and its closed-form counts,
  * the plain Bernoulli-SBM evidence ratio and its transition,
  * the degree-corrected (Poisson, configuration-null) evidence ratio,
  * the asymptotic slopes and the Ramsey community numbers.

Requires: networkx, mpmath.  Run:  python3 pseudofractal_verification.py
"""
import math
import networkx as nx
from mpmath import mp, mpf, loggamma, exp, log as mlog

mp.dps = 80  # 80-digit arithmetic


# ---------------------------------------------------------------- construction
def dgm(T):
    """Pseudofractal web after T generations, with branch labels.
    Rule: every edge spawns a vertex joined to both endpoints.
    branch[v] in {0,1,2} = descent branch; the 3 seed hubs are tagged -1."""
    G = nx.Graph()
    G.add_edges_from([(0, 1), (1, 2), (0, 2)])
    branch = {0: -1, 1: -1, 2: -1}
    eb = {(0, 1): 0, (1, 0): 0, (1, 2): 1, (2, 1): 1, (0, 2): 2, (2, 0): 2}
    nxt = 3
    for _ in range(T):
        for (u, v) in list(G.edges()):
            b = eb[(u, v)]
            w = nxt; nxt += 1
            G.add_edge(w, u); G.add_edge(w, v)
            branch[w] = b
            for e in ((w, u), (u, w), (w, v), (v, w)):
                eb[e] = b
    return G, branch


def branch_counts(T):
    """Closed-form (n1,n2,E1,E2,E12,kappa1,kappa2) for the branch bipartition."""
    n1 = (3**T + 5) // 2
    n2 = 3**T - 1
    E1 = 3**T + 2
    E12 = 2**(T + 2) - 4
    E2 = 2 * 3**T + 2 - 2**(T + 2)
    k1 = 2 * 3**T + 2**(T + 2)
    k2 = 4 * 3**T - 2**(T + 2)
    return n1, n2, E1, E2, E12, k1, k2


def Nof(T):
    return 3 * (3**T + 1) // 2


# --------------------------------------------------------- verify counts
print("== closed-form counts vs direct construction ==")
for T in range(2, 8):
    G, br = dgm(T)
    part = {v: (0 if br[v] in (0, -1) else 1) for v in G.nodes()}
    n1 = sum(part[v] == 0 for v in G.nodes()); n2 = len(G) - n1
    E1 = E2 = E12 = 0
    for u, v in G.edges():
        if part[u] == part[v]:
            E1 += part[u] == 0; E2 += part[u] == 1
        else:
            E12 += 1
    k = dict(G.degree())
    k1 = sum(k[v] for v in G.nodes() if part[v] == 0)
    k2 = sum(k[v] for v in G.nodes() if part[v] == 1)
    assert (n1, n2, E1, E2, E12, k1, k2) == branch_counts(T)
    assert (len(G), G.number_of_edges()) == (Nof(T), 3**(T + 1))
print("  OK through T=7\n")


# ---------------------------------------------------------- evidence ratios
def logB(a, b):
    return loggamma(a) + loggamma(b) - loggamma(a + b)


def logR_plain(T, alpha):
    n1, n2, E1, E2, E12, _, _ = branch_counts(T)
    a = mpf(alpha); N = n1 + n2
    m1 = n1 * (n1 - 1) // 2; m2 = n2 * (n2 - 1) // 2
    m12 = n1 * n2; M = N * (N - 1) // 2; E = E1 + E2 + E12
    num = (logB(E1 + a, m1 - E1 + a) + logB(E2 + a, m2 - E2 + a)
           + logB(E12 + a, m12 - E12 + a) + logB(n1 + a, n2 + a))
    den = 2 * logB(a, a) + logB(E + a, M - E + a) + logB(a, N + a)
    return (num - den).real


def logR_dc(T, alpha):
    _, _, E1, E2, E12, k1, k2 = branch_counts(T)
    a = mpf(alpha); m = E1 + E2 + E12; twom = k1 + k2
    O11 = mpf(k1 * k1) / (2 * twom)
    O22 = mpf(k2 * k2) / (2 * twom)
    O12 = mpf(k1 * k2) / twom
    Ot = O11 + O22 + O12  # == m
    def lZ(e, O):
        return loggamma(e + a) - (e + a) * mlog(O + a)
    occam = 2 * (a * mlog(a) - loggamma(a))
    return (lZ(E1, O11) + lZ(E2, O22) + lZ(E12, O12) - lZ(m, Ot) + occam).real


def Psplit(lr):
    return float(1 / (1 + exp(-mpf(lr))))


# ------------------------------------------------------------------- tables
print("== Table I: plain Bernoulli SBM ==")
print("  T     N    logR(.5)   logR(1)   logR(2)   P_split(1)")
for T in range(3, 9):
    r = [float(logR_plain(T, a)) for a in (0.5, 1, 2)]
    print(f"  {T:<2}{Nof(T):>6} {r[0]:>10.2f}{r[1]:>10.2f}{r[2]:>10.2f}"
          f"   {Psplit(logR_plain(T,1)):.4f}")

print("\n== Table II: degree-corrected (config null) ==")
print("  T     N    logR_dc(.5) logR_dc(1) logR_dc(2)")
for T in range(2, 9):
    r = [float(logR_dc(T, a)) for a in (0.5, 1, 2)]
    print(f"  {T:<2}{Nof(T):>6} {r[0]:>11.2f}{r[1]:>11.2f}{r[2]:>11.2f}")


# ------------------------------------------------------------- asymptotics
print("\n== asymptotic slopes (alpha=1) ==")
kp = math.log(3) - 2/3 * math.log(2)
kd = 2 * math.log(3) - 4/3 * math.log(2)
dp = float((logR_plain(31, 1) - logR_plain(30, 1)) / (Nof(31) - Nof(30)))
dd = float((logR_dc(31, 1) - logR_dc(30, 1)) / (Nof(31) - Nof(30)))
print(f"  plain: predicted ln3-(2/3)ln2 = {kp:.6f}   numerical = {dp:.6f}")
print(f"  dc:    predicted 2ln3-(4/3)ln2= {kd:.6f}   numerical = {dd:.6f}")
print(f"  ratio dc/plain = {kd/kp:.4f}  (exactly 2)")


# ---------------------------------------------------------- Ramsey numbers
def ramsey(fun, alpha, q):
    thr = math.log(q / (1 - q))
    for T in range(1, 40):
        if float(fun(T, alpha)) >= thr:
            return Nof(T)
    return None


print("\n== Table III: Ramsey community numbers r_kappa = N_T ==")
for alpha in (0.5, 1.0, 2.0):
    p = [ramsey(logR_plain, alpha, q) for q in (0.5, 0.9, 0.99, 0.999)]
    d = [ramsey(logR_dc, alpha, q) for q in (0.5, 0.9, 0.99, 0.999)]
    print(f"  alpha={alpha}: plain {p}   dc {d}")
