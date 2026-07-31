#!/usr/bin/env python3
"""Which branching factor generates the motif spectrum: <k^2>/<k> or <k^2-k>/<k>?

Chapter 7 writes the transfer matrix of a K-block degree-corrected fit as

    T_rs = (e_rs / S_r S_s) sqrt(Q_r Q_s)        Q_r = sum_{i in r} k_i^2

so that the one-block case has leading eigenvalue lambda = <k^2>/<k> and the
whole spectrum is lambda^L/(2L).  That is the branching factor of the
independent-link idealisation, where the two edges of a cycle at a node are
drawn independently and each carries a factor k_i.

But a cycle enters a node by one edge and leaves by another, and the model
actually fitted -- graph-tool's degree-corrected SBM -- holds the degree
sequence fixed.  With degrees fixed the two edges must use distinct stubs, so
the node contributes k_i(k_i - 1), not k_i^2, and

    T'_rs = (e_rs / S_r S_s) sqrt(Q'_r Q'_s)     Q'_r = sum_{i in r} k_i(k_i - 1)

with one-block eigenvalue kappa = <k^2 - k>/<k> = lambda - 1.  This is the
leading eigenvalue of the non-backtracking (Hashimoto) operator rather than of
the adjacency-like one, which is the natural object for cycles.

Asymptotically on the chapter's lattices lambda -> infinity and the distinction
vanishes, which is why the exact lattice calculations never saw it.  On real
networks lambda is around 3 and the two differ by a factor of 25 at L = 8.

This script decides between them on synthetic graphs whose ensemble is known
exactly, at n large enough that finite-size exclusion is negligible.

Usage:  python3 branching_verification.py [path-to-cyclecount-binary]
"""
import os
import subprocess
import sys

import numpy as np

BINARY = (sys.argv[1] if len(sys.argv) > 1
          else os.path.join(os.path.dirname(os.path.abspath(__file__)), "cyclecount"))
LMAX = 8
N = 40000
REPS = 3


def cycles(n, E, lmax=LMAX):
    payload = f"{n} {len(E)}\n" + "\n".join(f"{u} {v}" for u, v in E) + "\n"
    out = subprocess.run([BINARY, "-l", str(lmax)], input=payload,
                         capture_output=True, text=True, check=True).stdout
    got = {}
    for line in out.strip().splitlines():
        a, b = line.split()
        if a != "status":
            got[int(a)] = int(b)
        elif b != "ok":
            raise RuntimeError("cyclecount timed out")
    return got


def config_graph(n, sampler, seed):
    """A simple graph with the sampled degree sequence -- the configuration
    model conditioned on being simple, which is exactly what a degree-preserving
    rewiring of a real network samples."""
    import graph_tool.all as gt
    gt.seed_rng(seed)
    np.random.seed(seed)
    g = gt.random_graph(n, sampler, directed=False,
                        parallel_edges=False, self_loops=False, n_iter=100)
    E = sorted({(min(int(u), int(v)), max(int(u), int(v)))
                for u, v in g.get_edges()})
    k = np.zeros(n)
    for u, v in E:
        k[u] += 1
        k[v] += 1
    return E, k


def powerlaw_sampler(gamma, kmin, kmax, rng):
    ks = np.arange(kmin, kmax + 1)
    p = ks.astype(float) ** (-gamma)
    p /= p.sum()
    return lambda: int(rng.choice(ks, p=p))


def run(label, sampler, n=N, seed=0):
    rows = []
    for rep in range(REPS):
        E, k = config_graph(n, sampler, seed + rep)
        lam = (k ** 2).sum() / k.sum()
        kap = (k * (k - 1)).sum() / k.sum()
        rows.append((cycles(n, E), lam, kap, k.mean()))
    lam = np.mean([r[1] for r in rows])
    kap = np.mean([r[2] for r in rows])
    print(f"\n{label}   n={n}  <k>={np.mean([r[3] for r in rows]):.2f}  "
          f"lambda=<k^2>/<k>={lam:.3f}  kappa=<k^2-k>/<k>={kap:.3f}")
    print(f"  {'L':>3} {'counted':>12} {'lambda^L/2L':>13} {'ratio':>8} "
          f"{'kappa^L/2L':>13} {'ratio':>8}")
    err_lam, err_kap = [], []
    for L in range(3, LMAX + 1):
        c = np.mean([r[0].get(L, 0) for r in rows])
        pl, pk = lam ** L / (2 * L), kap ** L / (2 * L)
        err_lam.append(c / pl)
        err_kap.append(c / pk)
        print(f"  {L:>3} {c:>12.1f} {pl:>13.4g} {c / pl:>8.2f} "
              f"{pk:>13.4g} {c / pk:>8.2f}")
    print(f"  geometric-mean ratio over L=3..8:  lambda {np.exp(np.mean(np.log(err_lam))):.3f}"
          f"   kappa {np.exp(np.mean(np.log(err_kap))):.3f}")
    return err_lam, err_kap


def main():
    rng = np.random.default_rng(7)
    all_lam, all_kap = [], []

    # 4-regular: the textbook case.  Every node has k(k-1) = 12 ordered pairs of
    # stubs against k^2 = 16, so the two predictions differ by (4/3)^L.
    a, b = run("4-regular", lambda: 4)
    all_lam += a; all_kap += b

    a, b = run("Poisson <k>=5", lambda: max(1, int(rng.poisson(5))))
    all_lam += a; all_kap += b

    a, b = run("power law gamma=2.8, k in [2,120]",
               powerlaw_sampler(2.8, 2, 120, rng))
    all_lam += a; all_kap += b

    a, b = run("power law gamma=2.4, k in [2,60]",
               powerlaw_sampler(2.4, 2, 60, rng))
    all_lam += a; all_kap += b

    gl = np.exp(np.mean(np.log(all_lam)))
    gk = np.exp(np.mean(np.log(all_kap)))
    print(f"\noverall geometric-mean ratio counted/predicted")
    print(f"  lambda = <k^2>/<k>     {gl:.3f}   (chapter 7 as written)")
    print(f"  kappa  = <k^2-k>/<k>   {gk:.3f}   (excess degree)")
    print("\nkappa wins" if abs(np.log(gk)) < abs(np.log(gl)) else
          "\nlambda wins")


if __name__ == "__main__":
    main()
