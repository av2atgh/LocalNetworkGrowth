#!/usr/bin/env python3
"""Check the C cycle counter against the two independent references in the repo.

The C enumerator in cyclecount.c adds distance pruning to the plain DFS of
`true_cycles` in motif_spectrum.py, and pruning bugs are silent: they lose
cycles without ever producing a wrong-looking number.  So it is checked three
ways before being used on real data.

  1. against `true_cycles` itself, on random graphs of every density from tree
     to nearly complete, and on the two deterministic lattices;
  2. against the trace identities for L = 3, 4, 5 that chapter 7 already uses;
  3. against the diamond lattice's known exact spectrum -- support on
     L = 4, 8 only, with 4^t eight-cycles.

Usage:  python3 cyclecount_verification.py [path-to-cyclecount-binary]
"""
import subprocess
import sys

import numpy as np

from motif_spectrum import diamond, pseudofractal, true_cycles

BINARY = sys.argv[1] if len(sys.argv) > 1 else "./cyclecount"
LMAX = 8


def run_c(n, E, Lmax=LMAX, binary=BINARY):
    """Exact cycle counts L=3..Lmax from the C enumerator."""
    E = sorted(E)
    payload = f"{n} {len(E)}\n" + "\n".join(f"{u} {v}" for u, v in E) + "\n"
    out = subprocess.run([binary, "-l", str(Lmax)], input=payload,
                         capture_output=True, text=True, check=True).stdout
    counts, status = {}, None
    for line in out.strip().splitlines():
        a, b = line.split()
        if a == "status":
            status = b
        else:
            counts[int(a)] = int(b)
    if status != "ok":
        raise RuntimeError(f"cyclecount returned status {status}")
    return [counts.get(L, 0) for L in range(3, Lmax + 1)]


def trace_cycles(n, E):
    """C3, C4, C5 from traces -- the identities validated in chapter 7."""
    A = np.zeros((n, n))
    for u, v in E:
        A[u, v] = A[v, u] = 1.0
    d = A.sum(1)
    A2, A3 = A @ A, A @ A @ A
    A4, A5 = A3 @ A, A3 @ A @ A
    m = len(E)
    c3 = np.trace(A3) / 6
    c4 = (np.trace(A4) - 2 * (d ** 2).sum() + 2 * m) / 8
    c5 = (np.trace(A5) - 5 * np.trace(A3)
          - 5 * ((d - 2) * np.diag(A3)).sum()) / 10
    return [c3, c4, c5]


def random_graph(n, p, rng):
    return {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}


def main():
    rng = np.random.default_rng(20260728)
    failures = 0

    print("1. against true_cycles (plain DFS, no pruning)")
    cases = [("pseudofractal t=4", *pseudofractal(4)[:2]),
             ("pseudofractal t=5", *pseudofractal(5)[:2]),
             ("diamond t=4", *diamond(4)[:2]),
             ("diamond t=5", *diamond(5)[:2])]
    for label, n, E in [(l, n, E) for l, n, E in cases]:
        ref, got = true_cycles(n, E, LMAX)[3:], run_c(n, E)
        ok = ref == got
        failures += not ok
        print(f"   {'ok ' if ok else 'FAIL'} {label:<20} n={n:<5} {got}")

    for n, p in [(18, 0.1), (18, 0.25), (18, 0.5), (14, 0.8), (24, 0.15),
                 (30, 0.08), (12, 1.0)]:
        for rep in range(4):
            E = random_graph(n, p, rng)
            if not E:
                continue
            ref, got = true_cycles(n, E, LMAX)[3:], run_c(n, E)
            ok = ref == got
            failures += not ok
            if not ok or rep == 0:
                print(f"   {'ok ' if ok else 'FAIL'} "
                      f"G(n={n}, p={p}) m={len(E):<4} {got}")

    print("\n2. against the L=3,4,5 trace identities")
    for label, n, E in [("pseudofractal t=5", *pseudofractal(5)[:2]),
                        ("diamond t=5", *diamond(5)[:2]),
                        ("G(40, 0.15)", 40, random_graph(40, 0.15, rng)),
                        ("G(40, 0.30)", 40, random_graph(40, 0.30, rng))]:
        got, ref = run_c(n, E)[:3], trace_cycles(n, E)
        ok = all(abs(g - r) < 1e-6 for g, r in zip(got, ref))
        failures += not ok
        print(f"   {'ok ' if ok else 'FAIL'} {label:<20} "
              f"C3,C4,C5 = {got}  traces = {[round(r, 3) for r in ref]}")

    print("\n3. against the diamond lattice's known exact spectrum")
    for t in (3, 4, 5):
        n, E = diamond(t)[:2]
        got = run_c(n, E)
        expect_c8 = 4 ** t
        zeros_ok = got[0] == 0 and got[2] == 0 and got[3] == 0 and got[4] == 0
        ok = zeros_ok and got[5] == expect_c8
        failures += not ok
        print(f"   {'ok ' if ok else 'FAIL'} diamond t={t}  n={n:<5} "
              f"L=3,5,6,7 empty={zeros_ok}  C8={got[5]} (expect {expect_c8})")

    print(f"\n{'ALL CHECKS PASSED' if not failures else f'{failures} FAILURES'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
