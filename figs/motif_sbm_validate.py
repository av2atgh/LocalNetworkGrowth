#!/usr/bin/env python3
"""Does the motif-alphabet block model recover the motif a network was built from?

Runs the validation table of the chapter-7 working notes against ground truth
that is known exactly, before any real network is touched. For each system the
model is fitted twice -- with the alphabet ("edge",), which is an ordinary
degree-corrected block model, and with ("edge", "triangle", "square") -- and the
question is whether the enlarged alphabet pays for itself in description length,
and if so which motif it selects.

    positive controls   the answer is known and non-trivial
      planted 2-block   triangles in one block, squares in the other
      pseudofractal     exactly n-2 triangles          -> triangle
      BB(n,1)           Dorogovtsev-Mendes, triangles  -> triangle
      diamond           exactly 4^(t-1) squares        -> square
      DS(n,q)           squares, no triangles at all   -> square
      BB(n,2)           squares                        -> square

    negative controls   the motif is OUTSIDE the alphabet, and the honest
                        outcome is that nothing is selected
      BB(n,3)           pentagons
      BB(n,4)           hexagons

The negative controls matter as much as the positive ones. A method that reports
"square" for a pentagon-built network is hallucinating, and the alphabet is a
choice that has to be justified rather than a free parameter. The fullerenes are
the same kind of case -- pentagons and hexagons, nothing else -- and can be added
from the Netzschleuder cache if it is present.

Usage:  python3 motif_sbm_validate.py [--quick]
"""
import argparse
import itertools
import os
import sys
import time

from arity_models import bb, ds
from motif_sbm import MotifSBM
from motif_spectrum import diamond, pseudofractal

HERE = os.path.dirname(os.path.abspath(__file__))


def from_adj(adj):
    idx = {v: i for i, v in enumerate(sorted(adj))}
    E = sorted({(min(idx[u], idx[w]), max(idx[u], idx[w]))
                for u in adj for w in adj[u]})
    return len(adj), E


def case_lattice(build, t):
    n, E, _ = build(t)
    return n, sorted(E)


def run(label, n, edges, expect, quick=False):
    t0 = time.time()
    plain = MotifSBM(n, edges, alphabet=("edge",)).fit()
    s_plain = plain.description_length()

    full = MotifSBM(n, edges).fit()
    s_full = full.description_length()

    c = full.motif_counts()
    tri, sq = c.get("triangle", 0), c.get("square", 0)
    if s_full >= s_plain or (tri == 0 and sq == 0):
        got = "none"
    elif tri and not sq:
        got = "triangle"
    elif sq and not tri:
        got = "square"
    else:
        got = "triangle" if tri >= sq else "square"
    ok = got == expect
    print(f"  {'ok  ' if ok else 'FAIL'} {label:<28} n={n:<6} m={len(edges):<6} "
          f"Sigma {s_plain:9.1f} -> {s_full:9.1f}  "
          f"dS={s_full - s_plain:+8.1f}  tri={tri:<5} sq={sq:<5} "
          f"-> {got:<8} (expect {expect})  [{time.time() - t0:.0f}s]")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="smaller instances, for a fast check")
    args = ap.parse_args()
    small = args.quick
    ok = True

    print("positive controls -- the motif is in the alphabet\n")
    n, E, _ = MotifSBM.sample([100, 100],
                              {0: {"triangle": 50}, 1: {"square": 50}}, seed=11)
    # this one is reported separately below, since it has two answers at once
    m = MotifSBM(n, E).fit()
    per = {}
    for name, nodes in m.instances:
        if name != "edge":
            per.setdefault((m.blocks[nodes[0]], name), 0)
            per[(m.blocks[nodes[0]], name)] += 1
    blocks = len(set(m.blocks))
    good = blocks == 2 and all(len({k[1] for k in per if k[0] == r}) == 1
                               for r in set(m.blocks) if any(k[0] == r for k in per))
    ok &= good
    print(f"  {'ok  ' if good else 'FAIL'} {'planted 2-block':<28} n={n:<6} "
          f"m={len(E):<6} blocks={blocks}, motifs per block {per}")
    print("       (triangles planted in one block, squares in the other)\n")

    ok &= run("pseudofractal t=5", *case_lattice(pseudofractal, 5), "triangle")
    ok &= run("BB(n,1) Dorogovtsev-Mendes", *from_adj(bb(300 if small else 500, 1, 7)),
              "triangle")
    ok &= run("diamond t=4" if small else "diamond t=5",
              *case_lattice(diamond, 4 if small else 5), "square")
    ok &= run("DS(n, q=1/3)", *from_adj(ds(300 if small else 500, 1 / 3, 7)),
              "square")
    ok &= run("BB(n,2)", *from_adj(bb(300 if small else 500, 2, 7)), "square")

    print("\nnegative controls -- the motif is OUTSIDE the alphabet\n")
    ok &= run("BB(n,3) pentagons", *from_adj(bb(300 if small else 500, 3, 7)), "none")
    ok &= run("BB(n,4) hexagons", *from_adj(bb(300 if small else 500, 4, 7)), "none")

    cache = os.environ.get("CROSSOVER_CACHE", os.path.join(HERE, "_netcache"))
    ful = os.path.join(cache, "fullerene_structures__C240.txt")
    if os.path.exists(ful):
        with open(ful) as fh:
            nn, mm = (int(x) for x in fh.readline().split())
            EE = [tuple(int(x) for x in line.split()) for line in fh]
        ok &= run("fullerene C240", nn, EE, "none")

    print(f"\n{'ALL AS EXPECTED' if ok else 'SOME CASES DISAGREE'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
