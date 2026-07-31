#!/usr/bin/env python3
"""Does the square alphabet dissolve the diamond lattice's communities?

Section 8.6 finds one rule whose communities the motif alphabet abolishes rather
than delays: BB(n,2), which closes a square per step. Two explanations were open.
Either the square is what does it, or BB(n,2)'s reach is -- it lays a chain of two
nodes across a link, so the new path runs u-w1-w2-v and the interior nodes sit two
steps from an endpoint, where Chapter 6 found that a rule reaching three steps
builds communities far more reluctantly than one reaching two.

The diamond lattice separates them. It also builds squares, but from two chains of
ONE node each: every new node is adjacent to both endpoints of the link it
replaces, so nothing reaches past two steps. Same motif, same mean degree as
BB(n,2) to two decimals, opposite reach. If reach were the discriminator the
diamond would keep its communities.

It does not, which is what this script measures. Fits the flat block model, the
nested one, and the nested one with triangles and squares in its alphabet, to
diamond(t) for t = 3..6, and reports how many communities each returns together
with what the alphabet plants.

Note on why the lattice is deterministic here. A randomised diamond -- pick one
link at a time and replace it -- does not work, and the reason is worth recording:
subdividing a link that already lies in a square stretches that square into a
five-cycle, so the squares decay as fast as they are made and the rule ends up
with as many odd cycles as even ones. The lattice replaces every edge at once,
which is what preserves the parity. See generator_diamond in ramsey_motif.py.

Being deterministic, there is one network per size rather than an ensemble, so
this reports the community count directly instead of a Ramsey number. The fit is
stochastic and the diamond is known to trap it -- the branch cut puts every
four-cycle across the cut -- so each network is fitted several times and the fit
that wins on description length is the one reported.

Usage:  python3 diamond_motif.py [--tmax 6] [--restarts 3]
"""
import argparse
import os

import pandas as pd

from diamond_rg import diamond
from motif_spectrum import pseudofractal
from motif_sbm import MotifSBM

HERE = os.path.dirname(os.path.abspath(__file__))

# The two deterministic lattices are the matched pair the section needs: both
# replace every link at once, so both keep the parity a randomised rule destroys,
# and both reach exactly two steps -- but one closes triangles and the other
# closes squares. Chapter 6 puts a triangle rule at r_kappa = 108 and a square
# rule at 326, so if cycle length is what decides whether the alphabet can
# dissolve a community, these two should part company.
LATTICES = {"diamond": (diamond, "square"),
            "pseudofractal": (pseudofractal, "triangle")}


def build(lattice, t):
    """Both lattices, reduced to (n, edge list)."""
    if lattice == "pseudofractal":
        n, E, _ = pseudofractal(t)
        return n, sorted(E)
    r = diamond(t)
    g = r[0] if isinstance(r, tuple) else r
    idx = {v: i for i, v in enumerate(g.nodes())}
    return g.number_of_nodes(), [(idx[u], idx[v]) for u, v in g.edges()]


def fit(t, restarts, lattice="diamond"):
    n, edges = build(lattice, t)
    best = None
    for s in range(restarts):
        m = MotifSBM(n, edges)
        kf = len(set(m.dcsbm_blocks(nested=False, seed=s)))
        kd = len(set(m.dcsbm_blocks(seed=s)))
        m.fit()
        sigma = m.description_length()
        if best is None or sigma < best["sigma"]:
            c = m.motif_counts()
            best = dict(lattice=lattice, t=t, n=n, m=len(edges),
                        kappa_flat=kf, kappa_dcsbm=kd,
                        kappa=len(set(m.blocks)), n_sq=c.get("square", 0),
                        n_tri=c.get("triangle", 0), n_edge=c.get("edge", 0),
                        sigma=round(sigma, 2))
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tmax", type=int, default=6)
    ap.add_argument("--tmin", type=int, default=3)
    ap.add_argument("--restarts", type=int, default=3)
    ap.add_argument("--lattice", default="diamond", choices=tuple(LATTICES))
    a = ap.parse_args()
    out = os.path.join(HERE, "data", f"{a.lattice}_motif.csv")

    rows = []
    print(f"{'t':>2}{'n':>7}{'m':>7} | {'flat':>5}{'nested':>7}{'motif':>7} |"
          f" {'tri':>7}{'squares':>8}{'links':>7}")
    for t in range(a.tmin, a.tmax + 1):
        r = fit(t, a.restarts, a.lattice)
        rows.append(r)
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"{t:>2}{r['n']:>7}{r['m']:>7} | {r['kappa_flat']:>5}"
              f"{r['kappa_dcsbm']:>7}{r['kappa']:>7} | {r['n_tri']:>7}"
              f"{r['n_sq']:>8}{r['n_edge']:>7}", flush=True)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
