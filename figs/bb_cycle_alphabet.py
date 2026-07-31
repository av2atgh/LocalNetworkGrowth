#!/usr/bin/env python3
"""Where does BB(n,L) land once the alphabet can name its own cycle?

Section 8.5 puts BB(n,2) in the no-communities class: its squares are nearly
disjoint, the alphabet takes them up, and the partition is left with nothing to
describe. The obvious question is whether that is a fact about the square or a
fact about every bubble rule above L=1 -- except that it cannot be asked with the
alphabet of Chapter 8, which has no symbol for a cycle longer than four. Fitted
with link, triangle and square, BB(n,3) plants nothing at all and would be
recorded as a constant-density rule for want of a vocabulary rather than on the
evidence. That is the trap the chapter is about, sprung on itself.

So the L+2 cycle is added to motif_sbm's alphabet and each rule is fitted twice
on the same networks:

    control   edge, triangle, square         -- the rule's motif is unnameable,
                                                and the fit should reduce to the
                                                nested block model
    test      control + the L+2 cycle        -- the rule's own motif is in the
                                                alphabet

alongside the flat and nested block models, so that all four columns of the
master table can be filled. Writes data/ramsey_motif_scaling_bb{L}.csv in the
schema make_master_tables.py reads, with an extra n_cyc column.

L=3 is the pentagon and L=4 the hexagon. L=2 reproduces the square measurement
already in the chapter, and is worth running as a check that this pipeline and
ramsey_motif.py agree.

Usage:
    python3 bb_cycle_alphabet.py --L 4 --sizes 100,200,400,800,1600 --reps 20
    python3 bb_cycle_alphabet.py --L 4 --calibrate --n 1600
"""
import argparse
import os
import time
from multiprocessing import Pool

import pandas as pd

from arity_models import bb
from motif_sbm import MotifSBM

HERE = os.path.dirname(os.path.abspath(__file__))
CONTROL = ("edge", "triangle", "square")
# BB(n,L) closes a cycle of length L+2
CYCLE_OF = {2: "square", 3: "pentagon", 4: "hexagon", 5: "heptagon"}


def build(n, L, seed):
    adj = bb(n, L, seed)
    idx = {v: i for i, v in enumerate(sorted(adj))}
    edges = sorted({(min(idx[u], idx[w]), max(idx[u], idx[w]))
                    for u in adj for w in adj[u] if u != w})
    return len(adj), edges


def one(job):
    n, L, seed = job
    nn, edges = build(n, L, seed)
    m = len(edges)
    test = CONTROL if CYCLE_OF[L] in CONTROL else CONTROL + (CYCLE_OF[L],)

    base = MotifSBM(nn, edges)
    try:
        k_dc = len(set(base.dcsbm_blocks(seed=seed)))
        k_flat = len(set(base.dcsbm_blocks(nested=False, seed=seed)))
    except ImportError:
        k_dc = k_flat = float("nan")

    row = dict(model=f"BB(n,{L})", n=n, n_nodes=nn, seed=seed, m=m,
               kappa_dcsbm=k_dc, kappa_flat=k_flat)

    for tag, alpha in (("ctl", CONTROL), ("cyc", test)):
        t = time.time()
        f = MotifSBM(nn, edges, alphabet=alpha)
        f.fit()
        c = f.motif_counts()
        row.update({
            f"kappa_{tag}": len(set(f.blocks)),
            f"sigma_{tag}": round(f.description_length(), 2),
            f"tri_{tag}": c.get("triangle", 0),
            f"sq_{tag}": c.get("square", 0),
            f"cyc_{tag}": c.get(CYCLE_OF[L], 0),
            f"secs_{tag}": round(time.time() - t, 1),
        })
    # the master table's columns refer to the alphabet that can name this rule
    row["kappa"] = row["kappa_cyc"]
    row["sigma"] = row["sigma_cyc"]
    row["n_tri"] = row["tri_cyc"]
    row["n_sq"] = row["sq_cyc"]
    row["n_cyc"] = row["cyc_cyc"]
    row["cyc_len"] = L + 2
    row["named"] = round((L + 2) * row["cyc_cyc"] / m, 4)
    row["d_sigma"] = round(row["sigma_cyc"] - row["sigma_ctl"], 2)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int, default=4, choices=(2, 3, 4, 5))
    ap.add_argument("--sizes", default="100,200,400,800,1600")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--jobs", type=int, default=7)
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--n", type=int, default=1600)
    a = ap.parse_args()

    out = os.path.join(HERE, "data", f"ramsey_motif_scaling_bb{a.L}.csv")
    if a.calibrate:
        t = time.time()
        r = one((a.n, a.L, 12345))
        print(f"BB(n,{a.L}) n={a.n}: {time.time()-t:.0f}s  kappa flat "
              f"{r['kappa_flat']} nested {r['kappa_dcsbm']} ctl {r['kappa_ctl']} "
              f"+{CYCLE_OF[a.L]} {r['kappa_cyc']}   {r['cyc_cyc']} "
              f"{CYCLE_OF[a.L]}s ({r['named']:.0%} of links)  "
              f"dSigma {r['d_sigma']:+.0f}")
        return

    sizes = [int(x) for x in a.sizes.split(",")]
    done = pd.read_csv(out) if os.path.exists(out) else pd.DataFrame()
    have = set() if done.empty else set(zip(done.n, done.seed))
    jobs = [(n, a.L, 1000 * n + r) for n in sizes for r in range(a.reps)
            if (n, 1000 * n + r) not in have]
    print(f"{len(jobs)} networks on {a.jobs} workers", flush=True)

    rows = [] if done.empty else done.to_dict("records")
    with Pool(a.jobs) as pool:
        for r in pool.imap_unordered(one, jobs):
            rows.append(r)
            print(f"  n={r['n']:<5} kappa flat {r['kappa_flat']:3d} "
                  f"nested {r['kappa_dcsbm']:3d}  ctl {r['kappa_ctl']:3d} "
                  f"+cyc {r['kappa_cyc']:3d}   {r['cyc_cyc']:4d} "
                  f"{CYCLE_OF[r['cyc_len']-2]}s ({r['named']:.0%} of links)  "
                  f"dSigma {r['d_sigma']:+8.0f}  "
                  f"({r['secs_ctl']+r['secs_cyc']:.0f}s)", flush=True)
            pd.DataFrame(rows).to_csv(out, index=False)

    d = pd.DataFrame(rows)
    print(f"\n{'n':>6} {'flat':>6} {'nested':>7} {'ctl':>6} {'+cyc':>6} "
          f"{'P_flat':>7} {'P_nest':>7} {'P_cyc':>7} {'named':>7}")
    for n, g in d.groupby("n"):
        print(f"{n:6d} {g.kappa_flat.mean():6.1f} {g.kappa_dcsbm.mean():7.1f} "
              f"{g.kappa_ctl.mean():6.1f} {g.kappa_cyc.mean():6.1f} "
              f"{(g.kappa_flat>1).mean():7.2f} {(g.kappa_dcsbm>1).mean():7.2f} "
              f"{(g.kappa_cyc>1).mean():7.2f} {g.named.mean():7.0%}")


if __name__ == "__main__":
    main()
