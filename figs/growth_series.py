#!/usr/bin/env python3
"""Beta measured along a single real network's own growth, not across a census.

Section 7.12 compares two exponents that are not measured the same way. The
growth rules give beta from a *growth series*: one rule, one mechanism, the same
object at a sequence of sizes. The census gives beta from a *cross-section*:
eighty-one different networks, of different kinds, built by different processes
and sampled by different projects, with size as the only ordering. A power law
fitted across a cross-section estimates how the number of communities varies
between systems, which need not be how it varies within one.

The Internet at the autonomous-system level is the case where the distinction can
be tested, because it was recorded repeatedly as it grew. Two independent series:

  route_views   733 daily snapshots, 1997-11 to 2000-01, n = 3015 to 6474
  caida_as      122 snapshots, 2004 to 2007, n = 16301 to 26389

Each spans only a factor of two in size, which is why they were not used for this
in the first place; but each has many snapshots with very little scatter between
neighbours, because consecutive snapshots are largely the same network. The two
are not pooled: they are different measurement projects with different coverage,
and pooling them would reintroduce the sampling confound the exercise is meant to
remove.

Writes data/growth_series.csv, one row per snapshot, and prints the within-series
exponent. Resumes: existing rows are kept and only missing snapshots are fitted.

Usage:  python3 growth_series.py [--points 16]
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
import graph_tool.all as gt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data", "growth_series.csv")
SERIES = ("route_views", "caida_as")


def fit(g, reps=5):
    """The nested degree-corrected fit of Section 7.9, on the giant component.

    A single run of the minimiser is stochastic, and on these networks it
    scatters B by a factor of two at fixed n -- enough to swamp a slope measured
    over a size range of two. So run it several times and keep the fit that
    actually wins on description length, which is what the criterion says to do
    and what removes most of the scatter.
    """
    g = gt.extract_largest_component(g, prune=True)
    best = None
    for r in range(reps):
        gt.seed_rng(r)
        st = gt.minimize_nested_blockmodel_dl(g, state_args=dict(deg_corr=True))
        sigma = st.entropy() / np.log(2)
        if best is None or sigma < best[0]:
            b = np.asarray(st.get_bs()[0], dtype=int)[:g.num_vertices()]
            best = (sigma, len(np.unique(b)))
    return g.num_vertices(), g.num_edges(), best[1], best[0]


def snapshots(base, k):
    """k snapshots spread evenly through the series, by date order."""
    import graph_tool.collection as coll
    ks = sorted(x for x in coll.ns.keys() if x.startswith(base + "/"))
    idx = np.unique(np.linspace(0, len(ks) - 1, k).round().astype(int))
    return [ks[i] for i in idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--points", type=int, default=16)
    ap.add_argument("--reps", type=int, default=5)
    a = ap.parse_args()

    import graph_tool.collection as coll
    done = (pd.read_csv(OUT) if os.path.exists(OUT)
            else pd.DataFrame(columns=["name", "reps"]))
    done = done[done.get("reps", pd.Series(dtype=int)) >= a.reps]   # refit the singles
    rows = done.to_dict("records")
    have = set(done.name)

    for base in SERIES:
        for name in snapshots(base, a.points):
            if name in have:
                continue
            t = time.time()
            n, m, B, sigma = fit(coll.ns[name], a.reps)
            rows.append(dict(series=base, name=name, n=n, m=m, B_nested=B,
                             sigma=round(sigma, 1), reps=a.reps,
                             secs=round(time.time() - t, 1)))
            pd.DataFrame(rows).to_csv(OUT, index=False)
            print(f"  {name:<24} n={n:6d} m={m:7d} B={B:4d} "
                  f"B/n={B/n:.4f}  ({time.time()-t:.0f}s)", flush=True)

    d = pd.DataFrame(rows)
    print()
    for base in SERIES:
        s = d[d.series == base]
        if len(s) > 3:
            b = np.polyfit(np.log(s.n), np.log(s.B_nested), 1)[0]
            print(f"  {base:<12} {len(s):3d} snapshots, n = {s.n.min()}-{s.n.max()}, "
                  f"beta = {b:.3f}, B/n = {(s.B_nested/s.n).mean():.4f}")


if __name__ == "__main__":
    main()
