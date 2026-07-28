#!/usr/bin/env python3
"""Re-run chapter 6's community census with the nested block model.

Chapter 6 counts communities across ~90 real networks with a flat
degree-corrected block model and fits B ~ n^beta, getting beta ~ 0.61. Chapter 7
points out that a flat fit cannot resolve more than sqrt(N) groups however many
are present, so the instrument's own ceiling has the same functional form as the
fractal limit the exponent is being compared against, and the census cannot
certify an exponent above one half.

On the book's two growth rules that objection is now measured: over n = 400 to
1600 the flat model gives 0.50 and 0.60 while the nested model gives 0.94 and
0.97 on identical networks (ramsey_motif.py). This script asks the same of the
real networks, refitting every one of them both ways so the comparison is on the
same graphs rather than across runs, and so the flat column can be checked
against the B_mean the census already published.

Networks come from the edge-list cache built by crossover_scout.py (largest
connected component of the simple undirected graph, which is what the census
used); set CROSSOVER_CACHE if it lives elsewhere. Rows are written as they
finish, smallest first, so a partial run is still usable -- the largest networks
here reach two million nodes and a nested fit on those is not quick.

Usage:  python3 census_nested.py [--max-n N] [--reps R] [--out FILE]
"""
import argparse
import csv
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
CACHE = os.environ.get("CROSSOVER_CACHE", os.path.join(HERE, "_netcache"))
CENSUS = os.path.join(DATA, "community_communities_vs_n.csv")


def load_edges(name):
    path = os.path.join(CACHE, name.replace("/", "__") + ".txt")
    with open(path) as fh:
        n, m = (int(x) for x in fh.readline().split())
        E = [tuple(int(x) for x in line.split()) for line in fh]
    return n, E


def fit(n, E, nested, seed):
    import graph_tool.all as gt
    g = gt.Graph(directed=False)
    g.add_vertex(n)
    g.add_edge_list(E)
    gt.seed_rng(seed)
    if nested:
        st = gt.minimize_nested_blockmodel_dl(g, state_args=dict(deg_corr=True))
        b = np.asarray(st.get_bs()[0], dtype=int)[:n]
    else:
        st = gt.minimize_blockmodel_dl(g, state_args=dict(deg_corr=True))
        b = np.asarray(st.get_blocks().a, dtype=int)[:n]
    return len(np.unique(b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-n", type=int, default=10 ** 9)
    ap.add_argument("--reps", type=int, default=3,
                    help="runs per network below --reps-max-n nodes; 1 above")
    ap.add_argument("--reps-max-n", type=int, default=10000)
    ap.add_argument("--out", default=os.path.join(DATA, "census_nested.csv"))
    args = ap.parse_args()

    with open(CENSUS) as fh:
        census = {r["network"]: r for r in csv.DictReader(fh)}
    names = sorted(census, key=lambda x: int(census[x]["n"]))
    names = [x for x in names if int(census[x]["n"]) <= args.max_n]

    # resume: a nested fit on the largest networks here takes hours, so an
    # interrupted run picks up where it stopped rather than starting over
    rows = []
    if os.path.exists(args.out):
        with open(args.out) as fh:
            for r in csv.DictReader(fh):
                for k in ("n", "m", "reps"):
                    r[k] = int(r[k])
                for k in ("B_nested", "B_flat", "B_census", "secs"):
                    r[k] = float(r[k])
                rows.append(r)
        done = {r["network"] for r in rows}
        names = [x for x in names if x not in done]
        print(f"  resuming: {len(rows)} already done, {len(names)} to go")

    for name in names:
        n_c = int(census[name]["n"])
        reps = args.reps if n_c <= args.reps_max_n else 1
        try:
            n, E = load_edges(name)
        except FileNotFoundError:
            print(f"  skip {name}: not in cache", file=sys.stderr)
            continue
        t0 = time.time()
        try:
            nested = [fit(n, E, True, s) for s in range(reps)]
            flat = [fit(n, E, False, s) for s in range(reps)]
        except Exception as exc:
            print(f"  FAIL {name}: {type(exc).__name__}: {exc}", file=sys.stderr)
            continue
        rows.append(dict(network=name, n=n, m=len(E), reps=reps,
                         B_nested=float(np.mean(nested)),
                         B_flat=float(np.mean(flat)),
                         B_nested_runs=str(nested), B_flat_runs=str(flat),
                         B_census=float(census[name]["B_mean"]),
                         secs=round(time.time() - t0, 1)))
        print(f"  {name:<42} n={n:<8} nested={np.mean(nested):<8.1f} "
              f"flat={np.mean(flat):<7.1f} census={census[name]['B_mean']:<7} "
              f"[{time.time() - t0:.0f}s]", flush=True)
        with open(args.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)

    report(rows)
    print(f"\n{len(rows)} networks -> {args.out}")
    return 0


def report(rows):
    if len(rows) < 3:
        return
    print(f"\nB ~ n^beta over {len(rows)} networks, "
          f"n = {min(r['n'] for r in rows)} to {max(r['n'] for r in rows)}")
    x = np.log([r["n"] for r in rows])
    for key, lab in (("B_census", "flat, as published"),
                     ("B_flat", "flat, refitted"),
                     ("B_nested", "nested")):
        y = np.log([max(r[key], 1e-9) for r in rows])
        A = np.vstack([x, np.ones_like(x)]).T
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        resid = y - A @ coef
        se = np.sqrt((resid ** 2).sum() / (len(x) - 2)
                     * np.linalg.inv(A.T @ A)[0, 0])
        # bootstrap CI, as the chapter quotes one
        rng = np.random.default_rng(0)
        bs = []
        for _ in range(2000):
            i = rng.integers(0, len(x), len(x))
            c, *_ = np.linalg.lstsq(np.vstack([x[i], np.ones(len(i))]).T,
                                    y[i], rcond=None)
            bs.append(c[0])
        lo, hi = np.percentile(bs, [2.5, 97.5])
        print(f"  {lab:<20} beta = {coef[0]:.3f} +- {se:.3f}   "
              f"95% CI [{lo:.2f}, {hi:.2f}]")


if __name__ == "__main__":
    sys.exit(main())
