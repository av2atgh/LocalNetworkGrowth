"""Locating the condensation boundary with the triangle-cluster order parameter.

  fine    finer beta_-/beta_+ around the boundaries of opgrid.py, Delta = 0.5, 1,
          alpha_-/alpha_+ = 0, 0.5; n = 1e5 (4 seeds), 1e6 (2 seeds)
  alpha   Delta = 1, alpha_-/alpha_+ = 0.1 ... 0.9, beta_-/beta_+ = 0.7 ... 0.95;
          n = 1e5 (2 seeds), 1e6 (1 seed)
  crit    Delta = 1, alpha_- = 0, beta_-/beta_+ = 0.76 at n = 1e7

Observables as in delgrid.reduce.  Appends to data/bnd.csv (resumes).

    python bnd.py SCRATCHDIR fine|alpha|crit
"""
import os, sys, csv, subprocess
from scan import HERE
from delgrid import reduce

OUT = os.path.join(HERE, "data", "bnd.csv")


def plan(which):
    if which == "fine":
        pts = [(0.5, r, 0.0) for r in (0.66, 0.68, 0.70, 0.72, 0.74, 0.76)]
        pts += [(1.0, r, 0.0) for r in (0.70, 0.72, 0.74, 0.76, 0.78)]
        pts += [(D, r, 0.5) for D in (0.5, 1.0) for r in (0.84, 0.86, 0.88, 0.90)]
        return [(D, r, a, n, s) for n, ns in ((100000, 4), (1000000, 2)) for D, r, a in pts
                for s in range(1, ns + 1)]
    if which == "alpha":
        return [(1.0, r, a, n, s) for n, ns in ((100000, 2), (1000000, 1))
                for a in (0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9)
                for r in (0.7, 0.75, 0.8, 0.85, 0.9, 0.95) for s in range(1, ns + 1)]
    if which == "crit":
        return [(1.0, 0.76, 0.0, 10000000, 1)]
    raise SystemExit("plan: fine, alpha or crit")


def main():
    scratch, which = sys.argv[1], sys.argv[2]; os.makedirs(scratch, exist_ok=True)
    new = not os.path.exists(OUT)
    done = set() if new else {(float(r["D"]), float(r["ratio"]), float(r["a"]), int(r["n"]), int(r["seed"]))
                              for r in csv.DictReader(open(OUT))}
    f = open(OUT, "a", newline=""); wr = None
    for D, ratio, a, n, seed in plan(which):
        if (D, ratio, a, n, seed) in done: continue
        rp = D / (1 - ratio); rm = rp - D; z = 1 + a + rp + rm       # rates over alpha_+
        pre = os.path.join(scratch, f"bnd_{D}_{ratio}_{a}_{n}_{seed}")
        args = [os.path.join(HERE, "tc"), "w", f"{rp / z:.10f}", str(n), str(seed), pre, f"{rm / z:.10f}"]
        if a > 0: args.append(f"{a / z:.10f}")
        subprocess.run(args, check=True)
        row = dict(plan=which, D=D, ratio=ratio, a=a, n=n, seed=seed, **reduce(pre, n, rp, a))
        row["kpred"] = 2 * (1 + D) / (1 + a)
        if wr is None:
            wr = csv.DictWriter(f, list(row))
            if new: wr.writeheader()
        wr.writerow(row); f.flush()
        for ext in ("nodes", "edges", "ts", "sum"): os.remove(f"{pre}.{ext}")
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)


if __name__ == "__main__":
    main()
