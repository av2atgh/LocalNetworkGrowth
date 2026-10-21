"""Rule (w) with node deletion, beta_- = 0: does the condensate survive turnover
of nodes?

Parameters: u = beta_+/(alpha_+ + beta_+) and a = alpha_-/alpha_+ < 1.  Each
network is reduced to triangle-explicit observables, with no club-size choice:
  Y        inverse participation ratio of triangles over nodes, sum t_i^2/(sum t_i)^2
  trclust  edge-connected triangle clusters: fractions of triangles, nodes and
           links in the largest one (trclust.c)
plus <k> against 2(alpha_+ + beta_+)/(alpha_+ + alpha_-), k_max, and the density
and age (born / arrivals) of the sqrt(n) highest-degree nodes.
Appends to data/delgrid.csv (resumes).

    python delgrid.py SCRATCHDIR
"""
import os, sys, csv, subprocess
import numpy as np
from scan import HERE

OUT = os.path.join(HERE, "data", "delgrid.csv")
US = (0.3, 0.5)
AS = (0.0, 0.25, 0.5, 0.75, 0.9, 0.95)
SIZES = ((10000, 2), (100000, 2), (1000000, 1))


def reduce(pre, n, rp, a):
    nd = np.loadtxt(pre + ".nodes", dtype=np.int64)
    k, t, born = nd[:, 0], nd[:, 1].astype(float), nd[:, 4]
    s = dict(zip(*[iter(open(pre + ".sum").read().split())] * 2))
    T, ncl, tmax, nmax, emax = map(int, subprocess.run([os.path.join(HERE, "trclust"), pre + ".edges", str(n)],
                                                       capture_output=True, text=True, check=True).stdout.split())
    e = np.fromfile(pre + ".edges", dtype=np.int32).reshape(-1, 2)
    m = int(np.sqrt(n)); top = np.argsort(-k, kind="stable")[:m]
    intop = np.zeros(n, bool); intop[top] = True
    links_top = int((intop[e[:, 0]] & intop[e[:, 1]]).sum())
    arrivals = int(born.max()) + 1                      # born counts every node ever added
    return dict(E=len(e), kmean=float(s["kmean"]), kpred=2 * (1 + rp) / (1 + a), kmax=int(k.max()), T=T,
                Y=float((t ** 2).sum() / t.sum() ** 2) if t.sum() else float("nan"),
                clusters=ncl, fT=tmax / T if T else 0.0, fN=nmax / n, fE=emax / len(e),
                rho_sqrt=links_top / (m * (m - 1) / 2), phi_sqrt=links_top / len(e),
                age_sqrt=float(np.median(born[top]) / arrivals), deletions=int(s["deletions"]))


def main():
    scratch = sys.argv[1]; os.makedirs(scratch, exist_ok=True)
    new = not os.path.exists(OUT)
    done = set() if new else {(float(r["u"]), float(r["a"]), int(r["n"]), int(r["seed"]))
                              for r in csv.DictReader(open(OUT))}
    f = open(OUT, "a", newline=""); wr = None
    for n, ns in SIZES:
        for u in US:
            for a in AS:
                for seed in range(1, ns + 1):
                    if (u, a, n, seed) in done: continue
                    rp = u / (1 - u); z = 1 + a + rp                  # beta_+/alpha_+, total rate / alpha_+
                    pre = os.path.join(scratch, f"del_{u}_{a}_{n}_{seed}")
                    args = [os.path.join(HERE, "tc"), "w", f"{rp / z:.10f}", str(n), str(seed), pre, "0"]
                    if a > 0: args.append(f"{a / z:.10f}")
                    subprocess.run(args, check=True)
                    row = dict(u=u, a=a, n=n, seed=seed, **reduce(pre, n, rp, a))
                    if wr is None:
                        wr = csv.DictWriter(f, list(row))
                        if new: wr.writeheader()
                    wr.writerow(row); f.flush()
                    for ext in ("nodes", "edges", "ts", "sum"): os.remove(f"{pre}.{ext}")
                    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)


if __name__ == "__main__":
    main()
