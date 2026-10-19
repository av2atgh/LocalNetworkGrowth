"""Scan the decoupled triadic-closure model over u = beta/(alpha+beta) and n.

Runs ./tc for each (mode, u, n, seed) into a scratch directory, reduces each
network to one row of summary statistics, and appends it to data/scan.csv.

    python scan.py SCRATCHDIR
"""
import os, sys, subprocess, csv
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data", "scan.csv")
FIELDS = ["mode", "u", "n", "seed", "kmean", "transitivity", "assort", "meanC",
          "kmax", "beta_age", "ccdf_slope", "core_frac", "core_density",
          "top1pc_frac", "wratio_hub"]


def reduce(pre, mode, u, n, seed):
    k, t, knn, w = np.loadtxt(pre + ".nodes", dtype=np.int64).T
    e = np.fromfile(pre + ".edges", dtype=np.int32).reshape(-1, 2)
    toks = open(pre + ".sum").read().split()
    s = dict(zip(toks[0::2], toks[1::2]))
    N, E = len(k), len(e)
    C = np.where(k > 1, 2 * t / np.maximum(k * (k - 1), 1), 0).mean()
    # degree growth with age: <k> of nodes born in log bins of birth index
    b = np.unique(np.logspace(1, np.log10(N), 25).astype(int))
    x = np.sqrt(b[:-1] * b[1:]); mk = np.array([k[b[i]:b[i + 1]].mean() for i in range(len(b) - 1)])
    sel = (x > 100) & (x < N / 10)
    beta = -np.polyfit(np.log(x[sel]), np.log(mk[sel]), 1)[0]
    ks = np.sort(k); cc = 1 - np.arange(N) / N
    sel = (ks >= 10) & (ks <= max(30, k.max() / 10))
    slope = np.polyfit(np.log(ks[sel]), np.log(cc[sel]), 1)[0] if sel.sum() > 20 else np.nan
    o = np.argsort(-k)

    def club(sz):
        m = np.zeros(N, bool); m[o[:sz]] = True
        ein = (m[e[:, 0]] & m[e[:, 1]]).sum()
        return ein / E, 2 * ein / (sz * (sz - 1))
    core_frac, core_den = club(int(np.sqrt(N)))          # sqrt(n) richest nodes
    top1, _ = club(max(2, N // 100))
    # open-triad weight of hubs relative to the uncorrelated value k * 2W/(n<k>)
    W2 = w.sum()
    wr = (w[o[:100]] / (k[o[:100]] * W2 / k.sum())).mean()
    return dict(mode=mode, u=u, n=n, seed=seed, kmean=float(s["kmean"]),
                transitivity=float(s["transitivity"]), assort=float(s["assort"]),
                meanC=C, kmax=int(k.max()), beta_age=beta, ccdf_slope=slope,
                core_frac=core_frac, core_density=core_den, top1pc_frac=top1,
                wratio_hub=wr)


def main():
    scratch = sys.argv[1]
    os.makedirs(scratch, exist_ok=True)
    new = not os.path.exists(OUT)
    f = open(OUT, "a", newline=""); wr = csv.DictWriter(f, FIELDS)
    if new: wr.writeheader()
    jobs = [("l", 0.5)] + [(m, u) for m in "wp" for u in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7)]
    for n, seeds in ((10000, 4), (100000, 2), (1000000, 1)):
        for mode, u in jobs:
            for seed in range(1, seeds + 1):
                pre = os.path.join(scratch, f"{mode}_{u}_{n}_{seed}")
                subprocess.run([os.path.join(HERE, "tc"), mode, str(u), str(n), str(seed), pre], check=True)
                row = reduce(pre, mode, u, n, seed)
                wr.writerow(row); f.flush()
                for ext in ("nodes", "edges"): os.remove(f"{pre}.{ext}")
                print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)


if __name__ == "__main__":
    main()
