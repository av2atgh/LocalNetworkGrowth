"""How concentrated are the links?  phi(m) = fraction of links with both ends
among the m highest-degree nodes, for m = n^a, a = 0.5 ... 0.9.

A condensate holds a non-vanishing phi(n^a) for some a < 1 as n grows.  Runs
the rules along <k> = 4 (beta_+ = alpha + beta_-) at several r_- and n.

    python club.py SCRATCHDIR    ->  data/club.csv
"""
import os, sys, subprocess, csv
import numpy as np
from scan import HERE

OUT = os.path.join(HERE, "data", "club.csv")
AS = (0.5, 0.6, 0.7, 0.8, 0.9)
JOBS = [("w", 0.0), ("w", 0.5), ("w", 1.0), ("w", 2.0), ("w", 4.0), ("p", 0.0), ("p", 1.0),
        ("f", 0.0), ("c", 0.0), ("l", 0.0), ("f", 4.0), ("w", 8.0)]


def phis(pre):
    k = np.loadtxt(pre + ".nodes", usecols=0, dtype=np.int64)
    e = np.fromfile(pre + ".edges", dtype=np.int32).reshape(-1, 2)
    N = len(k); rank = np.empty(N, np.int64); rank[np.argsort(-k, kind="stable")] = np.arange(N)
    lo = np.maximum(rank[e[:, 0]], rank[e[:, 1]])        # a link is inside the top m iff lo < m
    return {f"phi_{a}": float((lo < int(N ** a)).mean()) for a in AS}, int(k.max())


def main():
    scratch = sys.argv[1]; os.makedirs(scratch, exist_ok=True)
    new = not os.path.exists(OUT)
    done = set() if new else {(r["mode"], float(r["r_minus"]), int(r["n"]), int(r["seed"]))
                              for r in csv.DictReader(open(OUT))}   # resume
    f = open(OUT, "a", newline="")
    wr = csv.DictWriter(f, ["mode", "r_minus", "n", "seed", "kmax"] + [f"phi_{a}" for a in AS])
    if new: wr.writeheader()
    for n, seeds in ((10000, 4), (100000, 2), (1000000, 1)):
        for mode, rm in JOBS:
            rp = 1 + rm; z = 1 + rp + rm
            for seed in range(1, seeds + 1):
                if (mode, rm, n, seed) in done: continue
                pre = os.path.join(scratch, f"club_{mode}_{rm}_{n}_{seed}")
                subprocess.run([os.path.join(HERE, "tc"), mode, f"{rp / z:.10f}", str(n), str(seed),
                                pre, f"{rm / z:.10f}"], check=True)
                ph, kmax = phis(pre)
                wr.writerow(dict(mode=mode, r_minus=rm, n=n, seed=seed, kmax=kmax, **ph)); f.flush()
                for ext in ("nodes", "edges"): os.remove(f"{pre}.{ext}")
                print(mode, rm, n, seed, kmax, {k: round(v, 3) for k, v in ph.items()}, flush=True)


if __name__ == "__main__":
    main()
