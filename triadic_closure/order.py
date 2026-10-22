"""Order parameter without a club-size exponent.

For each network the full club curve phi(m), the fraction of links with both
ends among the m highest-degree nodes, is recorded on a logarithmic grid of m.
From it:
  m_half   the smallest m with phi(m) >= Delta/[2(1+Delta)], half the share
           of the net closure links; condensation <=> m_half/n -> 0, i.e.
           m_half ~ n^{a*} with a* < 1, while a normal network has m_half ~ n;
  phi_eps  phi(eps n) for eps = 1e-3, 1e-2, 1e-1, whose eps -> 0 limit after
           n -> infinity is the condensate share.
Rule (w) on a grid of net growth Delta and turnover beta_-/beta_+, plus the
local rules (f) and LS(n,1) at u = 1/2 as normal baselines.
Appends to data/order.csv (resumes).

    python order.py SCRATCHDIR [D ...]
"""
import os, sys, csv, subprocess
import numpy as np
from scan import HERE

OUT = os.path.join(HERE, "data", "order.csv")
DS = (0.25, 0.5, 1.0, 2.0)
RATIOS = (0.0, 0.5, 0.67, 0.75, 0.8, 0.85, 0.9, 0.95)
SIZES = ((10000, 3), (100000, 2), (1000000, 1))
EPS = (1e-3, 1e-2, 1e-1)


def curve(pre, D):
    k = np.loadtxt(pre + ".nodes", usecols=0, dtype=np.int64)
    e = np.fromfile(pre + ".edges", dtype=np.int32).reshape(-1, 2)
    n = len(k); rank = np.empty(n, np.int64); rank[np.argsort(-k, kind="stable")] = np.arange(n)
    lo = np.sort(np.maximum(rank[e[:, 0]], rank[e[:, 1]]))   # a link is inside the top m iff lo < m
    phi = lambda m: np.searchsorted(lo, m) / len(lo)
    ms = np.unique(np.logspace(0, np.log10(n), 60).astype(np.int64))
    target = D / (2 * (1 + D))
    m_half = int(lo[int(np.ceil(target * len(lo))) - 1]) + 1    # smallest m with phi(m) >= target
    return dict(m_half=m_half, **{f"phi_eps{x:g}": phi(int(x * n)) for x in EPS},
                curve=" ".join(f"{m}:{phi(m):.5f}" for m in ms))


def main():
    scratch = sys.argv[1]; os.makedirs(scratch, exist_ok=True)
    Ds = [float(x) for x in sys.argv[2:]] or list(DS)
    plan = [("w", D, r, n, s) for D in Ds for r in RATIOS for n, ns in SIZES for s in range(1, ns + 1)]
    if not sys.argv[2:]:
        plan += [(m, 1.0, 0.0, n, s) for m in "fl" for n, ns in SIZES for s in range(1, ns + 1)]
    new = not os.path.exists(OUT)
    done = set() if new else {(r["mode"], float(r["D"]), float(r["ratio"]), int(r["n"]), int(r["seed"]))
                              for r in csv.DictReader(open(OUT))}
    f = open(OUT, "a", newline=""); wr = None
    for mode, D, ratio, n, s in plan:
        if (mode, D, ratio, n, s) in done: continue
        rp = D / (1 - ratio); rm = rp - D; z = 1 + rp + rm         # beta_+/alpha, beta_-/alpha
        pre = os.path.join(scratch, f"order_{mode}_{D}_{ratio}_{n}_{s}")
        args = [os.path.join(HERE, "tc"), mode, f"{rp / z:.10f}", str(n), str(s), pre]
        if rm > 0: args.append(f"{rm / z:.10f}")
        subprocess.run(args, check=True, env=dict(os.environ))
        row = dict(mode=mode, D=D, ratio=ratio, n=n, seed=s, **curve(pre, D))
        if wr is None:
            wr = csv.DictWriter(f, list(row))
            if new: wr.writeheader()
        wr.writerow(row); f.flush()
        for ext in ("nodes", "edges", "ts", "sum"): os.remove(f"{pre}.{ext}")
        print(mode, D, ratio, n, s, row["m_half"], round(row["m_half"] / n, 5),
              [round(row[f"phi_eps{x:g}"], 3) for x in EPS], flush=True)


if __name__ == "__main__":
    main()
