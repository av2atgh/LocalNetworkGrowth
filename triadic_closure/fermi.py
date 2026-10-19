"""Pair occupation against birth rank, rule (w).

Node ids are birth ranks.  For the 10 oldest nodes, the occupation P(j) of the
pairs they form with node j is fitted to a Fermi-Dirac form in eps = ln j,
P = 1/(exp((ln j - ln j*)/T) + 1), over the bins with 0 < P and j < n/4.5.
Also records the occupation among the 30 oldest nodes (degeneracy) and among the
first sqrt(n).  Appends to data/fermi.csv (resumes).

    python fermi.py SCRATCHDIR
"""
import os, sys, csv, subprocess
import numpy as np
from scipy.optimize import curve_fit

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data", "fermi.csv")


def plan():
    """(label, u_+, u_-, n, seed): size scan at u = 1/2, u scan, turnover at <k> = 4."""
    p = [("size", 0.5, 0.0, n, s) for n in (10000, 30000, 100000, 300000) for s in (1, 2, 3)]
    p += [("size", 0.5, 0.0, n, 1) for n in (1000000, 3000000)]
    p += [("u", u, 0.0, n, 1) for u in (0.3, 0.4, 0.6) for n in (100000, 1000000)]
    for rm in (0.1, 0.2, 0.3, 0.5, 1.0):                # beta_+ = alpha + beta_-
        rp = 1 + rm; z = 1 + rp + rm
        p += [("turnover", rp / z, rm / z, n, 1) for n in (100000, 1000000)]
    return p


def occupation(pre):
    e = np.fromfile(pre + ".edges", dtype=np.int32).reshape(-1, 2).astype(np.int64)
    n = e.max() + 1
    lo, hi = np.minimum(e[:, 0], e[:, 1]), np.maximum(e[:, 0], e[:, 1])
    occ = lambda m: np.sum(hi < m) / (m * (m - 1) / 2)          # density among the m oldest
    j = hi[lo < 10]
    edges = np.unique(np.floor(np.logspace(1, np.log10(n), 25)).astype(int))
    cnt, _ = np.histogram(j, bins=edges); N = 10 * np.diff(edges); P = cnt / N
    x = np.log(np.sqrt(edges[:-1] * edges[1:])); ok = (P > 0) & (x < np.log(n) - 1.5)
    sig = np.sqrt(np.maximum(P * (1 - P), 1 / N) / N)
    fd = lambda x, m, T: 1 / (np.exp((x - m) / T) + 1)
    try:
        (m, T), _ = curve_fit(fd, x[ok], P[ok], p0=(np.log(n) / 2, 0.8), sigma=sig[ok], maxfev=20000)
        chi = np.sum(((fd(x[ok], m, T) - P[ok]) / sig[ok]) ** 2) / (ok.sum() - 2)
    except RuntimeError:
        m = T = chi = float("nan")
    return dict(occ30=occ(30), occ_sqrt=occ(int(np.sqrt(n))), P_max=P.max(), jstar=np.exp(m), T=T, chi2=chi)


def main():
    scratch = sys.argv[1]; os.makedirs(scratch, exist_ok=True)
    new = not os.path.exists(OUT)
    done = set() if new else {(r["label"], float(r["u_plus"]), float(r["u_minus"]), int(r["n"]), int(r["seed"]))
                              for r in csv.DictReader(open(OUT))}
    f = open(OUT, "a", newline=""); wr = None
    for lab, up, um, n, s in plan():
        if (lab, round(up, 6), round(um, 6), n, s) in {(a, round(b, 6), round(c, 6), d, e) for a, b, c, d, e in done}:
            continue
        pre = os.path.join(scratch, f"fermi_{lab}_{up:.4f}_{um:.4f}_{n}_{s}")
        args = [os.path.join(HERE, "tc"), "w", f"{up:.10f}", str(n), str(s), pre]
        if um > 0: args.append(f"{um:.10f}")
        subprocess.run(args, check=True)
        row = dict(label=lab, u_plus=up, u_minus=um, n=n, seed=s, **occupation(pre))
        if wr is None:
            wr = csv.DictWriter(f, list(row))
            if new: wr.writeheader()
        wr.writerow(row); f.flush()
        for ext in ("nodes", "edges", "ts", "sum"): os.remove(f"{pre}.{ext}")
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)


if __name__ == "__main__":
    main()
