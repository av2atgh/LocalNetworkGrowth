"""Onset size n*(u) of the condensation: first N at which T/N exceeds theta.

Reads data/onset/*.ts and data/ts/w_*.ts, writes data/onset.csv and fits
ln n* against 1/u (essential singularity) and against ln u (power law).

    python onset.py
"""
import glob, os, re, csv
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def nstar(path, theta):
    d = np.loadtxt(path, skiprows=1)
    N, T = d[:, 0], d[:, 3]
    hit = np.nonzero((T / N >= theta) & (N >= 100))[0]
    return N[hit[0]] if len(hit) else np.nan, N[-1]


def main():
    rows = []
    files = glob.glob(os.path.join(HERE, "data/onset/w_*.ts")) + glob.glob(os.path.join(HERE, "data/ts/w_*.ts"))
    for p in files:
        u = float(re.search(r"_u([\d.]+)_", p).group(1))
        for theta in (1.0, 3.0):
            ns, nmax = nstar(p, theta)
            rows.append(dict(u=u, theta=theta, file=os.path.basename(p), nstar=ns, nmax=nmax))
    with open(os.path.join(HERE, "data/onset.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, rows[0].keys()); w.writeheader(); w.writerows(rows)
    for theta in (1.0, 3.0):
        print(f"theta = {theta}")
        us = sorted({r["u"] for r in rows})
        U, L = [], []
        for u in us:
            v = np.array([r["nstar"] for r in rows if r["u"] == u and r["theta"] == theta])
            ok = v[np.isfinite(v)]
            print(f"  u={u:<5} runs={len(v)} reached={len(ok)}  ln n* = "
                  + (f"{np.log(ok).mean():.2f} +- {np.log(ok).std():.2f}" if len(ok) else "--"))
            if len(ok) == len(v) and len(ok) >= 3: U.append(u); L.append(np.log(ok).mean())
        U, L = np.array(U), np.array(L)
        if len(U) >= 3:
            for name, x in (("1/u", 1 / U), ("(1-u)/u", (1 - U) / U), ("ln u", np.log(U))):
                c, res, *_ = np.polyfit(x, L, 1, full=True)
                print(f"  ln n* = {c[1]:.2f} + {c[0]:.3f} {name}   rms {np.sqrt(res[0] / len(U)) if len(res) else 0:.3f}")


if __name__ == "__main__":
    main()
