"""Onset of condensation measured relative to the plateau (revision, comment 3).

For each trajectory (data/onset/w_*.ts, data/ts/w_u*_n1e7_s1.ts, and at u = 0.05
data/ts/w_u0.05_n5e7_s*.ts), the plateau
level is the running minimum of T/N for N >= 1e3, and the onset n2* is the
first N at which T/N exceeds twice that minimum.  On the plateau the caricature
of Sec. VI uses epsilon = r_+/w, with r_+ = u/(1-u) and w = (Wall - 3T)/N the
open triads per node at the running minimum.  Fits ln n2* against ln u,
against 1/u and against 1/epsilon.  Writes data/rev_onset.csv.

    python onset2.py
"""
import glob, os, re, csv
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def onset(path):
    d = np.loadtxt(path, skiprows=1)
    N, T, W = d[:, 0], d[:, 3], d[:, 4]
    y = T / N
    ok = N >= 1000
    best, at = np.inf, None
    for i in np.nonzero(ok)[0]:
        if y[i] < best: best, at = y[i], i
        if y[i] >= 2 * best:
            return N[i], best, (W[at] - 3 * T[at]) / N[at], N[-1]
    return np.nan, best, (W[at] - 3 * T[at]) / N[at], N[-1]


def fit(x, y):
    c = np.polyfit(x, y, 1)
    return c, np.sqrt(np.mean((np.polyval(c, x) - y) ** 2))


def main():
    files = glob.glob(os.path.join(HERE, "data/onset/w_*.ts")) + glob.glob(os.path.join(HERE, "data/ts/w_u*_n1e7_s1.ts"))
    long = glob.glob(os.path.join(HERE, "data/ts/w_u0.05_n5e7_s*.ts"))      # second round: u = 0.05 to 5e7
    if long: files = [p for p in files if "w_u0.05_n1e7" not in p] + long   # the 1e7 run is a prefix of seed 1
    rows = []
    for p in sorted(files):
        u = float(re.search(r"_u([\d.]+)_", p).group(1))
        ns, plateau, w, nmax = onset(p)
        rows.append(dict(u=u, file=os.path.basename(p), nstar2=ns, plateau=plateau, w_open=w,
                         eps=u / (1 - u) / w, nmax=nmax))
    with open(os.path.join(HERE, "data/rev_onset.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, list(rows[0])); wr.writeheader(); wr.writerows(rows)
    U, L, EPS = [], [], []
    for u in sorted({r["u"] for r in rows}):
        rs = [r for r in rows if r["u"] == u]
        v = np.array([r["nstar2"] for r in rs]); good = v[np.isfinite(v)]
        e = np.mean([r["eps"] for r in rs]); pl = np.mean([r["plateau"] for r in rs])
        print(f"u={u:<5} runs={len(v)} reached={len(good)} plateau T/N={pl:.3f} w={np.mean([r['w_open'] for r in rs]):.2f} "
              f"eps={e:.4f} " + (f"ln n2* = {np.log(good).mean():.2f} +- {np.log(good).std():.2f}" if len(good) else ""))
        if len(good) == len(v) and len(good) and 0.08 <= u <= 0.3:
            U.append(u); L.append(np.log(good).mean()); EPS.append(e)
    U, L, EPS = map(np.array, (U, L, EPS))
    for name, x in (("power law, ln u", np.log(U)), ("essential, 1/u", 1 / U), ("caricature, 1/eps", 1 / EPS)):
        c, rms = fit(x, L)
        print(f"{name:<20} slope {c[0]:.3f} intercept {c[1]:.2f} rms {rms:.3f}")
    print("eps/u:", np.round(EPS / U, 3))


if __name__ == "__main__":
    main()
