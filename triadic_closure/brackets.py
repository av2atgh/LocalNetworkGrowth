"""All brackets of the condensation boundary of rule (w) (revision, comments 5 and 6).

Pools data/opgrid.csv and data/bnd.csv.  A point (Delta, alpha_-/alpha_+,
beta_-/beta_+) is condensed when the seed mean of P_triangle rises between
n = 1e5 and 1e6 or exceeds 0.9 at 1e6, dispersed when it falls.  For every line
of fixed Delta and alpha_-/alpha_+ the bracket is the last condensed and the
first dispersed turnover; at its ends the closure and opening rates per net
arrival, beta_+-/(alpha_+ - alpha_-), and the mean degree are given.
Writes data/rev_points.csv (every point, seeds per size) and data/rev_brackets.csv.

    python brackets.py
"""
import csv, os
from collections import defaultdict
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    g = defaultdict(list)
    for f in ("opgrid.csv", "bnd.csv"):
        for r in csv.DictReader(open(os.path.join(HERE, "data", f))):
            g[(float(r["D"]), float(r["a"]), float(r["ratio"]), int(r["n"]))].append(float(r["fT"]))
    pts = defaultdict(dict)
    for (D, a, x, n), v in g.items():
        pts[(D, a, x)][n] = (len(v), np.mean(v), np.std(v, ddof=1) / np.sqrt(len(v)) if len(v) > 1 else np.nan)
    rows = []
    for (D, a, x), d in sorted(pts.items()):
        if 100000 not in d or 1000000 not in d: continue
        lo, hi = d[100000], d[1000000]
        rows.append(dict(D=D, a=a, x=x, seeds5=lo[0], P5=lo[1], err5=lo[2], seeds6=hi[0], P6=hi[1], err6=hi[2],
                         cond=int(hi[1] > lo[1] or hi[1] > 0.9)))
    with open(os.path.join(HERE, "data", "rev_points.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, list(rows[0])); w.writeheader(); w.writerows(rows)
    br = []
    for D, a in sorted({(r["D"], r["a"]) for r in rows}):
        line = [r for r in rows if r["D"] == D and r["a"] == a]
        c = [r["x"] for r in line if r["cond"]]; s = [r["x"] for r in line if not r["cond"]]
        lo = max(c) if c else np.nan; hi = min([x for x in s if x > lo]) if s and c else np.nan
        flips = sum(line[i]["cond"] != line[i + 1]["cond"] for i in range(len(line) - 1))
        mid = (lo + hi) / 2 if np.isfinite(hi) else np.nan
        bp = D / (1 - mid) / (1 - a) if np.isfinite(mid) else np.nan
        br.append(dict(D=D, a=a, lo=lo, hi=hi, flips=flips, points=len(line), beta_plus_net=bp,
                       beta_minus_net=bp * mid if np.isfinite(mid) else np.nan, kmean=2 * (1 + D) / (1 + a)))
    with open(os.path.join(HERE, "data", "rev_brackets.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, list(br[0])); w.writeheader(); w.writerows(br)
    for r in br: print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})


if __name__ == "__main__":
    main()
