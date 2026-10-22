"""Runs for the revision (referee report on manuscript.tex).

Rule (w) at net closure per arrival Delta = 1 and alpha_- = 0 unless stated.
Every network is reduced with trclust (largest triangle cluster) and the link
origins written by tc (.eorig).  Appends to data/rev_<plan>.csv (resumes).

  traj   beta_-/beta_+ = 0.70 ... 0.85, two seeds, followed to n = 1e7; the
         same seed stopped at smaller n is a prefix of the same trajectory, so
         P_triangle is measured along it at n = 1e4, 3e4, ..., 1e7.  The 1e7 .ts
         is kept in data/ts/rev_traj_*.ts.
  seeds  beta_-/beta_+ = 0.70 ... 0.80 with 64, 32, 16 seeds at n = 1e4, 1e5, 1e6
  hubs   beta_-/beta_+ = 0 ... 0.9 at n = 1e5, 1e6 (one seed): closure gain and
         opening loss of the ten largest hubs
  tree   beta_+ = alpha_+ with beta_- = 2 beta_+ (tree phase) and beta_- = beta_+
         (critical), n = 1e4 ... 1e7, four seeds: k_max and T
  keep   networks kept in SCRATCH/keep for core.py: beta_- = 0 at n = 1e4, 1e5,
         1e6 and beta_-/beta_+ = 0.5, 0.67, 0.8, 0.89 at 1e6
  long   second round: beta_-/beta_+ = 0.78, seeds 1 and 2, to n = 3e7 (the 1e7
         runs of traj are prefixes); .ts kept in data/ts/rev_long_*.ts
  tags   second round, Fig. 4(b): the opgrid lines (Delta, alpha_-/alpha_+) =
         (0.5, 0), (1, 0.5), (0.5, 0.5) at n = 1e6, seed 1, as in opgrid.py, with
         the measured link origins (the line (1, 0) is in rev_hubs.csv)
  tagsdel  second round: beta_- = 0, u = 1/2, alpha_-/alpha_+ as in delgrid.py
         at n = 1e6, seed 1, and rule (p) at u = 1/2 (x = "p"), with link origins

    python rev.py SCRATCHDIR traj|seeds|hubs|tree|keep|long|tags|tagsdel
"""
import os, sys, csv, shutil, subprocess
import numpy as np
from scan import HERE


def rates(x, D=1.0):
    """(u_+, u_-) for turnover x = beta_-/beta_+ at net closure per arrival D, alpha_- = 0."""
    rp = D / (1 - x); rm = rp - D; z = 1 + rp + rm
    return rp / z, rm / z


def run(pre, n, seed, up, um, d=0.0, mode="w"):
    args = [os.path.join(HERE, "tc"), mode, f"{up:.10f}", str(n), str(seed), pre]
    if um > 0 or d > 0: args.append(f"{um:.10f}")
    if d > 0: args.append(f"{d:.10f}")
    subprocess.run(args, check=True)


def reduce(pre, n):
    T, ncl, tmax, nmax, emax = map(int, subprocess.run(
        [os.path.join(HERE, "trclust"), pre + ".edges", str(n), pre + ".mem"],
        capture_output=True, text=True, check=True).stdout.split())
    o = np.fromfile(pre + ".eorig", np.uint8).astype(bool)
    m = np.fromfile(pre + ".mem", np.uint8).astype(bool)
    nd = np.loadtxt(pre + ".nodes", dtype=np.int64, usecols=(0,))
    s = dict(zip(*[iter(open(pre + ".sum").read().split())] * 2))
    E = len(o)
    e = np.fromfile(pre + ".edges", dtype=np.int32).reshape(-1, 2)
    club = {}
    for q in (1, 3):                                    # links among the q% highest-degree nodes
        top = np.zeros(len(nd), bool); top[np.argsort(-nd, kind="stable")[:len(nd) * q // 100]] = True
        club[f"club{q}"] = float((top[e[:, 0]] & top[e[:, 1]]).sum() / E)
    return dict(E=E, T=T, kmax=int(nd.max()), **club, fT=tmax / T if T else 0.0, fN=nmax / n, fE=emax / E,
                closure_share=o.mean(),                                   # closure links / all links
                closure_in=(o & m).sum() / o.sum() if o.sum() else 0.0,   # closure links inside the cluster
                arrival_in=(~o & m).sum() / (~o).sum(),                   # arrival links inside the cluster
                cluster_closure=(o & m).sum() / m.sum() if m.sum() else 0.0,
                failed=int(s["failed"]), failed_rem=int(s["failed_rem"]))


def hubs(pre):
    """Gain w_a/W_open and loss 2 t_a/(3T) of the ten largest hubs (per closure, per opening)."""
    nd = np.loadtxt(pre + ".nodes", dtype=np.int64)
    k, t, w = nd[:, 0], nd[:, 1].astype(float), nd[:, 3].astype(float)
    W, T3 = w.sum() / 2, t.sum()                        # sum_a w_a = 2 W_open, sum_a t_a = 3T
    top = np.argsort(-k, kind="stable")[:10]
    g, l = w[top] / W, 2 * t[top] / T3
    return dict(k_hub=float(k[top].mean()), gain=float(g.mean()), loss=float(l.mean()),
                R=float(np.median(g / np.where(l > 0, l, np.nan))))


def clean(pre):
    for ext in ("nodes", "edges", "ts", "sum", "eorig", "mem"):
        if os.path.exists(f"{pre}.{ext}"): os.remove(f"{pre}.{ext}")


def plan(which):
    if which == "traj":
        sizes = (10000, 30000, 100000, 300000, 1000000, 3000000, 10000000)
        return [(x, n, s) for x in (0.70, 0.74, 0.76, 0.78, 0.80, 0.85) for s in (1, 2) for n in sizes]
    if which == "seeds":
        return [(x, n, s) for n, ns in ((10000, 64), (100000, 32), (1000000, 16))
                for x in (0.70, 0.72, 0.74, 0.76, 0.78, 0.80) for s in range(1001, 1001 + ns)]
    if which == "hubs":
        return [(x, n, 1) for n in (100000, 1000000) for x in (0.0, 0.3, 0.5, 0.6, 0.67, 0.7, 0.72, 0.74,
                                                                0.76, 0.78, 0.8, 0.85, 0.9)]
    if which == "tree":
        return [(kind, n, s) for kind in ("tree", "crit") for n in (10000, 100000, 1000000, 10000000)
                for s in (1, 2, 3, 4)]
    if which == "keep":
        return [(0.0, n, 1) for n in (10000, 100000, 1000000)] + [(x, 1000000, 1) for x in (0.5, 0.67, 0.8, 0.89)]
    if which == "long":
        return [(0.78, 30000000, s) for s in (1, 2)]
    if which == "tags":
        lines = {(0.5, 0.0): (0.0, 0.5, 0.66, 0.67, 0.68, 0.7, 0.72, 0.74, 0.75, 0.76, 0.8, 0.85, 0.9),
                 (1.0, 0.5): (0.0, 0.5, 0.67, 0.75, 0.8, 0.84, 0.85, 0.86, 0.88, 0.9),
                 (0.5, 0.5): (0.0, 0.5, 0.67, 0.75, 0.8, 0.85, 0.9)}
        return [(f"{D}_{a}_{x}", 1000000, 1) for (D, a), xs in lines.items() for x in xs]
    if which == "tagsdel":
        return [(f"del_{a}", 1000000, 1) for a in (0.0, 0.25, 0.5, 0.75, 0.9, 0.95)] + [("p", 1000000, 1)]
    raise SystemExit("plan: traj, seeds, hubs, tree, keep, long, tags or tagsdel")


def main():
    scratch, which = sys.argv[1], sys.argv[2]; os.makedirs(scratch, exist_ok=True)
    out = os.path.join(HERE, "data", f"rev_{which}.csv")
    new = not os.path.exists(out)
    done = set() if new else {tuple(r[k] for k in ("x", "n", "seed")) for r in csv.DictReader(open(out))}
    f = open(out, "a", newline=""); wr = None
    for x, n, seed in plan(which):
        if (str(x), str(n), str(seed)) in done: continue
        pre = os.path.join(scratch, f"{which}_{x}_{n}_{seed}")
        d, mode = 0.0, "w"
        if which == "tagsdel":                          # as in delgrid.py and rulesclust.py
            if x == "p": up, um, mode = 0.5, 0.0, "p"
            else:
                a = float(x.split("_")[1]); z = 2 + a
                up, um, d = 1 / z, 0.0, a / z
        elif which == "tree":
            up, um = (0.25, 0.5) if x == "tree" else (1 / 3, 1 / 3)
        elif which == "tags":                           # rates over alpha_+ as in opgrid.py
            D, a, r = map(float, x.split("_"))
            rp = D / (1 - r); rm = rp - D; z = 1 + a + rp + rm
            up, um, d = rp / z, rm / z, a / z
        else:
            up, um = rates(x)
        run(pre, n, seed, up, um, d, mode)
        row = dict(x=x, n=n, seed=seed, **reduce(pre, n))
        if which == "hubs": row.update(hubs(pre))
        if which == "traj" and n == 10000000:
            shutil.copy(pre + ".ts", os.path.join(HERE, "data", "ts", f"rev_traj_x{x}_s{seed}.ts"))
        if which == "long":
            shutil.copy(pre + ".ts", os.path.join(HERE, "data", "ts", f"rev_long_x{x}_s{seed}.ts"))
        if which == "keep":
            os.makedirs(os.path.join(scratch, "keep"), exist_ok=True)
            for ext in ("nodes", "edges", "eorig", "mem"):
                shutil.copy(f"{pre}.{ext}", os.path.join(scratch, "keep", f"x{x}_n{n}.{ext}"))
        if wr is None:
            wr = csv.DictWriter(f, list(row))
            if new: wr.writeheader()
        wr.writerow(row); f.flush()
        clean(pre)
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)


if __name__ == "__main__":
    main()
