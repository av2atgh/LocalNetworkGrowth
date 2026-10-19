"""Closure plus triangle opening at fixed density <k> = 4: beta_+ = alpha + beta_-.

For each rule and r_- = beta_-/alpha runs ./tc with u_+ = r_+/(1+r_++r_-),
u_- = r_-/(1+r_++r_-), reduces each network with scan.reduce and appends a
row to data/scan_rm.csv.

    python scan_rm.py SCRATCHDIR
"""
import os, sys, subprocess, csv
from scan import reduce, HERE

OUT = os.path.join(HERE, "data", "scan_rm.csv")
RM = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)


def main():
    scratch = sys.argv[1]
    os.makedirs(scratch, exist_ok=True)
    new = not os.path.exists(OUT)
    done = set() if new else {(r["mode"], float(r["r_minus"]), int(r["n"]), int(r["seed"]))
                              for r in csv.DictReader(open(OUT))}   # resume
    f = open(OUT, "a", newline=""); wr = None
    for n, seeds in ((100000, 2), (1000000, 1)):
        for mode in "wpcf":
            for rm in RM:
                rp = 1 + rm; z = 1 + rp + rm
                up, um = rp / z, rm / z
                for seed in range(1, seeds + 1):
                    if (mode, rm, n, seed) in done: continue
                    pre = os.path.join(scratch, f"{mode}_{rm}_{n}_{seed}")
                    subprocess.run([os.path.join(HERE, "tc"), mode, f"{up:.10f}", str(n), str(seed), pre,
                                    f"{um:.10f}"], check=True)
                    row = reduce(pre, mode, up, n, seed)
                    toks = open(pre + ".sum").read().split()
                    s = dict(zip(toks[0::2], toks[1::2]))
                    row = dict(r_minus=rm, r_plus=rp, u_minus=um, **row,
                               failed_rem=int(s["failed_rem"]))
                    if wr is None:
                        wr = csv.DictWriter(f, list(row))
                        if new: wr.writeheader()
                    wr.writerow(row); f.flush()
                    for ext in ("nodes", "edges"): os.remove(f"{pre}.{ext}")
                    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)


if __name__ == "__main__":
    main()
