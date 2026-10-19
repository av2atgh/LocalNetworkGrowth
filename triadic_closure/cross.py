"""Crossing beta_+ = beta_-: fixed r_+ = beta_+/alpha = 1, r_- = beta_-/alpha varied.

For r_- < r_+ triangles accumulate and <k> = 2(1 + r_+ - r_-); for r_- > r_+
removals are limited by the triangles available, the network stays a tree
(<k> -> 2) and closure-removal pairs rewire it.  Reduces each network with
scan.reduce plus the club fractions of club.phis, appends to data/cross.csv.

    python cross.py SCRATCHDIR
"""
import os, sys, subprocess, csv
from scan import reduce, HERE
from club import phis

OUT = os.path.join(HERE, "data", "cross.csv")
RP = 1.0
RM = (0.5, 0.8, 0.9, 1.0, 1.1, 1.25, 2.0, 4.0, 10.0)


def main():
    scratch = sys.argv[1]; os.makedirs(scratch, exist_ok=True)
    new = not os.path.exists(OUT)
    done = set() if new else {(r["mode"], float(r["r_minus"]), int(r["n"]), int(r["seed"]))
                              for r in csv.DictReader(open(OUT))}   # resume
    f = open(OUT, "a", newline=""); wr = None
    for n, seeds in ((100000, 2), (1000000, 1)):
        for mode in "wf":
            for rm in RM:
                z = 1 + RP + rm
                if mode == "f" and rm > RP: continue   # a local search for one of O(1) triangles never ends
                for seed in range(1, seeds + 1):
                    if (mode, rm, n, seed) in done: continue
                    pre = os.path.join(scratch, f"cross_{mode}_{rm}_{n}_{seed}")
                    subprocess.run([os.path.join(HERE, "tc"), mode, f"{RP / z:.10f}", str(n), str(seed),
                                    pre, f"{rm / z:.10f}"], check=True)
                    row = reduce(pre, mode, RP / z, n, seed)
                    ph, _ = phis(pre)
                    toks = open(pre + ".sum").read().split()
                    s = dict(zip(toks[0::2], toks[1::2]))
                    row = dict(r_plus=RP, r_minus=rm, **row, T=int(s["T"]), closures=int(s["closures"]),
                               removals=int(s["removals"]), failed_rem=int(s["failed_rem"]), **ph)
                    if wr is None:
                        wr = csv.DictWriter(f, list(row))
                        if new: wr.writeheader()
                    wr.writerow(row); f.flush()
                    for ext in ("nodes", "edges"): os.remove(f"{pre}.{ext}")
                    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)


if __name__ == "__main__":
    main()
