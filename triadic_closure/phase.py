"""Phase diagram of the global rule (w) in net growth D = (beta_+ - beta_-)/alpha,
which fixes <k> = 2(1 + D), and turnover r_- = beta_-/alpha.

Each point is run at n = 1e5 and 1e6 and reduced to kmax, T/N and the club
fractions phi(n^a); data/phase.csv.  The condensed phase keeps phi(n^0.8)
(and kmax/sqrt n) from shrinking with n; the hub-free phase has kmax ~ ln n.

    python phase.py SCRATCHDIR             # the grid at n = 1e5, 1e6
    python phase.py SCRATCHDIR D:R ...     # extra points D, r_- at n = 1e7
"""
import os, sys, subprocess, csv
from scan import HERE
from club import phis

OUT = os.path.join(HERE, "data", "phase.csv")
DS = (0.1, 0.25, 0.5, 1.0, 2.0)
RM = (0.0, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 16.0)


def main():
    scratch = sys.argv[1]; os.makedirs(scratch, exist_ok=True)
    new = not os.path.exists(OUT)
    done = set() if new else {(float(r["D"]), float(r["r_minus"]), int(r["n"]))
                              for r in csv.DictReader(open(OUT))}   # resume
    f = open(OUT, "a", newline=""); wr = None
    extra = [tuple(map(float, a.split(":"))) for a in sys.argv[2:]]
    plan = ([(10000000, D, rm) for D, rm in extra] if extra else
            [(n, D, rm) for n in (100000, 1000000) for D in DS for rm in RM])
    for n, D, rm in plan:
        if (D, rm, n) in done: continue
        rp = D + rm; z = 1 + rp + rm
        pre = os.path.join(scratch, f"phase_{D}_{rm}_{n}")
        subprocess.run([os.path.join(HERE, "tc"), "w", f"{rp / z:.10f}", str(n), "1", pre,
                        f"{rm / z:.10f}"], check=True)
        ph, kmax = phis(pre)
        toks = open(pre + ".sum").read().split()
        s = dict(zip(toks[0::2], toks[1::2]))
        row = dict(D=D, r_minus=rm, r_plus=rp, n=n, kmean=float(s["kmean"]), T_per_N=int(s["T"]) / n,
                   transitivity=float(s["transitivity"]), assort=float(s["assort"]), kmax=kmax, **ph)
        if wr is None:
            wr = csv.DictWriter(f, list(row))
            if new: wr.writeheader()
        wr.writerow(row); f.flush()
        for ext in ("nodes", "edges"): os.remove(f"{pre}.{ext}")
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)


if __name__ == "__main__":
    main()
