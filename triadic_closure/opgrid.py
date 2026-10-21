"""Rule (w) with triad opening and node deletion: where does the localized
triangle cluster disappear?

Parameters: net growth Delta = (beta_+ - beta_-)/alpha_+, turnover
beta_-/beta_+, and node deletion a = alpha_-/alpha_+ < 1.  Observables as in
delgrid.reduce; <k> is compared with 2(1 + Delta)/(1 + a).
Appends to data/opgrid.csv (resumes).

    python opgrid.py SCRATCHDIR
"""
import os, sys, csv, subprocess
from scan import HERE
from delgrid import reduce

OUT = os.path.join(HERE, "data", "opgrid.csv")
DS = (0.5, 1.0)
RATIOS = (0.0, 0.5, 0.67, 0.75, 0.8, 0.85, 0.9, 0.95)
AS = (0.0, 0.5)
SIZES = ((10000, 2), (100000, 2), (1000000, 1))


def main():
    scratch = sys.argv[1]; os.makedirs(scratch, exist_ok=True)
    new = not os.path.exists(OUT)
    done = set() if new else {(float(r["D"]), float(r["ratio"]), float(r["a"]), int(r["n"]), int(r["seed"]))
                              for r in csv.DictReader(open(OUT))}
    f = open(OUT, "a", newline=""); wr = None
    for n, ns in SIZES:
        for D in DS:
            for a in AS:
                for ratio in RATIOS:
                    for seed in range(1, ns + 1):
                        if (D, ratio, a, n, seed) in done: continue
                        rp = D / (1 - ratio); rm = rp - D; z = 1 + a + rp + rm   # rates over alpha_+
                        pre = os.path.join(scratch, f"op_{D}_{ratio}_{a}_{n}_{seed}")
                        args = [os.path.join(HERE, "tc"), "w", f"{rp / z:.10f}", str(n), str(seed), pre,
                                f"{rm / z:.10f}"]
                        if a > 0: args.append(f"{a / z:.10f}")
                        subprocess.run(args, check=True)
                        row = dict(D=D, ratio=ratio, a=a, n=n, seed=seed, **reduce(pre, n, rp, a))
                        row["kpred"] = 2 * (1 + D) / (1 + a)
                        if wr is None:
                            wr = csv.DictWriter(f, list(row))
                            if new: wr.writeheader()
                        wr.writerow(row); f.flush()
                        for ext in ("nodes", "edges", "ts", "sum"): os.remove(f"{pre}.{ext}")
                        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()},
                              flush=True)


if __name__ == "__main__":
    main()
