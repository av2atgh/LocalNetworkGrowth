"""Triangle clusters of the closure rules without opening, u = 1/2.

Fractions of triangles, nodes and links in the largest edge-connected triangle
cluster (trclust.c) at n = 1e4 (2 seeds), 1e5 (1 seed) and 1e6 (data/raw).
Writes data/rulesclust.csv.

    python rulesclust.py SCRATCHDIR
"""
import os, sys, csv, subprocess
from scan import HERE

OUT = os.path.join(HERE, "data", "rulesclust.csv")
SIZES = ((10000, 2), (100000, 1), (1000000, 1))


def main():
    scratch = sys.argv[1]; os.makedirs(scratch, exist_ok=True)
    with open(OUT, "w", newline="") as f:
        wr = csv.writer(f); wr.writerow(["mode", "n", "seed", "T", "clusters", "fT", "fN", "fE"])
        for m in "lfcpw":
            for n, ns in SIZES:
                for seed in range(1, ns + 1):
                    if n == 1000000:
                        pre = os.path.join(HERE, "data", "raw", f"{m}_u0.5_n1000000")
                    else:
                        pre = os.path.join(scratch, f"{m}_{n}_{seed}")
                        subprocess.run([os.path.join(HERE, "tc"), m, "0.5", str(n), str(seed), pre], check=True,
                                       stdout=subprocess.DEVNULL)
                    T, ncl, tmax, nmax, emax = map(int, subprocess.run(
                        [os.path.join(HERE, "trclust"), pre + ".edges", str(n)],
                        capture_output=True, text=True, check=True).stdout.split())
                    E = os.path.getsize(pre + ".edges") // 8
                    wr.writerow([m, n, seed, T, ncl, tmax / T, nmax / n, emax / E]); f.flush()


if __name__ == "__main__":
    main()
