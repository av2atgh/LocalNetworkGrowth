"""Opening in the hub caricature (revision, comments 1c and 2).

For the ten largest hubs a of a network: local clustering c_a, and tau_a, the
mean triangle count t_e of the links among a's neighbours.  With t = T/n and
w = W_open/n, the feedback on the summed degree s of a hub's neighbours runs
when beta_-/beta_+ < x* = 3 t (1 - c) / (w tau c).  Prints x* for each network.

    python hubfeed.py NETWORK_PREFIX ...
"""
import sys
import numpy as np


def main():
    for pre in sys.argv[1:]:
        e = np.fromfile(pre + ".edges", dtype=np.int32).reshape(-1, 2)
        nd = np.loadtxt(pre + ".nodes", dtype=np.int64)
        k, t, w = nd[:, 0], nd[:, 1].astype(float), nd[:, 3].astype(float)
        n = len(k); T = t.sum() / 3; W = w.sum() / 2
        nb = [set() for _ in range(n)]
        for a, b in e: nb[a].add(b); nb[b].add(a)
        cs, taus = [], []
        for a in np.argsort(-k, kind="stable")[:10]:
            N = list(nb[a]); far = [(b, c) for i, b in enumerate(N) for c in N[i + 1:] if c in nb[b]]
            cs.append(2 * len(far) / (len(N) * (len(N) - 1)))
            taus.append(np.mean([len(nb[b] & nb[c]) for b, c in far]) if far else np.nan)
        c, tau = np.mean(cs), np.nanmean(taus)
        print(f"{pre.split('/')[-1]}: T/n={T / n:.3f} W/n={W / n:.2f} c_hub={c:.3f} tau={tau:.1f} "
              f"x*={3 * T / n * (1 - c) / (W / n * tau * c):.3f}", flush=True)


if __name__ == "__main__":
    main()
