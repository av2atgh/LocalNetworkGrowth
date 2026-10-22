"""Size of the dense core without presupposing sqrt(n) (rule (w), beta_- = 0, u = 1/2).

For networks kept by rev.py (SCRATCH/keep/x0.0_n*.edges):
  kcore    nodes in the maximal k-core and its order k
  peel     densest subgraph by greedy peeling (Charikar's 2-approximation):
           size and link density of the prefix with the largest E/N
  rank50   largest rank r such that the r highest-degree nodes are linked
           with density >= 1/2
Writes data/rev_core.csv.

    python core.py SCRATCHDIR
"""
import os, sys, csv, heapq
import numpy as np
from scan import HERE


def load(pre):
    e = np.fromfile(pre + ".edges", dtype=np.int32).reshape(-1, 2)
    n = int(e.max()) + 1
    return n, e


def adjacency(n, e):
    order = np.argsort(np.concatenate([e[:, 0], e[:, 1]]), kind="stable")
    nb = np.concatenate([e[:, 1], e[:, 0]])[order]
    off = np.zeros(n + 1, np.int64); np.add.at(off, np.concatenate([e[:, 0], e[:, 1]]) + 1, 1)
    return np.cumsum(off), nb


def peel(n, e):
    """Remove a minimum-degree node at a time; return the order and the degeneracy (k-core) data."""
    off, nb = adjacency(n, e)
    deg = np.diff(off).astype(np.int64)
    alive = np.ones(n, bool)
    heap = [(int(d), i) for i, d in enumerate(deg)]; heapq.heapify(heap)
    order, kcore_at = [], []
    kmax = 0
    while heap:
        d, i = heapq.heappop(heap)
        if not alive[i] or d != deg[i]: continue
        kmax = max(kmax, d); alive[i] = False; order.append(i); kcore_at.append(kmax)
        for j in nb[off[i]:off[i + 1]]:
            if alive[j]:
                deg[j] -= 1; heapq.heappush(heap, (int(deg[j]), int(j)))
    return np.array(order), np.array(kcore_at)


def main():
    keep = os.path.join(sys.argv[1], "keep")
    rows = []
    for n_ in (10000, 100000, 1000000):
        pre = os.path.join(keep, f"x0.0_n{n_}")
        n, e = load(pre)
        order, kc = peel(n, e)
        K = kc[-1]
        ncore = int((kc == K).sum())                       # nodes removed at the final core order
        # densest prefix of the reversed peeling order
        pos = np.empty(n, np.int64); pos[order] = np.arange(n)
        last = np.minimum(pos[e[:, 0]], pos[e[:, 1]])     # a link survives until its first endpoint goes
        alive_links = n - 1 - last                         # links present when the remaining set has size s
        cnt = np.bincount(n - 1 - last, minlength=n)      # links whose earlier endpoint is at reversed rank r
        links_in_top = np.cumsum(cnt)                      # links inside the last s+1 nodes peeled
        sizes = np.arange(1, n + 1)
        dens = links_in_top / sizes
        s = int(np.argmax(dens)) + 1
        rho_peel = links_in_top[s - 1] / (s * (s - 1) / 2) if s > 1 else 0
        # rank50 on degree ranking
        k = np.bincount(e.ravel(), minlength=n)
        rank = np.empty(n, np.int64); rank[np.argsort(-k, kind="stable")] = np.arange(n)
        lr = np.maximum(rank[e[:, 0]], rank[e[:, 1]])
        inside = np.cumsum(np.bincount(lr, minlength=n))
        r = np.arange(1, n + 1); d = inside / np.maximum(r * (r - 1) / 2, 1)
        ok = np.where(d[1:] >= 0.5)[0]
        r50 = int(ok.max()) + 2 if len(ok) else 0
        rows.append(dict(n=n_, kcore_order=int(K), kcore_size=ncore, peel_size=s, peel_density=rho_peel,
                         rank50=r50, sqrt_n=int(np.sqrt(n_))))
        print(rows[-1], flush=True)
    with open(os.path.join(HERE, "data", "rev_core.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, list(rows[0])); wr.writeheader(); wr.writerows(rows)
    for col in ("kcore_size", "peel_size", "rank50"):
        y = [r[col] for r in rows]
        print(col, "exponent", np.polyfit(np.log([r["n"] for r in rows]), np.log(y), 1)[0].round(3))


if __name__ == "__main__":
    main()
