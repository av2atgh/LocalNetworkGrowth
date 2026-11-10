"""Power-study runner.  Appends one row per replicate to data/results.csv (resumes).

    ~/anaconda3/envs/gt/bin/python run.py [--reps R] [--workers 2] [--only SIZE,...] [--time]

Conditions (family, lam, contagion, dense):
  LS, DS mixes        lam in {0, .25, .5, .75, 1}
  SP mix              lam in {0, .5, 1}
  ASBM, BA            single condition each
  LSc, DSc contagion  lam in {0, .5, 1}
Sizes: 20, 35, 45, 100, 200, 500 (sparse); 20, 35, 45 dense (density 0.9).
"""
import os, sys, csv, time, argparse, zlib
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
OUT = os.path.join(HERE, "data", "results.csv")
H_HIST = 4

SPARSE_N = [20, 35, 45, 100, 200, 500]
DENSE_N = [20, 35, 45]
LAMS = [0.0, 0.25, 0.5, 0.75, 1.0]


def conditions(dense):
    c = [(f, l, False) for f in ("LS", "DS") for l in LAMS]
    c += [("SP", l, False) for l in (0.0, 0.5, 1.0)]
    c += [("ASBM", 1.0, False), ("BA", 0.0, False)]
    if not dense:
        c += [(f, l, True) for f in ("LS", "DS") for l in (0.0, 0.5, 1.0)]
    return c


SENS_N = [45, 100, 200]
SENS = [(0.5, 2), (0.95, 2), (0.8, 4), (0.8, 8)]  # (h, K) sensitivity sets;
# K = 8 means 8 latent types of which only type % 2 is observed (latent homophily)


def tasks(reps, sizes):
    out = []
    for n in sizes:
        for dense in ((False, True) if n in DENSE_N else (False,)):
            if not dense and n not in SPARSE_N:
                continue
            for fam, lam, cont in conditions(dense):
                for r in range(reps):
                    out.append((fam, lam, cont, n, dense, r, 0.8, 2))
        if n in SENS_N:
            for h, K in SENS:
                c = [(f, l) for f in ("LS", "DS") for l in (0.5, 1.0)] + [("ASBM", 1.0)]
                if K != 2:
                    c += [("LS", 0.0), ("DS", 0.0)]
                for fam, lam in c:
                    for r in range(reps):
                        out.append((fam, lam, False, n, False, r, h, K))
    return out


def seed_of(*key):
    return zlib.crc32(repr(key).encode()) & 0x7FFFFFFF


def one(task):
    import models, stats
    fam, lam, cont, n, dense, r, h, K = task
    t0 = time.time()
    s_attr = seed_of("attr", fam, lam, cont, n, dense, r, h, K)
    types0, pos = models.exogenous(n, K, fam, s_attr)
    t1 = int(round(0.75 * n)) if cont else None

    def build(hist):
        net = models.grow(fam, n, lam, types0, pos, seed_of("struct", s_attr, hist),
                          h=h, K=K, contagion=cont, t1=t1)
        e = net["edges"]
        if dense:
            e = models.densify(e, n, 0.9, seed_of("dense", s_attr, hist))
        return net, e

    net, e = build(0)
    row = dict(family=fam + ("c" if cont else ""), lam=lam, n=n, dense=int(dense), rep=r, h=h, K=K)
    obs = net["types"] % 2 if K == 8 else net["types"]
    st, b0 = stats.snapshot_stats(e, n, obs, pos, seed_of("fis", s_attr))
    row.update(st)
    parts = [b0]
    for hist in range(1, H_HIST):
        _, eh = build(hist)
        parts.append(stats.fit(stats.graph(eh, n), True, 1)[0])
    row["hist_ari"] = stats.hist_ari(parts)
    row["hist_nontrivial"] = np.mean([len(np.unique(p)) > 1 for p in parts])
    if cont:
        row.update(stats.temporal(net, n, t1))
    row["secs"] = time.time() - t0
    return row


COLS = None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=40)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--only", default="")
    ap.add_argument("--time", action="store_true")
    a = ap.parse_args()
    sizes = [int(x) for x in a.only.split(",")] if a.only else sorted(set(SPARSE_N + DENSE_N))
    if a.time:
        import graph_tool.all as gt
        for n in sizes:
            for task in [("LS", 0.0, False, n, False, 0, .8, 2), ("DS", 0.0, False, n, False, 0, .8, 2),
                         ("LS", 1.0, True, n, False, 0, .8, 2), ("SP", 0.0, False, n, n <= 45, 0, .8, 2)]:
                row = one(task)
                print(task, f"{row['secs']:.2f}s", {k: round(v, 3) for k, v in row.items()
                      if k in ("B_dc", "align_nmi", "beyond_dc", "logit_ratio", "hist_ari",
                               "fis_diff", "conv", "sel", "E")})
        return
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    done = set()
    if os.path.exists(OUT):
        for r in csv.DictReader(open(OUT)):
            done.add((r["family"], float(r["lam"]), int(r["n"]), int(r["dense"]), int(r["rep"]),
                      float(r["h"]), int(r["K"])))
    todo = [t for t in tasks(a.reps, sizes)
            if (t[0] + ("c" if t[2] else ""), t[1], t[3], int(t[4]), t[5], t[6], t[7]) not in done]
    print(f"{len(todo)} tasks to run", flush=True)
    allcols = ["family", "lam", "n", "dense", "rep", "h", "K", "E", "density", "B_dc", "B_ndc", "detect",
               "align_nmi", "align_ari", "align_nmi_ndc", "beyond_dc", "expl_dc", "beyond_ndc",
               "expl_ndc", "dcgain", "b_same", "b_cn", "b_deg", "b_dist", "logit_ratio",
               "fis_attr", "fis_topo", "fis_diff", "hist_ari", "hist_nontrivial",
               "conv", "sel", "n_conv_pairs", "secs"]
    new = not os.path.exists(OUT)
    f = open(OUT, "a", newline="")
    w = csv.DictWriter(f, allcols)
    if new:
        w.writeheader()
    from multiprocessing import Pool
    t0 = time.time()
    with Pool(a.workers) as pool:
        for i, row in enumerate(pool.imap_unordered(safe, todo, chunksize=1)):
            if row is None:
                continue
            w.writerow(row); f.flush()
            if i % 100 == 0:
                print(f"{i}/{len(todo)}  {time.time() - t0:.0f}s  last n={row['n']}", flush=True)


def safe(task):
    try:
        import graph_tool.all as gt
        gt.openmp_set_num_threads(1)
        gt.seed_rng(seed_of("gt", *task)); np.random.seed(seed_of("np", *task))
        return one(task)
    except Exception as ex:                     # skip, never halt the run
        print("FAILED", task, repr(ex), flush=True)
        return None


if __name__ == "__main__":
    main()
