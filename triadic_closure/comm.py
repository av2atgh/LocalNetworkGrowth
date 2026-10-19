"""Community structure with graph-tool (run in the gt conda environment).

For each network: the planted-partition fit (PPBlockState, assortative
communities only; B = 1 when there are none) with its modularity, and the
nested degree-corrected SBM with its bottom-level groups and the description
length gained over a single group.  The same fits on a configuration-model
rewiring of each network tell structure from degree sequence.  The best of
NREP fits (default 3) is kept.  Appends to data/comm.csv (resumes).

    ~/anaconda3/envs/gt/bin/python comm.py SCRATCHDIR N [NAME ...]
"""
import os, sys, csv, subprocess
import numpy as np
import graph_tool.all as gt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data", "comm.csv")
NREP = int(os.environ.get("NREP", 3))


def configs():
    """name -> (mode, u_+, u_-, beta_-/beta_+)"""
    c = {f"{m}_u0.5": (m, 0.5, 0.0, 0.0) for m in "lfcpw"}
    for rm in (0.5, 1, 2, 4, 6, 8, 16):            # <k> = 4 line, beta_+ = alpha + beta_-
        rp = 1 + rm; z = 1 + rp + rm
        c[f"w_k4_rm{rm:g}"] = ("w", rp / z, rm / z, rm / rp)
    c["w_crit"] = ("w", 1 / 3, 1 / 3, 1.0)
    c["w_tree"] = ("w", 0.25, 0.5, 2.0)
    c["w_rrt"] = ("w", 0.0, 0.0, float("nan"))
    return c


def graph(pre):
    e = np.fromfile(pre + ".edges", dtype=np.int32).reshape(-1, 2)
    g = gt.Graph(directed=False); g.add_edge_list(e)
    return g


def fits(g):
    pp = min((gt.minimize_blockmodel_dl(g, state=gt.PPBlockState) for _ in range(NREP)),
             key=lambda s: s.entropy())
    b = pp.get_blocks()
    nest = min((gt.minimize_nested_blockmodel_dl(g) for _ in range(NREP)), key=lambda s: s.entropy())
    lv = nest.get_levels()
    one = gt.BlockState(g, b=g.new_vp("int", 0)).entropy()   # DC-SBM with a single group
    a = b.a; e = g.get_edges()
    size = np.bincount(a) / g.num_vertices()
    big = np.argmax(size); kin = np.bincount(a, g.get_total_degrees(g.get_vertices()))
    return dict(B_pp=pp.get_nonempty_B(), Q_pp=gt.modularity(g, b), S1_pp=size.max(),
                in_pp=np.mean(a[e[:, 0]] == a[e[:, 1]]),             # fraction of links inside groups
                kbar_rest=np.delete(kin, big).sum() / max(1e-9, (1 - size.max()) * g.num_vertices()),
                B_sbm=lv[0].get_nonempty_B(),
                levels=sum(1 for s in lv if s.get_nonempty_B() > 1),
                dl_gain=1 - nest.entropy() / one)


def main():
    scratch, n = sys.argv[1], int(sys.argv[2]); os.makedirs(scratch, exist_ok=True)
    gt.seed_rng(1); np.random.seed(1)
    cf = configs(); names = sys.argv[3:] or list(cf)
    new = not os.path.exists(OUT)
    done = set() if new else {(r["name"], int(r["n"])) for r in csv.DictReader(open(OUT))}
    f = open(OUT, "a", newline=""); wr = None
    for name in names:
        if (name, n) in done: continue
        mode, up, um, ratio = cf[name]
        pre = os.path.join(scratch, f"{name}_{n}")
        args = [os.path.join(HERE, "tc"), mode, f"{up:.10f}", str(n), "1", pre]
        if um > 0: args.append(f"{um:.10f}")
        subprocess.run(args, check=True)
        g = graph(pre)
        row = dict(name=name, n=n, ratio=ratio, E=g.num_edges(), **fits(g))
        gt.random_rewire(g, model="configuration", n_iter=10)
        row.update({k + "_null": v for k, v in fits(g).items()})
        if wr is None:
            wr = csv.DictWriter(f, list(row))
            if new: wr.writeheader()
        wr.writerow(row); f.flush()
        for ext in ("nodes", "edges", "ts", "sum"): os.remove(f"{pre}.{ext}")
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)


if __name__ == "__main__":
    main()
