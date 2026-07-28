#!/usr/bin/env python3
"""The Ramsey community number of the motif-based method, and how kappa grows.

Chapter 6 defines the communities likelihood and the Ramsey community number
relative to a detection method f_kappa,

    P_kappa(n) = Prob{ kappa >= 2 },      r_kappa = min{ n : P_kappa(n) >= 1-eps }

with eps = 0.05, and stresses that r_kappa is a property of the rule *and* the
method. Chapter 6 measured it with a stochastic block model and got r_kappa = 81
for LS(l=1) and 98 for DS(q=0.5). This script asks the same two questions of the
motif-alphabet block model of motif_sbm.py, which describes a network as blocks
whose links are grouped into edges, triangles and squares:

  1. r_kappa, the size past which communities appear in 95% of realisations;
  2. how the mean number of communities grows with n.

Both are reported for the motif method and, from the same realisations, for the
ordinary nested degree-corrected block model, so the two can be compared on
identical networks rather than across separate runs. The block-model column is
also the check that the pipeline reproduces chapter 6.

The two growth rules are taken verbatim from the source repository for the
Ramsey work, github.com/av2atgh/ramsey_netcom (libs.py), so that the models are
exactly the ones the book measured:

    LS(n, l)   local search: pick a random node, take an l-step random walk,
               link the newcomer to every node visited. For l = 1 this is the
               triadic-closure rule.
    DS(n, q)   duplication-split: with probability q copy a random node's links,
               otherwise split one of its links in two.

Usage:
    python3 ramsey_motif.py --stage ramsey    # fine grid, many realisations
    python3 ramsey_motif.py --stage scaling   # coarse grid, larger n
"""
import argparse
import csv
import math
import os
import sys
import time

import numpy as np

from motif_sbm import MotifSBM

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")


# ------------------------------------------------------------------ models
# Verbatim from github.com/av2atgh/ramsey_netcom/libs.py, so that the rules are
# identical to the ones chapter 6 measured. Only the docstrings are new.

def generator_local_search(n, params, seed):
    """LS(n, d): each newcomer links to every node on a d-step random walk."""
    d = params["d"]
    neighbours = [[1], [0]]
    np.random.seed(seed=seed)
    for i in range(2, n):
        visited = [np.random.randint(i)]
        for step in range(d):
            j = visited[step]
            l = np.random.randint(len(neighbours[j]))
            k = neighbours[j][l]
            visited.append(k)
        visited_unique = list(np.unique(visited))
        for j in visited_unique:
            neighbours[j].append(i)
        neighbours.append(visited_unique)
    return neighbours


def generator_dup_split(n, params, seed):
    """DS(n, q): duplicate a random node's links, or split one of them in two."""
    np.random.seed(seed=seed)
    p = params["p"] if "p" in params else 0
    q = params["q"]
    neighbours = [[3, 1], [0, 2], [1, 3], [2, 0]]
    is_duplicate = np.random.random(n) < q
    for i in range(4, n):
        j = np.random.randint(i)
        degree_j = len(neighbours[j])
        if is_duplicate[i]:
            for k in neighbours[j]:
                neighbours[k].append(i)
            new_neighbours = [k for k in neighbours[j]]
            if p > 0 and np.random.random() < p:
                neighbours[j].append(i)
                new_neighbours.append(j)
            neighbours.append(new_neighbours)
        else:
            degree_j = len(neighbours[j])
            l = np.random.randint(degree_j)
            k = neighbours[j][l]
            neighbours[j][l] = i
            neighbours[k][neighbours[k].index(j)] = i
            neighbours.append([j, k])
    return neighbours


# LS(l=1) and DS(q=0.5) are the pair chapter 6 tabulates, with published Ramsey
# numbers of 81 and 98 under a block model; keeping the same parameters is what
# makes those numbers a check on this pipeline rather than a loose comparison.
# DS(q=0.3) was measured first and is kept because less duplication turns out to
# move r_kappa a long way.
def generator_dup_divergence(n, params, seed):
    """DD(n, q): duplicate a random node, then drop each copied link with
    probability q. networkx parametrises this by the retention probability, so
    q here is its 1 - p; q is the probability of *removing* a copied link, to
    match the gene-duplication chapter's reading of divergence."""
    import networkx as nx
    g = nx.duplication_divergence_graph(n, 1.0 - params["q"], seed=seed)
    g.remove_edges_from(nx.selfloop_edges(g))
    idx = {v: i for i, v in enumerate(sorted(g))}
    nb = [[] for _ in range(len(g))]
    for u, v in g.edges():
        nb[idx[u]].append(idx[v])
        nb[idx[v]].append(idx[u])
    return nb


def generator_bubble(n, params, seed):
    """BB(n, L): add a chain of L nodes across a random link, closing a fresh
    cycle of length L+2. Same rule as arity_models.bb."""
    from arity_models import bb
    adj = bb(n, params["L"], seed)
    idx = {v: i for i, v in enumerate(sorted(adj))}
    nb = [[] for _ in range(len(adj))]
    for u in adj:
        for w in adj[u]:
            nb[idx[u]].append(idx[w])
    return nb


MODELS = {
    "LS(n,1)": (generator_local_search, {"d": 1}),
    "DS(n,0.5)": (generator_dup_split, {"q": 0.5}),
    "DS(n,0.3)": (generator_dup_split, {"q": 0.3}),
    "DD(n,0.5)": (generator_dup_divergence, {"q": 0.5}),
    "BB(n,2)": (generator_bubble, {"L": 2}),
}
DEFAULT = ("LS(n,1)", "DS(n,0.5)")


def to_edges(neighbours):
    """Simple undirected edge list; the generators may repeat a link."""
    return sorted({(min(i, j), max(i, j))
                   for i, lst in enumerate(neighbours) for j in lst if i != j})


# ------------------------------------------------------------------ measure

def one(model, n, seed):
    gen, params = MODELS[model]
    nb = gen(n, params, seed)
    # some rules overshoot the requested size -- the bubble rule adds L nodes at
    # a time and stops once it is past n -- so the graph is built on however many
    # nodes came back, while n stays the label the sizes are grouped by
    n_actual = len(nb)
    edges = to_edges(nb)
    m = MotifSBM(n_actual, edges)
    # Two block-model answers on the same network. The nested one is also one of
    # the motif model's starting partitions, so it costs nothing extra. The flat
    # one is what chapter 6's census used, and its resolution ceiling is sqrt(N)
    # against the nested model's N/log N, so the two report different numbers of
    # communities on the same graph -- quoting them interchangeably is how the
    # exponents in chapter 6 and in this table come to disagree.
    try:
        k_dc = len(set(m.dcsbm_blocks(seed=seed)))
        k_flat = len(set(m.dcsbm_blocks(nested=False, seed=seed)))
    except ImportError:
        k_dc = k_flat = float("nan")
    m.fit()
    c = m.motif_counts()
    return dict(model=model, n=n, n_nodes=n_actual, seed=seed, m=len(edges),
                kappa=len(set(m.blocks)), kappa_dcsbm=k_dc, kappa_flat=k_flat,
                n_edge=c.get("edge", 0), n_tri=c.get("triangle", 0),
                n_sq=c.get("square", 0), sigma=round(m.description_length(), 2))


def sweep(sizes, reps, out, models=DEFAULT):
    rows = []
    for model in models:
        for n in sizes:
            t0 = time.time()
            for r in range(reps):
                rows.append(one(model, n, 1000 * n + r))
            k = [x["kappa"] for x in rows[-reps:]]
            kd = [x["kappa_dcsbm"] for x in rows[-reps:]]
            p = sum(1 for v in k if v >= 2) / reps
            print(f"  {model:<10} n={n:<6} P_kappa={p:.2f}  "
                  f"<kappa>={np.mean(k):5.2f}  <kappa_dcsbm>={np.mean(kd):5.2f}  "
                  f"[{time.time() - t0:.0f}s]", flush=True)
            with open(out, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
    return rows


# ------------------------------------------------------------------ analysis

def ramsey(rows, key="kappa", eps=0.05):
    """Smallest measured n with P_kappa >= 1-eps, plus a logistic interpolation
    between the bracketing sizes so the answer is not purely grid-limited."""
    out = {}
    for model in sorted({r["model"] for r in rows}):
        sub = {}
        for r in rows:
            if r["model"] != model:
                continue
            sub.setdefault(r["n"], []).append(r[key])
        ns = sorted(sub)
        ps = [(n, sum(1 for v in sub[n] if v >= 2) / len(sub[n])) for n in ns]
        hit = [n for n, p in ps if p >= 1 - eps]
        # first n from which the criterion holds and never fails again
        rk = None
        for n, p in ps:
            if p >= 1 - eps and all(q >= 1 - eps for m, q in ps if m >= n):
                rk = n
                break
        out[model] = dict(curve=ps, r_kappa=rk, first_hit=hit[0] if hit else None)
    return out


def scaling(rows, key="kappa", nmin=100):
    out = {}
    for model in sorted({r["model"] for r in rows}):
        sub = {}
        for r in rows:
            if r["model"] == model and r["n"] >= nmin:
                sub.setdefault(r["n"], []).append(r[key])
        ns = sorted(sub)
        if len(ns) < 3:
            continue
        x = np.log([n for n in ns])
        y = np.log([np.mean(sub[n]) for n in ns])
        A = np.vstack([x, np.ones_like(x)]).T
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        resid = y - A @ coef
        se = np.sqrt((resid ** 2).sum() / max(len(x) - 2, 1)
                     * np.linalg.inv(A.T @ A)[0, 0])
        out[model] = dict(beta=coef[0], se=se, sizes=ns,
                          means=[float(np.mean(sub[n])) for n in ns])
    return out


def report(rows):
    print("\nP_kappa(n) and the Ramsey community number, eps = 0.05")
    for key, lab in (("kappa", "motif alphabet"), ("kappa_dcsbm", "block model")):
        res = ramsey(rows, key)
        print(f"\n  --- {lab} ---")
        for model, d in res.items():
            curve = "  ".join(f"{n}:{p:.2f}" for n, p in d["curve"])
            print(f"  {model:<10} r_kappa = {d['r_kappa']}")
            print(f"             {curve}")

    print("\n<kappa> against n")
    for key, lab in (("kappa", "motif alphabet"), ("kappa_dcsbm", "block model")):
        sc = scaling(rows, key)
        print(f"\n  --- {lab} ---")
        for model, d in sc.items():
            pts = "  ".join(f"{n}:{mu:.1f}" for n, mu in zip(d["sizes"], d["means"]))
            print(f"  {model:<10} <kappa> ~ n^{d['beta']:.3f} +- {d['se']:.3f}")
            print(f"             {pts}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("ramsey", "scaling", "report"),
                    default="ramsey")
    ap.add_argument("--reps", type=int)
    ap.add_argument("--sizes", help="comma-separated override for the n grid")
    # semicolon, not comma: the model names contain commas themselves
    ap.add_argument("--models", help="semicolon-separated subset of the models")
    ap.add_argument("--out")
    args = ap.parse_args()

    if args.stage == "ramsey":
        sizes = [10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100, 120, 150, 200]
        reps = args.reps or 100
        out = args.out or os.path.join(DATA, "ramsey_motif_threshold.csv")
    elif args.stage == "scaling":
        sizes = [100, 200, 400, 800, 1600]
        reps = args.reps or 20
        out = args.out or os.path.join(DATA, "ramsey_motif_scaling.csv")
    else:
        import glob
        rows = []
        for p in sorted(glob.glob(os.path.join(DATA, "ramsey_motif_*.csv"))):
            if True:
                with open(p) as fh:
                    for r in csv.DictReader(fh):
                        rows.append({k: (int(v) if k in ("n", "seed", "m", "kappa",
                                                         "kappa_dcsbm", "kappa_flat",
                                                         "n_edge", "n_tri", "n_sq")
                                         and v not in ("", "nan") else v)
                                     for k, v in r.items()})
        report(rows)
        return 0

    if args.sizes:
        sizes = [int(x) for x in args.sizes.split(",")]
    models = tuple(args.models.split(";")) if args.models else DEFAULT
    print(f"stage={args.stage}  sizes={sizes}  reps={reps}  models={models}")
    rows = sweep(sizes, reps, out, models)
    report(rows)
    print(f"\n{len(rows)} realisations -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
