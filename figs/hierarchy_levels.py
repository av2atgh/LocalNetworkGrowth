#!/usr/bin/env python3
"""Is the nested hierarchy self-similar, or only its bottom level?

Section 7.12 measures one number: B, the number of groups at the bottom level of
the nested fit, which grows in proportion to n so that B/n is constant. That is a
statement about one level. The nested model fits a whole hierarchy -- groups of
groups of groups -- and if the limit of a locally grown network really is a nested
structure rather than a kernel, the interesting question is whether the same
constant governs every level or only the first.

So take the full hierarchy from the fit, list the level sizes
n = B_0 > B_1 > B_2 > ... > 1, and look at the branching ratios B_l / B_{l+1}.
Three outcomes are possible and they mean different things:

  constant ratio      the hierarchy is self-similar: one number characterises the
                      rule at every scale, and the depth grows like log n. The
                      limit object is a tree with a fixed branching factor.

  ratio drifts        the bottom level is special and the higher levels are doing
                      something else; "constant density" is then a fact about
                      communities, not about the hierarchy.

  depth fixed         the hierarchy has a bounded number of levels and all the
                      growth is in the bottom one, which would make the nesting a
                      convenience rather than the structure.

Runs the growth rules at a range of sizes and the real growth series' snapshots.

Usage:  python3 hierarchy_levels.py [--reps 5]
"""
import argparse
import os

import numpy as np
import pandas as pd
import graph_tool.all as gt

from ramsey_motif import MODELS, to_edges

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data", "hierarchy_levels.csv")

RULES = ("LS(n,1)", "DS(n,0.5)", "DD(n,0.5)", "BB(n,2)")
SIZES = (200, 400, 800, 1600)


def levels(g, reps):
    """Level sizes of the best nested fit, by description length."""
    g = gt.extract_largest_component(g, prune=True)
    best = None
    for r in range(reps):
        gt.seed_rng(r)
        st = gt.minimize_nested_blockmodel_dl(g, state_args=dict(deg_corr=True))
        s = st.entropy() / np.log(2)
        if best is None or s < best[0]:
            bs = [len(np.unique(np.asarray(b, dtype=int))) for b in st.get_bs()]
            best = (s, bs)
    ls = [g.num_vertices()] + best[1]
    # graph-tool pads the top of the hierarchy with levels of one group; they
    # carry no information and would report a branching ratio of exactly 1.
    while len(ls) > 1 and ls[-1] == ls[-2]:
        ls.pop()
    return ls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=5)
    a = ap.parse_args()
    rows = []

    for rule in RULES:
        gen, params = MODELS[rule]
        for n in SIZES:
            for seed in range(3):
                nb = gen(n, params, seed)
                g = gt.Graph(directed=False)
                g.add_vertex(len(nb))
                g.add_edge_list(to_edges(nb))
                ls = levels(g, a.reps)
                rows.append(dict(source=rule, n=g.num_vertices(), seed=seed,
                                 depth=len(ls) - 1, levels=json_ls(ls)))
                print(f"  {rule:<11} n={n:5d} s{seed}  levels {ls}", flush=True)
                pd.DataFrame(rows).to_csv(OUT, index=False)

    print("\nbranching ratios B_l / B_(l+1), averaged over seeds:")
    for rule in RULES:
        for n in SIZES:
            sub = [eval(r["levels"]) for r in rows
                   if r["source"] == rule and r["n"] == n]
            if not sub:
                continue
            d = max(len(x) for x in sub)
            rat = []
            for l in range(d - 1):
                v = [x[l] / x[l + 1] for x in sub if len(x) > l + 1]
                rat.append(np.mean(v))
            print(f"  {rule:<11} n={n:5d}  " +
                  "  ".join(f"{r:5.2f}" for r in rat))


def json_ls(ls):
    return "[" + ", ".join(str(x) for x in ls) + "]"


if __name__ == "__main__":
    main()


# --- control: does the prior invent the levels above the communities? --------
# The branching ratios above the bottom level come out at 2-4 on every rule, which
# looks like a self-similar hierarchy. Before believing it, fit the same model to
# networks that have no hierarchy above their bottom level by construction.
#
# Result: on Erdos-Renyi, where there are no communities at all, the nested model
# correctly returns a single level -- it does not hallucinate structure. But on a
# planted partition of 64 equal groups with nothing above them, it recovers the 64
# exactly and then builds a 4-4-4 tree on top. So the upper branching is the
# prior's own scaffolding, not a property of the network. The bottom level is a
# measurement; the levels above it are a description device.

def controls(n=1600, reps=3):
    rows = []

    def er(seed, k=4):
        rng = np.random.default_rng(seed)
        e = rng.integers(0, n, (int(n * k / 2), 2))
        e = e[e[:, 0] != e[:, 1]]
        g = gt.Graph(directed=False); g.add_vertex(n); g.add_edge_list(e)
        gt.remove_parallel_edges(g); return g

    def planted(seed, B=64, k=6, p_in=0.9):
        rng = np.random.default_rng(seed)
        b, e = np.arange(n) % B, []
        for _ in range(int(n * k / 2)):
            i = int(rng.integers(n))
            j = (int(rng.choice(np.where(b == b[i])[0])) if rng.random() < p_in
                 else int(rng.integers(n)))
            if i != j:
                e.append((i, j))
        g = gt.Graph(directed=False); g.add_vertex(n); g.add_edge_list(e)
        gt.remove_parallel_edges(g); return g

    for name, fn in (("ER", er), ("planted-64", planted)):
        for s in range(3):
            ls = levels(fn(s), reps)
            rows.append(dict(source=f"control:{name}", n=ls[0], seed=s,
                             depth=len(ls) - 1, levels=json_ls(ls)))
            print(f"  {name:<12} s{s} levels {ls}", flush=True)
    return pd.DataFrame(rows)
