#!/usr/bin/env python3
"""The bubble family in one table: Chapter 9's master table.

Seven rules measured across four runs with three different column schemas, all
normalised here to the same six quantities:

    <k>          mean degree, 2(L+1)/L for BB(n,L) and one link more for the
                 two variants of L=2 that add one
    bipartite    by two-colouring, not inferred from the cycle spectrum
    r_kappa      onset under the nested block model, P_kappa >= 0.95
    beta         growth of the community count, nested and with the rule's own
                 motif in the alphabet, fitted over n >= 400 so that every rule
                 is past its onset and none of them is contributing a stretch of
                 kappa = 1 to the slope
    named        share of links the alphabet succeeds in planting, at the
                 largest size fitted

Writes tab-bubble.tex.
"""
import os
from collections import deque

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
ROOT = os.path.dirname(HERE)
FIT_FROM = 400


def beta(d, col, lo=FIT_FROM):
    s = d[d.n >= lo].groupby("n")[col].mean()
    s = s[s > 0]
    return np.polyfit(np.log(s.index.values), np.log(s.values), 1)[0] if len(s) > 2 else np.nan


def onset(d, col):
    p = d.groupby("n")[col].apply(lambda s: (s > 1).mean())
    ok = [n for n in p.index if p[n] >= 0.95 and all(p[m] >= 0.95 for m in p.index if m >= n)]
    return ok[0] if ok else None


def bipartite(L, chord=False):
    """Two-colour a grown instance rather than read it off the cycle spectrum."""
    import random
    from arity_models import bb
    if not chord:
        adj = bb(600, L, 3)
    else:
        from bb2_chord import grow
        adj = grow(600, "bb2chord", 3)
    col, rng = {}, random.Random(0)
    for s in adj:
        if s in col:
            continue
        col[s] = 0
        q = deque([s])
        while q:
            u = q.popleft()
            for w in adj[u]:
                if w not in col:
                    col[w] = 1 - col[u]
                    q.append(w)
                elif col[w] == col[u]:
                    return False
    return True


def rows():
    out = []

    # --- BB(n,1) and BB(n,2): the runs that predate the cycle family
    import glob
    # the suffix has to match exactly: "bb3" as a substring would also drop
    # ramsey_motif_scaling_bb3200.csv, which is BB(n,2) at n = 3200
    skip = ("bb3.csv", "bb4.csv", "bb5.csv")
    old = pd.concat([pd.read_csv(f) for f in
                     glob.glob(f"{DATA}/ramsey_motif_scaling*.csv") +
                     glob.glob(f"{DATA}/ramsey_motif_threshold*.csv")
                     if not f.endswith(skip)],
                    ignore_index=True)
    for L in (1, 2):
        g = old[old.model == f"BB(n,{L})"]
        top = g[g.n == g.n.max()]
        out.append(dict(
            rule=f"$BB(n,{L})$", cyc=L + 2, k=2 * (L + 1) / L,
            bip=bipartite(L), rk=onset(g, "kappa_dcsbm"),
            b_nest=beta(g, "kappa_dcsbm"), b_mot=beta(g, "kappa"),
            named=(3 * top.n_tri + 4 * top.n_sq).sum() / top.m.sum(),
            bimodal=(L == 2)))

    # --- BB(n,3), BB(n,4), BB(n,5): fitted with their own cycle in the alphabet
    for L in (3, 4, 5):
        g = pd.read_csv(f"{DATA}/ramsey_motif_scaling_bb{L}.csv")
        kcyc = "kappa_pent" if L == 3 else "kappa_cyc"
        ncyc = "pent_pent" if L == 3 else "cyc_cyc"
        top = g[g.n == g.n.max()]
        out.append(dict(
            rule=f"$BB(n,{L})$", cyc=L + 2, k=2 * (L + 1) / L,
            bip=bipartite(L), rk=onset(g, "kappa_dcsbm"),
            b_nest=beta(g, "kappa_dcsbm"), b_mot=beta(g, kcyc),
            named=((L + 2) * top[ncyc]).sum() / top.m.sum(),
            bimodal=(L == 4)))

    # --- the two variants of L = 2
    ch = pd.read_csv(f"{DATA}/bb2_chord.csv")
    # neither variant lays down a cycle, so neither can be drawn as one: the
    # chorded footprint is K_4 minus an edge and the other is a square with a
    # link running off it
    for tag, label, bip, glyph in (
            ("bb2par", r"$BB(n,2)+$ a link", True, r"\sqstub"),
            ("bb2chord", r"$BB(n,2)+$ a diagonal", False, r"\sqdiag")):
        g = ch[ch.rule == tag]
        top = g[g.n == g.n.max()]
        out.append(dict(
            rule=label, cyc=glyph, k=g.mean_k.mean(), bip=bip,
            rk=onset(g, "kappa_dcsbm"),
            b_nest=beta(g, "kappa_dcsbm"), b_mot=beta(g, "kappa_kite"),
            named=top.named_kite.mean(), bimodal=False))
    return out


def main():
    r = rows()
    # the sweep first, in order of L, then the two variants of L = 2 that add a
    # link -- they are interventions on the family rather than members of it
    order = [0, 1, 2, 3, 4, 5, 6]
    L = [r"\begin{center}", r"\footnotesize\setlength{\tabcolsep}{3pt}",
         r"\begin{tabular}{lccrrrrr}", r"\hline",
         r"rule & footprint & bip. & $\langle k\rangle$ & $r_\kappa$ "
         r"& $\beta$ nested & $\beta+$mot. & named\\", r"\hline"]
    for i in order:
        x = r[i]
        bm = "---" if x["bimodal"] else f"${x['b_mot']:.2f}$"
        rk = "---" if x["rk"] is None else f"${x['rk']}$"
        # the footprint column is drawn rather than counted: a pentagon is
        # quicker to recognise than the numeral 5, and parity is what the column
        # is for. Rules whose footprint is a cycle get the polygon; the two that
        # lay down something else carry their own glyph.
        glyph = x["cyc"] if isinstance(x["cyc"], str) else f"\\ngon{{{x['cyc']}}}"
        L.append(f"{x['rule']} & {glyph} & "
                 f"{'yes' if x['bip'] else 'no'} & "
                 f"${x['k']:.1f}$ & {rk} & ${x['b_nest']:.2f}$ & {bm} & "
                 f"${100*x['named']:.0f}\\%$" r"\\")
        if i == 4:                      # rule off the sweep from the variants
            L.append(r"\hline")
    L += [r"\hline", r"\end{tabular}", r"\end{center}"]
    open(os.path.join(ROOT, "tab-bubble.tex"), "w").write("\n".join(L) + "\n")

    d = pd.DataFrame(r)
    d["named"] = (100 * d.named).round(0)
    print(d.to_string())


if __name__ == "__main__":
    main()
