#!/usr/bin/env python3
"""Figure: the alternation, and the diagonal that breaks it.

Chapter 9 sweeps BB(n,L) from L=1 to 5 with the alphabet extended so that each
rule's own L+2 cycle can be named, and finds that naming it abolishes the
communities when L is even and does nothing when L is odd. Both panels plot the
same quantity, which is the one the chapter turns on:

    kappa(+motif) / kappa(nested)

the community count the alphabet reports divided by the count the same nested
block model reports on the same networks. A rule whose communities survive sits
on one. A rule whose communities are carried off falls away from it. Plotting the
ratio rather than the two counts separately is what makes five rules legible in
one panel, and it divides out the fact that the rules differ several-fold in how
many communities they have to begin with -- which is a difference the chapter
needs the reader to see is *not* what decides the outcome.

  (a) L = 1..5. The odd rules lie on one at every size; the even rules peel away
      downward and keep going. Nothing in the coverage or the community counts
      separates these five curves; the parity of L does.

  (b) BB(n,2) against its two variants, both of which add one link per step. The
      bipartite one leaves the alphabet nothing separable to name and so cannot
      test anything; the diagonal destroys bipartiteness and the communities come
      back, while the alphabet takes up a larger share of that network than of
      any other in the book.

Style follows mkfigs.py (grayscale, serif, book column width).

Usage:  python3 mkfigs_bubbles.py   (writes fig-bb-alternation.pdf here)
"""
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from mkfigs import W, panel, save  # noqa: F401  (import sets book rcParams)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

# L, marker, face, label -- odd L filled, even L open, so parity is visible in
# the marker before the reader reaches the legend
SWEEP = (
    (1, "o", "black", r"$L=1$, triangle"),
    (2, "s", "white", r"$L=2$, square"),
    (3, "^", "0.45", r"$L=3$, pentagon"),
    (4, "D", "white", r"$L=4$, hexagon"),
    (5, "v", "0.75", r"$L=5$, heptagon"),
)

VARIANTS = (
    ("bb2", "s", "white", r"$BB(n,2)$"),
    ("bb2par", "P", "0.6", r"$+$ a link, bipartite"),
    ("bb2chord", "o", "black", r"$+$ a diagonal"),
)


def ratio(d, nested, motif):
    """Mean kappa under each method at each size, then their ratio."""
    a = d.groupby("n")[nested].mean()
    b = d.groupby("n")[motif].mean()
    keep = a > 0
    return a.index.values[keep], (b[keep] / a[keep]).values


def sweep_frame(L):
    if L in (1, 2):
        import glob
        # match the suffix exactly: "bb3" as a substring would also exclude
        # ramsey_motif_scaling_bb3200.csv, which is BB(n,2) at n = 3200
        f = [x for x in glob.glob(f"{DATA}/ramsey_motif_scaling*.csv")
             if not any(x.endswith(b) for b in ("bb3.csv", "bb4.csv", "bb5.csv"))]
        d = pd.concat([pd.read_csv(x) for x in f], ignore_index=True)
        return d[d.model == f"BB(n,{L})"], "kappa_dcsbm", "kappa"
    d = pd.read_csv(f"{DATA}/ramsey_motif_scaling_bb{L}.csv")
    return d, "kappa_dcsbm", ("kappa_pent" if L == 3 else "kappa_cyc")


def main():
    fig, axes = plt.subplots(2, 1, figsize=(W, 4.4))

    ax = axes[0]
    for i, (L, mk, fc, lab) in enumerate(SWEEP):
        d, nested, motif = sweep_frame(L)
        x, y = ratio(d, nested, motif)
        ax.plot(x, y, marker=mk, mfc=fc, mec="black", color="0.45",
                lw=0.7, ms=3.4, label=lab, zorder=3)
    ax.axhline(1.0, color="black", lw=0.6, ls=":", zorder=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(0.006, 2.4)
    ax.set_ylabel(r"$\kappa$ with motifs $/\ \kappa$ nested")
    ax.set_xlabel(r"$n$")
    ax.legend(loc="lower left", ncol=2, frameon=False)
    panel(ax, "(a)")

    ax = axes[1]
    ch = pd.read_csv(f"{DATA}/bb2_chord.csv")
    for rule, mk, fc, lab in VARIANTS:
        d = ch[ch.rule == rule]
        x, y = ratio(d, "kappa_dcsbm", "kappa_kite")
        ax.plot(x, y, marker=mk, mfc=fc, mec="black", color="0.45",
                lw=0.7, ms=3.4, label=lab, zorder=3)
    ax.axhline(1.0, color="black", lw=0.6, ls=":", zorder=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(0.006, 2.4)
    ax.set_ylabel(r"$\kappa$ with motifs $/\ \kappa$ nested")
    ax.set_xlabel(r"$n$")
    ax.legend(loc="lower left", frameon=False)
    panel(ax, "(b)")

    fig.tight_layout(h_pad=1.6)
    save(fig, "fig-bb-alternation.pdf")


if __name__ == "__main__":
    main()
