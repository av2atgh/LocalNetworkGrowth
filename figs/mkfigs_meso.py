#!/usr/bin/env python3
"""Figure: motifs and communities counted as one thing.

A planted motif and a community are the same kind of object seen at two sizes. A
motif is a small, fixed, homogeneous group of nodes whose internal wiring is
given by a symbol; a community is a larger, heterogeneous group whose internal
wiring is given by a block of the count matrix. Both are meso-structure -- both
are the description saying "these nodes belong together, and here is how they are
joined" -- and the description length treats them as competing accounts of the
same links, which is why naming motifs can take communities away.

If that reading is right, the quantity to watch is not the community count but
the total. This figure plots both.

  (a) the number of motifs planted, against size
  (b) motifs plus communities, against size

Both come out proportional to n for every rule in the family, with a slope of one
to within a few hundredths, while the community count on its own runs from
n^0.25 to n^0.93 depending on the parity of L (Figure 9.2). What the alternation
moves is how the meso-structure is split between the two kinds of symbol, not how
much of it there is.

Style follows mkfigs.py (grayscale, serif, book column width).

Usage:  python3 mkfigs_meso.py   (writes fig-bb-meso.pdf here)
"""
import glob
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from mkfigs import W, panel, save  # noqa: F401  (import sets book rcParams)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

SWEEP = ((1, "o", "black"), (2, "s", "white"), (3, "^", "0.45"),
         (4, "D", "white"), (5, "v", "0.75"))


def series(L):
    """Motifs planted and communities reported, averaged at each size."""
    if L in (1, 2):
        f = [x for x in glob.glob(f"{DATA}/ramsey_motif_scaling*.csv")
             if not any(x.endswith(b) for b in ("bb3.csv", "bb4.csv", "bb5.csv"))]
        d = pd.concat([pd.read_csv(x) for x in f], ignore_index=True)
        d = d[d.model == f"BB(n,{L})"].copy()
        d["M"] = d.n_tri + d.n_sq
        kap = "kappa"
    else:
        d = pd.read_csv(f"{DATA}/ramsey_motif_scaling_bb{L}.csv").copy()
        tag = "pent" if L == 3 else "cyc"
        d["M"] = d[f"{tag}_{tag}"] + d[f"tri_{tag}"] + d[f"sq_{tag}"]
        kap = "kappa_pent" if L == 3 else "kappa_cyc"
    g = d.groupby("n").agg(M=("M", "mean"), k=(kap, "mean"))
    return g.index.values, g.M.values, g.k.values


def main():
    fig, axes = plt.subplots(2, 1, figsize=(W, 4.3))

    for ax, which, lab in ((axes[0], "M", "motifs planted"),
                           (axes[1], "Mk", "motifs $+$ communities")):
        for L, mk, fc in SWEEP:
            n, M, k = series(L)
            y = M if which == "M" else M + k
            ax.plot(n, y, marker=mk, mfc=fc, mec="black", color="0.45",
                    lw=0.7, ms=3.4, label=rf"$L={L}$", zorder=3)
        # slope one, for the eye: every rule is parallel to it
        x0 = np.array([90, 3600])
        ax.plot(x0, x0 * 0.42, color="black", lw=0.6, ls=":", zorder=1)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$n$")
        ax.set_ylabel(lab)
        panel(ax, "(a)" if which == "M" else "(b)")
    axes[0].legend(loc="upper left", ncol=2, frameon=False)

    fig.tight_layout(h_pad=1.2)
    save(fig, "fig-bb-meso.pdf")


if __name__ == "__main__":
    main()
