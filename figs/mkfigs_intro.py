#!/usr/bin/env python3
"""Figure for the Introduction: what the data demand.

Plots the four real networks measured by fetch_intro_data.py, which draws them
from the Netzschleuder repository (networks.skewed.com) through the graph-tool
interface. This script needs only the committed CSVs -- no graph-tool, no network
connection.

Style follows mkfigs.py (grayscale, 5x8in trim, Computer Modern); importing it
sets the shared rcParams.

Usage:  python3 mkfigs_intro.py      (writes fig-intro-data.pdf next to this file)
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from mkfigs import W, sc, panel, save  # noqa: F401  (import sets book rcParams)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

# Order matters: the three technological/biological networks first, the social
# one last, so the legend reads as the two families the text describes.
ORDER = ["Internet (AS)", "Gnutella", "Yeast proteins", "cond-mat co-authorship"]


def fig_intro():
    d = pd.read_csv(os.path.join(DATA, "intro_curves.csv"))
    fig, axes = plt.subplots(4, 1, figsize=(W * 0.74, W * 1.86))

    def series(ax, q):
        for i, lab in enumerate(ORDER):
            s = d[(d.label == lab) & (d.quantity == q)]
            s = s[s.value > 0]
            sc(ax, s.k.values, s.value.values, i, lab, ms=2.6)

    # (a) degree distribution -- the scale-free property
    ax = axes[0]
    series(ax, "pk")
    ax.set(xscale="log", yscale="log", xlabel=r"$k$", ylabel=r"$p(k)$")
    ax.legend(loc="lower left", frameon=False, fontsize=5.4)
    panel(ax, "a", x=-0.17)

    # (b) clustering hierarchy -- <c>(k) falls, roughly as 1/k
    ax = axes[1]
    series(ax, "ck")
    kk = np.array([2.0, 300.0])
    ax.plot(kk, 0.9 / kk, "k--", lw=0.8, zorder=1)
    # Park the label below the guide's right end, clear of every series.
    ax.annotate(r"$k^{-1}$", xy=(300, 0.9 / 300), xytext=(330, 0.0016), fontsize=6)
    ax.set(xscale="log", yscale="log", xlabel=r"$k$", ylabel=r"$\langle c\rangle(k)$")
    panel(ax, "b", x=-0.17)

    # (c) the same, corrected for what the neighbours' degrees actually allow.
    # Shares panel (b)'s y-limits so the flattening is read off directly rather
    # than hidden by a rescaled axis.
    ax = axes[2]
    series(ax, "ctilde")
    ax.plot(kk, 0.9 / kk, "k--", lw=0.8, zorder=1)
    ax.set(
        xscale="log",
        yscale="log",
        xlabel=r"$k$",
        ylabel=r"$\langle \tilde{c}\rangle(k)$",
        ylim=axes[1].get_ylim(),
    )
    panel(ax, "c", x=-0.17)

    # (d) degree correlations -- the two families part company here
    ax = axes[3]
    series(ax, "knn")
    ax.set(
        xscale="log",
        yscale="log",
        xlabel=r"$k$",
        ylabel=r"$\langle k_{nn}\rangle(k)$",
    )
    panel(ax, "d", x=-0.17)

    fig.subplots_adjust(hspace=0.46)
    save(fig, "fig-intro-data.pdf")


if __name__ == "__main__":
    fig_intro()
