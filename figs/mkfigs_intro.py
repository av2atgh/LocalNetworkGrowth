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

# Half-width panels on a 5x8 page cannot carry the full names; the captions do.
SHORT = {
    "Internet (AS)": "Internet",
    "Gnutella": "Gnutella",
    "Yeast proteins": "yeast",
    "cond-mat co-authorship": "cond-mat",
}


def _load():
    return pd.read_csv(os.path.join(DATA, "intro_curves.csv"))


def _series(ax, d, q, ms=2.4):
    for i, lab in enumerate(ORDER):
        s = d[(d.label == lab) & (d.quantity == q)]
        s = s[s.value > 0]
        sc(ax, s.k.values, s.value.values, i, SHORT[lab], ms=ms)


# --------------------------------------------------------------------------
# Figure for section 1.4: the degree distribution and the degree correlations
# -- the scale-free property, and where the two families part company.
# --------------------------------------------------------------------------
def fig_degree():
    d = _load()
    fig, axes = plt.subplots(1, 2, figsize=(W, W * 0.47))

    # (a) degree distribution -- broad and hub-dominated
    ax = axes[0]
    _series(ax, d, "pk")
    ax.set(xscale="log", yscale="log", xlabel=r"$k$", ylabel=r"$p(k)$")
    ax.legend(loc="lower left", frameon=False, fontsize=5.6)
    panel(ax, "a")

    # (b) degree correlations -- falling for the first three, rising for cond-mat
    ax = axes[1]
    _series(ax, d, "knn")
    ax.set(
        xscale="log", yscale="log", xlabel=r"$k$", ylabel=r"$\langle k_{nn}\rangle(k)$"
    )
    panel(ax, "b")

    fig.subplots_adjust(wspace=0.36)
    save(fig, "fig-intro-degree.pdf")


# --------------------------------------------------------------------------
# Figure for section 1.2: the clustering hierarchy, before and after correcting
# for what the neighbours' degrees actually allow. The two panels share a
# vertical scale, which is the whole point -- the flattening must be readable
# directly, not hidden by a rescaled axis.
# --------------------------------------------------------------------------
def fig_clustering():
    d = _load()
    fig, axes = plt.subplots(1, 2, figsize=(W, W * 0.47))
    kk = np.array([2.0, 300.0])

    # (a) the standard coefficient: the hierarchy
    ax = axes[0]
    _series(ax, d, "ck")
    ax.plot(kk, 0.9 / kk, "k--", lw=0.8, zorder=1)
    ax.annotate(r"$k^{-1}$", xy=(300, 0.9 / 300), xytext=(190, 0.0075), fontsize=6)
    ax.set(xscale="log", yscale="log", xlabel=r"$k$", ylabel=r"$\langle c\rangle(k)$")
    # Gnutella and yeast occupy the lower left; the hubs' corner is empty.
    ax.legend(loc="upper right", frameon=False, fontsize=5.6)
    panel(ax, "a")

    # (b) corrected: the hierarchy largely evaporates
    ax = axes[1]
    _series(ax, d, "ctilde")
    ax.plot(kk, 0.9 / kk, "k--", lw=0.8, zorder=1)
    ax.set(
        xscale="log",
        yscale="log",
        xlabel=r"$k$",
        ylabel=r"$\langle \tilde{c}\rangle(k)$",
        ylim=axes[0].get_ylim(),
    )
    panel(ax, "b")

    fig.subplots_adjust(wspace=0.36)
    save(fig, "fig-intro-clustering.pdf")


if __name__ == "__main__":
    fig_degree()
    fig_clustering()
