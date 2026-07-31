#!/usr/bin/env python3
"""Figures for the chapter "The limit of a network".

Two figures, both self-contained -- no external data except the community
counts already committed for chapter 6 (community_communities_vs_n.csv), which
panel 2(b) re-plots against the statistical resolution scale rather than
against the fractal sqrt(n) of Figure com-count.

  fig-gr-limit.pdf       the dense limit sharpens, the sparse limit blanks
  fig-gr-resolution.pdf  minimax risk vs number of blocks, and k*(n)

Style follows mkfigs.py (grayscale, 5x8in trim, Computer Modern) -- importing
it sets the shared rcParams.

Usage:  python3 mkfigs_graphon.py   (writes fig-gr-*.pdf next to this script)
"""

import os

import numpy as np
import matplotlib.pyplot as plt

from mkfigs import W, panel, save, read  # noqa: F401  (import sets book rcParams)

HERE = os.path.dirname(os.path.abspath(__file__))

SEED = 11


# --------------------------------------------------------------------------
# Figure 1: what the adjacency matrix looks like as the network grows, under
# the two scalings. Dense (p fixed): the picture sharpens onto a step function
# -- the graphon. Sparse (mean degree fixed): the same block structure is
# present at every size, but the picture fades to blank, which is the whole
# difficulty. Both rows use the identical 2-block assortative structure.
# --------------------------------------------------------------------------
def _sbm(n, p_in, p_out, rng):
    """Symmetric 2-block adjacency matrix, blocks contiguous."""
    half = n // 2
    block = np.zeros((n, n), dtype=bool)
    block[:half, :half] = True
    block[half:, half:] = True
    p = np.where(block, p_in, p_out)
    a = rng.random((n, n)) < p
    a = np.triu(a, 1)
    return a | a.T


def fig_limit():
    sizes = [60, 240, 960]
    fig, axes = plt.subplots(2, 3, figsize=(W, W * 0.74))
    rng = np.random.default_rng(SEED)

    # Row 1 -- dense: probabilities held fixed, so the mean degree grows with n.
    for ax, n in zip(axes[0], sizes):
        a = _sbm(n, 0.62, 0.18, rng)
        ax.imshow(a, cmap="binary", interpolation="nearest", aspect="equal")
        ax.set_title(f"$n={n}$", fontsize=7, pad=2.5)

    # Row 2 -- sparse: mean degree held fixed at 6, so the probabilities fall
    # as 1/n. Same in/out contrast (a factor 3.4) at every size.
    kbar = 6.0
    for ax, n in zip(axes[1], sizes):
        # p_in / p_out = 3.4 with mean degree kbar: solve for the pair.
        r = 3.4
        p_out = 2 * kbar / (n * (1 + r))
        a = _sbm(n, r * p_out, p_out, rng)
        ax.imshow(a, cmap="binary", interpolation="nearest", aspect="equal")

    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_linewidth(0.4)

    axes[0, 0].set_ylabel("dense\n$p$ fixed", fontsize=7)
    axes[1, 0].set_ylabel("sparse\n$\\langle k\\rangle$ fixed", fontsize=7)
    panel(axes[0, 0], "(a)", x=-0.30)
    panel(axes[1, 0], "(b)", x=-0.30)

    fig.subplots_adjust(wspace=0.10, hspace=0.10)
    save(fig, "fig-gr-limit.pdf")


# --------------------------------------------------------------------------
# Figure 2: the resolution question. (a) the Gao-Lu-Zhou minimax risk for a
# k-block model, n^{-1} log k + k^2/n^2, whose two terms balance near
# sqrt(n log n); (b) that optimum against n, with the community counts of the
# real networks laid over it.
# --------------------------------------------------------------------------
def risk(k, n, alpha=1.0):
    """Mean-squared risk of a k-block fit to a Holder-alpha smooth graphon.

    Two estimation terms from Gao-Lu-Zhou -- log(k)/n for learning the node
    labels and k^2/n^2 for the block parameters themselves -- plus the
    approximation bias k^{-2 alpha} of describing a smooth graphon by k steps.
    The estimation terms alone increase with k and so have no interior
    minimum: it is the bias, falling as k grows, that creates the optimum.
    """
    return k ** (-2.0 * alpha) + np.log(k) / n + k**2 / n**2


def kstar(n, alpha=1.0):
    """Number of blocks minimising the risk, found on a fine grid."""
    k = np.arange(2, max(8, int(30 * np.sqrt(n))))
    return k[np.argmin(risk(k, n, alpha))]


def fig_resolution():
    fig, axes = plt.subplots(1, 2, figsize=(W, W * 0.46))

    # (a) risk vs k for three sizes; the minimum walks right as n grows.
    ax = axes[0]
    for n, ls in [(1e3, ":"), (1e4, "--"), (1e5, "-")]:
        k = np.logspace(0.3, np.log10(3 * np.sqrt(n)), 400)
        r = risk(k, n)
        ax.loglog(k, r / r.min(), ls, color="black", lw=0.9,
                  label=f"$n=10^{{{int(np.log10(n))}}}$")
        ks = kstar(n)
        ax.plot([ks], [risk(ks, n) / r.min()], "o", color="black", ms=3.0)
    ax.set_xlabel("number of blocks $k$")
    ax.set_ylabel("risk / minimum")
    ax.set_ylim(0.8, 60)
    ax.legend(frameon=False, fontsize=6, loc="upper center", handlelength=1.6)
    panel(ax, "(a)")

    # (b) the optimum k*(n), the fractal sqrt(n), and the measured counts.
    ax = axes[1]
    ns = np.logspace(1.5, 6.5, 60)
    ax.loglog(ns, [kstar(n) for n in ns], "-", color="black", lw=1.0,
              label=r"$k^*\!\propto\!\sqrt{n}$")

    df = read("community_communities_vs_n.csv")
    df = df[df["B_mean"] > 0]
    ax.loglog(df["n"], df["B_mean"], "o", mfc="none", mec="0.35",
              ms=2.6, mew=0.5, label="real networks")

    # The fitted power law of chapter 6, refitted here from the same CSV so the
    # two chapters cannot drift apart: B = 0.22 n^0.61.
    lx, ly = np.log(df["n"].values), np.log(df["B_mean"].values)
    beta, a = np.polyfit(lx, ly, 1)
    ax.loglog(ns, np.exp(a) * ns**beta, "--", color="black", lw=0.9,
              label=rf"$B\propto n^{{{beta:.2f}}}$")

    ax.set_xlabel("nodes $n$")
    ax.set_ylabel("blocks")
    # The points run corner to corner, so the legend needs headroom made for
    # it rather than a corner chosen for it.
    ax.set_ylim(top=3e5)
    ax.legend(frameon=False, fontsize=6, loc="upper left", handlelength=1.6,
              labelspacing=0.3, borderaxespad=0.4)
    panel(ax, "(b)")

    fig.subplots_adjust(wspace=0.40)
    save(fig, "fig-gr-resolution.pdf")


if __name__ == "__main__":
    fig_limit()
    fig_resolution()
