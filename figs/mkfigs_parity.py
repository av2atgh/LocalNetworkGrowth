#!/usr/bin/env python3
"""Schematic: what the bubble rule lays down, and where the colouring fails.

The measured figure of Chapter 9 shows the alternation but cannot show why the
rules divide the way they do, because the reason is geometric. This one draws it.

Every step of BB(n,L) -- and every link of DBB(n,L) at every generation -- lays a
chain of L new nodes across an existing link, so the footprint is a cycle of
length L+2 in which one link is old and L+1 are new. Two-colour it. For even L
the cycle is even and the colouring closes; for odd L it does not, and exactly
one link ends up joining two nodes of the same colour, wherever the colouring is
started. That link is what makes the network non-bipartite, and it is the only
thing that distinguishes the rules that keep their communities from the rules
that lose them.

  (a) the footprint for L = 1..5, two-coloured, with the frustrated link dashed
      where there is one. Odd L is frustrated; even L is not.

  (b) the intervention of Section 9.6. A diagonal of the square joins two nodes
      the colouring has put on the same side, so adding it destroys the
      bipartition without changing the chain length, the reach, or the number of
      footprints laid per step.

Style follows mkfigs.py (grayscale, serif, book column width).

Usage:  python3 mkfigs_parity.py   (writes fig-bb-parity.pdf here)
"""
import os

import numpy as np
import matplotlib.pyplot as plt

from mkfigs import W, panel, save  # noqa: F401  (import sets book rcParams)

HERE = os.path.dirname(os.path.abspath(__file__))

R = 1.0
NODE = 62          # marker area
LW_NEW = 0.9
LW_OLD = 2.0


def ring(k):
    """Positions of a k-cycle with nodes 0 and k-1 straddling the bottom, so
    the old link -- the one the chain was laid across -- is always the base."""
    a = np.radians(-90 + 180.0 / k + np.arange(k) * 360.0 / k)
    return np.c_[R * np.cos(a), R * np.sin(a)]


def frustrated_mark(ax, a, b):
    """A short bar across the middle of a link whose ends share a colour."""
    a, b = np.asarray(a), np.asarray(b)
    mid = (a + b) / 2.0
    d = b - a
    perp = np.array([-d[1], d[0]])
    perp = perp / np.hypot(*perp) * 0.17
    ax.plot(*zip(mid - perp, mid + perp), lw=1.5, color="0.5",
            solid_capstyle="butt", zorder=4)


def draw_footprint(ax, L, label):
    """The cycle of length L+2: one old link, L+1 new ones, two-coloured."""
    k = L + 2
    p = ring(k)
    colour = [j % 2 for j in range(k)]          # walk the cycle, alternating
    # links: (j, j+1) are the new chain; (k-1, 0) is the old link
    for j in range(k):
        u, v = j, (j + 1) % k
        old = (j == k - 1)
        ax.plot(*zip(p[u], p[v]), lw=LW_OLD if old else LW_NEW,
                color="black", solid_capstyle="round", zorder=2)
        # frustration is marked at the midpoint rather than by restyling the
        # link, because the link is already carrying old-against-new in its
        # weight and one line cannot encode two things
        if colour[u] == colour[v]:
            frustrated_mark(ax, p[u], p[v])
    for j in range(k):
        ax.scatter(*p[j], s=NODE, zorder=3,
                   facecolor="black" if colour[j] else "white",
                   edgecolor="black", linewidths=0.8)
    ax.set_aspect("equal")
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.28, 1.92)
    ax.axis("off")
    ax.text(0, 1.58, label, ha="center", va="center", fontsize=7)


def draw_chord(ax, chord, label):
    """The square, with or without a diagonal."""
    p = ring(4)
    colour = [j % 2 for j in range(4)]
    for j in range(4):
        u, v = j, (j + 1) % 4
        ax.plot(*zip(p[u], p[v]), lw=LW_OLD if j == 3 else LW_NEW,
                color="black", solid_capstyle="round", zorder=2)
    if chord:
        # a diagonal joins nodes two apart, which the alternation has given the
        # same colour -- so it is frustrated by construction
        ax.plot(*zip(p[1], p[3]), lw=LW_NEW, color="black", zorder=2)
        frustrated_mark(ax, p[1], p[3])
    for j in range(4):
        ax.scatter(*p[j], s=NODE, zorder=3,
                   facecolor="black" if colour[j] else "white",
                   edgecolor="black", linewidths=0.8)
    ax.set_aspect("equal")
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.28, 1.92)
    ax.axis("off")
    ax.text(0, 1.58, label, ha="center", va="center", fontsize=7)


def main():
    fig = plt.figure(figsize=(W, 2.35))
    gs = fig.add_gridspec(2, 6, height_ratios=[1, 1],
                          hspace=0.22, wspace=0.05)

    for i, L in enumerate((1, 2, 3, 4, 5)):
        ax = fig.add_subplot(gs[0, i])
        draw_footprint(ax, L, rf"$L={L}$")
        if i == 0:
            panel(ax, "(a)", x=-0.30)

    # the key at the end of the top row, so the two rows share it
    ax = fig.add_subplot(gs[0, 5])
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(-1.28, 1.92)
    ax.plot([0.05, 0.45], [0.55, 0.55], lw=LW_OLD, color="black")
    ax.text(0.55, 0.55, "old", va="center", fontsize=6)
    ax.plot([0.05, 0.45], [0.05, 0.05], lw=LW_NEW, color="black")
    ax.text(0.55, 0.05, "new", va="center", fontsize=6)
    ax.plot([0.05, 0.45], [-0.45, -0.45], lw=LW_NEW, color="black")
    ax.plot([0.25, 0.25], [-0.62, -0.28], lw=1.5, color="0.5",
            solid_capstyle="butt")
    ax.text(0.55, -0.45, "same\ncolour", va="center", fontsize=6)

    for i, (chord, lab) in enumerate(((False, r"$BB(n,2)$"),
                                      (True, r"$+$ a diagonal"))):
        ax = fig.add_subplot(gs[1, 2 * i + 1:2 * i + 3])
        draw_chord(ax, chord, lab)
        if i == 0:
            panel(ax, "(b)", x=-0.18)

    save(fig, "fig-bb-parity.pdf")


if __name__ == "__main__":
    main()
