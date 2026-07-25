#!/usr/bin/env python3
"""Figures for the chapter "Boosting shortest path multiplicity".

Adapted from generate_data_and_figs.py of
    github.com/av2atgh/ramsey_netcom/tree/master/local_multi_path_2026
the source repository of

    A. Vazquez, "Local network evolution rules drive shortest path
    multiplicity", arXiv:2605.25237.

The CSVs under data/ are copied verbatim from that repository; this script only
re-plots them. It does NOT regenerate the simulations (that takes days and needs
graph-tool -- see the upstream repo for the generators).

Differences from the paper's figures, deliberate, so the book does not simply
reprint them:
  * grayscale throughout -- the book is printed black and white, so series are
    separated by marker shape and fill rather than by color;
  * panels regrouped for the 5x8in trim (4.05in text width): the paper's
    six-panel Fig. 1 becomes a four-panel figure, and its four-panel bubble
    figure (L=1,3 / L=2,4) becomes a single panel carrying all four L;
  * Computer Modern maths to match the book's body type;
  * the deterministic bubble model shares a figure with the stochastic one.

Usage:  python3 mkfigs.py          (writes fig-mu-*.pdf next to this script)
"""

import os

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

# 5x8in book: textwidth = 5 - 0.55 - 0.4 = 4.05in. Stay just inside it.
W = 3.9

plt.rcParams.update(
    {
        "font.size": 7,
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "axes.linewidth": 0.6,
        "axes.labelsize": 7.5,
        "axes.titlesize": 8,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "legend.fontsize": 6.5,
        "legend.handletextpad": 0.4,
        "legend.borderaxespad": 0.3,
        "lines.markeredgewidth": 0.6,
        "figure.dpi": 300,
        "savefig.dpi": 300,
    }
)

# Grayscale marker vocabulary: shape + fill carry the identity, not color.
MK = [
    dict(marker="o", mfc="black", mec="black"),
    dict(marker="s", mfc="white", mec="black"),
    dict(marker="^", mfc="0.55", mec="black"),
    dict(marker="D", mfc="white", mec="0.35"),
    dict(marker="v", mfc="0.8", mec="black"),
]


def sc(ax, x, y, i, label=None, ms=3.4):
    st = MK[i % len(MK)]
    ax.plot(x, y, ls="none", ms=ms, label=label, zorder=3, **st)


def panel(ax, letter, x=-0.22):
    # ax.text rather than set_title: some panels also carry a descriptive
    # title, and two set_title(loc="left") calls would overwrite each other.
    ax.text(
        x,
        1.02,
        letter,
        transform=ax.transAxes,
        fontweight="bold",
        fontsize=8,
        va="bottom",
        ha="left",
    )


def save(fig, name):
    out = os.path.join(HERE, name)
    fig.savefig(out, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)
    print("wrote", os.path.relpath(out, HERE))


def read(name):
    return pd.read_csv(os.path.join(DATA, name))


# --------------------------------------------------------------------------
# Figure 1: the random baseline. Three degree-distribution regimes plus the
# mean-field collapse. (Paper Fig. 1, six panels -> four.)
# --------------------------------------------------------------------------
def fig_random():
    fig, axes = plt.subplots(2, 2, figsize=(W, W * 0.82))

    # (a) LS(n,1)* -- gamma = 5 > 3 -> saturates.
    ax = axes[0, 0]
    df = read("model-ls_d1_nr100_vs_n_log.csv")
    x, y = np.log(df.n), df.multipath_rnd
    # Fit the log law on the small-n points only, to expose the saturation.
    p = np.poly1d(np.polyfit(x[:-3], y[:-3], 1))
    ax.plot(df.n, p(x), "k-.", lw=0.8, label=r"$a+b\ln n$")
    sc(ax, df.n, y, 0, r"$LS(n,1)^*$")
    ax.set(xscale="log", xlabel=r"$n$", ylabel=r"$\langle\mu\rangle$")
    ax.legend(loc="upper left", frameon=False)
    panel(ax, "a")

    # (b) DS(n,1/3)* -- gamma = 3 -> logarithmic, mean-field slope 1/e.
    ax = axes[0, 1]
    df = read("model-ds_q1_3_nr100_vs_n_log.csv")
    x, y = np.log(df.n), df.multipath_rnd
    b = np.mean(y - x / np.e)
    ax.plot(df.n, b + x / np.e, "k-.", lw=0.8, label=r"$a+(1/e)\ln n$")
    sc(ax, df.n, y, 1, r"$DS(n,1/3)^*$")
    ax.set(xscale="log", xlabel=r"$n$", ylabel=r"$\langle\mu\rangle$")
    ax.legend(loc="upper left", frameon=False)
    panel(ax, "b")

    # (c) DS(n,2/5)* -- gamma = 5/2 < 3 -> power law, alpha = 3/gamma - 1 = 1/5.
    ax = axes[1, 0]
    df = read("model-ds_q2_5_nr100_vs_n_log.csv")
    x, y = np.log(df.n), np.log(df.multipath_rnd)
    a = np.mean(y - x / 5.0)
    ax.plot(df.n, np.exp(a + x / 5.0), "k--", lw=0.8, label=r"$b\,n^{1/5}$")
    sc(ax, df.n, df.multipath_rnd, 2, r"$DS(n,2/5)^*$")
    ax.set(xscale="log", yscale="log", xlabel=r"$n$", ylabel=r"$\langle\mu\rangle$")
    # Plain numbers on the log y-axis; "3 x 10^0" is noise at this range.
    ax.set_yticks([1, 2, 3, 4])
    ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.legend(loc="upper left", frameon=False)
    panel(ax, "c")

    # (d) Collapse: every model against the mean-field line c_2/e.
    ax = axes[1, 1]
    series = [
        ("model-ls_d1_nr100_vs_n_log.csv", "multipath_rnd", r"$LS(n,1)^*$"),
        ("model-ds_q1_3_nr100_vs_n_log.csv", "multipath_rnd", r"$DS(n,1/3)^*$"),
        ("model-ds_q2_5_nr100_vs_n_log.csv", "multipath_rnd", r"$DS(n,2/5)^*$"),
        ("model-bb_L1_nr100_vs_n_log.csv", "multipath_rnd", r"$BB(n,1)^*$"),
        ("model-ba_m2_a2_nr100_vs_n_log.csv", "multipath", r"$BA(n,2)$"),
    ]
    for i, (f, col, lab) in enumerate(series):
        df = read(f)
        sc(ax, df.mean_kk, df[col], i, lab, ms=3.0)
    c2 = np.array([1.2, 10.5])
    ax.plot(c2, c2 / np.e, "k-", lw=0.9, label=r"$c_2/e$")
    ax.set(xlabel=r"$c_2$", ylabel=r"$\langle\mu\rangle$")
    # Six entries and a diagonal cloud: no corner is free, so buy headroom
    # above the data and lay the key out in two columns.
    ax.set_ylim(0.7, 5.9)
    ax.legend(
        loc="upper left", frameon=False, ncol=2, fontsize=5.4, columnspacing=0.6
    )
    panel(ax, "d")

    fig.subplots_adjust(wspace=0.34, hspace=0.42)
    save(fig, "fig-mu-random.pdf")


# --------------------------------------------------------------------------
# Figure 2: local rules, unrewired. LS -> log-quadratic, DS -> exponential;
# and each against the inferred number of communities. (Paper Fig. 2.)
# --------------------------------------------------------------------------
def fig_local():
    fig, axes = plt.subplots(2, 2, figsize=(W, W * 0.82))

    # (a) LS(n,1): log-quadratic in n.
    ax = axes[0, 0]
    df = read("model-ls_d1_nr100_vs_n_log.csv")
    x, y = np.log(df.n), df.multipath
    p = np.poly1d(np.polyfit(x, y, 2))
    xx = np.linspace(x.min(), x.max(), 400)
    ax.plot(np.exp(xx), p(xx), "k-.", lw=0.8, label=r"$a+b\ln n+c(\ln n)^2$")
    sc(ax, df.n, y, 0, r"$LS(n,1)$")
    ax.set(xscale="log", xlabel=r"$n$", ylabel=r"$\langle\mu\rangle$")
    ax.legend(loc="upper left", frameon=False)
    panel(ax, "a")

    # (b) LS(n,1): linear in the number of communities.
    ax = axes[0, 1]
    d1 = df.loc[df.unique_mean > 1]
    p = np.poly1d(np.polyfit(d1.unique_mean, d1.multipath, 1))
    xx = np.linspace(d1.unique_mean.min(), d1.unique_mean.max(), 200)
    ax.plot(xx, p(xx), "k-", lw=0.8, label="linear")
    sc(ax, df.unique_mean, df.multipath, 0, r"$LS(n,1)$")
    ax.set(xlabel=r"$\langle\kappa\rangle$", ylabel=r"$\langle\mu\rangle$")
    ax.legend(loc="upper left", frameon=False)
    panel(ax, "b")

    # (c) DS(n,1/3): exponential in n -- note the linear x-axis.
    ax = axes[1, 0]
    df = read("model-ds_q1_3_nr100_vs_n_log.csv")
    p = np.poly1d(np.polyfit(df.n, np.log(df.multipath), 1))
    xx = np.linspace(df.n.min(), df.n.max(), 400)
    ax.plot(xx, np.exp(p(xx)), "k--", lw=0.8, label=r"$a\,e^{bn}$")
    sc(ax, df.n, df.multipath, 1, r"$DS(n,1/3)$")
    ax.set(yscale="log", xlabel=r"$n$", ylabel=r"$\langle\mu\rangle$")
    ax.legend(loc="upper left", frameon=False)
    panel(ax, "c")

    # (d) DS(n,1/3): exponential in the number of communities.
    ax = axes[1, 1]
    d1 = df.loc[df.unique_mean > 1]
    p = np.poly1d(np.polyfit(d1.unique_mean, np.log(d1.multipath), 1))
    xx = np.linspace(d1.unique_mean.min(), d1.unique_mean.max(), 200)
    ax.plot(xx, np.exp(p(xx)), "k--", lw=0.8, label=r"$a\,e^{b\kappa}$")
    sc(ax, df.unique_mean, df.multipath, 1, r"$DS(n,1/3)$")
    ax.set(yscale="log", xlabel=r"$\langle\kappa\rangle$", ylabel=r"$\langle\mu\rangle$")
    ax.legend(loc="upper left", frameon=False)
    panel(ax, "d")

    fig.subplots_adjust(wspace=0.34, hspace=0.42)
    save(fig, "fig-mu-local.pdf")


# --------------------------------------------------------------------------
# Figure 3: cycle length (bubble model) and the role of randomness.
# Paper splits L into two 2x2 panels; here all four L share one panel so the
# even/odd split is visible at a glance.
# --------------------------------------------------------------------------
def fig_bubble():
    fig, axes = plt.subplots(1, 2, figsize=(W, W * 0.45))

    # (a) BB(n,L), L = 1..4 in one panel: even L sits above odd L.
    ax = axes[0]
    for i, L in enumerate([1, 2, 3, 4]):
        df = read(f"model-bb_L{L}_nr100_vs_n_log.csv")
        x, y = np.log(df.n), df.multipath
        p = np.poly1d(np.polyfit(x, y, 2))
        xx = np.linspace(x.min(), x.max(), 400)
        ax.plot(np.exp(xx), p(xx), "k-.", lw=0.6, zorder=2)
        sc(ax, df.n, y, i, rf"$L={L}$", ms=3.0)
    ax.set(xscale="log", yscale="log", xlabel=r"$n$", ylabel=r"$\langle\mu\rangle$")
    ax.legend(loc="upper left", frameon=False, ncol=2, columnspacing=0.8)
    panel(ax, "a")

    # (b) Deterministic vs randomized bubble. Both are power laws; the
    # deterministic one is the shallower, so randomness accelerates. The exact
    # solution of the deterministic bubble gives alpha = 0.100 (root of a cubic),
    # against 3/gamma - 1 = 0.161 for its degree-preserving randomization.
    ax = axes[1]
    df = read("model-bbd_nr1_vs_n.csv")
    # Row df.n=1 is the t=0 seed triangle (n=3); n_t = (3^t + 3)/2 thereafter.
    n = (3 ** df.n + 3) // 2
    xx = np.linspace(np.log(n.min()), np.log(n.max()), 400)

    a_rnd = np.mean(np.log(df.multipath_rnd) - 0.161 * np.log(n))
    ax.plot(np.exp(xx), np.exp(a_rnd + 0.161 * xx), "k--", lw=0.8, label=r"$n^{0.16}$")
    sc(ax, n, df.multipath_rnd, 0, r"$BBD(n,1)^*$", ms=3.0)

    a_det = np.mean(np.log(df.multipath) - 0.100 * np.log(n))
    ax.plot(np.exp(xx), np.exp(a_det + 0.100 * xx), "k-.", lw=0.8, label=r"$n^{0.10}$")
    sc(ax, n, df.multipath, 1, r"$BBD(n,1)$", ms=3.0)
    ax.set(xscale="log", yscale="log", xlabel=r"$n$", ylabel=r"$\mu$")
    # Plain numbers on the narrow log y-axis; the default "1.75 x 10^0" ticks are noise.
    ax.set_yticks([1, 1.5, 2, 2.5])
    ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.legend(loc="upper left", frameon=False)
    panel(ax, "b")

    fig.subplots_adjust(wspace=0.36)
    save(fig, "fig-mu-bubble.pdf")


# --------------------------------------------------------------------------
# Figure 4: real networks, against the mean-field expectation c_2/e.
# (Paper Fig. 5.)
# --------------------------------------------------------------------------
def fig_real():
    # Full natural width: the chapter includes this at width=\linewidth, so a
    # narrower figure would simply be scaled UP, magnifying its height with it.
    # Three stacked panels plus a long caption must fit one 6.8in text block, so
    # the aspect ratio is what matters here, not the nominal size.
    fig, axes = plt.subplots(3, 1, figsize=(W, W * 1.18))

    def trend(ax, x, y, kind, ls):
        """Fit the trend the paper reports: power law (log-log) or exponential."""
        x, y = np.asarray(x, float), np.asarray(y, float)
        if kind == "power":
            p = np.poly1d(np.polyfit(np.log(x), np.log(y), 1))
            xx = np.linspace(np.log(x.min()), np.log(x.max()), 400)
            ax.plot(np.exp(xx), np.exp(p(xx)), ls, lw=0.8, color="0.3")
        else:  # exponential in n
            p = np.poly1d(np.polyfit(x, np.log(y), 1))
            xx = np.linspace(x.min(), x.max(), 400)
            ax.plot(xx, np.exp(p(xx)), ls, lw=0.8, color="0.3")

    def one(ax, df, kind, xscale, letter, title, loc="upper left"):
        mu = df.multiplicity / df.connected_pairs
        mf = df.mean_excess_degree / np.e
        trend(ax, df.n_nodes, mu, kind, "--")
        trend(ax, df.n_nodes, mf, kind, "-.")
        sc(ax, df.n_nodes, mu, 0, "data", ms=2.4)
        sc(ax, df.n_nodes, mf, 1, r"mean field, $c_2/e$", ms=2.4)
        # mu here is multiplicity/connected_pairs = <mu>_c, averaged over the
        # connected pairs only (Eq. path-muc), since real networks are not connected.
        ax.set(xscale=xscale, yscale="log", xlabel=r"$n$", ylabel=r"$\langle\mu\rangle_c$")
        ax.legend(loc=loc, frameon=False)
        ax.set_title(title, loc="left", fontsize=6.5, pad=3)
        panel(ax, letter, x=-0.19)

    # (a) Internet, autonomous-system level: power law, data below mean field.
    df = read("as733_metrics.csv")
    one(axes[0], df, "power", "log", "a", "Internet (autonomous systems)")

    # (b) Protein interaction networks: exponential, the duplication signature.
    df = read("biogrid_interactome_metrics.csv")
    df = df.loc[(df.n_nodes > 999) & ~df.organism.str.contains("irus")]
    one(axes[1], df, "exp", "linear", "b", "Protein interaction", loc="lower right")

    # (c) cond-mat co-authorship: data above mean field (assortative).
    df = read("condmat_coauthor_metrics.csv")
    one(axes[2], df, "power", "log", "c", "cond-mat co-authorship")

    fig.subplots_adjust(hspace=0.45)
    save(fig, "fig-mu-real.pdf")


# --------------------------------------------------------------------------
# Figure 5: the diamond hierarchical lattice -- an exactly solvable ceiling.
# Construction adapted from diamond_lattice_fig.py; the multiplicities are
# computed exactly (arbitrary-precision BFS) by multiplicity_diamond.py.
# --------------------------------------------------------------------------
def _diamond_embed(t, off_frac=0.32):
    """Geometric embedding of G_t for b=s=2 (poles 0=A top, 1=B bottom)."""
    import math

    pos = {0: (0.0, 1.0), 1: (0.0, -1.0)}
    edges = [(0, 1)]
    nxt = 2
    for _ in range(t):
        new = []
        for u, v in edges:
            xu, yu = pos[u]
            xv, yv = pos[v]
            mx, my = (xu + xv) / 2.0, (yu + yv) / 2.0
            dx, dy = xv - xu, yv - yu
            L = math.hypot(dx, dy)
            px, py = -dy / L, dx / L  # perpendicular unit
            off = off_frac * L
            ml, mr = nxt, nxt + 1
            nxt += 2
            pos[ml] = (mx + px * off, my + py * off)
            pos[mr] = (mx - px * off, my - py * off)
            new += [(u, ml), (ml, v), (u, mr), (mr, v)]
        edges = new
    return edges, pos


def fig_diamond():
    import multiplicity_diamond as md

    fig = plt.figure(figsize=(W, W * 0.78))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.25], hspace=0.45, wspace=0.25)

    # (a-c) construction for t = 0, 1, 2
    for j, t in enumerate((0, 1, 2)):
        ax = fig.add_subplot(gs[0, j])
        edges, pos = _diamond_embed(t)
        for u, v in edges:
            ax.plot(
                [pos[u][0], pos[v][0]],
                [pos[u][1], pos[v][1]],
                "-",
                color="0.45",
                lw=0.7,
                zorder=1,
            )
        inner = [k for k in pos if k not in (0, 1)]
        ax.plot(
            [pos[k][0] for k in inner],
            [pos[k][1] for k in inner],
            ls="none",
            marker="o",
            ms=2.2,
            mfc="0.55",
            mec="black",
            mew=0.35,
            zorder=2,
        )
        ax.plot(
            [pos[0][0], pos[1][0]],
            [pos[0][1], pos[1][1]],
            ls="none",
            marker="o",
            ms=4.0,
            mfc="black",
            mec="black",
            zorder=3,
        )
        ax.annotate("A", pos[0], textcoords="offset points", xytext=(4, 1), fontsize=6)
        ax.annotate("B", pos[1], textcoords="offset points", xytext=(4, -6), fontsize=6)
        n_t = (2 * 4**t + 4) // 3
        ax.set_title(rf"$t={t}$ ($n_t={n_t}$)", fontsize=6.5, pad=2)
        m = 1.18
        ax.set(xlim=(-m, m), ylim=(-m, m))
        ax.set_aspect("equal")
        ax.axis("off")
        if j == 0:
            panel(ax, "a", x=-0.05)

    # (d) exact average multiplicity vs n: stretched exponential in sqrt(n).
    ax = fig.add_subplot(gs[1, :])
    ts = range(1, 7)
    n, mu = [], []
    for t in ts:
        n_, _S, avg = md.total_and_avg(t)
        n.append(n_)
        mu.append(avg)
    n = np.array(n, float)
    mu = np.array(mu, float)
    xx = np.linspace(np.sqrt(n.min()), np.sqrt(n.max()), 400)
    b = np.sqrt(1.5) * np.log(2)  # exact rate, ~0.849
    a = 2 * 1.576 / 3  # kappa fixed point -> amplitude ~1.05
    ax.plot(xx**2, a * np.exp(b * xx) / xx**2, "k--", lw=0.8, label=r"$a\,e^{b\sqrt{n}}/n$")
    sc(ax, n, mu, 0, r"$\langle\mu\rangle_t$ (exact)", ms=3.4)
    ax.set(xscale="log", yscale="log", xlabel=r"$n$", ylabel=r"$\langle\mu\rangle$")
    ax.legend(loc="upper left", frameon=False)
    panel(ax, "b", x=-0.11)

    save(fig, "fig-mu-diamond.pdf")


if __name__ == "__main__":
    fig_random()
    fig_local()
    fig_bubble()
    fig_real()
    fig_diamond()
