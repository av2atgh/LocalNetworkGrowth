#!/usr/bin/env python3
"""Figures for the chapter "Emergence of communities".

Backends are the authors' own verification scripts, copied verbatim from the
three "wants to be broken" papers and imported here rather than reimplemented:

  ring_sbm_verification.py      github.com/av2atgh/ring            arXiv:2607.01967
  pseudofractal_verification.py github.com/av2atgh/pseudofractalweb arXiv:2607.03010
  diamond_rg.py                 github.com/av2atgh/diamond          arXiv:2607.05954

Everything is closed-form Bayesian evidence evaluated at 80-digit precision
(mpmath); no graph is simulated, so these run in seconds.

Style follows mkfigs.py (grayscale, 5x8in trim, Computer Modern) -- importing it
sets the shared rcParams. As in chapter 7 the panels are regrouped relative to
the papers: the book prints black and white, and the papers' colour-coded
multi-alpha panels do not survive that, so each figure here carries one prior
(alpha=1) and separates series by marker shape and fill.

Usage:  python3 mkfigs_com.py       (writes fig-com-*.pdf next to this script)
"""

import importlib.util
import os

import numpy as np
import matplotlib.pyplot as plt

from mkfigs import W, sc, panel, save  # noqa: F401  (import sets book rcParams)

HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ring = _load("ring_sbm_verification")
pf = _load("pseudofractal_verification")
dia = _load("diamond_rg")

ALPHA = 1  # one prior throughout; the papers show alpha-independence separately


# --------------------------------------------------------------------------
# Figure 1: the ring. (a) the c=1 / c=2 contrast the paper headlines;
# (b) the general (c-1) n ln 2 law -- NOT in the paper; verified here for
# c = 1..4, and the reason the cycle is exactly marginal rather than merely
# unfavourable.
# --------------------------------------------------------------------------
def fig_ring():
    fig, axes = plt.subplots(1, 2, figsize=(W, W * 0.46))

    # (a) P(split) vs n for the cycle and the next-nearest-neighbour ring.
    ax = axes[0]
    ns = [n for n in range(6, 260, 2)]
    for i, c in enumerate((1, 2)):
        x = [n for n in ns if ring.valid(n, c)]
        y = [ring.prob_split(n, c, ALPHA) for n in x]
        sc(ax, x, y, i, rf"$c={c}$", ms=2.2)
    ax.axhline(0.99, color="0.6", lw=0.5, ls=":")
    rk = ring.kappa(2, ALPHA, 0.99)
    ax.axvline(rk, color="black", lw=0.7, ls="-.")
    ax.annotate(
        rf"$r_\kappa={rk}$",
        xy=(rk, 0.45),
        xytext=(rk * 1.5, 0.38),
        fontsize=6,
        arrowprops=dict(arrowstyle="-", lw=0.5, color="black"),
    )
    ax.set(xscale="log", xlabel=r"$n$", ylabel=r"$P(\mathrm{split})$", ylim=(-0.05, 1.18))
    # c=2 only reaches the top beyond n~35, so the upper-left corner is free.
    ax.legend(loc="upper left", frameon=False)
    panel(ax, "a")

    # (b) log R vs n for c = 1..4, against the (c-1) n ln 2 prediction.
    ax = axes[1]
    ns = np.array([n for n in range(8, 1030, 2)])
    for i, c in enumerate((1, 2, 3, 4)):
        x = np.array([n for n in ns if ring.valid(n, c)])
        y = np.array([float(ring.log_ratio(int(n), c, ALPHA)) for n in x])
        ax.plot(x, (c - 1) * x * np.log(2), "-", color="0.55", lw=0.7, zorder=1)
        sc(ax, x[::6], y[::6], i, rf"$c={c}$", ms=2.4)
    ax.axhline(0.0, color="black", lw=0.5)
    ax.set(xlabel=r"$n$", ylabel=r"$\log R$", ylim=(-120, 2200))
    ax.legend(loc="upper left", frameon=False, ncol=2, columnspacing=0.7)
    panel(ax, "b")

    fig.subplots_adjust(wspace=0.38)
    save(fig, "fig-com-ring.pdf")


# --------------------------------------------------------------------------
# Figure 2: communities or degrees? The pseudofractal web under a plain block
# model versus one corrected for the degree sequence. (Paper Fig. 1a.)
# --------------------------------------------------------------------------
def fig_degree():
    fig, ax = plt.subplots(figsize=(W * 0.66, W * 0.46))

    Ts = list(range(2, 9))
    n = [pf.Nof(t) for t in Ts]
    # Both series saturate at 1, so they collide there. Draw the plain one as a
    # large open circle and the corrected one as a small filled square, so the
    # overlap reads as a dot inside a ring rather than one series erasing the
    # other.
    styles = [
        dict(marker="o", mfc="white", mec="black", ms=4.6),
        dict(marker="s", mfc="black", mec="black", ms=2.4),
    ]
    for st, (fun, lab) in zip(
        styles, [(pf.logR_plain, "plain"), (pf.logR_dc, "degree-corrected")]
    ):
        y = [pf.Psplit(fun(t, ALPHA)) for t in Ts]
        ax.plot(n, y, ls="none", label=lab, zorder=3, **st)

    for x, ls in [(1095, "-."), (42, "--")]:
        ax.axvline(x, color="black", lw=0.7, ls=ls)
    ax.annotate(r"$r_\kappa=42$", xy=(42, 0.80), xytext=(50, 0.74), fontsize=6)
    ax.annotate(r"$r_\kappa=1095$", xy=(1095, 0.42), xytext=(1250, 0.36), fontsize=6)
    ax.set(
        xscale="log",
        xlabel=r"$n$",
        ylabel=r"$P(\mathrm{split})$",
        ylim=(-0.05, 1.12),
    )
    ax.legend(loc="center left", frameon=False)

    save(fig, "fig-com-degree.pdf")


# --------------------------------------------------------------------------
# Figure 3: the diamond lattice. (a) the evidence crossing that IS r_kappa;
# (b) the RG flow of the block densities to the community fixed point (K, 0).
# --------------------------------------------------------------------------
def fig_rg():
    fig, axes = plt.subplots(1, 2, figsize=(W, W * 0.46))
    q = 0.99
    thr = float(np.log(q / (1 - q)))

    # (a) log evidence vs size; r_kappa is literally where it clears the
    # threshold, so mark both crossings.
    ax = axes[0]
    ts = list(range(1, 9))
    n = [dia.Nof(t) for t in ts]
    for i, (fun, lab) in enumerate(
        [(dia.logR_dc_cf, "degree-corrected"), (dia.logR_plain_cf, "plain")]
    ):
        y = [float(fun(t, ALPHA)) for t in ts]
        sc(ax, n, y, i, lab, ms=3.0)
    ax.axhline(thr, color="black", lw=0.7, ls=":")
    ax.annotate(r"$\ln\frac{q}{1-q}$", xy=(2.2e4, thr), xytext=(6e3, 40), fontsize=5.5)
    # Crossings: r_kappa = 44 (degree-corrected) and 684 (plain). Named in the
    # caption rather than annotated, to keep the panel legible at this size.
    for x, ls in [(44, "--"), (684, "-.")]:
        ax.axvline(x, color="black", lw=0.6, ls=ls)
    ax.set(xscale="log", yscale="symlog", xlabel=r"$n$", ylabel=r"$\log R$")
    # Evidence climbs left-to-right, so the lower-right corner is the free one.
    ax.legend(loc="lower right", frameon=False)
    panel(ax, "a")

    # (b) the flow to the community fixed point (u_in, u_cross) -> (K, 0) = (2, 0).
    ax = axes[1]
    ts = list(range(1, 10))
    u_in, u_x = [], []
    for t in ts:
        # densities() -> (u11, u22, u12). Follow diamond_rg's own __main__ and
        # read u_in off block 2, the pure bundle; block 1 carries the two poles
        # and is atypical.
        _u11, u22, u12 = dia.densities(t)
        u_in.append(float(u22))
        u_x.append(float(u12))
    ax.plot(u_in, u_x, "-", color="0.55", lw=0.7, zorder=1)
    sc(ax, u_in, u_x, 0, r"$t=1,2,\ldots$", ms=3.0)
    ax.plot([2.0], [0.0], marker="*", ms=8, mfc="black", mec="black", ls="none",
            zorder=4, label=r"fixed point $(K,0)$")
    ax.set(xlabel=r"$u_{\mathrm{in}}$", ylabel=r"$u_{\mathrm{cross}}$")
    # The flow runs top-left to bottom-right, leaving the lower-left free.
    ax.legend(loc="lower left", frameon=False)
    panel(ax, "b")

    fig.subplots_adjust(wspace=0.38)
    save(fig, "fig-com-rg.pdf")


if __name__ == "__main__":
    fig_ring()
    fig_degree()
    fig_rg()
