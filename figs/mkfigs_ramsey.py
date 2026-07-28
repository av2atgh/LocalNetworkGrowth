#!/usr/bin/env python3
"""Figure: what the detection method does to the Ramsey number and the exponent.

Plots data/ramsey_motif_*.csv, produced by ramsey_motif.py. Four panels, a row
per growth rule, each row carrying chapter 6's two questions side by side, with
all three methods measured on identical networks.

  left column   the communities likelihood P_kappa(n), with the 95% line whose
                crossing defines the Ramsey community number. Adding triangles
                and squares to the alphabet pushes local search far to the right
                -- its clustering IS triangles, and a method that can name a
                triangle stops counting it as evidence for a community -- while
                leaving duplication-split, which makes none, where it was.

  right column  the number of communities against size, with the best power
                law fitted to each method over n >= 400 -- the range the chapter
                quotes. The flat model comes out near one half, its own sqrt(n)
                resolution ceiling; the nested and motif models come out near
                one. The exponent is a property of the instrument at least as
                much as of the rule.

DS(n,0.5) is on top and LS(n,1) below, so the row that moves is the lower one.

Style follows mkfigs.py (grayscale, 5x8in trim, Computer Modern).

Usage:  python3 mkfigs_ramsey.py   (writes fig-gr-ramsey.pdf here)
"""
import csv
import glob
import os

import numpy as np
import matplotlib.pyplot as plt

from mkfigs import W, panel, save  # noqa: F401  (import sets book rcParams)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

INT = ("n", "seed", "m", "kappa", "kappa_dcsbm", "kappa_flat",
       "n_edge", "n_tri", "n_sq")

# key, marker, colour, label -- shared by every panel so the legend reads once
METHODS = (("kappa_flat", "v", "0.62", "flat block model"),
           ("kappa_dcsbm", "s", "0.38", "nested block model"),
           ("kappa", "o", "k", r"nested $+$ motifs"))

ROWS = (("DS(n,0.5)", "(a)", "(b)"), ("LS(n,1)", "(c)", "(d)"))

# the chapter quotes the exponent over this range, so the figure fits it there
FIT_FROM = 400.0


def load(pattern):
    rows = []
    for p in sorted(glob.glob(os.path.join(DATA, pattern))):
        with open(p) as fh:
            for r in csv.DictReader(fh):
                rows.append({k: (int(v) if k in INT and v not in ("", "nan")
                                 else v) for k, v in r.items()})
    return rows


def curve(rows, model, key, reduce_fn):
    sub = {}
    for r in rows:
        if r["model"] == model and isinstance(r.get(key), int):
            sub.setdefault(r["n"], []).append(r[key])
    ns = sorted(sub)
    return np.array(ns, float), np.array([reduce_fn(sub[n]) for n in ns])


def main():
    thr = load("ramsey_motif_threshold*.csv")     # fine grid, 100 realisations
    sca = load("ramsey_motif_scaling*.csv")       # coarse grid, to n = 1600
    fig, axes = plt.subplots(2, 2, figsize=(W, W * 0.86))
    p_two = lambda v: sum(1 for x in v if x >= 2) / len(v)

    for row, (model, la, lb) in enumerate(ROWS):
        # ---- left: the communities likelihood
        ax = axes[row, 0]
        for key, mk, col, lab in METHODS:
            x, y = curve(thr, model, key, p_two)
            ax.plot(x, y, mk + "-", ms=2.2, lw=0.7, color=col, mfc=col,
                    mec=col, label=lab)
        ax.axhline(0.95, color="k", lw=0.5, ls=":")
        ax.set_xscale("log")
        ax.set_ylabel(r"$P_\kappa(n)$")
        ax.set_ylim(-0.04, 1.34)
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.text(0.97, 0.06, model, transform=ax.transAxes, ha="right",
                va="bottom", fontsize=6)
        if row == 0:
            ax.legend(frameon=False, loc="upper left", handlelength=1.3,
                      fontsize=4.8, borderaxespad=0.2)
        panel(ax, la)

        # ---- right: how many communities, with a fitted exponent per method
        ax = axes[row, 1]
        for key, mk, col, lab in METHODS:
            x, y = curve(sca, model, key, np.mean)
            ax.plot(x, y, mk, ms=2.6, color=col, mfc=col, mec=col, ls="none")
            # fitted over the same range the chapter quotes
            fit = x >= FIT_FROM
            b, a = np.polyfit(np.log(x[fit]), np.log(y[fit]), 1)
            xx = np.array([x[fit].min(), x[fit].max()])
            ax.plot(xx, np.exp(a) * xx ** b, "-", lw=0.7, color=col,
                    label=rf"{lab}, $n^{{{b:.2f}}}$")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylabel(r"$\langle\kappa\rangle$")
        ax.set_xlim(75, 2600)
        ax.text(0.97, 0.06, model, transform=ax.transAxes, ha="right",
                va="bottom", fontsize=6)
        ax.legend(frameon=False, loc="upper left", handlelength=1.3,
                  fontsize=4.6, borderaxespad=0.2)
        panel(ax, lb)

    for ax in axes[1]:
        ax.set_xlabel(r"network size $n$")
    for ax in axes[0]:
        ax.set_xticklabels([])

    fig.tight_layout(pad=0.3, w_pad=1.0, h_pad=0.6)
    save(fig, "fig-gr-ramsey.pdf")


if __name__ == "__main__":
    main()
