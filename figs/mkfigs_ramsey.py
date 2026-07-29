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
                law fitted to each method over every measured size. The flat
                model comes out near its own sqrt(n) resolution ceiling; the
                nested and motif models come out near one. The exponent is a
                property of the instrument at least as much as of the rule.

$DS(n,0.5)$ is on top and $LS(n,1)$ below, so the row that moves is the lower one.

Style follows mkfigs.py (grayscale, 5x8in trim, Computer Modern).

Usage:  python3 mkfigs_ramsey.py   (writes fig-gr-ramsey.pdf here)
"""
import csv
import glob
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter

from mkfigs import W, panel, save  # noqa: F401  (import sets book rcParams)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

INT = ("n", "seed", "m", "kappa", "kappa_dcsbm", "kappa_flat",
       "n_edge", "n_tri", "n_sq")

# key, marker, colour, label -- shared by every panel so the legend reads once
METHODS = (("kappa_flat", "v", "0.62", "flat block model"),
           ("kappa_dcsbm", "s", "0.38", "nested block model"),
           ("kappa", "o", "k", r"nested $+$ motifs"))

# (data key, panel labels, the book's own math styling for the rule)
# The communities likelihood belongs with the motif chapter, which is about when
# communities appear; the growth of their number belongs with the chapter on
# resolution, which is about how many there are. So the two are drawn separately.
LIKELIHOOD = {
    "fig-gr-ramsey.pdf": (("DS(n,0.5)", "(a)", r"$DS(n,0.5)$"),
                          ("LS(n,1)", "(b)", r"$LS(n,1)$")),
    # the two rules whose motif is the square: one that makes them separably,
    # one that piles them into bicliques
    "fig-gr-squares.pdf": (("BB(n,2)", "(a)", r"$BB(n,2)$"),
                           ("DD(n,0.5)", "(b)", r"$DD(n,0.5)$")),
}
SCALING = (("LS(n,1)", "(a)", r"$LS(n,1)$"), ("DS(n,0.5)", "(b)", r"$DS(n,0.5)$"),
           ("BB(n,2)", "(c)", r"$BB(n,2)$"), ("DD(n,0.5)", "(d)", r"$DD(n,0.5)$"))

# fitted over every measured size, as the chapter quotes it
FIT_FROM = 0.0


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


def reps(rows, model):
    """How many realisations went into each point, for the caption."""
    sub = {}
    for r in rows:
        if r["model"] == model:
            sub[r["n"]] = sub.get(r["n"], 0) + 1
    counts = sorted(set(sub.values()))
    return counts, sorted(sub)


def _pk_rows(thr, sca):
    """P_kappa from the threshold runs, extended above their grid by the scaling
    runs. Sizes the fine grid already covers are not pooled with the coarser
    ones: mixing twenty noisier realisations into a hundred moves the 95%
    crossing that defines the Ramsey number."""
    have = {(r["model"], r["n"]) for r in thr}
    return thr + [r for r in sca if (r["model"], r["n"]) not in have]


def likelihood_figure(out, rows):
    thr, sca = load("ramsey_motif_threshold*.csv"), load("ramsey_motif_scaling*.csv")
    both = _pk_rows(thr, sca)
    p_two = lambda v: sum(1 for x in v if x >= 2) / len(v)
    fig, axes = plt.subplots(1, 2, figsize=(W, W * 0.50))
    for ax, (model, lab, tag) in zip(axes, rows):
        for key, mk, col, name in METHODS:
            x, y = curve(both, model, key, p_two)
            ax.plot(x, y, mk + "-", ms=2.2, lw=0.7, color=col, mfc=col,
                    mec=col, label=name)
        ax.axhline(0.95, color="k", lw=0.5, ls=":")
        ax.set_xscale("log")
        ax.set_xlabel(r"size $n$")
        ax.set_ylabel(r"$P_\kappa(n)$")
        ax.set_ylim(-0.04, 1.34)
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.text(0.97, 0.97, tag, transform=ax.transAxes, ha="right", va="top",
                fontsize=6.5)
        if lab == "(a)":
            ax.legend(frameon=False, loc="upper left", handlelength=1.3,
                      fontsize=4.8, borderaxespad=0.2)
        panel(ax, lab)
    fig.tight_layout(pad=0.3, w_pad=1.0)
    save(fig, out)


def scaling_figure(out="fig-gr-scaling.pdf"):
    sca = load("ramsey_motif_scaling*.csv")
    fig, axes = plt.subplots(2, 2, figsize=(W, W * 0.86))
    for ax, (model, lab, tag) in zip(axes.ravel(), SCALING):
        for key, mk, col, name in METHODS:
            x, y = curve(sca, model, key, np.mean)
            ax.plot(x, y, mk, ms=2.4, color=col, mfc=col, mec=col, ls="none")
            # a power law is only fitted where the mean actually rises with size
            if np.all(np.diff(y) > 0):
                b, a = np.polyfit(np.log(x), np.log(y), 1)
                ax.plot(x, np.exp(a) * x ** b, "-", lw=0.7, color=col,
                        label=rf"{name}, $n^{{{b:.2f}}}$")
            else:
                ax.plot([], [], mk, ms=2.4, color=col, mfc=col, mec=col,
                        ls="none", label=rf"{name}, bimodal")
        ax.set_xscale("log")
        ax.set_yscale("log")
        # decade ticks only: DD spans less than a decade and matplotlib would
        # otherwise label the minor ticks, which then run into one another
        ax.set_xlim(75, 1.35 * max(curve(sca, model, "kappa", np.mean)[0]))
        ax.xaxis.set_major_locator(LogLocator(base=10.0))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_ylabel(r"$\langle\kappa\rangle$")
        ax.text(0.97, 0.06, tag, transform=ax.transAxes, ha="right",
                va="bottom", fontsize=6.5)
        ax.legend(frameon=False, loc="upper left", handlelength=1.3,
                  fontsize=4.4, borderaxespad=0.2)
        panel(ax, lab)
    for ax in axes[1]:
        ax.set_xlabel(r"size $n$")
    fig.tight_layout(pad=0.3, w_pad=1.0, h_pad=0.6)
    save(fig, out)


def main():
    for out, rows in LIKELIHOOD.items():
        likelihood_figure(out, rows)
    scaling_figure()


if __name__ == "__main__":
    main()
