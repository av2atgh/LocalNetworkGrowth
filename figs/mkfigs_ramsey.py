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
PAIRS = {
    "fig-gr-ramsey.pdf": (("DS(n,0.5)", "(a)", "(b)", r"$DS(n,0.5)$"),
                          ("LS(n,1)", "(c)", "(d)", r"$LS(n,1)$")),
    # the two rules whose motif is the square: one that makes them separably,
    # one that piles them into bicliques
    "fig-gr-squares.pdf": (("BB(n,2)", "(a)", "(b)", r"$BB(n,2)$"),
                           ("DD(n,0.5)", "(c)", "(d)", r"$DD(n,0.5)$")),
}

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


def figure(out, rows):
    thr = load("ramsey_motif_threshold*.csv")     # fine grid, 100 realisations
    sca = load("ramsey_motif_scaling*.csv")       # coarse grid, larger n
    fig, axes = plt.subplots(2, 2, figsize=(W, W * 0.86))
    p_two = lambda v: sum(1 for x in v if x >= 2) / len(v)

    # P_kappa is taken from the threshold runs, a hundred networks a size, and
    # extended above their grid by the scaling runs, twenty a size. Sizes the
    # threshold grid already covers are NOT pooled with the coarser ones: mixing
    # twenty noisier realisations into a hundred moves the 95% crossing that
    # defines the Ramsey number, and the tables quote the threshold runs.
    have = {(r["model"], r["n"]) for r in thr}
    both = thr + [r for r in sca if (r["model"], r["n"]) not in have]

    for row, (model, la, lb, tag) in enumerate(rows):
        # ---- left: the communities likelihood
        ax = axes[row, 0]
        for key, mk, col, lab in METHODS:
            x, y = curve(both, model, key, p_two)
            ax.plot(x, y, mk + "-", ms=2.2, lw=0.7, color=col, mfc=col,
                    mec=col, label=lab)
        ax.axhline(0.95, color="k", lw=0.5, ls=":")
        ax.set_xscale("log")
        ax.set_ylabel(r"$P_\kappa(n)$")
        ax.set_ylim(-0.04, 1.34)
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        # top right: the curves saturate at 1 and the axis runs to 1.34, so
        # this corner is clear in both rows where the lower one is not
        ax.text(0.97, 0.97, tag, transform=ax.transAxes, ha="right",
                va="top", fontsize=6.5)
        if row == 0:
            ax.legend(frameon=False, loc="upper left", handlelength=1.3,
                      fontsize=4.8, borderaxespad=0.2)
        panel(ax, la)

        # ---- right: how many communities, with a fitted exponent per method
        ax = axes[row, 1]
        for key, mk, col, lab in METHODS:
            x, y = curve(sca, model, key, np.mean)
            ax.plot(x, y, mk, ms=2.6, color=col, mfc=col, mec=col, ls="none")
            # A power law is only fitted where the mean actually rises with size.
            # For BB(n,2) under the motif alphabet it does not: the answer is
            # bimodal, either one community or as many as the block model finds,
            # and the mean of the two turns over. Quoting an exponent for that
            # would be quoting the mixing ratio, not a law.
            fit = x >= FIT_FROM
            if np.all(np.diff(y[fit]) > 0):
                b, a = np.polyfit(np.log(x[fit]), np.log(y[fit]), 1)
                xx = np.array([x[fit].min(), x[fit].max()])
                ax.plot(xx, np.exp(a) * xx ** b, "-", lw=0.7, color=col,
                        label=rf"{lab}, $n^{{{b:.2f}}}$")
            else:
                ax.plot([], [], mk, ms=2.6, color=col, mfc=col, mec=col,
                        ls="none", label=rf"{lab}, bimodal")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylabel(r"$\langle\kappa\rangle$")
        ax.set_xlim(75, 1.7 * max(x))
        ax.text(0.97, 0.06, tag, transform=ax.transAxes, ha="right",
                va="bottom", fontsize=6.5)
        ax.legend(frameon=False, loc="upper left", handlelength=1.3,
                  fontsize=4.6, borderaxespad=0.2)
        panel(ax, lb)

    for ax in axes[1]:
        ax.set_xlabel(r"network size $n$")
    for ax in axes[0]:
        ax.set_xticklabels([])

    fig.tight_layout(pad=0.3, w_pad=1.0, h_pad=0.6)
    save(fig, out)
    for model, *_ in rows:
        rt, nt = reps(both, model)
        rs, ns = reps(sca, model)
        print(f"    {model}: P_kappa from {rt} realisations per size, "
              f"n = {min(nt)}..{max(nt)}; <kappa> from {rs} per size, "
              f"n = {min(ns)}..{max(ns)}")


def main():
    for out, rows in PAIRS.items():
        figure(out, rows)


if __name__ == "__main__":
    main()
