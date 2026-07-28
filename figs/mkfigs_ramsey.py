#!/usr/bin/env python3
"""Figure: what the detection method does to the Ramsey number and the exponent.

Plots data/ramsey_motif_*.csv, produced by ramsey_motif.py. Two panels, one for
each of chapter 6's two questions, both measured with three methods on identical
networks.

  (a) the communities likelihood P_kappa(n). Adding triangles and squares to the
      alphabet pushes the local-search curve far to the right -- its clustering
      is triangles, and an alphabet containing the triangle explains that away as
      planted motifs rather than reading it as community structure -- while
      leaving duplication-split, which makes no triangles, where it was.

  (b) the number of communities against size. The flat block model tracks its own
      sqrt(n) resolution ceiling; the nested and motif models both grow almost in
      proportion to n. The exponent is a property of the instrument at least as
      much as of the rule.

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
    fig, axes = plt.subplots(1, 2, figsize=(W, W * 0.48))

    # ---- (a) the communities likelihood
    ax = axes[0]
    p_two = lambda v: sum(1 for x in v if x >= 2) / len(v)
    for model, mk, col in (("LS(n,1)", "o", "k"), ("DS(n,0.5)", "s", "0.45")):
        for key, fill, ls in (("kappa_dcsbm", col, "--"), ("kappa", "none", "-")):
            x, y = curve(thr, model, key, p_two)
            lab = (f"{model}, block model" if key == "kappa_dcsbm"
                   else f"{model}, + motifs")
            ax.plot(x, y, mk + ls, ms=2.4, lw=0.7, color=col, mfc=fill,
                    mec=col, mew=0.5, label=lab)
    ax.axhline(0.95, color="k", lw=0.5, ls=":")
    ax.set_xscale("log")
    ax.set_xlabel(r"network size $n$")
    ax.set_ylabel(r"$P_\kappa(n)$")
    # headroom above the curves so the legend sits on empty axes
    ax.set_ylim(-0.03, 1.42)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.legend(frameon=False, loc="upper left", handlelength=1.4, fontsize=4.8)
    panel(ax, "(a)")

    # ---- (b) how many communities
    ax = axes[1]
    for key, mk, col, lab in (("kappa_flat", "v", "0.6", "flat block model"),
                              ("kappa_dcsbm", "s", "0.35", "nested block model"),
                              ("kappa", "o", "k", "nested $+$ motifs")):
        x, y = curve(sca, "LS(n,1)", key, np.mean)
        ax.plot(x, y, mk + "-", ms=2.4, lw=0.7, color=col, mfc=col, mec=col,
                label=lab)
    xx = np.array([100.0, 1600.0])
    ax.plot(xx, 2.6 * (xx / 100) ** 1.0, ls="-", lw=0.5, color="0.7")
    ax.plot(xx, 4.9 * (xx / 100) ** 0.5, ls="--", lw=0.5, color="0.7")
    ax.text(1700, 40, r"$n$", fontsize=5.5, color="0.5")
    ax.text(1700, 19, r"$\sqrt{n}$", fontsize=5.5, color="0.5")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"network size $n$")
    ax.set_ylabel(r"communities $\langle\kappa\rangle$")
    ax.set_xlim(80, 2600)
    ax.legend(frameon=False, loc="upper left", handlelength=1.4, fontsize=4.8)
    panel(ax, "(b)")

    fig.tight_layout(pad=0.3, w_pad=1.0)
    save(fig, "fig-gr-ramsey.pdf")


if __name__ == "__main__":
    main()
