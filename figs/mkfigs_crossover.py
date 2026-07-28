#!/usr/bin/env python3
"""Figure for the L* crossover test of chapter 7 on real networks.

Plots data/crossover_results.csv, produced by crossover_measure.py.  Two panels,
one for each half of the chapter's claim.

  (a) the transfer matrix is right.  A degree-preserving rewiring's own cycle
      counts against the geometric profile its transfer matrix predicts, with
      the corrected branching factor kappa = <k^2-k>/<k> and with the <k^2>/<k>
      of the chapter as written.  The corrected one sits on 1 at every length;
      the other drifts away by an order of magnitude.

  (b) the deficit is a level, not a slope.  ln(true_L / rewired_L) against L.
      The chapter predicts a line of slope -ln kappa crossing zero at
      L* = ln n/ln kappa (dashed, drawn for each network shown).  What happens
      instead is a large and nearly flat offset: real networks carry many times
      more cycles than their rewirings at every length alike.

Style follows mkfigs.py (grayscale, 5x8in trim, Computer Modern).

Usage:  python3 mkfigs_crossover.py   (writes fig-gr-crossover.pdf here)
"""
import os

import numpy as np
import matplotlib.pyplot as plt

from mkfigs import W, panel, save, read  # noqa: F401  (import sets rcParams)

LMIN, LMAX = 3, 8

# Four systems spanning the range of n and kappa that the test can reach, and
# spanning the outcome: yeast is the one network that does what the chapter
# predicts, the road network and the two large graphs do not.
SHOW = [("interactome_yeast", "yeast interactome", "o", "k"),
        ("chicago_road", "Chicago streets", "s", "0.35"),
        ("gnutella/31", "Gnutella overlay", "^", "0.55"),
        ("soc_net_comms/amazon", "Amazon co-purchase", "D", "0.7")]


def series(row, prefix):
    L, y = [], []
    for ell in range(LMIN, LMAX + 1):
        v = row.get(f"{prefix}{ell}")
        if v is None or (isinstance(v, float) and not np.isfinite(v)) or v == "":
            continue
        L.append(ell)
        y.append(float(v))
    return np.array(L), np.array(y)


def main():
    df = read("crossover_results.csv").set_index("network")
    fig, axes = plt.subplots(1, 2, figsize=(W, W * 0.48))

    # ---- (a) the null's own cycles against its transfer-matrix profile
    ax = axes[0]
    for kind, marker, colour, lab in (
            ("kap", "o", "k", r"$\kappa=\langle k^2-k\rangle/\langle k\rangle$"),
            ("lam", "v", "0.6", r"$\lambda=\langle k^2\rangle/\langle k\rangle$")):
        xs, ys = [], []
        for name in df.index:
            row = df.loc[name]
            if row["n"] < 500:            # finite-n exclusion swamps the test
                continue
            base = row["kap_rank1"] if kind == "kap" else row["lam_rank1"]
            L, a = series(row, "A")
            keep = a > 0
            xs += list(L[keep])
            ys += list(a[keep] / (base ** L[keep] / (2 * L[keep])))
        ax.plot(np.array(xs) + (0.12 if kind == "lam" else -0.12), ys, marker,
                ms=2.2, mfc="none" if kind == "lam" else colour, mec=colour,
                mew=0.5, ls="none", label=lab)
    ax.axhline(1.0, color="k", lw=0.6, ls=":")
    ax.set_yscale("log")
    ax.set_xlabel(r"cycle length $L$")
    ax.set_ylabel(r"rewired count $/$ $\mathrm{(profile)}^L/2L$")
    ax.set_ylim(2e-2, 5e1)
    ax.legend(frameon=False, loc="lower left", handlelength=1.0, fontsize=5.5)
    panel(ax, "(a)")

    # ---- (b) the deficit against the predicted line
    ax = axes[1]
    for name, lab, marker, colour in SHOW:
        if name not in df.index:
            continue
        row = df.loc[name]
        L, t = series(row, "C")
        _, a = series(row, "A")
        m = min(len(L), len(a))
        L, t, a = L[:m], t[:m], a[:m]
        keep = (t > 0) & (a > 0)
        ax.plot(L[keep], np.log(t[keep] / a[keep]), marker + "-", ms=2.6,
                lw=0.7, color=colour, mfc=colour, label=lab)
        # what chapter 7 predicts: slope -ln kappa through zero at ln n/ln kappa
        kap = row["kap_rank1"]
        lstar = np.log(row["n"]) / np.log(kap)
        xx = np.array([LMIN, LMAX])
        ax.plot(xx, -np.log(kap) * (xx - lstar), ls="--", lw=0.5, color=colour)
    ax.axhline(0.0, color="k", lw=0.6, ls=":")
    ax.set_xlabel(r"cycle length $L$")
    ax.set_ylabel(r"$\ln(\,\mathrm{true}_L / \mathrm{rewired}_L)$")
    ax.set_ylim(-6, 11)
    ax.legend(frameon=False, loc="upper right", handlelength=1.2, fontsize=5.5)
    panel(ax, "(b)")

    fig.tight_layout(pad=0.3, w_pad=1.0)
    save(fig, "fig-gr-crossover.pdf")


if __name__ == "__main__":
    main()
