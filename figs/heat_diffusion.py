#!/usr/bin/env python3
r"""
heat_diffusion.py -- classical diffusion on the growth rules of this book, for
the chapter "Diffusion and the scales it finds".

Three instruments, in order of how much they know:

  * the stationary distribution of a random walk, k_i/2m, which knows the
    degrees and nothing else -- and the mean return time 2m/k_i, which is
    exactly as blind;

  * the mean first passage time between distinct nodes, which is not blind, and
    whose asymmetry defines the random-walk centrality of Noh & Rieger,
    Phys. Rev. Lett. 92, 118701 (2004);

  * the spectral entropy and specific heat of the Laplacian density operator
    rho(tau) = e^{-tau L}/Tr e^{-tau L}, following Villegas et al., Nature
    Physics 19, 445 (2023) and Phys. Rev. Research 4, 033196 (2022).  With
    S = -sum mu log mu / log n and C = -dS/dlog tau, a flat C means no
    characteristic scale and a peak in C means one.

The chapter's quantitative claim lives on the (b,s) diamond, where both of the
following are derivable and are checked here:

    peak spacing in tau  =  s^2          (mass x resistance per generation)
    C log n plateau      =  d_s/2 = ln(bs) / (2 ln s)

Simulation output is cached in data/heat-diffusion.csv; delete it or pass
--force to re-measure.

Usage:  python3 heat_diffusion.py [--force]
"""

import os
import sys

import numpy as np
import networkx as nx
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
CSV = os.path.join(HERE, "data", "heat-diffusion.csv")
DCSV = os.path.join(HERE, "data", "heat-diamond.csv")

from qwalk_ipr import (RULES, pseudofractal, rec_triangle, diamond, giant,
                       MK, panel, save, W, plt)

NPTS = 1500
TAU_LO, TAU_HI = 1e-3, 1e6


# ------------------------------------------------------------------ the walk

def spectrum(g):
    """Eigenvalues of the combinatorial Laplacian L = D - A."""
    lam = np.linalg.eigvalsh(nx.laplacian_matrix(g).toarray().astype(float))
    lam[lam < 0] = 0.0
    return lam


def heat(g, n_points=NPTS, lo=TAU_LO, hi=TAU_HI, lam=None):
    """Spectral entropy S(tau) and specific heat C(tau) = -dS/dlog tau.

    S is normalized by log n, as in Villegas et al., so that S runs from 1 at
    tau = 0 (every node its own island) to 0 as tau -> infinity (diffusion has
    reached everywhere).
    """
    n = g.number_of_nodes()
    if lam is None:
        lam = spectrum(g)
    tau = np.exp(np.linspace(np.log(lo), np.log(hi), n_points))
    # exp(-lam tau) underflows harmlessly; the zero mode keeps Z >= 1
    E = np.exp(-np.outer(tau, lam))
    Z = E.sum(1, keepdims=True)
    mu = E / Z
    S = -(mu * np.log(mu + 1e-300)).sum(1) / np.log(n)
    C = -np.gradient(S, np.log(tau))
    return tau, S, C


def peaks(tau, C, thresh=0.04, sep=0.7):
    """Local maxima of C above a floor and separated in log tau."""
    idx = [i for i in range(2, len(C) - 2)
           if C[i] > thresh and C[i] >= C[i - 1] and C[i] >= C[i + 1]
           and C[i] > C[i - 2] and C[i] > C[i + 2]]
    keep = []
    for i in sorted(idx, key=lambda j: -C[j]):
        if all(abs(np.log(tau[i]) - np.log(t)) > sep for t, _ in keep):
            keep.append((tau[i], C[i]))
    return sorted(keep)


def plateau(tau, C, n, lo, hi):
    """Mean of C log n over a window, the quantity that equals d_s/2."""
    m = (tau > lo) & (tau < hi)
    return float((C[m] * np.log(n)).mean()) if m.sum() > 10 else float("nan")


# ------------------------------------------------------ first passage times

def mfpt(g):
    """Mean first passage time matrix, T[i, j] = steps from i to j.

    For a connected undirected graph, T_ij = 2m (Lp_jj - Lp_ij) with Lp the
    Moore-Penrose pseudoinverse of L = D - A.  Verified against the two cases
    with closed forms: the complete graph, where T_ij = n - 1 for i != j, and
    the ring, where T_ij = d(i,j) (n - d(i,j)).
    """
    n = g.number_of_nodes()
    m = g.number_of_edges()
    L = nx.laplacian_matrix(g).toarray().astype(float)
    Lp = np.linalg.pinv(L)
    d = np.diag(Lp)
    return 2 * m * (d[None, :] - Lp)


def check_mfpt():
    """The two closed forms, as a test of the formula above."""
    out = []
    K = nx.complete_graph(9)
    T = mfpt(K)
    off = T[~np.eye(9, dtype=bool)]
    out.append(("complete K9", 8.0, float(off.mean()), float(off.std())))
    R = nx.cycle_graph(21)
    T = mfpt(R)
    pred = [d * (21 - d) for d in range(1, 11)]
    got = [T[0, d] for d in range(1, 11)]
    out.append(("ring C21", float(np.mean(pred)), float(np.mean(got)),
                float(np.max(np.abs(np.array(pred) - np.array(got))))))
    return out


def rw_centrality(g):
    """Noh-Rieger random-walk centrality, C_i = P_i^inf / tau_i.

    tau_i is read off the MFPT asymmetry, which is the definition that does not
    require summing the relaxation series: T_ij - T_ji = 1/C_j - 1/C_i, so
    1/C_i is fixed up to a constant by averaging over j.  The constant drops out
    of every comparison made in the chapter, so it is set by the mean.
    """
    T = mfpt(g)
    inv = (T - T.T).mean(1)          # = 1/C_i + const, up to sign convention
    inv = inv - inv.mean()
    return -inv                      # larger => more central


# ------------------------------------------------------------------ the cases

def build_cases():
    """(label, class, graph) for every network the chapter measures."""
    n = 1000
    out = []
    for lab in ("LS(n,1)", "BB(n,1)", "BB(n,2)", "BB(n,3)", "BB(n,5)",
                "DS(n,0.5)", "DD(n,0.5)"):
        out.append((lab, "local", giant(RULES[lab][0](n, 1))))
    out.append(("BA(n,2)", "control", giant(RULES["BA(n,2)"][0](n, 1))))
    out.append(("ER(n,4)", "control", giant(RULES["ER(n,4)"][0](n, 1))))
    out.append(("ring", "control", nx.cycle_graph(n)))
    out.append(("2D lattice", "control", nx.convert_node_labels_to_integers(
        nx.grid_2d_graph(32, 32))))
    out.append(("pseudofractal", "lattice", pseudofractal(6)))
    out.append(("rec.triangle", "lattice", rec_triangle(8)))
    out.append(("diamond(2,2)", "lattice", giant(diamond(6, 2, 2))))
    out.append(("diamond(2,3)", "lattice", giant(diamond(4, 2, 3))))
    return out


DIAMONDS = [(2, 2, (3, 4, 5, 6)), (2, 3, (2, 3, 4)), (2, 4, (2, 3)),
            (3, 2, (2, 3, 4)), (4, 2, (2, 3))]


def measure():
    rows = []
    for lab, cls, g in build_cases():
        n = g.number_of_nodes()
        lam = spectrum(g)
        tau, S, C = heat(g, lam=lam)
        pk = peaks(tau, C)
        k = np.array([d for _, d in g.degree()], float)
        rows.append(dict(
            label=lab, cls=cls, n=n, m=g.number_of_edges(), k=k.mean(),
            k2_over_k2=(k ** 2).mean() / k.mean() ** 2,   # degree heterogeneity
            C_of_1=float(np.interp(1.0, tau, C)),
            Cmax=float(C.max()), tau_Cmax=float(tau[int(np.argmax(C))]),
            n_peaks=len(pk),
            peak_taus=";".join(f"{t:.4g}" for t, _ in pk),
            plateau=plateau(tau, C, n, 3.0, n),
            flatness=float(C[(tau > 1) & (tau < n)].std()
                           / max(C[(tau > 1) & (tau < n)].mean(), 1e-12)),
            lam2=float(np.sort(lam)[1]), lam_max=float(lam.max())))
        print(f"  {lab:15s} n={n:5d} peaks={len(pk)} "
              f"plateau={rows[-1]['plateau']:.3f}", flush=True)
    pd.DataFrame(rows).to_csv(CSV, index=False)
    print("wrote", os.path.relpath(CSV, HERE))

    drows = []
    for b, s, ts in DIAMONDS:
        for t in ts:
            g = giant(diamond(t, b, s))
            n = g.number_of_nodes()
            if n > 2800:
                continue
            tau, S, C = heat(g)
            pk = [p[0] for p in peaks(tau, C)]
            ratios = [pk[i + 1] / pk[i] for i in range(len(pk) - 1)]
            drows.append(dict(
                b=b, s=s, t=t, n=n, ds_half=np.log(b * s) / (2 * np.log(s)),
                plateau=plateau(tau, C, n, 3.0, s ** (2 * t) / 8),
                n_peaks=len(pk),
                ratio_mean=float(np.mean(ratios)) if ratios else float("nan"),
                s2=s * s,
                peak_taus=";".join(f"{x:.4g}" for x in pk)))
            print(f"  ({b},{s}) t={t} n={n:5d} peaks={len(pk)} "
                  f"plateau={drows[-1]['plateau']:.3f} "
                  f"pred={drows[-1]['ds_half']:.3f}", flush=True)
    pd.DataFrame(drows).to_csv(DCSV, index=False)
    print("wrote", os.path.relpath(DCSV, HERE))
    return pd.read_csv(CSV), pd.read_csv(DCSV)


# ------------------------------------------------------------------- figures

def fig_entropy(df):
    """S and C against tau, for one local rule and the two controls that
    bracket it: a lattice with a scale and a random graph without one."""
    fig, axes = plt.subplots(1, 2, figsize=(W, W * 0.44))
    show = [("BB(n,1)", "$BB(n,1)$"), ("ring", "ring"), ("ER(n,4)", "$ER(n,4)$")]
    cases = {lab: g for lab, _, g in build_cases()}
    for i, (lab, nice) in enumerate(show):
        g = cases[lab]
        tau, S, C = heat(g)
        axes[0].plot(tau, S, lw=0.9, ls=["-", "--", ":"][i], c="0.2", label=nice)
        axes[1].plot(tau, C * np.log(g.number_of_nodes()), lw=0.9,
                     ls=["-", "--", ":"][i], c="0.2", label=nice)
    for ax, yl in ((axes[0], "$S(\\tau)$"), (axes[1], "$C(\\tau)\\,\\log n$")):
        ax.set_xscale("log")
        ax.set_xlabel(r"$\tau$")
        ax.set_ylabel(yl)
        ax.set_xlim(1e-2, 1e5)
    axes[0].legend(frameon=False, loc="lower left", fontsize=5.4,
                   handletextpad=0.4, borderpad=0.1)
    panel(axes[0], "(a)", x=-0.30)
    panel(axes[1], "(b)", x=-0.30)
    fig.tight_layout(pad=0.3)
    save(fig, "fig-hc-entropy.pdf")


def fig_diamond(dd):
    """The two derivable results: the ladder, and its plateau height."""
    fig, axes = plt.subplots(1, 2, figsize=(W, W * 0.46))

    ax = axes[0]
    for i, t in enumerate((4, 5, 6)):
        g = giant(diamond(t, 2, 2))
        tau, S, C = heat(g)
        # the three curves are distinguished by line style and named in the
        # caption: there is no clear space left in the panel for a legend
        ax.plot(tau, C * np.log(g.number_of_nodes()), lw=0.8,
                ls=["-", "--", ":"][i], c="0.2")
    ax.axhline(1.0, lw=0.6, ls="-", c="0.6")
    ax.text(3e3, 1.07, r"$d_s/2=1$", fontsize=5.6, color="0.4")
    ax.set_xscale("log"); ax.set_xlim(1e-1, 1e6)
    ax.set_xlabel(r"$\tau$"); ax.set_ylabel(r"$C(\tau)\,\log n$")
    panel(ax, "(a)", x=-0.30)

    ax = axes[1]
    for i, (b, s) in enumerate([(2, 2), (2, 3), (2, 4), (3, 2), (4, 2)]):
        d = dd[(dd.b == b) & (dd.s == s)].sort_values("n")
        if not len(d):
            continue
        ax.plot(d.ds_half, d.plateau, ls="none", ms=3.2,
                label=f"$({b},{s})$", **MK[i % len(MK)])
    lim = [0.6, 2.6]
    ax.plot(lim, lim, ls=":", lw=0.7, c="0.35")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel(r"$d_s/2 = \ln(bs)/2\ln s$")
    ax.set_ylabel(r"measured $C\log n$")
    ax.legend(frameon=False, loc="upper left", fontsize=5.2,
              handletextpad=0.3, labelspacing=0.2, borderpad=0.1)
    panel(ax, "(b)", x=-0.30)

    fig.tight_layout(pad=0.3)
    save(fig, "fig-hc-diamond.pdf")


def fig_rules(df):
    """What the specific heat does and does not separate."""
    fig, axes = plt.subplots(1, 2, figsize=(W, W * 0.46))
    loc = df[df.cls == "local"]
    ctl = df[df.cls == "control"]
    lat = df[df.cls == "lattice"]

    ax = axes[0]
    for i, (d, lab) in enumerate([(loc, "local rules"), (ctl, "controls"),
                                  (lat, "lattices")]):
        ax.plot(d.k2_over_k2, d.Cmax, ls="none", ms=3.2, label=lab, **MK[i])
    ax.set_xscale("log")
    ax.set_xlabel(r"$\langle k^2\rangle/\langle k\rangle^2$")
    ax.set_ylabel(r"$\max_\tau C$")
    ax.legend(frameon=False, loc="lower right", fontsize=5.2,
              handletextpad=0.3, borderpad=0.1)
    panel(ax, "(a)", x=-0.30)

    ax = axes[1]
    for i, (d, lab) in enumerate([(loc, "local rules"), (ctl, "controls"),
                                  (lat, "lattices")]):
        ax.plot(d.k2_over_k2, d.n_peaks, ls="none", ms=3.2, label=lab, **MK[i])
    ax.set_xscale("log")
    ax.set_xlabel(r"$\langle k^2\rangle/\langle k\rangle^2$")
    ax.set_ylabel("peaks in $C$")
    ax.set_ylim(-0.4, 5.4)
    panel(ax, "(b)", x=-0.30)

    fig.tight_layout(pad=0.3)
    save(fig, "fig-hc-rules.pdf")


# -------------------------------------------------------------------- tables

TEX = {"LS(n,1)": r"$LS(n,1)$", "BB(n,1)": r"$BB(n,1)$", "BB(n,2)": r"$BB(n,2)$",
       "BB(n,3)": r"$BB(n,3)$", "BB(n,5)": r"$BB(n,5)$",
       "DS(n,0.5)": r"$DS(n,0.5)$", "DD(n,0.5)": r"$DD(n,0.5)$",
       "BA(n,2)": r"$BA(n,2)$", "ER(n,4)": r"$ER(n,4)$", "ring": "ring",
       "2D lattice": "square lattice", "pseudofractal": "pseudofractal",
       "rec.triangle": "rec.\\ triangle", "diamond(2,2)": r"diamond $(2,2)$",
       "diamond(2,3)": r"diamond $(2,3)$"}


def table(df):
    out = [r"\begin{center}", r"\footnotesize\setlength{\tabcolsep}{4pt}",
           r"\begin{tabular}{lrrrrrr}", r"\hline",
           r"network & $n$ & $\langle k^2\rangle/\langle k\rangle^2$ & "
           r"$C\log n$ & $\max C$ & peaks & flatness\\", r"\hline"]
    for cls, head in (("local", r"\emph{local rules}"),
                      ("control", r"\emph{controls}"),
                      ("lattice", r"\emph{deterministic lattices}")):
        d = df[df.cls == cls]
        if not len(d):
            continue
        out.append(r"\multicolumn{7}{l}{" + head + r"}\\")
        for _, r in d.iterrows():
            out.append(f"\\quad {TEX.get(r.label, r.label)} & ${r.n:.0f}$ & "
                       f"${r.k2_over_k2:.2f}$ & ${r.plateau:.2f}$ & "
                       f"${r.Cmax:.2f}$ & ${r.n_peaks:.0f}$ & "
                       f"${r.flatness:.2f}$" + r"\\")
    out += [r"\hline", r"\end{tabular}", r"\end{center}"]
    p = os.path.join(os.path.dirname(HERE), "tab-hc.tex")
    open(p, "w").write("\n".join(out) + "\n")
    print("wrote", os.path.relpath(p, os.path.dirname(HERE)))


def table_diamond(dd):
    out = [r"\begin{center}", r"\footnotesize\setlength{\tabcolsep}{4pt}",
           r"\begin{tabular}{lrrrrrr}", r"\hline",
           r"$(b,s)$ & $t$ & $n$ & $d_s/2$ & $C\log n$ & peaks & "
           r"$\tau$ ratio\\", r"\hline"]
    for _, r in dd.iterrows():
        # a dash where the quantity does not exist at this size: the ratio needs
        # two peaks to be a ratio, and the plateau needs a scaling window, which
        # at t = 2 is empty because the lattice is crossed at tau ~ s^(2t) before
        # diffusion has escaped the smallest cell
        rat = "---" if r.ratio_mean != r.ratio_mean else f"${r.ratio_mean:.1f}$"
        plat = "---" if r.plateau != r.plateau else f"${r.plateau:.3f}$"
        out.append(f"$({r.b:.0f},{r.s:.0f})$ & ${r.t:.0f}$ & ${r.n:.0f}$ & "
                   f"${r.ds_half:.3f}$ & {plat} & "
                   f"${r.n_peaks:.0f}$ & {rat} \\quad$(s^2{{=}}{r.s2:.0f})$"
                   + r"\\")
    out += [r"\hline", r"\end{tabular}", r"\end{center}"]
    p = os.path.join(os.path.dirname(HERE), "tab-hc-diamond.tex")
    open(p, "w").write("\n".join(out) + "\n")
    print("wrote", os.path.relpath(p, os.path.dirname(HERE)))


def main():
    force = "--force" in sys.argv
    if not force and os.path.exists(CSV) and os.path.exists(DCSV):
        df, dd = pd.read_csv(CSV), pd.read_csv(DCSV)
        print("read cached CSVs")
    else:
        df, dd = measure()
    print("\n=== MFPT formula against its two closed forms ===")
    for name, pred, got, err in check_mfpt():
        print(f"  {name:12s} predicted {pred:8.3f}  measured {got:8.3f}  "
              f"max abs err {err:.2e}")
    pd.set_option("display.width", 200)
    print("\n=== networks ===")
    print(df[["label", "cls", "n", "k2_over_k2", "plateau", "Cmax",
              "n_peaks", "flatness"]].to_string(
        index=False, float_format=lambda x: f"{x:8.3f}"))
    print("\n=== diamonds ===")
    print(dd[["b", "s", "t", "n", "ds_half", "plateau", "n_peaks",
              "ratio_mean", "s2"]].to_string(
        index=False, float_format=lambda x: f"{x:8.3f}"))
    fig_entropy(df)
    fig_diamond(dd)
    fig_rules(df)
    table(df)
    table_diamond(dd)


if __name__ == "__main__":
    main()
