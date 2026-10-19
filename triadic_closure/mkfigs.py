"""Figures for manuscript.tex.

    python mkfigs.py
      fig-phase.pdf       phase diagram and collapse of the club share
      fig-degree.pdf      log-binned degree distributions
      fig-transition.pdf  condensation without opening: trajectories and onset size
      fig-turnover.pdf    club fractions of the condensed phase under turnover at <k> = 4
      fig-pertriad.pdf    <k> along per-triad-rate runs
      fig-corr.pdf        assortativity over the phase diagram and k_nn(k)

Inputs: data/raw/*.nodes (networks of 1e6 nodes; manuscript.tex, Reproducibility),
data/ts/*.ts, data/onset.csv (onset.py), data/phase.csv (phase.py),
data/cross.csv (cross.py), data/club.csv (club.py).
"""
import csv, glob, os, re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
plt.rcParams.update({"font.size": 8.5, "font.family": "serif", "mathtext.fontset": "cm",
                     "axes.linewidth": 0.6, "legend.fontsize": 6.5, "legend.frameon": False,
                     "lines.linewidth": 1.1, "xtick.major.width": 0.5, "ytick.major.width": 0.5})
RAW = os.path.join(HERE, "data", "raw")
RULES = [("l", r"$LS(n,1)$", "0.0", "-", "o"),
         ("f", "friend of a friend (f)", "0.55", "--", "s"),
         ("c", "introducer (c)", "0.55", ":", "^"),
         ("p", "random potential link (p)", "tab:orange", "-", "D"),
         ("w", "random open triad (w)", "tab:blue", "-", "v")]
MK_D = ("o", "s", "^", "D", "v")              # one symbol per net growth Delta (black-and-white print)
MK = ("o", "s", "^", "D", "v", "<", ">", "p", "h", "X")


def n_handles():
    """Legend entries for the three sizes: line style and fill (symbol shape gives Delta)."""
    return [Line2D([], [], color="0.4", ls="--", lw=0.8, label=r"$n=10^5$ (open)"),
            Line2D([], [], color="0.4", ls="-", lw=0.8, label=r"$n=10^6$ (filled)"),
            Line2D([], [], color="0.4", ls="", marker="o", ms=4.5, mec="k", mew=0.5,
                   label=r"$n=10^7$ (large)")]


def by_delta(ax, rows, Ds, cols, y):
    """y(row) against beta_-/beta_+ for each Delta and n."""
    for D, c, mk in zip(Ds, cols, MK_D):
        for n, ls in ((100000, "--"), (1000000, "-"), (10000000, "")):
            sel = sorted([r for r in rows if float(r["D"]) == D and int(r["n"]) == n],
                         key=lambda r: float(r["r_minus"]))
            if not sel: continue
            big = n == 10 ** 7
            ax.plot([float(r["r_minus"]) / float(r["r_plus"]) for r in sel], [y(r) for r in sel],
                    ls, marker=mk, color=c, ms=4.5 if big else 2.6, lw=0.8,
                    mfc="white" if n == 100000 else c, mec="k" if big else c, mew=0.5 if big else 0.7)
C_TREE, C_FREE, C_COND = "0.80", "tab:green", "tab:purple"


def ts(path):
    d = np.loadtxt(path, skiprows=1)
    return d[:, 0], d


def degrees(name):
    return np.loadtxt(os.path.join(RAW, name + ".nodes"), usecols=0, dtype=np.int64)


def logbin(k, ratio=1.35):
    """P(k) from integer degrees in logarithmic bins: counts / (bin width * N)."""
    edges = np.unique(np.floor(ratio ** np.arange(0, 80)).astype(np.int64))
    edges = edges[edges <= k.max()]
    edges = np.append(edges, k.max() + 1)
    h, _ = np.histogram(k, bins=edges)
    w = np.diff(edges); x = np.sqrt(edges[:-1] * (edges[1:] - 1))
    ok = h > 0
    return x[ok], h[ok] / (w[ok] * len(k))


def finish(fig, ax, name):
    for a_, t in zip(np.ravel(ax), "abcdef"): a_.set_title(f"({t})", fontsize=8.5)
    fig.tight_layout(pad=0.3)
    fig.savefig(os.path.join(HERE, name))
    plt.close(fig)


# ------------------------------------------------------------------ fig 1
def fig_phase():
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.9))
    rows = list(csv.DictReader(open(os.path.join(HERE, "data/phase.csv"))))
    Ds = sorted({float(r["D"]) for r in rows})
    cols = cm.Purples(np.linspace(0.4, 1.0, len(Ds)))
    psi = lambda r: float(r["phi_0.8"]) * (1 + float(r["D"])) / float(r["D"])
    # (a) the plane
    b = np.linspace(0, 20, 50)
    ax[0].fill_between(b, b, 20, color=C_TREE, lw=0)
    ax[0].fill_between(b, 0.83 * b, b, color=C_FREE, alpha=0.3, lw=0)
    ax[0].fill_between(b, 0, 0.83 * b, color=C_COND, alpha=0.2, lw=0)
    ax[0].plot(b, b, "-", color="0.3", lw=0.7)
    ax[0].plot(b, 0.83 * b, "--", color="0.3", lw=0.7)
    for r in rows:
        if int(r["n"]) != 1000000: continue
        i = Ds.index(float(r["D"])); c = cols[i]
        ax[0].plot(float(r["r_plus"]), float(r["r_minus"]), MK_D[i], ms=2.8, color=c,
                   mfc=c if psi(r) > 0.9 else "white", mew=0.7)
    ax[0].set(xlabel=r"$\beta_+/\alpha$", ylabel=r"$\beta_-/\alpha$",
              xlim=(0, 20), ylim=(0, 20))
    ax[0].legend(handles=[Patch(color=C_TREE, label="tree"),
                          Patch(color=C_FREE, alpha=0.3, label="hub-free"),
                          Patch(color=C_COND, alpha=0.2, label="condensed"),
                          Line2D([], [], color="0.3", lw=0.7, label=r"$\beta_-=\beta_+$"),
                          Line2D([], [], color="0.3", lw=0.7, ls="--", label=r"$\beta_-=0.83\,\beta_+$"),
                          Line2D([], [], ls="", marker="h", color="0.3", ms=3.2, label=r"$\psi>0.9$ (filled)"),
                          Line2D([], [], ls="", marker="h", color="0.3", mfc="white", ms=3.2,
                                 label=r"$\psi<0.9$ (open)")],
                 loc="upper left", fontsize=6.2, handlelength=1.6,
                 frameon=True, facecolor="white", edgecolor="0.6", framealpha=1)
    # (b) collapse of the club share
    by_delta(ax[1], rows, Ds, cols, psi)
    ax[1].axvspan(0.80, 0.86, color="0.85", lw=0)
    ax[1].set(xlabel=r"$\beta_-/\beta_+$", ylabel=r"$\psi$",
              ylim=(0, 1.45))
    hd = [Line2D([], [], color=c, marker=mk, ms=3, lw=1.2, label=rf"$\Delta={D:g}$")
          for D, c, mk in zip(Ds, cols, MK_D)]
    hd += n_handles() + [Patch(color="0.85", label="boundary")]
    ax[1].legend(handles=hd, loc="lower left", fontsize=6, ncol=2, columnspacing=0.8, handlelength=1.8)
    finish(fig, ax, "fig-phase.pdf")


# ------------------------------------------------------------------ fig 2
def fig_degree():
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.7))
    for m, lab, col, ls, mk in RULES:
        x, p = logbin(degrees(f"{m}_u0.5_n1000000"))
        ax[0].loglog(x, p, ls, color=col, marker=mk, ms=2.2, lw=0.9, label=lab)
    kk = np.array([20.0, 150.0]); ax[0].loglog(kk, 40 * kk ** -5.0, color="0.3", lw=0.6, ls="-.", label=r"$k^{-5}$")
    kk = np.array([12.0, 700.0]); ax[0].loglog(kk, 0.012 / kk, color="0.3", lw=0.6, ls=(0, (5, 2, 1, 2, 1, 2)),
                                               label=r"$k^{-1}$")
    ax[0].set(xlabel="$k$", ylabel="$P(k)$", ylim=(1e-10, 1))
    ax[0].legend(loc="lower left", fontsize=6.2)
    runs = [("w_u0.5_n1000000", r"$\beta_-/\alpha=0$", cm.Purples(1.0), "-"),
            ("w_k4_rm1_n1000000", r"$\beta_-/\alpha=1$", cm.Purples(0.75), "-"),
            ("w_k4_rm4_n1000000", r"$\beta_-/\alpha=4$", cm.Purples(0.5), "-"),
            ("w_k4_rm8_n1000000", r"$\beta_-/\alpha=8$", C_FREE, "-"),
            ("w_crit_rp1_rm1_n1000000", r"$\beta_+=\beta_-=\alpha$", "0.3", "--"),
            ("w_tree_rp1_rm2_n1000000", r"$\beta_+=\alpha,\ \beta_-=2\alpha$", "0.0", "-"),
            ("w_rrt_n1000000", r"random tree ($\beta_\pm=0$)", "0.6", ":")]
    for (name, lab, col, ls), mk in zip(runs, MK):
        x, p = logbin(degrees(name))
        ax[1].loglog(x, p, ls, color=col, marker=mk, ms=2.2, lw=0.9, label=lab)
    ax[1].set(xlabel="$k$", ylabel="$P(k)$", ylim=(1e-10, 1))
    ax[1].legend(loc="lower left", fontsize=6)
    finish(fig, ax, "fig-degree.pdf")


# ------------------------------------------------------------------ fig 3
def fig_transition():
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.7))
    wf = glob.glob(os.path.join(HERE, "data/ts/w_u*_n1e7_s1.ts"))
    wf.sort(key=lambda p: float(re.search(r"u([\d.]+)_", p).group(1)))
    blues = cm.Blues(np.linspace(0.35, 1.0, len(wf)))
    for p, col, mk in zip(wf, blues, MK):
        u = re.search(r"u([\d.]+)_", p).group(1)
        N, d = ts(p)
        ax[0].loglog(N, d[:, 3] / N, color=col, marker=mk, ms=2.6, markevery=(0.02, 0.12), label=f"(w) $u={u}$")
    for m, lab, col, ls, mk in RULES[:3]:
        N, d = ts(os.path.join(HERE, f"data/ts/{m}_u0.5_n1e7_s1.ts"))
        ax[0].loglog(N, d[:, 3] / N, ls, color=col, lw=0.8, label=lab + r", $u=1/2$")
    x = np.array([1e3, 1e7]); ax[0].loglog(x, 1.0 * x ** 0.5, color="0.3", lw=0.5, ls="-.", label=r"$n^{1/2}$")
    ax[0].set(xlabel="$n$", ylabel="$T/n$", ylim=(1e-2, 1e6))
    ax[0].legend(loc="upper left", fontsize=6.2, ncol=2, columnspacing=0.6)
    rows = [r for r in csv.DictReader(open(os.path.join(HERE, "data/onset.csv"))) if r["theta"] == "1.0"]
    U, M, F = [], [], []
    first = True
    for u in sorted({float(r["u"]) for r in rows}):
        v = np.array([float(r["nstar"]) for r in rows if float(r["u"]) == u]); ok = v[np.isfinite(v)]
        if len(ok) == 0:
            nmax = max(float(r["nmax"]) for r in rows if float(r["u"]) == u)
            ax[1].plot(1 / u, np.log(nmax), "v", color="0.5", ms=4, mfc="none", label=r"no onset up to this $n$")
            continue
        U.append(u); M.append(np.log(ok).mean()); F.append(len(ok) == len(v))
        ax[1].errorbar(1 / u, M[-1], yerr=np.log(ok).std() if len(ok) > 1 else 0, fmt="o", color="tab:blue",
                       ms=3, capsize=1.5, lw=0.7, label="simulation, mean $\\pm$ s.d." if first else None)
        first = False
    U, M, F = np.array(U), np.array(M), np.array(F)
    sel = F & (U >= 0.1) & (U <= 0.3)       # censored runs (no onset yet) keep u out of the fit
    c = np.polyfit(np.log(U[sel]), M[sel], 1); c2 = np.polyfit(1 / U[sel], M[sel], 1)
    xx = np.linspace(2.5, 22, 100)
    ax[1].plot(xx, c[1] + c[0] * np.log(1 / xx), color="0.3", lw=0.6, label=rf"fit $n^*\propto u^{{{c[0]:.1f}}}$")
    ax[1].plot(xx, c2[1] + c2[0] * xx, "--", color="0.3", lw=0.6,
               label=rf"fit $\ln n^*={c2[1]:.1f}+{c2[0]:.2f}/u$")
    ax[1].set(xlabel="$1/u$", ylabel=r"$\ln n^*$", ylim=(4, 22))
    ax[1].legend(loc="lower right", fontsize=6.7)
    finish(fig, ax, "fig-transition.pdf")
    return c, c2


# ------------------------------------------------------------------ fig 4
def fig_turnover():
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.7))
    rows = list(csv.DictReader(open(os.path.join(HERE, "data/club.csv"))))
    A = (0.5, 0.6, 0.7, 0.8, 0.9)
    mean_phi = lambda m, rm, n, a: np.mean([float(r[f"phi_{a}"]) for r in rows if r["mode"] == m
                                           and float(r["r_minus"]) == rm and int(r["n"]) == n])
    wr = sorted({float(r["r_minus"]) for r in rows if r["mode"] == "w"})
    wc = cm.Purples(np.linspace(0.35, 1.0, len(wr)))[::-1]
    for rm, col, mk in zip(wr, wc, MK):
        ax[0].plot(A, [mean_phi("w", rm, 1000000, a) for a in A], "-", marker=mk, color=col, ms=2.8, lw=0.8,
                   label=rf"(w) $\beta_-/\alpha={rm:g}$")
        ns = sorted({int(r["n"]) for r in rows if r["mode"] == "w" and float(r["r_minus"]) == rm})
        ax[1].semilogx(ns, [mean_phi("w", rm, n, 0.8) for n in ns], "-", marker=mk, color=col, ms=2.8, lw=0.8)
    for m, lab, col, ls, mk in RULES[:3]:
        ax[0].plot(A, [mean_phi(m, 0.0, 1000000, a) for a in A], ls, color=col, lw=0.9, label=lab)
        ns = sorted({int(r["n"]) for r in rows if r["mode"] == m and float(r["r_minus"]) == 0})
        ax[1].semilogx(ns, [mean_phi(m, 0.0, n, 0.8) for n in ns], ls, color=col, lw=0.9, label=lab)
    for a_ in ax: a_.axhline(0.5, color="tab:red", lw=0.5, ls="-.", label="net closure share $1/2$")
    ax[0].set(xlabel=r"$a$", ylabel=r"$\phi(n^a)$", ylim=(0, 1.0))
    ax[1].set(xlabel="$n$", ylabel=r"$\phi(n^{0.8})$", ylim=(0, 0.95))
    ax[0].legend(loc="upper left", fontsize=6.2, ncol=2, columnspacing=0.6, handlelength=1.6)
    ax[1].legend(loc="upper right", fontsize=6.7)
    finish(fig, ax, "fig-turnover.pdf")


# ------------------------------------------------------------------ fig 5
def fig_pertriad():
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    bs = {}
    for p in glob.glob(os.path.join(HERE, "data/ts/r_b*_n1e*_s1.ts")):   # keep the longest run per beta_+/alpha
        b = float(re.search(r"b([\d.]+)_", p).group(1))
        if b not in bs or os.path.getsize(p) > os.path.getsize(bs[b]): bs[b] = p
    greens = cm.Greens(np.linspace(0.35, 1.0, len(bs)))
    for (b, p), col, mk in zip(sorted(bs.items()), greens, MK):
        N, d = ts(p)
        ax.semilogx(N, d[:, 2], color=col, marker=mk, ms=2.4, markevery=(0.02, 0.1), label=rf"$\beta_+/\alpha={b:g}$")
    ax.axhline(2.58, color="0.5", lw=0.5, ls="--", label="MF-max, 2.58")
    ax.set(xlabel="$n$", ylabel=r"$\langle k\rangle$", ylim=(1.9, 17))
    ax.legend(loc="upper left", fontsize=6, handlelength=1.6, labelspacing=0.25)
    fig.tight_layout(pad=0.3)
    fig.savefig(os.path.join(HERE, "fig-pertriad.pdf"))
    plt.close(fig)


# ------------------------------------------------------------------ degree correlations
def knn(name, ratio=1.35):
    """k_nn(k) <k>/<k^2> in the log bins of logbin()."""
    d = np.loadtxt(os.path.join(RAW, name + ".nodes"), usecols=(0, 2))
    k, s = d[:, 0].astype(np.int64), d[:, 1]
    edges = np.unique(np.floor(ratio ** np.arange(0, 80)).astype(np.int64))
    edges = np.append(edges[edges <= k.max()], k.max() + 1)
    i = np.digitize(k, edges) - 1
    cnt = np.bincount(i, minlength=len(edges) - 1)
    val = np.bincount(i, s / k, minlength=len(edges) - 1)
    x = np.sqrt(edges[:-1] * (edges[1:] - 1)); ok = cnt > 0
    return x[ok], val[ok] / cnt[ok] * k.mean() / (k.astype(float) ** 2).mean()


def fig_corr():
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.7))
    rows = list(csv.DictReader(open(os.path.join(HERE, "data/phase.csv"))))
    Ds = sorted({float(r["D"]) for r in rows})
    cols = cm.Purples(np.linspace(0.4, 1.0, len(Ds)))
    by_delta(ax[0], rows, Ds, cols, lambda r: float(r["assort"]))
    cr = sorted([r for r in csv.DictReader(open(os.path.join(HERE, "data/cross.csv")))
                 if r["mode"] == "w" and int(r["n"]) == 1000000 and float(r["r_minus"]) <= 2],
                key=lambda r: float(r["r_minus"]))
    ax[0].plot([float(r["r_minus"]) for r in cr], [float(r["assort"]) for r in cr], "-", marker="X", color="0.2",
               ms=3.2, lw=0.8)
    ax[0].axvspan(0.80, 0.86, color="0.85", lw=0)
    ax[0].axhline(0, color="0.6", lw=0.5, ls=":")
    ax[0].set(xlim=(-0.05, 2.12), ylim=(-0.25, 0.75), xlabel=r"$\beta_-/\beta_+$", ylabel="$r$")
    hd = [Line2D([], [], color=c, marker=mk, ms=3, lw=1.2, label=rf"$\Delta={D:g}$")
          for D, c, mk in zip(Ds, cols, MK_D)]
    hd += n_handles() + [Line2D([], [], color="0.2", ls="-", marker="X", ms=3.2, lw=0.8,
                                label=r"$\beta_+=\alpha$, $n=10^6$"),
                         Patch(color="0.85", label="boundary")]
    ax[0].legend(handles=hd, loc="upper right", fontsize=6.2, handlelength=1.8, labelspacing=0.3)
    runs = [("w_u0.5_n1000000", r"(w) $\beta_-/\alpha=0$", cm.Purples(1.0), "-"),
            ("w_k4_rm1_n1000000", r"(w) $\beta_-/\alpha=1$", cm.Purples(0.75), "-"),
            ("w_k4_rm4_n1000000", r"(w) $\beta_-/\alpha=4$", cm.Purples(0.5), "-"),
            ("w_k4_rm8_n1000000", r"(w) $\beta_-/\alpha=8$", C_FREE, "-"),
            ("w_tree_rp1_rm2_n1000000", r"(w) $\beta_+=\alpha,\ \beta_-=2\alpha$", "0.0", "-"),
            ("w_rrt_n1000000", r"random tree ($\beta_\pm=0$)", "0.6", ":"),
            ("l_u0.5_n1000000", r"$LS(n,1)$", "0.3", "--"),
            ("f_u0.5_n1000000", "friend of a friend (f)", "0.55", "-.")]
    for (name, lab, col, ls), mk in zip(runs, MK):
        x, y = knn(name)
        ax[1].loglog(x, y, ls, color=col, marker=mk, ms=2.2, lw=0.9, label=lab)
    ax[1].axhline(1, color="0.6", lw=0.5, ls=":")
    ax[1].set(xlabel="$k$", ylabel=r"$k_{nn}(k)\,\langle k\rangle/\langle k^2\rangle$", ylim=(5e-3, 8))
    ax[1].legend(loc="lower right", fontsize=6.2)
    finish(fig, ax, "fig-corr.pdf")


if __name__ == "__main__":
    fig_phase()
    fig_degree()
    print(fig_transition())
    fig_turnover()
    fig_pertriad()
    fig_corr()
