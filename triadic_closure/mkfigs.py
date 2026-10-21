"""Figures for manuscript.tex.

    python mkfigs.py
      fig-phase.pdf       phase diagram from the triangle-cluster order parameter P_triangle
      fig-degree.pdf      log-binned degree distributions
      fig-transition.pdf  condensation without opening: trajectories and onset size
      fig-cluster.pdf     nodes and links of the largest triangle cluster

Inputs: data/raw/*.nodes (networks of 1e6 nodes; manuscript.tex, Reproducibility),
data/ts/*.ts, data/onset.csv (onset.py), data/opgrid.csv, data/bnd.csv,
data/delgrid.csv (opgrid.py, bnd.py, delgrid.py), data/rulesclust.csv (rulesclust.py).
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
def order_rows():
    """Rule (w) runs reduced to triangle clusters: opgrid, bnd, and the beta_- = 0 runs of delgrid at u = 1/2."""
    rows = []
    for f in ("opgrid", "bnd"):
        rows += list(csv.DictReader(open(os.path.join(HERE, f"data/{f}.csv"))))
    rows += [dict(r, D="1.0", ratio="0.0") for r in csv.DictReader(open(os.path.join(HERE, "data/delgrid.csv")))
             if r["u"] == "0.5"]
    return rows


def mean_by(rows, col):
    """Seed mean of a column per (Delta, alpha_-/alpha_+, beta_-/beta_+, n)."""
    g = {}
    for r in rows:
        g.setdefault((float(r["D"]), float(r["a"]), float(r["ratio"]), int(r["n"])), []).append(float(r[col]))
    return {k: np.mean(v) for k, v in g.items()}


def condensed(fT):
    """Phase of each point from the trend of P_triangle between n = 1e5 and 1e6: True when it rises (or is ~1)."""
    out = {}
    for (D, a, x, n) in fT:
        if n == 1000000 and (D, a, x, 100000) in fT:
            out[(D, a, x)] = fT[(D, a, x, n)] >= 0.9 or fT[(D, a, x, n)] > fT[(D, a, x, 100000)]
    return out


def brackets(cls, D):
    """Per alpha_-/alpha_+: largest condensed and smallest dispersed beta_-/beta_+ above it."""
    out = []
    for a in sorted({k[1] for k in cls if k[0] == D}):
        xs = sorted((k[2], c) for k, c in cls.items() if k[0] == D and k[1] == a)
        lo = max([x for x, c in xs if c], default=None)
        hi = min([x for x, c in xs if not c and x > lo], default=None)
        if lo is not None and lo > 0: out.append((a, lo, hi))
    return out


def fig_phase():
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.9))
    rows = order_rows(); fT = mean_by(rows, "fT"); cls = condensed(fT)
    # (a) the plane at Delta = 1, with the Delta = 1/2 brackets
    br = brackets(cls, 1.0)
    fit = [(a, lo, hi) for a, lo, hi in br if hi is not None]
    xx = np.array([1 - a for a, _, _ in fit]); yy = np.array([1 - (lo + hi) / 2 for _, lo, hi in fit])
    c = (xx * yy).sum() / (xx ** 2).sum()
    A = np.linspace(0, 1, 50); xc = 1 - c * (1 - A)
    ax[0].fill_betweenx(A, 0, xc, color=C_COND, alpha=0.2, lw=0)
    ax[0].fill_betweenx(A, xc, 1, color=C_FREE, alpha=0.3, lw=0)
    ax[0].fill_betweenx(A, 1, 1.1, color=C_TREE, lw=0)
    ax[0].plot(xc, A, "--", color="0.3", lw=0.7)
    for D, mk, col in ((1.0, "o", cm.Purples(0.9)), (0.5, "s", cm.Purples(0.55))):
        for (D_, a, x), cnd in cls.items():
            if D_ != D: continue
            ax[0].plot(x, a, mk, ms=3.0 if D == 1 else 2.6, color=col, mfc=col if cnd else "white", mew=0.7)
    ax[0].set(xlabel=r"$\beta_-/\beta_+$", ylabel=r"$\alpha_-/\alpha_+$", xlim=(-0.02, 1.1), ylim=(-0.03, 1))
    ax[0].legend(handles=[Patch(color=C_COND, alpha=0.2, label="condensed"),
                          Patch(color=C_FREE, alpha=0.3, label="dispersed"),
                          Patch(color=C_TREE, label="tree"),
                          Line2D([], [], color="0.3", lw=0.7, ls="--",
                                 label="boundary fit"),
                          Line2D([], [], ls="", marker="o", color=cm.Purples(0.9), ms=3, label=r"$\Delta=1$"),
                          Line2D([], [], ls="", marker="s", color=cm.Purples(0.55), ms=2.6, label=r"$\Delta=1/2$"),
                          Line2D([], [], ls="", marker="h", color="0.3", ms=3.2, label=r"$P_\triangle$ rising (filled)"),
                          Line2D([], [], ls="", marker="h", color="0.3", mfc="white", ms=3.2,
                                 label=r"$P_\triangle$ falling (open)")],
                 loc="upper left", bbox_to_anchor=(0.06, 0.99), fontsize=6.0, handlelength=1.6, labelspacing=0.3,
                 frameon=True, facecolor="white", edgecolor="0.6", framealpha=1)
    # (b) P_triangle against beta_-/beta_+ at Delta = 1, three sizes, alpha_- = 0 and alpha_+/2
    for a, col in ((0.0, cm.Purples(0.9)), (0.5, "tab:orange")):
        for n, ls, mk in ((10000, ":", "o"), (100000, "--", "s"), (1000000, "-", "^")):
            pts = sorted((x, v) for (D, a_, x, n_), v in fT.items() if D == 1.0 and a_ == a and n_ == n)
            ax[1].plot(*zip(*pts), ls, marker=mk, color=col, ms=2.6, lw=0.8, mfc="white" if n < 1000000 else col)
    for (D, a, x, n), v in fT.items():
        if n == 10000000:
            ax[1].plot(x, v, "D", color=cm.Purples(0.9), ms=4.5, mec="k", mew=0.5)
    ax[1].set(xlabel=r"$\beta_-/\beta_+$", ylabel=r"$P_\triangle$", xlim=(-0.02, 1.0), ylim=(-0.03, 1.05))
    hd = [Line2D([], [], color=cm.Purples(0.9), lw=1.2, label=r"$\alpha_-=0$"),
          Line2D([], [], color="tab:orange", lw=1.2, label=r"$\alpha_-=\alpha_+/2$"),
          Line2D([], [], color="0.4", ls=":", marker="o", mfc="white", ms=2.6, lw=0.8, label=r"$n=10^4$"),
          Line2D([], [], color="0.4", ls="--", marker="s", mfc="white", ms=2.6, lw=0.8, label=r"$n=10^5$"),
          Line2D([], [], color="0.4", ls="-", marker="^", ms=2.6, lw=0.8, label=r"$n=10^6$"),
          Line2D([], [], color=cm.Purples(0.9), ls="", marker="D", mec="k", mew=0.5, ms=4.5, label=r"$n=10^7$")]
    ax[1].legend(handles=hd, loc="lower left", fontsize=6.2, handlelength=1.8, labelspacing=0.3)
    finish(fig, ax, "fig-phase.pdf")
    return c, br, brackets(cls, 0.5)


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
    runs = [("w_u0.5_n1000000", r"$\beta_-/\alpha_+=0$", cm.Purples(1.0), "-"),
            ("w_k4_rm1_n1000000", r"$\beta_-/\alpha_+=1$", cm.Purples(0.75), "-"),
            ("w_k4_rm4_n1000000", r"$\beta_-/\alpha_+=4$", cm.Greens(0.5), "-"),
            ("w_k4_rm8_n1000000", r"$\beta_-/\alpha_+=8$", C_FREE, "-"),
            ("w_crit_rp1_rm1_n1000000", r"$\beta_+=\beta_-=\alpha_+$", "0.3", "--"),
            ("w_tree_rp1_rm2_n1000000", r"$\beta_+=\alpha_+,\ \beta_-=2\alpha_+$", "0.0", "-"),
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
def fig_cluster():
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.7))
    rows = order_rows(); fN = mean_by(rows, "fN"); fE = mean_by(rows, "fE")
    # (a) P_bullet (node fraction) against n
    xs = (0.0, 0.5, 0.67, 0.75, 0.8)
    for x, col, mk in zip(xs, [cm.Purples(v) for v in (1.0, 0.8, 0.6, 0.45)] + [C_FREE], MK):
        ns = sorted(n for (D, a, x_, n) in fN if D == 1.0 and a == 0 and x_ == x and n <= 1000000)
        ax[0].loglog(ns, [max(fN[(1.0, 0.0, x, n)], 1e-6) for n in ns], "-", marker=mk, color=col, ms=2.8, lw=0.8,
                     label=rf"(w) $\beta_-/\beta_+={x:g}$")
    ns = sorted(n for (D, a, x_, n) in fN if D == 1.0 and a == 0.5 and x_ == 0)
    ax[0].loglog(ns, [fN[(1.0, 0.5, 0.0, n)] for n in ns], "-", marker="X", color="tab:brown", ms=3, lw=0.8,
                 label=r"(w) $\beta_-=0$, $\alpha_-=\alpha_+/2$")
    rc = list(csv.DictReader(open(os.path.join(HERE, "data/rulesclust.csv"))))
    for m, lab, col, ls, mk in RULES[:4]:
        ns = sorted({int(r["n"]) for r in rc if r["mode"] == m})
        ax[0].loglog(ns, [np.mean([float(r["fN"]) for r in rc if r["mode"] == m and int(r["n"]) == n]) for n in ns],
                     ls, marker=mk, color=col, ms=2.6, lw=0.9, mfc="white", label=lab)
    x = np.array([1e4, 1e6]); ax[0].loglog(x, 6 * x ** -0.5, color="0.3", lw=0.6, ls="-.", label=r"$n^{-1/2}$")
    ax[0].set(xlabel="$n$", ylabel=r"$P_\bullet$", ylim=(3e-8, 3))
    ax[0].legend(loc="lower left", fontsize=5.8, ncol=2, columnspacing=0.6, handlelength=1.6, labelspacing=0.25)
    # (b) share of the net closure links inside the cluster, n = 1e6
    for (D, a), col, mk in (((1.0, 0.0), cm.Purples(0.9), "o"), ((0.5, 0.0), cm.Purples(0.55), "s"),
                            ((1.0, 0.5), "tab:orange", "^"), ((0.5, 0.5), "tab:red", "D")):
        pts = sorted((x, v * (1 + D) / D) for (D_, a_, x, n), v in fE.items() if D_ == D and a_ == a and n == 1000000)
        ax[1].plot(*zip(*pts), "-", marker=mk, color=col, ms=2.8, lw=0.8,
                   label=rf"$\Delta={D:g}$, $\alpha_-/\alpha_+={a:g}$")
    ax[1].axhline(1, color="0.6", lw=0.5, ls=":")
    ax[1].set(xlabel=r"$\beta_-/\beta_+$", ylabel=r"$P_{\bullet{-}\bullet}\,(1+\Delta)/\Delta$", xlim=(-0.02, 1.0), ylim=(-0.03, 1.15))
    ax[1].legend(loc="lower left", fontsize=6.2, handlelength=1.8, labelspacing=0.3)
    finish(fig, ax, "fig-cluster.pdf")


if __name__ == "__main__":
    print(fig_phase())
    fig_degree()
    print(fig_transition())
    print(fig_cluster())
