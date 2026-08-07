#!/usr/bin/env python3
r"""
qwalk_ipr.py -- localization of continuous-time quantum walks on the growth
rules of this book, for the chapter "Localization of quantum walks".

The measurement follows Boettcher & Porter, Phys. Rev. E (2025),
arXiv:2412.04325: the Hamiltonian is the symmetric normalized Laplacian
H = D^{-1/2}(D-A)D^{-1/2}, the walk starts localized on a node j, and the
confinement is read off the inverse participation ratio

    IPR_j(t) = sum_i |<i|exp(-iHt)|j>|^4 ,

time-averaged over a late window.  The localization volume is L = 1/IPR, the
effective number of nodes the walker occupies.  Their paper varies the mean
clustering coefficient; this script varies instead the cycle each rule closes
and the community structure it builds, which the clustering cannot separate.

Everything is generated here -- there is no upstream data to copy.  Simulation
output is cached in data/qwalk-ipr.csv so that re-plotting is instant; delete
that file (or pass --force) to re-measure.

Usage:
    python3 qwalk_ipr.py            # measure if needed, then plot + tabulate
    python3 qwalk_ipr.py --force    # re-measure from scratch
    python3 qwalk_ipr.py --plot     # plot from the cache only
"""

import os
import sys
import itertools

import numpy as np
import networkx as nx
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
CSV = os.path.join(DATA, "qwalk-ipr.csv")

sys.path.insert(0, HERE)

REPS = 10
SIZES = (100, 200, 400, 800)
NSTART = 24                      # start nodes sampled per network
WINDOW = (400.0, 800.0)          # the late window the IPR is averaged over
NT = 240                         # time samples in that window


# ------------------------------------------------------------------ the walk

def ipr(g, window=WINDOW, nt=NT, nstart=NSTART, seed=0):
    """Time-averaged IPR_j over `window`, for a sample of start nodes j.

    Returns one value per sampled start node.  The walk is propagated by
    diagonalizing H once and phasing the eigenbasis, which is exact -- there is
    no integrator error to worry about, only the finite averaging window.
    """
    A = nx.to_numpy_array(g)
    k = A.sum(1)
    if (k == 0).any():
        raise ValueError("isolated node: H is undefined")
    dm = 1.0 / np.sqrt(k)
    H = (dm[:, None] * (np.diag(k) - A)) * dm[None, :]
    lam, V = np.linalg.eigh(H)
    ts = np.linspace(window[0], window[1], nt)
    n = len(k)
    js = (np.arange(n) if n <= nstart else
          np.random.default_rng(seed).choice(n, nstart, replace=False))
    ph = np.exp(-1j * np.outer(ts, lam))                 # nt x n
    out = np.empty(len(js))
    for a, j in enumerate(js):
        col = V @ (ph * V[j, :]).T                       # n x nt
        out[a] = (np.abs(col) ** 4).sum(0).mean()
    return out


def giant(g):
    g = g.subgraph(max(nx.connected_components(g), key=len)).copy()
    g.remove_edges_from(nx.selfloop_edges(g))
    return nx.convert_node_labels_to_integers(g)


# ------------------------------------------------------------------ the rules
# The stochastic rules are the generators chapters 6-8 measured, imported so
# that the networks here are the same networks those chapters fitted.

from ramsey_motif import (generator_local_search, generator_dup_split,
                          generator_dup_divergence, to_edges)
from arity_models import bb
from bb2_chord import grow as bb2_grow


def _nb(nb):
    g = nx.Graph()
    g.add_nodes_from(range(len(nb)))
    g.add_edges_from(to_edges(nb))
    return giant(g)


def _adj(adj):
    g = nx.Graph()
    g.add_nodes_from(adj)
    for u in adj:
        for w in adj[u]:
            g.add_edge(u, w)
    return giant(g)


def pseudofractal(t):
    """Dorogovtsev-Goltsev-Mendes web: every link spawns a node joined to both
    of its ends.  n_t = 3(3^t+1)/2, m_t = 3^{t+1}."""
    g = nx.Graph([(0, 1), (1, 2), (2, 0)])
    for _ in range(t):
        for u, v in list(g.edges()):
            w = g.number_of_nodes()
            g.add_edge(u, w)
            g.add_edge(v, w)
    return g


def rec_triangle(d):
    """The recursive triangle web of Boettcher & Porter: n = 3*2^d, m =
    3(2^{d+1}-1).  One new node across each existing link, until the node count
    has doubled -- so the triangles are added a generation at a time rather than
    to every link at once, as the pseudofractal does."""
    g = nx.Graph([(0, 1), (1, 2), (2, 0)])
    for _ in range(d):
        target = 2 * g.number_of_nodes()
        for u, v in list(g.edges()):
            if g.number_of_nodes() >= target:
                break
            w = g.number_of_nodes()
            g.add_edge(u, w)
            g.add_edge(v, w)
    return g


def diamond(t, b, s):
    """(b,s) diamond: every link becomes b parallel paths of s links.  Simple
    graph, so s=1 collapses -- its role as the seam-keeping case is played by
    the plain ring, which chapter 6 already identifies as its counterpart."""
    g = nx.Graph([(0, 1)])
    for _ in range(t):
        h = nx.Graph()
        h.add_nodes_from(g)
        nxt = g.number_of_nodes()
        for u, v in g.edges():
            for _ in range(b):
                path = [u] + list(range(nxt, nxt + s - 1)) + [v]
                nxt += s - 1
                h.add_edges_from(zip(path, path[1:]))
        g = h
    return g


# label -> (builder(n, seed), cycle closed, <n_B> from tab-onset, class)
RULES = {
    "LS(n,1)":   (lambda n, s: _nb(generator_local_search(n, {"d": 1}, s)), 3, 29, "comm"),
    "BB(n,1)":   (lambda n, s: _adj(bb(n, 1, s)), 3, 34, "comm"),
    "BB(n,2)":   (lambda n, s: _adj(bb(n, 2, s)), 4, 69, "none"),
    "BB(n,3)":   (lambda n, s: _adj(bb(n, 3, s)), 5, 120, "comm"),
    "BB(n,4)":   (lambda n, s: _adj(bb(n, 4, s)), 6, 304, "none"),
    "BB(n,5)":   (lambda n, s: _adj(bb(n, 5, s)), 7, 271, "comm"),
    "BB(n,6)":   (lambda n, s: _adj(bb(n, 6, s)), 8, None, "?"),
    "DS(n,1/3)": (lambda n, s: _nb(generator_dup_split(n, {"q": 1 / 3}, s)), 4, 55, "comm"),
    "DS(n,0.5)": (lambda n, s: _nb(generator_dup_split(n, {"q": 0.5}, s)), 4, 20, "comm"),
    "DD(n,0.5)": (lambda n, s: _nb(generator_dup_divergence(n, {"q": 0.5}, s)), None, 26, "comm"),
    "BB2+link":  (lambda n, s: _adj(bb2_grow(n, "bb2par", s)), 4, None, "comm"),
    "BB2+diag":  (lambda n, s: _adj(bb2_grow(n, "bb2chord", s)), 4, None, "comm"),
    "BA(n,2)":   (lambda n, s: giant(nx.barabasi_albert_graph(n, 2, seed=s)), None, None, "none"),
    "ER(n,4)":   (lambda n, s: giant(nx.gnm_random_graph(n, 2 * n, seed=s)), None, None, "none"),
    "ring":      (lambda n, s: nx.cycle_graph(n), None, None, "none"),
}

DETERMINISTIC = {
    "pseudofractal": [pseudofractal(t) for t in (3, 4, 5)],
    "rec.triangle":  [rec_triangle(d) for d in (4, 5, 6, 7)],
    "diamond(2,2)":  [diamond(t, 2, 2) for t in (3, 4, 5)],
    "diamond(2,3)":  [diamond(t, 2, 3) for t in (2, 3, 4)],
    "diamond(2,4)":  [diamond(t, 2, 4) for t in (2, 3)],
    "diamond(3,2)":  [diamond(t, 3, 2) for t in (2, 3, 4)],
    "diamond(4,2)":  [diamond(t, 4, 2) for t in (2, 3)],
}


# ------------------------------------------------------------------ twins

def twin_classes(g):
    """False twins: nodes with identical open neighbourhoods.  Each class of
    size c contributes c-1 eigenvectors of H at eigenvalue exactly 1, each
    supported on two nodes -- see the chapter's calculation box."""
    d = {}
    for v in g:
        d.setdefault(frozenset(g[v]), []).append(v)
    return [c for c in d.values() if len(c) > 1]


def merge_twins(g):
    """Keep one representative of every twin class."""
    h = g.copy()
    h.remove_nodes_from([w for cl in twin_classes(g) for w in cl[1:]])
    return giant(h)


# ------------------------------------------------------------------ measure

def measure():
    rows = []

    def add(rule, n, seed, g, **kw):
        # `n` is the requested size and is the grouping key: the giant component
        # of a duplication network falls a little short of it, and grouping by
        # the realised count would scatter each ensemble across neighbouring
        # columns.  `n_real` keeps the number actually walked on.
        v = ipr(g)
        rows.append(dict(rule=rule, n=n or g.number_of_nodes(),
                         n_real=g.number_of_nodes(), seed=seed,
                         C=nx.average_clustering(g),
                         k=2 * g.number_of_edges() / g.number_of_nodes(),
                         ipr_mean=v.mean(), ipr_max=v.max(),
                         ipr_min=v.min(), **kw))

    for rule, (build, cyc, nB, cls) in RULES.items():
        for n in SIZES:
            for seed in range(1 if rule == "ring" else REPS):
                add(rule, n, seed, build(n, seed + 1), cycle=cyc, nB=nB, cls=cls)
        print("done", rule, flush=True)

    for rule, graphs in DETERMINISTIC.items():
        for g in graphs:
            if g.number_of_nodes() <= 1200:
                add(rule, 0, 0, g, cycle=3 if "tri" in rule or "pseudo" in rule else 4,
                    nB=None, cls="none")
        print("done", rule, flush=True)

    # the twin experiment: duplication with and without its exact symmetries
    for q in (0.3, 0.5, 0.7):
        for n in (200, 400, 800):
            for seed in range(REPS):
                g = _nb(generator_dup_divergence(n, {"q": q}, seed + 1))
                tc = twin_classes(g)
                frac = sum(len(c) for c in tc) / g.number_of_nodes()
                add(f"DD(n,{q})", n, seed, g, cycle=None, nB=None, cls="comm",
                    twinned=frac)
                add(f"DD(n,{q})-merged", n, seed, merge_twins(g), cycle=None,
                    nB=None, cls="comm", twinned=0.0)
        print("done twins q =", q, flush=True)

    # window convergence: is the saturation a t -> infinity statement, or an
    # artifact of averaging too early?
    for rule in ("BB(n,2)", "BB(n,5)", "LS(n,1)", "ring"):
        build = RULES[rule][0]
        for w in ((100.0, 200.0), (400.0, 800.0), (1600.0, 3200.0)):
            for seed in range(3):
                g = build(800, seed + 1)
                v = ipr(g, window=w)
                rows.append(dict(rule=rule, n=800, seed=seed, C=np.nan, k=np.nan,
                                 ipr_mean=v.mean(), ipr_max=v.max(),
                                 ipr_min=v.min(), window=w[0], cls="window"))
        print("done window", rule, flush=True)

    df = pd.DataFrame(rows)
    os.makedirs(DATA, exist_ok=True)
    df.to_csv(CSV, index=False)
    print("wrote", os.path.relpath(CSV, HERE))
    return df


# ------------------------------------------------------------------ report

def report(df):
    d = df[df.cls != "window"]
    g = d.groupby(["rule", "n"]).agg(
        C=("C", "mean"), k=("k", "mean"),
        ipr=("ipr_mean", "mean"), ipr_sd=("ipr_mean", "std"),
        iprmax=("ipr_max", "mean"), reps=("ipr_mean", "size")).reset_index()
    g["L"] = 1.0 / g.ipr
    pd.set_option("display.width", 200)
    print("\n=== localization volume L = 1/IPR ===")
    print(g.to_string(index=False, float_format=lambda x: f"{x:8.3f}"))

    w = df[df.cls == "window"]
    if len(w):
        print("\n=== averaging-window convergence (n = 800) ===")
        print(w.groupby(["rule", "window"]).ipr_mean.mean()
              .apply(lambda x: 1 / x).to_string(float_format=lambda x: f"{x:7.1f}"))

    if "twinned" in df:
        t = df.dropna(subset=["twinned"])
        if len(t):
            print("\n=== twins ===")
            print(t.groupby(["rule", "n"]).agg(
                twinned=("twinned", "mean"), ipr=("ipr_mean", "mean"),
                iprmax=("ipr_max", "mean")).to_string(
                float_format=lambda x: f"{x:7.3f}"))


# ------------------------------------------------------------------ figures
# Grayscale throughout, as everywhere else in the book: series are separated by
# marker shape and fill, never by colour.

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

W = 3.9                                  # 4.05in text width on the 5x8 trim

plt.rcParams.update({
    "font.size": 7, "font.family": "serif", "mathtext.fontset": "cm",
    "axes.linewidth": 0.6, "axes.labelsize": 7.5, "axes.titlesize": 8,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.direction": "in", "ytick.direction": "in",
    "legend.fontsize": 6.0, "legend.handletextpad": 0.4,
    "legend.borderaxespad": 0.3, "lines.markeredgewidth": 0.6,
    "figure.dpi": 300, "savefig.dpi": 300,
})

MK = [dict(marker="o", mfc="black", mec="black"),
      dict(marker="s", mfc="white", mec="black"),
      dict(marker="^", mfc="0.55", mec="black"),
      dict(marker="D", mfc="white", mec="0.35"),
      dict(marker="v", mfc="0.8", mec="black"),
      dict(marker="P", mfc="black", mec="black"),
      dict(marker="X", mfc="white", mec="black")]


def panel(ax, letter, x=-0.24):
    ax.text(x, 1.02, letter, transform=ax.transAxes, fontweight="bold",
            fontsize=8, va="bottom", ha="left")


def save(fig, name):
    out = os.path.join(HERE, name)
    fig.savefig(out, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)
    print("wrote", os.path.relpath(out, HERE))


def agg(df):
    d = df[df.cls != "window"]
    g = d.groupby(["rule", "n"]).agg(
        C=("C", "mean"), k=("k", "mean"), ipr=("ipr_mean", "mean"),
        sd=("ipr_mean", "std"), iprmax=("ipr_max", "mean"),
        n_real=("n_real", "mean"), cycle=("cycle", "first"),
        nB=("nB", "first"), cls=("cls", "first")).reset_index()
    g["L"] = 1.0 / g.ipr
    return g


def fig_density():
    """The phenomenon itself: a walker that stays put, next to one that does
    not.  Same panel layout as Boettcher & Porter's Fig. 5, in grayscale."""
    from matplotlib.colors import LogNorm
    fig, axes = plt.subplots(1, 2, figsize=(W, W * 0.50))
    ims = []
    for ax, (lab, g) in zip(axes, [
            ("$BB(n,1)$, a local rule", RULES["BB(n,1)"][0](150, 1)),
            ("the ring", nx.cycle_graph(150))]):
        A = nx.to_numpy_array(g)
        k = A.sum(1)
        dm = 1.0 / np.sqrt(k)
        H = (dm[:, None] * (np.diag(k) - A)) * dm[None, :]
        lam, V = np.linalg.eigh(H)
        # start where the walker is most trapped, as the paper does
        v = ipr(g, nstart=g.number_of_nodes())
        j = int(np.argmax(v))
        ts = np.linspace(0, 120, 400)
        psi = V @ (np.exp(-1j * np.outer(ts, lam)) * V[j, :]).T   # n x nt
        p = np.abs(psi) ** 2
        # order nodes by graph distance from j, so "staying put" is visible as a
        # band rather than scattered across an arbitrary labelling
        dist = nx.single_source_shortest_path_length(g, j)
        order = sorted(g.nodes(), key=lambda w: dist.get(w, 10**6))
        # log colour scale: the trapped walker holds O(0.1) on a couple of nodes
        # while the ring spreads to O(0.01) over hundreds, and a linear scale
        # that shows one washes the other out entirely
        ims.append(ax.pcolormesh(ts, np.arange(g.number_of_nodes()), p[order],
                                 cmap="Greys", norm=LogNorm(vmin=1e-4, vmax=1.0),
                                 rasterized=True, shading="auto"))
        ax.set_xlabel("$t$")
        ax.set_title(lab, fontsize=7, pad=3)
    axes[0].set_ylabel("node, by distance from start")
    axes[1].set_yticklabels([])
    panel(axes[0], "(a)", x=-0.34)
    panel(axes[1], "(b)", x=-0.10)
    cb = fig.colorbar(ims[0], ax=axes, fraction=0.045, pad=0.02)
    cb.set_label(r"$|\psi_i(t)|^2$", fontsize=6.5)
    cb.ax.tick_params(labelsize=5.5)
    save(fig, "fig-qw-density.pdf")


def fig_scaling(g):
    fig, axes = plt.subplots(1, 2, figsize=(W, W * 0.52))

    # (a) the dichotomy: ell saturates for local rules, grows like n otherwise
    ax = axes[0]
    # LS(n,1) is left out: it lands within a node or two of BB(n,1) and BB(n,2)
    # and only crowds them.  The full set is in the table.
    series = [("ring", "ring"), ("ER(n,4)", "$ER(n,4)$"),
              ("BB(n,1)", "$BB(n,1)$"), ("BB(n,2)", "$BB(n,2)$"),
              ("DD(n,0.5)", "$DD(n,0.5)$")]
    for i, (r, lab) in enumerate(series):
        d = g[g.rule == r].sort_values("n")
        delocal = r in ("ring", "ER(n,4)")
        ax.plot(d.n, d.L, ls="-" if delocal else "none", lw=0.7, ms=3.2,
                color="0.45" if delocal else "none", label=lab,
                **MK[i % len(MK)])
    ax.plot([95, 850], [33, 295], ls=":", lw=0.7, c="0.25", zorder=1)
    ax.text(420, 105, r"$\propto n$", fontsize=6.5, color="0.25")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("$n$"); ax.set_ylabel(r"$\ell = 1/\overline{\mathrm{IPR}}$")
    ax.set_xticks([100, 200, 400, 800])
    ax.set_xticklabels(["100", "200", "400", "800"])
    ax.set_xticks([], minor=True)
    ax.set_yticks([5, 10, 20, 50, 100, 200])
    ax.set_yticklabels(["5", "10", "20", "50", "100", "200"])
    ax.set_yticks([], minor=True)
    ax.set_xlim(88, 1000); ax.set_ylim(3.5, 700)
    ax.legend(frameon=False, loc="upper left", fontsize=5.4,
              handletextpad=0.3, labelspacing=0.22, borderpad=0.1)
    panel(ax, "(a)")

    # (b) the footprint law: L against the length of the cycle the rule closes
    ax = axes[1]
    bb = g[(g.n == 800) & g.rule.str.match(r"BB\(n,\d\)$")].copy()
    bb["c"] = bb.cycle
    ax.plot(bb.c, bb.L, ls="none", ms=3.6, **MK[0])
    for _, r in bb.iterrows():
        ax.annotate(f"$L{{=}}{int(r.c)-2}$", (r.c, r.L), fontsize=5.5,
                    textcoords="offset points", xytext=(2.5, -4.5))
    fit = np.polyfit(bb.c, bb.L, 1)
    xs = np.array([2.7, 8.3])
    ax.plot(xs, np.polyval(fit, xs), ls="--", lw=0.7, c="0.35")
    ax.text(0.05, 0.88, rf"$\ell \simeq {fit[0]:.0f}c - {-fit[1]:.0f}$",
            transform=ax.transAxes, fontsize=6.5)
    ax.set_xlabel("cycle length $c$")
    ax.set_ylabel(r"$\ell$")
    ax.set_xlim(2.5, 8.6); ax.set_ylim(0, 52)
    panel(ax, "(b)")

    fig.tight_layout(pad=0.3)
    save(fig, "fig-qw-scaling.pdf")


def logn(ax, sizes):
    """Log size axis with only the measured sizes labelled -- matplotlib's minor
    decade labels collide badly at this figure width."""
    ax.set_xscale("log")
    ax.set_xticks(list(sizes))
    ax.set_xticklabels([str(s) for s in sizes])
    ax.set_xticks([], minor=True)
    ax.set_xlim(min(sizes) * 0.82, max(sizes) * 1.22)


def fig_tests(g, df):
    fig, axes = plt.subplots(1, 3, figsize=(W, W * 0.42))

    # (a) clustering explains nothing: the whole range of ell sits at zero
    # clustering, and the two rules that differ most in clustering agree.
    ax = axes[0]
    d = g[(g.n == 800) & (g.cls != "window")]
    ax.plot(d.C, d.L, ls="none", ms=3.0, **MK[1])
    for lab, r, off in (("ring", "ring", (4, 1)),
                        ("$ER$", "ER(n,4)", (4, -7)),
                        ("$DD$", "DD(n,0.5)", (4, -2)),
                        ("$BB(n,2)$", "BB(n,2)", (5, 1)),
                        ("$BB(n,1)$", "BB(n,1)", (-4, -9))):
        row = d[d.rule == r]
        if len(row):
            ax.annotate(lab, (row.C.iloc[0], row.L.iloc[0]), fontsize=5.2,
                        textcoords="offset points", xytext=off)
    ax.set_xlabel(r"$\langle c\rangle$"); ax.set_ylabel(r"$\ell$")
    ax.set_yscale("log")
    ax.set_yticks([5, 10, 20, 50, 100, 200])
    ax.set_yticklabels(["5", "10", "20", "50", "100", "200"])
    ax.set_yticks([], minor=True)
    ax.set_xlim(-0.12, 0.95); ax.set_ylim(3.5, 700)
    panel(ax, "(a)", x=-0.46)

    # (b) the same square, community class flipped
    ax = axes[1]
    for i, (r, lab) in enumerate([("BB(n,2)", "no comm."),
                                  ("BB2+link", r"$+$ link"),
                                  ("BB2+diag", r"$+$ diagonal")]):
        d = g[g.rule == r].sort_values("n")
        ax.plot(d.n, d.L, ls="none", ms=3.2, label=lab, **MK[i])
    ax.set_xlabel("$n$"); ax.set_ylabel(r"$\ell$")
    logn(ax, SIZES)
    ax.set_ylim(0, 46)
    ax.legend(frameon=False, loc="upper left", fontsize=5.4,
              handletextpad=0.3, labelspacing=0.25, borderpad=0.1)
    panel(ax, "(b)", x=-0.40)

    # (c) twins: duplication's exact symmetries, kept and removed
    ax = axes[2]
    t = df.dropna(subset=["twinned"]) if "twinned" in df else pd.DataFrame()
    if len(t):
        for i, q in enumerate((0.3, 0.5, 0.7)):
            a = t[t.rule == f"DD(n,{q})"].groupby("n").ipr_mean.mean()
            b = t[t.rule == f"DD(n,{q})-merged"].groupby("n").ipr_mean.mean()
            ax.plot(a.index, 1 / a.values, ls="none", ms=3.2,
                    label=f"$q={q}$", **MK[i])
            ax.plot(b.index, 1 / b.values, ls="none", ms=3.2, mfc="none",
                    mec=MK[i].get("mec", "black"), marker=MK[i]["marker"])
        # filled vs hollow is explained in the caption; two more text labels
        # in a panel this narrow only collide with the legend
    ax.set_xlabel("$n$"); ax.set_ylabel(r"$\ell$")
    logn(ax, (200, 400, 800))
    ax.set_ylim(0, 26)
    ax.legend(frameon=False, loc="upper right", fontsize=5.4,
              handletextpad=0.3, labelspacing=0.25, borderpad=0.1)
    panel(ax, "(c)", x=-0.40)

    fig.tight_layout(pad=0.3)
    save(fig, "fig-qw-tests.pdf")


ORDER = ["LS(n,1)", "BB(n,1)", "BB(n,2)", "BB(n,3)", "BB(n,4)", "BB(n,5)",
         "BB(n,6)", "DS(n,1/3)", "DS(n,0.5)", "DD(n,0.5)",
         "BB2+link", "BB2+diag", "BA(n,2)", "ER(n,4)", "ring"]
TEX = {"LS(n,1)": r"$LS(n,1)$", "BB(n,1)": r"$BB(n,1)$", "BB(n,2)": r"$BB(n,2)$",
       "BB(n,3)": r"$BB(n,3)$", "BB(n,4)": r"$BB(n,4)$", "BB(n,5)": r"$BB(n,5)$",
       "BB(n,6)": r"$BB(n,6)$", "DS(n,1/3)": r"$DS(n,1/3)$",
       "DS(n,0.5)": r"$DS(n,0.5)$", "DD(n,0.5)": r"$DD(n,0.5)$",
       "BB2+link": r"$BB(n,2)+$ a link", "BB2+diag": r"$BB(n,2)+$ a diagonal",
       "BA(n,2)": r"$BA(n,2)$", "ER(n,4)": r"$ER(n,4)$", "ring": "ring"}
MOT = {3: r"\ngon{3}", 4: r"\ngon{4}", 5: r"\ngon{5}", 6: r"\ngon{6}",
       7: r"\ngon{7}", 8: r"\ngon{8}"}


def table(g):
    """tab-qw.tex: the master table of the chapter."""
    out = [r"\begin{center}", r"\footnotesize\setlength{\tabcolsep}{3pt}",
           r"\begin{tabular}{llrrrrrrr}", r"\hline",
           r" & & & & \multicolumn{4}{c}{$\ell=1/\overline{\mathrm{IPR}}$} & \\",
           r"rule & motif & $\langle c\rangle$ & $\langle k\rangle$ & $100$ & "
           r"$200$ & $400$ & $800$ & $\langle n_B\rangle$\\", r"\hline"]
    for r in ORDER:
        d = g[g.rule == r].set_index("n").sort_index()
        if not len(d):
            continue
        cyc = d.cycle.iloc[0]
        mot = MOT.get(int(cyc), "---") if cyc == cyc else "---"
        if r == "DD(n,0.5)":
            mot = r"$K_{2,d}$"
        if r == "BB2+link":
            mot = r"\sqstub"
        if r == "BB2+diag":
            mot = r"\sqdiag"
        nB = d.nB.iloc[0]
        cells = " & ".join(f"${d.L[n]:.0f}$" if n in d.index else "---"
                           for n in SIZES)
        # clustering and mean degree quoted at the largest size, so that the
        # whole row describes the same networks the last ell column does
        big = d.iloc[-1]
        out.append(f"{TEX[r]} & {mot} & ${big.C:.2f}$ & ${big.k:.1f}$ & "
                   f"{cells} & {f'${nB:.0f}$' if nB == nB else '---'}" + r"\\")
    out += [r"\hline", r"\end{tabular}", r"\end{center}"]
    p = os.path.join(os.path.dirname(HERE), "tab-qw.tex")
    open(p, "w").write("\n".join(out) + "\n")
    print("wrote", os.path.relpath(p, os.path.dirname(HERE)))


# Density of meso-structure, (kappa_0 + kappa)/n, published in chapter 9 for the
# bubble family: motifs plus communities per node, the invariant that does not
# care which vocabulary names the structure.
MESO = {"BB(n,1)": 0.38, "BB(n,2)": 0.20, "BB(n,3)": 0.15,
        "BB(n,4)": 0.12, "BB(n,5)": 0.09}


def table_meso(g):
    """tab-qw-meso.tex: the walker's territory measured in meso-units instead of
    in nodes.  ell varies threefold across the family; ell * b_0 does not."""
    out = [r"\begin{center}", r"\footnotesize\setlength{\tabcolsep}{5pt}",
           r"\begin{tabular}{lrrrr}", r"\hline",
           r"rule & $c$ & $b_0$ & $\ell$ & $\ell_0$\\", r"\hline"]
    vals = []
    for r, b0 in MESO.items():
        d = g[(g.rule == r) & (g.n == 800)]
        if not len(d):
            continue
        L = d.L.iloc[0]
        vals.append(L * b0)
        out.append(f"{TEX[r]} & ${int(d.cycle.iloc[0])}$ & ${b0:.2f}$ & "
                   f"${L:.0f}$ & ${L*b0:.1f}$" + r"\\")
    out += [r"\hline", r"\end{tabular}", r"\end{center}"]
    p = os.path.join(os.path.dirname(HERE), "tab-qw-meso.tex")
    open(p, "w").write("\n".join(out) + "\n")
    print("wrote", os.path.relpath(p, os.path.dirname(HERE)))
    if vals:
        print(f"  ell*b0: mean {np.mean(vals):.2f}, range "
              f"{min(vals):.2f}-{max(vals):.2f}, "
              f"while ell spans {g[(g.rule.isin(MESO)) & (g.n==800)].L.min():.0f}"
              f"-{g[(g.rule.isin(MESO)) & (g.n==800)].L.max():.0f}")


def table_det(g):
    """tab-qw-det.tex: the deterministic lattices, including the paper's own."""
    out = [r"\begin{center}", r"\footnotesize\setlength{\tabcolsep}{5pt}",
           r"\begin{tabular}{lrrrr}", r"\hline",
           r"lattice & $n$ & $\langle c\rangle$ & $\ell$ & $\ell_{\min}$\\",
           r"\hline"]
    names = {"pseudofractal": "pseudofractal", "rec.triangle": "recursive triangle",
             "diamond(2,2)": r"diamond $(b,s)=(2,2)$",
             "diamond(2,3)": r"diamond $(2,3)$", "diamond(2,4)": r"diamond $(2,4)$",
             "diamond(3,2)": r"diamond $(3,2)$", "diamond(4,2)": r"diamond $(4,2)$"}
    for r, lab in names.items():
        d = g[g.rule == r].sort_values("n")
        for _, row in d.iterrows():
            out.append(f"{lab} & ${row.n:.0f}$ & ${row.C:.2f}$ & "
                       f"${row.L:.1f}$ & ${1/row.iprmax:.1f}$" + r"\\")
            lab = ""
    out += [r"\hline", r"\end{tabular}", r"\end{center}"]
    p = os.path.join(os.path.dirname(HERE), "tab-qw-det.tex")
    open(p, "w").write("\n".join(out) + "\n")
    print("wrote", os.path.relpath(p, os.path.dirname(HERE)))


def main():
    force = "--force" in sys.argv
    if not force and os.path.exists(CSV):
        df = pd.read_csv(CSV)
        print("read", os.path.relpath(CSV, HERE))
    else:
        df = measure()
    report(df)
    g = agg(df)
    fig_density()
    fig_scaling(g)
    fig_tests(g, df)
    table(g)
    table_meso(g)
    table_det(g)


if __name__ == "__main__":
    main()
