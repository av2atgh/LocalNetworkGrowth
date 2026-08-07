#!/usr/bin/env python3
r"""
qwalk_real.py -- do real protein interaction networks carry the twin symmetry
that traps a quantum walker, and does removing it delocalize them?

The chapter's duplication section proves a theorem about models: two nodes with
identical neighbourhoods force an eigenvector of H = D^{-1/2}(D-A)D^{-1/2} at
eigenvalue exactly 1 supported on two nodes, and merging the twins of a
DD(n,q) network more than doubles its localization volume.  Whether real
interactomes show it was left open.  This script closes it.

The confound that has to be handled, and the reason the answer is not simply
"yes": in a sparsely sampled interactome most proteins are seen with a single
partner, and any two degree-one proteins hanging off the same hub are false
twins by definition.  That is a sampling artifact, not a duplication.  So every
quantity here is reported twice -- over all twins, and over twins of degree at
least two, which share two or more partners and cannot be produced by a single
lucky bait-prey hit.  The second is the number that bears on duplication.

Networks come from Netzschleuder (networks.skewed.de) over its HTTP API rather
than through graph_tool.collection.ns, so this runs without graph-tool; the
CSV bundles are cached under data/netzschleuder/.  Physical-interaction
interactomes are measured against non-biological controls of comparable size,
because the claim is about duplication and not about being a network.

Usage:
    python3 qwalk_real.py [--force]
"""

import io
import os
import sys
import csv
import zipfile
import urllib.request

import numpy as np
import networkx as nx
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
CACHE = os.path.join(HERE, "data", "netzschleuder")
CSV = os.path.join(HERE, "data", "qwalk-real.csv")

from qwalk_ipr import ipr, giant, twin_classes, MK, panel, save, W, plt

BASE = "https://networks.skewed.de/net"

# (netzschleuder key, subnet, label, class).  "ppi" is a physical protein
# interaction network -- the ones gene duplication is a theory of.  "control" is
# anything of comparable size grown by something else.
NETS = [
    ("interactome_pdz",     None,      "PDZ domains",        "ppi"),
    ("collins_yeast",       None,      "yeast (Collins)",    "ppi"),
    ("interactome_yeast",   None,      "yeast (Coulomb)",    "ppi"),
    ("interactome_stelzl",  None,      "human (Stelzl)",     "ppi"),
    ("interactome_figeys",  None,      "human (Figeys)",     "ppi"),
    ("interactome_vidal",   None,      "human (Vidal)",      "ppi"),
    ("celegans_interactomes", "wi2007", "C. elegans (WI-2007)", "ppi"),
    ("celegans_interactomes", "WI8",    "C. elegans (WI8)",   "ppi"),
    # the Internet (n = 22963) and cond-mat co-authorship (n = 36458) are the
    # controls one would want most, and both are far past what a dense
    # eigendecomposition can do here; they are named rather than fetched.
    ("power",               None,      "power grid",         "control"),
    ("euroroad",            None,      "Euroroad",           "control"),
]


def fetch(key, sub=None):
    """Edge list of one Netzschleuder network, cached."""
    sub = sub or key.split("/")[-1]
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, f"{key.replace('/', '_')}__{sub}.csv.zip")
    if not os.path.exists(path):
        url = f"{BASE}/{key.split('/')[0]}/files/{sub}.csv.zip"
        print("  fetching", url, flush=True)
        with urllib.request.urlopen(url, timeout=180) as r:
            data = r.read()
        with open(path, "wb") as f:
            f.write(data)
    with zipfile.ZipFile(path) as z:
        with z.open("edges.csv") as f:
            rows = list(csv.reader(io.TextIOWrapper(f, "utf-8")))
    header = rows[0]
    src = next(i for i, h in enumerate(header) if "source" in h.lower())
    tgt = next(i for i, h in enumerate(header) if "target" in h.lower())
    g = nx.Graph()
    g.add_edges_from((r[src], r[tgt]) for r in rows[1:] if len(r) > max(src, tgt))
    return giant(g)


def twins(g, kmin=1):
    """False-twin classes restricted to nodes of degree >= kmin.

    kmin=1 counts every identical-neighbourhood pair; kmin=2 keeps only twins
    that share at least two partners, which a single spurious bait-prey hit
    cannot manufacture.
    """
    d = {}
    for v in g:
        if g.degree(v) >= kmin:
            d.setdefault(frozenset(g[v]), []).append(v)
    return [c for c in d.values() if len(c) > 1]


def merged(g, kmin=1):
    h = g.copy()
    h.remove_nodes_from([w for cl in twins(g, kmin) for w in cl[1:]])
    return giant(h)


def deg1_mult(g):
    """Multiplicity of eigenvalue 1 of the normalized Laplacian."""
    A = nx.to_numpy_array(g)
    k = A.sum(1)
    dm = 1.0 / np.sqrt(k)
    H = (dm[:, None] * (np.diag(k) - A)) * dm[None, :]
    lam = np.linalg.eigvalsh(H)
    return int((np.abs(lam - 1.0) < 1e-8).sum())


def measure():
    rows = []
    for key, sub, label, cls in NETS:
        try:
            g = fetch(key, sub)
        except Exception as e:                       # a missing net must not
            print(f"  SKIP {label}: {e}", flush=True)  # kill the whole sweep
            continue
        n = g.number_of_nodes()
        if n > 5200:
            print(f"  SKIP {label}: n={n} too large for dense eigh", flush=True)
            continue
        w = (2.0 * n, 4.0 * n)
        v = ipr(g, window=w)
        t1, t2 = twins(g, 1), twins(g, 2)
        f1 = sum(len(c) for c in t1) / n
        f2 = sum(len(c) for c in t2) / n
        row = dict(label=label, cls=cls, key=key, sub=sub or "", n=n,
                   m=g.number_of_edges(), k=2 * g.number_of_edges() / n,
                   C=nx.average_clustering(g),
                   frac_deg1=sum(1 for _, d in g.degree() if d == 1) / n,
                   twin_all=f1, twin_k2=f2,
                   mult1=deg1_mult(g) / n,
                   ipr=v.mean(), iprmax=v.max())
        row["L"] = 1 / row["ipr"]
        # the demolition, both ways
        for kmin, tag in ((1, "all"), (2, "k2")):
            h = merged(g, kmin)
            vh = ipr(h, window=(2.0 * h.number_of_nodes(),
                                4.0 * h.number_of_nodes()))
            row[f"L_merged_{tag}"] = 1 / vh.mean()
            row[f"n_merged_{tag}"] = h.number_of_nodes()
        rows.append(row)
        print(f"  {label:22s} n={n:5d} twins {f1:5.1%}/{f2:5.1%}  "
              f"ell {row['L']:6.1f} -> {row['L_merged_all']:6.1f}/"
              f"{row['L_merged_k2']:6.1f}", flush=True)

        # degree-preserving rewiring: same degrees, nothing else kept
        r = g.copy()
        try:
            nx.double_edge_swap(r, nswap=20 * r.number_of_edges(),
                                max_tries=400 * r.number_of_edges(), seed=1)
        except nx.NetworkXError:
            pass
        r = giant(r)
        nr = r.number_of_nodes()
        vr = ipr(r, window=(2.0 * nr, 4.0 * nr))
        tr1, tr2 = twins(r, 1), twins(r, 2)
        rows.append(dict(
            label=label, cls="rewired", key=key, sub=sub or "", n=nr,
            m=r.number_of_edges(), k=2 * r.number_of_edges() / nr,
            C=nx.average_clustering(r),
            frac_deg1=sum(1 for _, d in r.degree() if d == 1) / nr,
            twin_all=sum(len(c) for c in tr1) / nr,
            twin_k2=sum(len(c) for c in tr2) / nr,
            mult1=deg1_mult(r) / nr, ipr=vr.mean(), iprmax=vr.max(),
            L=1 / vr.mean()))
        print(f"    rewired             n={nr:5d} twins "
              f"{rows[-1]['twin_all']:5.1%}/{rows[-1]['twin_k2']:5.1%}  "
              f"ell {rows[-1]['L']:6.1f}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(CSV, index=False)
    print("wrote", os.path.relpath(CSV, HERE))
    return df


# ------------------------------------------------------------------- reporting

def report(df):
    pd.set_option("display.width", 220)
    cols = ["label", "cls", "n", "k", "C", "frac_deg1", "twin_all", "twin_k2",
            "mult1", "L", "L_merged_all", "L_merged_k2"]
    print("\n=== real networks ===")
    print(df[cols].to_string(index=False,
                             float_format=lambda x: f"{x:7.3f}"))
    for cls in ("ppi", "control"):
        d = df[df.cls == cls]
        if not len(d):
            continue
        print(f"\n{cls}: twins(all) {d.twin_all.mean():.1%}, "
              f"twins(k>=2) {d.twin_k2.mean():.1%}, "
              f"mult(1)/n {d.mult1.mean():.1%}, "
              f"ell {d.L.mean():.1f} -> {d.L_merged_k2.mean():.1f} "
              f"(x{(d.L_merged_k2/d.L).mean():.2f})")


def fig_real(df):
    fig, axes = plt.subplots(1, 3, figsize=(W, W * 0.40))
    ppi = df[df.cls == "ppi"]
    ctl = df[df.cls == "control"]
    rw = df[df.cls == "rewired"].set_index("label")

    # (a) the headline: interactomes localize, comparable controls do not
    ax = axes[0]
    for i, (d, lab) in enumerate([(ppi, "interactomes"), (ctl, "controls")]):
        ax.plot(d.n, d.L, ls="none", ms=3.2, label=lab, **MK[i])
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("$n$"); ax.set_ylabel(r"$\ell$")
    ax.set_xticks([200, 1000, 4000])
    ax.set_xticklabels(["200", "1000", "4000"])
    ax.set_yticks([2, 10, 100]); ax.set_yticklabels(["2", "10", "100"])
    ax.set_xticks([], minor=True); ax.set_yticks([], minor=True)
    ax.set_ylim(1.5, 700)
    ax.legend(frameon=False, loc="upper left", fontsize=5.0,
              handletextpad=0.3, borderpad=0.1)
    panel(ax, "(a)", x=-0.44)

    # (b) the symmetry fingerprint, and how much of it the degrees explain
    ax = axes[1]
    for i, (d, lab) in enumerate([(ppi, "interactomes"), (ctl, "controls")]):
        ax.plot(d.twin_all * 100, d.mult1 * 100, ls="none", ms=3.2, label=lab,
                **MK[i])
    y = [rw.mult1.get(l, np.nan) * 100 for l in ppi.label]
    x = [rw.twin_all.get(l, np.nan) * 100 for l in ppi.label]
    ax.plot(x, y, ls="none", ms=3.2, mfc="none", mec="0.45", marker="o",
            label="rewired")
    ax.set_xlabel("twinned (%)")
    ax.set_ylabel("mult. at $\\lambda{=}1$ (%)")
    ax.legend(frameon=False, loc="upper left", fontsize=5.0,
              handletextpad=0.3, borderpad=0.1)
    panel(ax, "(b)", x=-0.44)

    # (c) against the degree-preserving rewiring -- no intervention on the
    # network itself, so this is the comparison the section leans on
    ax = axes[2]
    for i, (d, lab) in enumerate([(ppi, "interactomes"), (ctl, "controls")]):
        ax.plot(d.L.values, [rw.L.get(l, np.nan) for l in d.label],
                ls="none", ms=3.2, label=lab, **MK[i])
    lim = [1.2, 800]
    ax.plot(lim, lim, ls=":", lw=0.7, c="0.35")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xticks([2, 10, 100]); ax.set_xticklabels(["2", "10", "100"])
    ax.set_yticks([2, 10, 100]); ax.set_yticklabels(["2", "10", "100"])
    ax.set_xticks([], minor=True); ax.set_yticks([], minor=True)
    ax.set_xlabel(r"$\ell$, as measured"); ax.set_ylabel(r"$\ell$, rewired")
    panel(ax, "(c)", x=-0.44)

    fig.tight_layout(pad=0.3)
    save(fig, "fig-qw-real.pdf")


def table(df):
    """tab-qw-real.tex.

    The last column is the degree-preserving rewiring, not the twin merge.  The
    merge deletes up to a third of the nodes and nearly all the leaves, so it
    cannot separate "removed a symmetry" from "removed the periphery"; the
    rewiring changes no node at all and is the comparison the section rests on.
    """
    rw = df[df.cls == "rewired"].set_index("label")
    out = [r"\begin{center}", r"\footnotesize\setlength{\tabcolsep}{3pt}",
           r"\begin{tabular}{lrrrrrrr}", r"\hline",
           r" & & & \multicolumn{2}{c}{twinned} & mult. & "
           r"\multicolumn{2}{c}{$\ell$}\\",
           r"network & $n$ & $\langle c\rangle$ & all & $k\!\geq\!2$ & "
           r"$\lambda{=}1$ & meas. & rewired\\", r"\hline"]
    for cls, head in (("ppi", r"\emph{protein interaction}"),
                      ("control", r"\emph{controls}")):
        d = df[df.cls == cls].sort_values("n")
        if not len(d):
            continue
        out.append(r"\multicolumn{8}{l}{" + head + r"}\\")
        for _, r in d.iterrows():
            lab = r.label.replace("C. elegans", r"\emph{C.\ elegans}")
            lr = rw.L.get(r.label, float("nan"))
            out.append(
                f"\\quad {lab} & ${r.n:.0f}$ & ${r.C:.2f}$ & "
                f"${r.twin_all*100:.0f}\\%$ & ${r.twin_k2*100:.0f}\\%$ & "
                f"${r.mult1*100:.0f}\\%$ & ${r.L:.1f}$ & "
                f"${lr:.1f}$" + r"\\")
    out += [r"\hline", r"\end{tabular}", r"\end{center}"]
    p = os.path.join(os.path.dirname(HERE), "tab-qw-real.tex")
    open(p, "w").write("\n".join(out) + "\n")
    print("wrote", os.path.relpath(p, os.path.dirname(HERE)))


def main():
    if "--force" not in sys.argv and os.path.exists(CSV):
        df = pd.read_csv(CSV)
        print("read", os.path.relpath(CSV, HERE))
    else:
        df = measure()
    report(df)
    fig_real(df)
    table(df)


if __name__ == "__main__":
    main()
