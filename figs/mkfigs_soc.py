#!/usr/bin/env python3
"""Figures for the chapter "Communities in primate and human societies".

Re-plots saved results of the archeology/ workstreams (the Proc. R. Soc. B
manuscript archeology/manuscript.tex); nothing is recomputed here:

  archeology/0_power/data/{power,results}.csv           synthetic societies
  archeology/1_fission/results/ngogo_temporal.csv       Ngogo fission by year
  archeology/4_timeseries/results/dynamics_*            Chaco arrow of time

Differences from the manuscript's figures: grayscale (series by marker shape
and fill), 5x8in trim, Computer Modern maths. The human-support bar panel of
the paper is not redrawn; its numbers are given in the text.

Usage:  ~/anaconda3/envs/gt/bin/python mkfigs_soc.py   (writes fig-soc-*.pdf)
"""

import json
import os
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from mkfigs import W, MK, panel, save  # noqa: F401  (import sets book rcParams)

HERE = os.path.dirname(os.path.abspath(__file__))
A = os.path.join(os.path.dirname(HERE), "archeology")
H = 1.85
TK = [20, 50, 100, 200, 500]


def line(ax, x, y, i, **kw):
    st = MK[i % len(MK)]
    ax.plot(x, y, ls="-", color="black", lw=0.8, ms=3.2, zorder=3, **st, **kw)


def logn(ax):
    ax.set_xscale("log")
    ax.set_xticks(TK)
    ax.set_xticklabels(TK)
    ax.minorticks_off()
    ax.set_xlim(16, 620)


# --------------------------------------------------------------------------
# fig-soc-power: what a small society can reveal
# --------------------------------------------------------------------------
pw = pd.read_csv(os.path.join(A, "0_power/data/power.csv"))
pw = pw[(pw.dense == 0) & (pw.h == 0.8) & (pw.K == 2) & (pw.contrast == "LSc0_vs_LSc1")]
rs = pd.read_csv(os.path.join(A, "0_power/data/results.csv"))
rs = rs[(rs.dense == 0) & (rs.h == 0.8) & (rs.K == 2)]

fig, ax = plt.subplots(1, 2, figsize=(W, H))
a = ax[0]
for i, st in enumerate(["conv", "align_ari", "b_same"]):
    t = pw[pw.stat == st].sort_values("n")
    line(a, t.n, t.auc, i)
a.axhline(0.5, c="0.5", lw=0.5, ls=":")
a.text(22, 0.97, "temporal order", fontsize=6.5, ha="left", va="bottom")
a.text(46, 0.57, "community–trait\noverlap", fontsize=6.5, ha="left", va="bottom")
a.text(22, 0.17, "trait similarity\nof ties", fontsize=6.5, ha="left", va="top")
logn(a)
a.set_ylim(-0.05, 1.08)
a.set_xlabel("group size $n$")
a.set_ylabel("AUC")
panel(a, "a")

b = ax[1]
for i, lam in [(0, 0.0), (1, 1.0)]:
    t = rs[(rs.family == "LS") & (rs.lam == lam)].groupby("n").detect.mean()
    line(b, t.index, t.values, i)
b.axvspan(81, 98, color="0.88", zorder=0, lw=0)
b.text(89, 1.04, r"$r_\kappa$", ha="center", va="bottom", fontsize=7)
b.text(330, 0.90, r"$\lambda=0$", fontsize=6.5, ha="center", va="top")
b.text(330, 0.22, r"$\lambda=1$", fontsize=6.5, ha="center", va="top")
logn(b)
b.set_ylim(-0.05, 1.13)
b.set_xlabel("group size $n$")
b.set_ylabel(r"$P(B>1)$")
panel(b, "b")
fig.tight_layout(w_pad=1.2)
save(fig, "fig-soc-power.pdf")

# --------------------------------------------------------------------------
# fig-soc-ngogo: who went where
# --------------------------------------------------------------------------
t = pd.read_csv(os.path.join(A, "1_fission/results/ngogo_temporal.csv"))
t = t[(t.n_lab1 > 0) & (t.n_eval >= 20)]
s = t.dropna(subset=["space_balacc"])

fig, a = plt.subplots(1, 1, figsize=(W, H))
a.axvspan(2015, 2017.4, color="0.9", zorder=0, lw=0)
a.text(2016.2, 0.16, "split", ha="center", fontsize=6.5)
line(a, t.year, t.neighvote_balacc, 0)
line(a, s.year, s.space_balacc, 1)
line(a, s.year, s.attr_balacc, 2)
a.axhline(0.5, c="0.5", lw=0.5, ls=":")
a.text(2003.2, 0.86, "social network", fontsize=6.5)
a.text(2009.0, 1.0, "ranging area", fontsize=6.5, ha="right", va="center")
a.text(2009.0, 0.42, "sex and age", fontsize=6.5, ha="right", va="center")
a.set_ylim(0.1, 1.08)
a.set_xlim(2002.5, 2017.5)
a.set_xticks(range(2003, 2018, 2))
a.set_xlabel("year of the network")
a.set_ylabel("balanced accuracy")
fig.tight_layout()
save(fig, "fig-soc-ngogo.pdf")

# --------------------------------------------------------------------------
# fig-soc-chaco: an arrow of time
# --------------------------------------------------------------------------
R = os.path.join(A, "4_timeseries/results")
with open(os.path.join(R, "dynamics_thr0.75_null200.pkl"), "rb") as f:
    d = pickle.load(f)
d = [r for r in d if r["ds"] == "chaco" and r["obs"]["n"] >= 30]
d.sort(key=lambda r: (r["p1"] - r["p0"], r["p0"]))
nd = np.load(os.path.join(R, "dynamics_joint_null500_draws.npy"))[:, 0]
with open(os.path.join(R, "dynamics_joint_null500.json")) as f:
    obs = json.load(f)["asym_all"]["obs"]

fig, ax = plt.subplots(1, 2, figsize=(W, H), gridspec_kw={"width_ratios": [1.5, 1]})
a = ax[0]
x = np.arange(len(d))
for i, (key, off, nm) in enumerate([("beta_fwd", -0.16, "forward"), ("beta_bwd", 0.16, "reversed")]):
    v = np.array([r["obs"][key] for r in d])
    ci = np.array([r["obs"]["boot"][key] for r in d])
    st = MK[i]
    a.errorbar(x + off, v, yerr=[v - ci[:, 0], ci[:, 1] - v], ls="none", ecolor="black",
               elinewidth=0.6, capsize=0, ms=3.2, label=nm, color="black", **st)
a.axhline(0, c="0.5", lw=0.5)
a.axvline(3.5, c="0.5", lw=0.5, ls="--")
a.text(1.5, 0.33, "50 yr", ha="center", fontsize=6.5)
a.text(5.5, 0.33, "100 yr", ha="center", fontsize=6.5)
a.set_xticks(x)
a.set_xticklabels([str(r["p0"]) for r in d], fontsize=6, rotation=45)
a.set_xlabel("earlier slice (AD)")
a.set_ylabel("convergence effect")
a.set_ylim(-0.9, 0.42)
a.legend(loc="lower center", ncol=2, frameon=False)
panel(a, "a", x=-0.26)

b = ax[1]
b.hist(nd, bins=30, color="0.7", edgecolor="white", lw=0.3)
b.axvline(obs, c="black", lw=1.4)
b.set_xlim(nd.min() - 0.005, obs + 0.03)
b.text(obs - 0.006, b.get_ylim()[1] * 0.95, "observed", ha="right", va="top", fontsize=6.5)
b.set_xlabel("forward excess")
b.set_yticks([])
b.spines["left"].set_visible(False)
b.spines["top"].set_visible(False)
b.spines["right"].set_visible(False)
panel(b, "b", x=-0.08)
fig.tight_layout(w_pad=1.0)
save(fig, "fig-soc-chaco.pdf")
