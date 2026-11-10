"""Power summaries and figures from data/results.csv.  Any python with numpy/pandas/
scipy/matplotlib (gt env is fine).

Writes data/power.csv (binary contrasts), data/lambda.csv (lambda recovery),
data/confusion.csv (mechanism classification), figs/*.pdf|png.
"""
import os, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "data")
F = os.path.join(HERE, "figs")
rng = np.random.default_rng(0)
NBOOT = 400

# a priori direction: +1 = larger under homophily, -1 = larger under network-driven
STATS = {
    "align_nmi": +1, "align_ari": +1, "beyond_dc": -1, "beyond_ndc": -1, "expl_dc": +1,
    "logit_ratio": +1, "abs_b_cn": -1, "b_same": +1, "hist_ari": +1, "fis_attr": +1, "fis_diff": +1,
    "conv": -1, "sel": +1,
}
LABEL = {
    "align_nmi": "a NMI(SBM, attr)", "align_ari": "a ARI(SBM, attr)",
    "beyond_dc": "b DL beyond attr (DC)", "beyond_ndc": "b DL beyond attr (NDC)",
    "expl_dc": "b DL explained by attr", "logit_ratio": "c logit same/(same+CN)",
    "abs_b_cn": "c |logit CN coef|", "b_same": "c logit same coef", "hist_ari": "d ARI across histories",
    "fis_attr": "e ARI(cut, attr)", "fis_diff": "e ARI(cut,attr)-ARI(cut,topo)",
    "conv": "f tie->similarity", "sel": "f similarity->tie",
}


def auc(x0, x1):
    """P(X1 > X0) + 0.5 P(tie)."""
    x0 = np.asarray(x0); x1 = np.asarray(x1)
    allx = np.concatenate([x0, x1])
    r = pd.Series(allx).rank().values
    r1 = r[len(x0):].sum()
    return (r1 - len(x1) * (len(x1) + 1) / 2) / (len(x0) * len(x1))


def contrast(df0, df1, stat, sign):
    """AUC (predicted direction) with bootstrap CI, and one-sided power at alpha = 0.05
    for both nulls.  NaNs are set to the pooled median (uninformative)."""
    x0 = sign * df0[stat].values.astype(float)
    x1 = sign * df1[stat].values.astype(float)
    nanfrac = np.mean(np.isnan(np.concatenate([x0, x1])))
    if nanfrac == 1:
        return dict(auc=np.nan, lo=np.nan, hi=np.nan, pow_h=np.nan, pow_n=np.nan, nanfrac=1.0)
    med = np.nanmedian(np.concatenate([x0, x1]))
    x0 = np.where(np.isnan(x0), med, x0); x1 = np.where(np.isnan(x1), med, x1)
    a = auc(x0, x1)
    bs = [auc(rng.choice(x0, len(x0)), rng.choice(x1, len(x1))) for _ in range(NBOOT)]
    # null = network-driven (x0): reject when x1 > its 95th percentile; and vice versa
    pow_h = np.mean(x1 > np.quantile(x0, 0.95))
    pow_n = np.mean(x0 < np.quantile(x1, 0.05))
    return dict(auc=a, lo=np.quantile(bs, 0.025), hi=np.quantile(bs, 0.975),
                pow_h=pow_h, pow_n=pow_n, nanfrac=nanfrac)


def sel_rows(df, fam, lam, h=0.8, K=2, dense=0):
    m = (df.family == fam) & np.isclose(df.lam, lam) & (df.dense == dense)
    if fam not in ("LSc", "DSc", "BA") and not (fam in ("LS", "DS") and lam == 0 and K == 2):
        m &= np.isclose(df.h, h)
    return df[m & (df.K == K)]


def power_table(df):
    rows = []
    contrasts = [  # (name, null(network-like), alternative(homophily-like), stats)
        ("LS0_vs_LS1", ("LS", 0.0), ("LS", 1.0)),
        ("DS0_vs_DS1", ("DS", 0.0), ("DS", 1.0)),
        ("LS0_vs_LS.5", ("LS", 0.0), ("LS", 0.5)),
        ("DS0_vs_DS.5", ("DS", 0.0), ("DS", 0.5)),
        ("LS0_vs_ASBM", ("LS", 0.0), ("ASBM", 1.0)),
        ("SP0_vs_SP1", ("SP", 0.0), ("SP", 1.0)),
        ("LS0_vs_SP0", ("LS", 0.0), ("SP", 0.0)),
        ("LSc0_vs_LSc1", ("LSc", 0.0), ("LSc", 1.0)),
        ("DSc0_vs_DSc1", ("DSc", 0.0), ("DSc", 1.0)),
    ]
    for dense in (0, 1):
        for n in sorted(df[df.dense == dense].n.unique()):
            sub = df[(df.n == n)]
            for hK in ((0.8, 2), (0.5, 2), (0.95, 2), (0.8, 4), (0.8, 8)):
                h, K = hK
                for name, c0, c1 in contrasts:
                    if hK != (0.8, 2) and not name.startswith(("LS0_vs_LS", "DS0_vs_DS", "LS0_vs_ASBM")):
                        continue
                    d0 = sel_rows(sub, *c0, h=h, K=K, dense=dense)
                    d1 = sel_rows(sub, *c1, h=h, K=K, dense=dense)
                    if len(d0) < 10 or len(d1) < 10:
                        continue
                    for st, sign in STATS.items():
                        if st in ("conv", "sel") and not name.startswith(("LSc", "DSc")):
                            continue
                        r = contrast(d0, d1, st, sign)
                        rows.append(dict(contrast=name, n=n, dense=dense, h=h, K=K, stat=st,
                                         n0=len(d0), n1=len(d1), **r))
    return pd.DataFrame(rows)


def fill(X):
    """Median-impute NaNs (all-NaN columns -> 0) and standardize."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        med = np.nanmedian(X, 0)
    med = np.where(np.isnan(med), 0, med)
    X = np.where(np.isnan(X), med, X)
    return (X - X.mean(0)) / (X.std(0) + 1e-12)


def lambda_table(df):
    rows = []
    feats = ["align_nmi", "beyond_dc", "expl_dc", "logit_ratio", "hist_ari", "fis_diff"]
    for dense in (0, 1):
        for fam in ("LS", "DS"):
            for n in sorted(df[df.dense == dense].n.unique()):
                d = df[(df.family == fam) & (df.n == n) & (df.dense == dense) & (df.K == 2)
                       & (np.isclose(df.h, 0.8) | (df.lam == 0))]
                d = d[d.lam.isin([0, .25, .5, .75, 1])]
                if d.lam.nunique() < 3:
                    continue
                for st in feats:
                    x = d[st].values.astype(float); y = d.lam.values
                    ok = ~np.isnan(x)
                    rho = spearmanr(x[ok], y[ok]).statistic if ok.sum() > 10 else np.nan
                    bs = []
                    for _ in range(200):
                        i = rng.integers(0, ok.sum(), ok.sum())
                        bs.append(spearmanr(x[ok][i], y[ok][i]).statistic)
                    rows.append(dict(family=fam, n=n, dense=dense, stat=st, rho=rho,
                                     lo=np.nanquantile(bs, .025), hi=np.nanquantile(bs, .975)))
                # all features: 5-fold CV linear prediction of lambda
                X = fill(d[feats].values.astype(float))
                y = d.lam.values
                fold = rng.permutation(len(y)) % 5
                pred = np.zeros(len(y))
                for k in range(5):
                    tr, te = fold != k, fold == k
                    A = np.column_stack([np.ones(tr.sum()), X[tr]])
                    w = np.linalg.solve(A.T @ A + 1e-2 * np.eye(A.shape[1]), A.T @ y[tr])
                    pred[te] = np.column_stack([np.ones(te.sum()), X[te]]) @ w
                rmse = np.sqrt(np.mean((np.clip(pred, 0, 1) - y) ** 2))
                rows.append(dict(family=fam, n=n, dense=dense, stat="ALL_cv_rmse", rho=rmse,
                                 lo=np.nan, hi=np.nan))
    return pd.DataFrame(rows)


CLASSES = {"network": [("LS", 0.0), ("DS", 0.0)],
           "homophily": [("LS", 1.0), ("DS", 1.0), ("ASBM", 1.0)],
           "space": [("SP", 0.0)],
           "none(BA)": [("BA", 0.0)]}
CFEATS = ["align_nmi", "beyond_dc", "expl_dc", "logit_ratio", "b_cn", "b_dist", "hist_ari",
          "fis_diff", "dcgain", "detect"]


def softmax_fit(X, y, C, l2=1.0):
    d = X.shape[1]

    def f(w):
        W = w.reshape(C, d)
        Z = X @ W.T; Z -= Z.max(1, keepdims=True)
        P = np.exp(Z); P /= P.sum(1, keepdims=True)
        loss = -np.log(P[np.arange(len(y)), y] + 1e-12).sum() + l2 * (W[:, 1:] ** 2).sum()
        G = (P - np.eye(C)[y]).T @ X; G[:, 1:] += 2 * l2 * W[:, 1:]
        return loss, G.ravel()
    return minimize(f, np.zeros(C * d), jac=True, method="L-BFGS-B").x.reshape(C, d)


def confusion(df, feats=CFEATS):
    rows = []
    names = list(CLASSES)
    for dense in (0, 1):
        for n in sorted(df[df.dense == dense].n.unique()):
            parts = []
            for ci, cname in enumerate(names):
                for fam, lam in CLASSES[cname]:
                    d = sel_rows(df[df.n == n], fam, lam, dense=dense)
                    # balance: homophily has three sources, network two
                    d = d.iloc[: 300 // len(CLASSES[cname])]
                    parts.append(d.assign(cls=ci))
            d = pd.concat(parts)
            X = fill(d[feats].values.astype(float))
            X = np.column_stack([np.ones(len(X)), X])
            y = d.cls.values
            fold = rng.permutation(len(y)) % 5
            pred = np.zeros(len(y), int)
            for k in range(5):
                W = softmax_fit(X[fold != k], y[fold != k], len(names))
                pred[fold == k] = np.argmax(X[fold == k] @ W.T, 1)
            for i, a in enumerate(names):
                for j, b in enumerate(names):
                    m = y == i
                    rows.append(dict(n=n, dense=dense, true=a, pred=b,
                                     frac=np.mean(pred[m] == j) if m.any() else np.nan,
                                     count=int(m.sum())))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ figures

def figures(df, pw, lt, cf):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    os.makedirs(F, exist_ok=True)
    grays = ["0.0", "0.35", "0.6", "0.8"]
    marks = "osD^v<>p*h"

    # Fig 1: AUC vs n, network vs homophily
    panels = [("LS0_vs_LS1", 0, "LS: λ=0 vs λ=1 (sparse)"), ("DS0_vs_DS1", 0, "DS: λ=0 vs λ=1 (sparse)"),
              ("LS0_vs_LS1", 1, "LS: λ=0 vs λ=1 (density 0.9)"),
              ("LSc0_vs_LSc1", 0, "LS + attribute dynamics: copy vs drift")]
    fig, axes = plt.subplots(2, 2, figsize=(8.5, 6.2), sharey=True)
    show = ["align_nmi", "align_ari", "beyond_dc", "logit_ratio", "hist_ari", "fis_attr"]
    for ax, (cname, dense, title) in zip(axes.ravel(), panels):
        sts = show + (["conv", "sel"] if cname.startswith("LSc") else [])
        for k, st in enumerate(sts):
            d = pw[(pw.contrast == cname) & (pw.stat == st) & (pw.dense == dense)
                   & np.isclose(pw.h, .8) & (pw.K == 2)].sort_values("n")
            ax.errorbar(d.n * (1 + 0.025 * (k - 3)), d.auc, yerr=[d.auc - d.lo, d.hi - d.auc],
                        marker=marks[k], color=str(0.72 * k / max(1, len(sts) - 1)),
                        mfc="white" if k % 2 else None, ms=4, lw=1, capsize=1.5,
                        label=LABEL[st])
        ax.axhline(0.5, color="0.7", lw=0.6, ls=":")
        ax.axhline(0.8, color="0.85", lw=0.6)
        ax.set_xscale("log")
        ticks = [20, 35, 45] if dense else [20, 45, 100, 200, 500]
        ax.set_xticks(ticks); ax.set_xticklabels([str(t) for t in ticks])
        ax.minorticks_off()
        ax.set_xlabel("n"); ax.set_title(title, fontsize=9)
    for ax in axes[:, 0]:
        ax.set_ylabel("AUC (predicted direction)")
    axes[0, 0].set_ylim(0.0, 1.03)
    axes[1, 1].legend(fontsize=6.5, frameon=False, loc="lower left", ncol=2)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(F, f"fig-auc-vs-n.{ext}"), dpi=200)
    plt.close(fig)

    # Fig 2: statistic vs lambda (LS), boxes at n = 45 and 200
    fig, axes = plt.subplots(2, 4, figsize=(10, 4.6))
    for row, n in enumerate((45, 200)):
        for ax, st in zip(axes[row], ["align_ari", "expl_dc", "logit_ratio", "fis_attr"]):
            d = df[(df.family == "LS") & (df.n == n) & (df.dense == 0) & (df.K == 2)
                   & (np.isclose(df.h, .8) | (df.lam == 0))]
            data = [d[np.isclose(d.lam, l)][st].dropna().values for l in [0, .25, .5, .75, 1]]
            ax.boxplot(data, labels=["0", ".25", ".5", ".75", "1"], widths=0.5, showfliers=False,
                       medianprops=dict(color="k"), boxprops=dict(color="0.3"),
                       whiskerprops=dict(color="0.3"), capprops=dict(color="0.3"))
            ax.set_title(f"{LABEL[st]}, n={n}", fontsize=8)
            if row == 1:
                ax.set_xlabel("λ (homophily share)")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(F, f"fig-stat-vs-lambda.{ext}"), dpi=200)
    plt.close(fig)

    # Fig 3: confusion matrices
    show_n = [(20, 0), (45, 0), (100, 0), (500, 0), (45, 1)]
    fig, axes = plt.subplots(1, len(show_n), figsize=(11, 2.6))
    names = list(CLASSES)
    for ax, (n, dense) in zip(axes, show_n):
        d = cf[(cf.n == n) & (cf.dense == dense)]
        if d.empty:
            continue
        M =d.pivot(index="true", columns="pred", values="frac").loc[names, names].values
        ax.imshow(M, cmap="Greys", vmin=0, vmax=1)
        for i in range(len(names)):
            for j in range(len(names)):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if M[i, j] > 0.55 else "black")
        ax.set_xticks(range(len(names))); ax.set_xticklabels([s[:4] for s in names], fontsize=7)
        ax.set_yticks(range(len(names))); ax.set_yticklabels([s[:4] for s in names], fontsize=7)
        ax.set_title(f"n={n}" + (" dense" if dense else ""), fontsize=9)
        ax.set_xlabel("predicted", fontsize=7)
    axes[0].set_ylabel("true")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(F, f"fig-confusion.{ext}"), dpi=200)
    plt.close(fig)

    # Fig 4: detection probability (B>1) vs n by mechanism
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    for k, (fam, lam, lab) in enumerate([("LS", 0, "LS λ=0"), ("DS", 0, "DS λ=0"),
                                         ("LS", 1, "LS λ=1 (PA-homophily)"),
                                         ("ASBM", 1, "attribute SBM"), ("SP", 0, "spatial"),
                                         ("BA", 0, "BA")]):
        d = df[(df.family == fam) & np.isclose(df.lam, lam) & (df.dense == 0) & (df.K == 2)
               & (np.isclose(df.h, .8) | (df.lam == 0) | (df.family == "BA"))]
        g = d.groupby("n").detect.mean()
        ax.plot(g.index, g.values, marker=marks[k], color=str(0.7 * k / 5),
                mfc="white" if k % 2 else None, ms=4, lw=1, label=lab)
    ax.set_xscale("log"); ax.set_xticks([20, 45, 100, 200, 500])
    ax.set_xticklabels(["20", "45", "100", "200", "500"])
    ax.set_xlabel("n"); ax.set_ylabel("P(DC-SBM finds B>1)")
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(F, f"fig-detect.{ext}"), dpi=200)
    plt.close(fig)


def main():
    df = pd.read_csv(os.path.join(D, "results.csv"))
    df["abs_b_cn"] = df.b_cn.abs()
    pw = power_table(df); pw.to_csv(os.path.join(D, "power.csv"), index=False)
    lt = lambda_table(df); lt.to_csv(os.path.join(D, "lambda.csv"), index=False)
    cf = confusion(df); cf.to_csv(os.path.join(D, "confusion.csv"), index=False)
    figures(df, pw, lt, cf)
    # console digest
    pd.set_option("display.width", 200)
    m = pw[np.isclose(pw.h, .8) & (pw.K == 2)]
    for c in ("LS0_vs_LS1", "DS0_vs_DS1", "LS0_vs_LS.5", "LS0_vs_ASBM", "SP0_vs_SP1",
              "LS0_vs_SP0", "LSc0_vs_LSc1", "DSc0_vs_DSc1"):
        t = m[m.contrast == c].pivot_table(index="stat", columns=["dense", "n"], values="auc")
        print(f"\nAUC {c}\n", t.round(2))
    for c in ("LS0_vs_LS1", "DS0_vs_DS1", "LSc0_vs_LSc1"):
        for p in ("pow_h", "pow_n"):
            t = m[m.contrast == c].pivot_table(index="stat", columns=["dense", "n"], values=p)
            print(f"\n{p} {c}\n", t.round(2))
    for c in ("LS0_vs_LS1", "DS0_vs_DS1"):
        print(f"\nAUC sensitivity ({c})")
        s = pw[(pw.contrast == c) & (pw.dense == 0) & pw.n.isin([45, 100, 200])]
        print(s.pivot_table(index="stat", columns=["h", "K", "n"], values="auc").round(2).to_string())
    print("\nconfusion\n", cf.pivot_table(index=["dense", "n", "true"], columns="pred",
                                          values="frac").round(2).to_string())
    print("\nlambda\n", lt.pivot_table(index=["family", "stat"], columns=["dense", "n"],
                                     values="rho").round(2).to_string())
    print("\nbalanced accuracy", cf[cf.true == cf.pred].groupby(["dense", "n"]).frac.mean().round(2))
    print("\ndetect", df[(df.K == 2)].groupby(["family", "lam", "dense", "n"]).detect.mean()
          .unstack(["dense", "n"]).round(2).to_string())


if __name__ == "__main__":
    main()
