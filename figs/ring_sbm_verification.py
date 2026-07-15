#!/usr/bin/env python3
"""
ring_sbm_verification.py
========================
Verification of the split-vs-no-split model comparison for the partition of a
circulant ring network C_n(1,...,c) under a Bernoulli stochastic block model
with symmetric Beta(alpha, alpha) priors on all connection probabilities and on
the label prior pi.

Hypotheses compared
-------------------
  H_half : balanced contiguous cut,  n1 = n/2   (two arcs of size m = n/2)
  H_zero : no partition,             n1 = 0      (the whole ring is one block)

Edge counts (valid for n >> c, i.e. the two seams do not interact)
-------------------------------------------------------------------
  interior arc block of size n_i:   E_i  = c*n_i - c(c+1)/2   present edges
                                    N_i  = C(n_i, 2)          possible edges
  crossing block (half cut):        E_12 = c(c+1)             present edges
                                    N_12 = n1*n2              possible edges
  no-partition single block:        E    = c*n               present edges (full cycle)
                                    N    = C(n, 2)            possible edges

Marginal likelihood (evidence) contributions
---------------------------------------------
Integrating each theta ~ Beta(alpha, alpha) against theta^E (1-theta)^(N-E)
gives B(E+alpha, N-E+alpha). A block that carries no data (E=0 out of N=0)
integrates to 1, i.e. contributes a bare prior-normaliser B(alpha, alpha) that
does NOT cancel -> this is the Occam factor penalising H_zero's two unused
parameters (theta_1, theta_12).

The global normaliser Z and the prefactor 1/B(alpha,alpha)^4 cancel in the ratio
R = P(n/2) / P(0), so only R is needed:

        B(E_i+a, N_i-E_i+a)^2  * B(E12+a, N12-E12+a) * B(m+a, m+a)
  R  =  ----------------------------------------------------------------
        B(a,a)^2 * B(cn+a, C(n,2)-cn+a) * B(a, n+a)

  P(split) = P(n/2) / [P(n/2) + P(0)] = R / (1 + R) = 1 / (1 + 1/R).

All arithmetic is done in log space with mpmath for numerical stability.

Usage:  python3 ring_sbm_verification.py
Requires: mpmath   (pip install mpmath).  matplotlib is optional (for figures).
"""

import math
import mpmath as mp

mp.mp.dps = 80  # decimal digits of working precision


# ----------------------------------------------------------------------
# Core quantities
# ----------------------------------------------------------------------
def logB(x, y):
    """log of the Beta function B(x, y) = Gamma(x)Gamma(y)/Gamma(x+y)."""
    return mp.loggamma(x) + mp.loggamma(y) - mp.loggamma(x + y)


def log_evidence_half(n, c, a):
    """log P(n/2) up to the constant 1/(Z * B(a,a)^4) that cancels in the ratio."""
    m = n // 2
    E_i = c * m - c * (c + 1) / 2.0            # present edges in each arc
    N_i = (m - c) * (m - c - 1) / 2.0          # = C(m,2) - E_i  = C(m-c, 2)
    within = 2 * logB(E_i + a, N_i + a)        # two identical arc blocks
    E12 = c * (c + 1)                          # crossing edges (both seams)
    between = logB(E12 + a, m * m - E12 + a)    # out of n1*n2 = m*m slots
    label = logB(m + a, m + a)                 # balanced label prior
    return within + between + label


def log_evidence_zero(n, c, a):
    """log P(0) up to the same cancelling constant."""
    ring = logB(c * n + a, n * (n - 1) / 2.0 - c * n + a)  # full cycle: cn edges
    label = logB(a, n + a)                                  # all nodes one label
    unspent = 2 * logB(a, a)   # theta_1, theta_12 carry no data -> Occam factor
    return unspent + ring + label


def log_ratio(n, c, a):
    """log R = log P(n/2) - log P(0)."""
    v = log_evidence_half(n, c, a) - log_evidence_zero(n, c, a)
    return float(v.real) if hasattr(v, "real") else float(v)


def prob_split(n, c, a):
    """P(n/2) / [P(n/2) + P(0)] = 1 / (1 + exp(-log R))."""
    lr = log_ratio(n, c, a)
    return float(1.0 / (1.0 + mp.e ** (-lr)))


def valid(n, c):
    """Arc edge-counting requires each block larger than c and >= c+1 crossings' worth."""
    m = n // 2
    return n % 2 == 0 and m > c and m * m > c * (c + 1)


# ----------------------------------------------------------------------
# 1. Main table:  log R and P(split) across n
# ----------------------------------------------------------------------
def table_main():
    print("=" * 66)
    print("TABLE 1 :  log R  and  P(split)=R/(1+R)   (positive log R => split)")
    print("=" * 66)
    print(f"{'c':>2} {'alpha':>6} {'n':>6} {'log R':>14} {'P(split)':>13}")
    for c in (1, 2):
        for a in (0.5, 1.0, 2.0):
            for n in (8, 16, 32, 64, 128, 256, 512, 1024):
                if not valid(n, c):
                    continue
                print(f"{c:>2} {a:>6.1f} {n:>6} {log_ratio(n,c,a):>14.4f} "
                      f"{prob_split(n,c,a):>13.6g}")
            print()


# ----------------------------------------------------------------------
# 2. c=1 decay exponent:  log R ~ A - B log n ,  claim B = 2*alpha + 3
# ----------------------------------------------------------------------
def table_c1_exponent():
    print("=" * 66)
    print("TABLE 2 :  c=1 decay exponent  (log R ~ A - B log n)")
    print("=" * 66)
    print(f"{'alpha':>6} {'B (measured)':>14} {'2*alpha+3':>12}")
    for a in (0.5, 1.0, 2.0, 3.0):
        B = -(log_ratio(1024, 1, a) - log_ratio(512, 1, a)) / math.log(2)
        print(f"{a:>6.1f} {B:>14.3f} {2*a+3:>12.1f}")
    print()


# ----------------------------------------------------------------------
# 3. c=2 crossover threshold n* (smallest even n with log R > 0)
# ----------------------------------------------------------------------
def table_c2_threshold():
    print("=" * 66)
    print("TABLE 3 :  c=2 crossover threshold n* (smallest even n, log R>0)")
    print("=" * 66)
    print(f"{'alpha':>6} {'n*':>6}")
    for a in (0.3, 0.5, 1.0, 2.0, 3.0):
        nstar = next((n for n in range(2 * 2 + 2, 2000, 2)
                      if valid(n, 2) and log_ratio(n, 2, a) > 0), None)
        print(f"{a:>6.1f} {str(nstar):>6}")
    print()


# ----------------------------------------------------------------------
# 4. c=2 growth slope  d(log R)/dn -> ln 2
# ----------------------------------------------------------------------
def table_c2_slope():
    print("=" * 66)
    print("TABLE 4 :  c=2 large-n slope  d(log R)/dn  ->  ln 2 = %.6f" % math.log(2))
    print("=" * 66)
    print(f"{'alpha':>6} {'n':>8} {'slope':>12}")
    for a in (1.0,):
        for n in (2048, 8192, 32768, 131072):
            s = (log_ratio(2 * n, 2, a) - log_ratio(n, 2, a)) / n
            print(f"{a:>6.1f} {n:>8} {s:>12.6f}")
    print()


# ----------------------------------------------------------------------
# 5. Ramsey community number kappa  (arXiv:2501.17042)
#    kappa_q = smallest even n at which P(split) >= q, i.e. the minimum
#    network size at which the two-community description takes over.
# ----------------------------------------------------------------------
def kappa(c, a, q, nmax=20000):
    for n in range(2 * c + 2, nmax, 2):
        if valid(n, c) and prob_split(n, c, a) >= q:
            return n
    return None  # None == infinity (communities never emerge)


def table_kappa():
    levels = (0.5, 0.90, 0.99, 0.999, 0.9999)
    print("=" * 66)
    print("TABLE 5 :  Ramsey community number  kappa_q")
    print("           (smallest even n with P(split) >= q; 'inf' = never)")
    print("=" * 66)
    header = f"{'c':>2} {'alpha':>6}" + "".join(f"{'q='+str(q):>10}" for q in levels)
    print(header)
    for c in (1, 2):
        for a in (0.3, 0.5, 1.0, 2.0, 3.0):
            cells = "".join(f"{(str(kappa(c,a,q)) if kappa(c,a,q) else 'inf'):>10}"
                            for q in levels)
            print(f"{c:>2} {a:>6.1f}{cells}")
    print()


# ----------------------------------------------------------------------
# 6. Optional figures
# ----------------------------------------------------------------------
def make_figures():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[figures skipped: matplotlib unavailable: {e}]")
        return

    plt.rcParams.update({"font.size": 11, "mathtext.fontset": "cm",
                         "axes.titlesize": 12})
    ns = [n for n in range(6, 260, 2)]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.6))

    for c, color in ((1, "#c0392b"), (2, "#2c6fbb")):
        for a, ls in ((0.5, ":"), (1.0, "-"), (2.0, "--")):
            xs = [n for n in ns if valid(n, c)]
            ax1.plot(xs, [prob_split(n, c, a) for n in xs], ls, color=color,
                     lw=1.6, label=rf"$c={c},\ \alpha={a}$")
            ax2.plot(xs, [log_ratio(n, c, a) for n in xs], ls, color=color, lw=1.6)

    k99 = kappa(2, 1.0, 0.99)
    if k99:
        ax1.axvline(k99, color="#2c6fbb", lw=0.9, ls="-.")
        ax1.annotate(rf"$\kappa\approx{k99}$", xy=(k99, 0.5),
                     xytext=(k99 + 10, 0.42), color="#2c6fbb", fontsize=11)
    ax1.axhline(0.99, color="grey", lw=0.6, ls=":")
    ax1.axhline(0.5, color="grey", lw=0.8)
    ax1.set_xlabel(r"$n$"); ax1.set_ylabel(r"$P(\mathrm{split})=R/(1+R)$")
    ax1.set_title("Two-community posterior weight")
    ax1.legend(fontsize=8, ncol=2, loc="center right"); ax1.set_ylim(-0.03, 1.06)

    ax2.axhline(0.0, color="grey", lw=0.8)
    ax2.set_xlabel(r"$n$"); ax2.set_ylabel(r"$\log R=\log[P(n/2)/P(0)]$")
    ax2.set_title("Log evidence ratio")
    ax2.set_ylim(-40, 60)

    fig.tight_layout()
    fig.savefig("ring_sbm_figure.png", dpi=200, bbox_inches="tight")
    fig.savefig("ring_sbm_figure.pdf", bbox_inches="tight")
    print("[wrote ring_sbm_figure.png and .pdf]")


if __name__ == "__main__":
    table_main()
    table_c1_exponent()
    table_c2_threshold()
    table_c2_slope()
    table_kappa()
    make_figures()
