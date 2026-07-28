#!/usr/bin/env python3
"""The L* crossover experiment: does a block-model fit undercount short cycles
and overcount long ones, with the crossover where chapter 7 says?

Chapter 7's transfer matrix.  For a model with independent links and expected
adjacency P, the expected number of L-cycles is tr(P^L)/(2L), and for a K-block
degree-corrected fit P collapses onto a K x K matrix

    T_rs = (e_rs / S_r S_s) sqrt(Q_r Q_s)     Q_r = sum of k^2 in block r
                                              S_r = sum of k   in block r

whose leading eigenvalue generates the whole motif spectrum as a geometric
profile in L.  One block is the rank-one baseline and needs no fit at all.  The
prediction is that a real network is undercounted while the profile is below n
and overcounted after, so

    ln(true_L / predicted_L)  is linear in L with slope  -ln lambda,
    crossing zero at          L* = ln n / ln lambda.

Which branching factor.  Chapter 7 takes Q_r = sum k_i^2, giving one-block
eigenvalue lambda = <k^2>/<k>.  That is right for the independent-link
idealisation but not for the model actually fitted, which holds the degree
sequence fixed: a cycle enters a node by one edge and leaves by another, so the
node contributes k_i(k_i - 1) and not k_i^2.  The correct generator is

    T'_rs = (e_rs / S_r S_s) sqrt(Q'_r Q'_s)   Q'_r = sum_{i in r} k_i(k_i - 1)

with one-block eigenvalue kappa = <k^2 - k>/<k> = lambda - 1 -- the leading
eigenvalue of the non-backtracking operator.  branching_verification.py settles
this on synthetic ensembles: kappa reproduces exact counts to 2% at every
L = 3..8, lambda is off by 2x to 10x.  Asymptotically lambda -> infinity on the
chapter's lattices and the two agree, which is why the exact lattice
calculations never saw it; on real networks lambda is around 3 and the two
differ by a factor of 25 at L = 8.  Both are carried here and reported
separately.

Why no asymptotic formula is used on its own.  tr(P^L)/(2L) counts closed walks,
not cycles, and the two only agree when the profile is small next to n.  On the
karate club the rank-one formula predicts 7.2e5 eight-cycles where an actual
degree-preserving rewiring of it has 1.4e4.  Using a formula as the null would
manufacture a steep negative slope out of nothing but finite size.  So each
model is represented by its own *empirical* null, sampled by rewiring, and every
formula is tested against that null rather than assumed:

    null A   degree-preserving rewiring ("configuration")
             -- the rank-one model, and the control the chapter asks for: same
                degrees, no locality
    null B   rewiring that preserves degrees *and* the fitted block matrix
             ("constrained-configuration" with block_membership)
             -- the fitted degree-corrected SBM, sampled rather than linearised

That splits the chapter's claim into two independently falsifiable pieces:

    transfer-matrix theory    ln(null_L / lambda^L 2L) flat in L
                              -- does the fitted model itself produce lambda^L/2L?
    locality deficit          ln(true_L / null_L) linear with slope -ln lambda

and the crossover of the second is compared against both forms of L*:

    (a) operational   L* = ln n / ln lambda,  lambda measured directly
    (b) headline      L* = (gamma - 1)/(3 - gamma),  from the degree exponent

(b) rests on lambda ~ n^{(3-gamma)/(gamma-1)}, which presumes a clean power law
with no structural cutoff, so lambda is always measured and gamma is fitted
independently by tail MLE.

Usage:  python3 crossover_measure.py [--networks a,b,c] [--reps 3] [--no-sbm]
"""
import argparse
import csv
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
CACHE = os.environ.get("CROSSOVER_CACHE", os.path.join(HERE, "_netcache"))
BINARY = os.environ.get("CYCLECOUNT", os.path.join(HERE, "cyclecount"))

LMIN, LMAX = 3, 8
CYCLE_BUDGET = float(os.environ.get("CYCLE_BUDGET", 120))   # seconds, backstop
SBM_MAX_N = int(os.environ.get("SBM_MAX_N", 60000))
# Enumeration cost is bounded below by the answer, and the answer at length L is
# of order kappa^L/(2L), so the reachable depth is fixed by kappa before any
# counting starts.  Capping on the predicted count rather than on elapsed time
# keeps the cap reproducible across machines; the time budget is only a
# backstop.  Whatever the cap gives is reported per network as lmax_reached --
# nothing is silently truncated.
MAX_PRED = float(os.environ.get("MAX_PRED", 2e7))


def depth_cap(kap, lmax=LMAX):
    """Deepest L worth attempting at this branching factor."""
    reach = [L for L in range(LMIN, lmax + 1) if kap ** L / (2 * L) <= MAX_PRED]
    return max(reach) if reach else LMIN + 2


# ---------------------------------------------------------------- graphs

def load_edges(name):
    path = os.path.join(CACHE, name.replace("/", "__") + ".txt")
    with open(path) as fh:
        n, m = (int(x) for x in fh.readline().split())
        E = [tuple(int(x) for x in line.split()) for line in fh]
    assert len(E) == m
    return n, E


def degrees(n, E):
    k = np.zeros(n)
    for u, v in E:
        k[u] += 1
        k[v] += 1
    return k


def to_gt(n, E):
    import graph_tool.all as gt
    g = gt.Graph(directed=False)
    g.add_vertex(n)
    g.add_edge_list(E)
    return g


def rewired(n, E, seed, blocks=None):
    """A sample from the null: degrees preserved exactly, and if `blocks` is
    given the block matrix e_rs preserved exactly too.  Parallel edges and
    self-loops stay off, so the sample is a simple graph and is comparable with
    the original edge for edge."""
    import graph_tool.all as gt
    g = to_gt(n, E)
    gt.seed_rng(seed)
    np.random.seed(seed)
    if blocks is None:
        gt.random_rewire(g, model="configuration", n_iter=20, edge_sweep=True)
    else:
        bp = g.new_vertex_property("int")
        bp.a = blocks
        gt.random_rewire(g, model="constrained-configuration", n_iter=20,
                         edge_sweep=True, block_membership=bp)
    return sorted({(min(int(u), int(v)), max(int(u), int(v)))
                   for u, v in g.get_edges()})


# ---------------------------------------------------------------- cycles

def true_cycles(n, E, lmax=LMAX, budget=CYCLE_BUDGET):
    """Exact cycle counts, as deep in L as the budget allows.

    Enumeration cost is bounded below by the answer, so deep L is out of reach
    on some graphs.  Rather than lose everything to one timeout, climb: a run
    with -l L returns complete counts for 3..L, and cost is dominated by the
    last length, so climbing costs about twice a single run."""
    payload = f"{n} {len(E)}\n" + "\n".join(f"{u} {v}" for u, v in E) + "\n"
    counts, reached, spent = {}, 0, 0.0
    for L in range(LMIN, lmax + 1):
        left = budget - spent
        if left <= 1:
            break
        t = time.time()
        out = subprocess.run([BINARY, "-l", str(L), "-t", f"{left:.1f}"],
                             input=payload, capture_output=True, text=True)
        spent += time.time() - t
        if out.returncode != 0:
            break
        got, status = {}, None
        for line in out.stdout.strip().splitlines():
            a, b = line.split()
            if a == "status":
                status = b
            else:
                got[int(a)] = int(b)
        if status != "ok":
            break
        counts, reached = got, L
    return counts, reached


def null_cycles(n, E, reps, seed, lmax, blocks=None):
    """Mean exact cycle counts over `reps` samples of the null ensemble."""
    acc, reached = {L: [] for L in range(LMIN, lmax + 1)}, lmax
    for r in range(reps):
        nE = rewired(n, E, seed + r, blocks)
        c, got = true_cycles(n, nE, lmax=lmax)
        reached = min(reached, got)
        for L in range(LMIN, got + 1):
            acc[L].append(c.get(L, 0))
    return ({L: float(np.mean(v)) for L, v in acc.items() if v}, reached)


# ---------------------------------------------------------------- models

def transfer_matrix(n, E, k, blocks, excess=True):
    """The K x K generator of the fitted model's whole motif spectrum,

        T_rs = (e_rs / S_r S_s) sqrt(Q_r Q_s)

    with Q_r = sum_{i in r} k_i(k_i - 1) when `excess` (degrees held fixed, the
    model that is actually fitted) and Q_r = sum_{i in r} k_i^2 otherwise
    (chapter 7 as written, the independent-link idealisation)."""
    _, b = np.unique(blocks, return_inverse=True)
    K = b.max() + 1
    e = np.zeros((K, K))
    for u, v in E:
        e[b[u], b[v]] += 1
        e[b[v], b[u]] += 1
    S = np.array([k[b == r].sum() for r in range(K)])
    w = k * (k - 1) if excess else k ** 2
    Q = np.array([w[b == r].sum() for r in range(K)])
    good = S > 0
    e, S, Q = e[np.ix_(good, good)], S[good], Q[good]
    return e / np.outer(S, S) * np.sqrt(np.outer(Q, Q))


def leading(T):
    return float(max(abs(np.linalg.eigvals(T))))


def asymptotic(lam, lmax=LMAX):
    """lambda^L/(2L): the leading term of tr(T^L)/(2L)."""
    return {L: lam ** L / (2 * L) for L in range(LMIN, lmax + 1)}


def fit_nested_sbm(n, E, seed=0):
    """Nested degree-corrected SBM.  Nested, not flat: the flat model's
    resolution ceiling scales as sqrt(N), the same functional form as the
    quantity under test (Peixoto, PRX 2014)."""
    import graph_tool.all as gt
    gt.seed_rng(seed)
    state = gt.minimize_nested_blockmodel_dl(to_gt(n, E),
                                             state_args=dict(deg_corr=True))
    return np.asarray(state.get_bs()[0], dtype=int)[:n]


# ---------------------------------------------------------------- gamma

def fit_gamma(k):
    """Discrete power-law tail MLE with k_min chosen by KS distance
    (Clauset-Shalizi-Newman).  Deliberately independent of lambda: form (b) of
    the prediction and the lambda-gamma scaling relation are separate claims and
    must not share an estimate."""
    from scipy.optimize import minimize_scalar
    from scipy.special import zeta

    ks = np.sort(k[k >= 1].astype(int))
    cands = np.unique(ks)
    cands = cands[cands <= np.percentile(ks, 99)]
    if len(cands) > 60:
        cands = cands[np.linspace(0, len(cands) - 1, 60).astype(int)]

    best = (np.inf, np.nan, np.nan, 0)
    for kmin in cands:
        tail = ks[ks >= kmin]
        nt = len(tail)
        if nt < 50:
            continue
        lsum = np.log(tail).sum()
        res = minimize_scalar(
            lambda a: nt * np.log(zeta(a, kmin)) + a * lsum,
            bounds=(1.05, 6.0), method="bounded")
        a = res.x
        vals, cnts = np.unique(tail, return_counts=True)
        emp = np.cumsum(cnts) / nt
        theo = 1.0 - zeta(a, vals + 1) / zeta(a, kmin)
        d = float(np.max(np.abs(emp - theo)))
        if d < best[0]:
            best = (d, float(a), int(kmin), nt)
    d, gamma, kmin, ntail = best
    return dict(gamma=gamma, gamma_kmin=kmin, gamma_ntail=ntail, gamma_ks=d)


# ---------------------------------------------------------------- the test

def loglin(num, den, lmax):
    """ln(num_L/den_L) against L: slope, standard error, zero crossing.

    Lengths where either count is zero are dropped -- the log is undefined --
    but reported, because an empty length is the strongest possible deficit and
    is exactly what no smooth geometric profile can represent."""
    Ls, y, dropped = [], [], []
    for L in range(LMIN, lmax + 1):
        a, b = num.get(L, 0), den.get(L, 0)
        if a <= 0 or b <= 0:
            dropped.append(L)
            continue
        Ls.append(L)
        y.append(np.log(a) - np.log(b))
    out = dict(npts=len(Ls), dropped=",".join(map(str, dropped)),
               r3=y[0] if Ls and Ls[0] == LMIN else np.nan)
    if len(Ls) < 3:
        out.update(slope=np.nan, slope_se=np.nan, cross=np.nan)
        return out
    X, y = np.array(Ls, float), np.array(y)
    A = np.vstack([X, np.ones_like(X)]).T
    coef = np.linalg.lstsq(A, y, rcond=None)[0]
    resid = y - A @ coef
    dof = len(X) - 2
    s2 = (resid ** 2).sum() / dof if dof > 0 else np.nan
    se = np.sqrt(s2 * np.linalg.inv(A.T @ A)[0, 0]) if dof > 0 else np.nan
    out.update(slope=float(coef[0]), slope_se=float(se),
               cross=float(-coef[1] / coef[0]) if coef[0] else np.nan)
    return out


def put(row, prefix, d):
    for key, val in d.items():
        row[f"{prefix}_{key}"] = val


def measure(name, reps=3, do_sbm=True, seed=1):
    n, E = load_edges(name)
    k = degrees(n, E)
    m = len(E)
    lam1 = float((k ** 2).sum() / k.sum())              # chapter 7 as written
    kap1 = float((k * (k - 1)).sum() / k.sum())         # excess degree, correct
    row = dict(network=name, n=n, m=m, mean_k=round(k.mean(), 4),
               max_k=int(k.max()), lam_rank1=lam1, kap_rank1=kap1,
               # P_ij = k_i k_j / 2m exceeds 1 above this: the independent-link
               # model is not even well defined there
               rank1_pmax=round(float(k.max() ** 2 / (2 * m)), 4),
               lstar_operational=float(np.log(n) / np.log(kap1)) if kap1 > 1 else np.inf,
               lstar_operational_lam=float(np.log(n) / np.log(lam1)),
               minus_ln_kap1=float(-np.log(kap1)) if kap1 > 1 else np.nan,
               minus_ln_lam1=float(-np.log(lam1)))
    row.update(fit_gamma(k))
    g = row["gamma"]
    row["lstar_headline"] = (g - 1) / (3 - g) if 1 < g < 3 else np.inf

    cap = depth_cap(kap1)
    row["depth_cap"] = cap
    t = time.time()
    true, reached = true_cycles(n, E, lmax=cap)
    row["cycle_secs"] = round(time.time() - t, 1)
    if reached < LMIN + 2:
        row["lmax_reached"] = reached
        return row

    # ---- null A: same degrees, no locality (the chapter's control)
    nullA, rA = null_cycles(n, E, reps, seed, reached)
    lmax = min(reached, rA)
    row["lmax_reached"] = lmax
    for L in range(LMIN, LMAX + 1):
        row[f"C{L}"] = true.get(L, "")
        row[f"A{L}"] = round(nullA[L], 2) if L in nullA and L <= lmax else ""

    put(row, "deg", loglin(true, nullA, lmax))               # locality vs degrees
    put(row, "asy1", loglin(true, asymptotic(lam1), lmax))   # spec's literal test
    put(row, "asyK", loglin(true, asymptotic(kap1), lmax))   # same, corrected
    # do the fitted models' own cycles follow the geometric profile at all?
    put(row, "val1", loglin(nullA, asymptotic(lam1), lmax))
    put(row, "valK", loglin(nullA, asymptotic(kap1), lmax))

    # ---- null B: same degrees and the same fitted block matrix
    if do_sbm and n <= SBM_MAX_N:
        t = time.time()
        blocks = fit_nested_sbm(n, E, seed)
        kapS = leading(transfer_matrix(n, E, k, blocks, excess=True))
        lamS = leading(transfer_matrix(n, E, k, blocks, excess=False))
        row.update(sbm_blocks=int(len(np.unique(blocks))),
                   lam_sbm=lamS, kap_sbm=kapS,
                   sbm_secs=round(time.time() - t, 1),
                   minus_ln_kapS=float(-np.log(kapS)) if kapS > 1 else np.nan,
                   lstar_sbm=float(np.log(n) / np.log(kapS)) if kapS > 1 else np.inf)
        nullB, rB = null_cycles(n, E, reps, seed + 100, lmax, blocks=blocks)
        lmaxB = min(lmax, rB)
        for L in range(LMIN, LMAX + 1):
            row[f"B{L}"] = round(nullB[L], 2) if L in nullB and L <= lmaxB else ""
        put(row, "sbm", loglin(true, nullB, lmaxB))
        put(row, "asySK", loglin(true, asymptotic(kapS), lmaxB))
        put(row, "valSK", loglin(nullB, asymptotic(kapS), lmaxB))
        put(row, "valS", loglin(nullB, asymptotic(lamS), lmaxB))
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--networks")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--no-sbm", action="store_true")
    ap.add_argument("--max-n", type=int, default=400000)
    ap.add_argument("--out", default=os.path.join(DATA, "crossover_results.csv"))
    args = ap.parse_args()

    if args.networks:
        names = args.networks.split(",")
    else:
        with open(os.path.join(DATA, "crossover_candidates.csv")) as fh:
            names = [r["network"] for r in csv.DictReader(fh)
                     if r["tractable"] == "1" and int(r["n"]) <= args.max_n]

    out = []
    for name in names:
        try:
            row = measure(name, reps=args.reps, do_sbm=not args.no_sbm)
        except Exception as exc:
            print(f"  FAIL {name}: {type(exc).__name__}: {exc}", file=sys.stderr)
            continue
        out.append(row)
        nan = float("nan")
        print(f"  {name:<42} n={row['n']:<7} kap={row['kap_rank1']:<7.2f} "
              f"L<={row['lmax_reached']:<2} "
              f"deg={row.get('deg_slope', nan):>7.3f} "
              f"sbm={row.get('sbm_slope', nan):>7.3f} "
              f"(-ln kap={row['minus_ln_kap1']:>6.3f}) "
              f"valK={row.get('valK_slope', nan):>7.3f}", flush=True)
        keys = sorted({k for r in out for k in r})
        with open(args.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(out)
    print(f"\n{len(out)} networks -> {args.out}")


if __name__ == "__main__":
    main()
