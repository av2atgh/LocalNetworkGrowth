#!/usr/bin/env python3
"""Read data/crossover_results.csv and decide what chapter 7's L* prediction did.

The chapter's own specification splits the outcome three ways, and this script
keeps the pieces separate because they can fail separately.

  TEST 1  transfer-matrix theory.  Does a fitted model's own cycle spectrum
          follow the geometric profile its transfer matrix predicts?  Measured
          as the slope of ln(null_L / kappa^L 2L) against L, on rewirings of the
          real network rather than on a formula.  Flat means yes.

  TEST 2  the locality deficit.  Slope of ln(true_L / null_L) against L, which
          the chapter predicts is exactly -ln kappa.  Reported as the ratio of
          the measured slope to -ln kappa: 1 confirms, 0 means the real network
          tracks its own rewiring, and anything else is a number the chapter
          does not predict.

  TEST 3  where the crossover falls, against the two forms of the prediction,
              (a) operational  L* = ln n / ln kappa
              (b) headline     L* = (gamma - 1)/(3 - gamma)
          and TEST 4, the scaling relation kappa ~ n^{(3-gamma)/(gamma-1)} that
          turns (a) into (b), regressed within bands of similar gamma.

Usage:  python3 crossover_analysis.py [results.csv]
"""
import csv
import sys

import numpy as np

PATH = sys.argv[1] if len(sys.argv) > 1 else "data/crossover_results.csv"


def num(x, default=np.nan):
    try:
        v = float(x)
        return v if np.isfinite(v) else default
    except (TypeError, ValueError):
        return default


def load(path):
    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        for key in list(r):
            if key not in ("network", "dropped") and not key.endswith("dropped"):
                r[key] = num(r[key])
    return rows


def summarise(label, vals, target=None):
    v = np.array([x for x in vals if np.isfinite(x)])
    if not len(v):
        print(f"  {label:<44} (no data)")
        return
    q = np.percentile(v, [25, 50, 75])
    s = (f"  {label:<44} n={len(v):<4} median={q[1]:>7.3f}  "
         f"IQR [{q[0]:>6.3f}, {q[2]:>6.3f}]")
    if target is not None:
        s += f"   target {target}"
    print(s)


def main():
    rows = load(PATH)
    ok = [r for r in rows if np.isfinite(r.get("deg_slope", np.nan))]
    print(f"{len(rows)} networks measured, {len(ok)} with a usable slope "
          f"(>= 3 lengths with cycles in both graph and null)\n")

    # ---------------------------------------------------------- TEST 1
    print("TEST 1  does a fitted model's own spectrum follow its transfer matrix?")
    print("        slope of ln(null_L / profile_L) vs L; flat (0) = the profile is right")
    summarise("rank-one, kappa = <k^2-k>/<k>  (corrected)",
              [r.get("valK_slope") for r in ok], target="0")
    summarise("rank-one, lambda = <k^2>/<k>   (chapter 7 as written)",
              [r.get("val1_slope") for r in ok], target="0")
    summarise("nested DC-SBM, kappa form",
              [r.get("valSK_slope") for r in ok], target="0")
    summarise("nested DC-SBM, lambda form",
              [r.get("valS_slope") for r in ok], target="0")

    # ---------------------------------------------------------- TEST 2
    print("\nTEST 2  the locality deficit: slope of ln(true_L / null_L) vs L")
    print("        chapter 7 predicts this equals -ln kappa, i.e. ratio 1")
    for tag, lab in (("deg", "vs degree-preserving rewiring (rank-one)"),
                     ("sbm", "vs block-preserving rewiring (nested DC-SBM)")):
        ratio, raw = [], []
        for r in ok:
            s, t = r.get(f"{tag}_slope"), r.get("minus_ln_kap1")
            raw.append(s)
            if np.isfinite(s) and np.isfinite(t) and t != 0:
                ratio.append(s / t)
        summarise(f"{lab}: raw slope", raw)
        summarise(f"{lab}: slope / (-ln kappa)", ratio, target="1")

    sig = [r for r in ok
           if np.isfinite(r.get("deg_slope")) and np.isfinite(r.get("deg_slope_se"))
           and r["deg_slope_se"] > 0
           and abs(r["deg_slope"] - r["minus_ln_kap1"]) > 2 * r["deg_slope_se"]]
    print(f"\n  networks whose slope differs from -ln kappa by more than 2 "
          f"standard errors: {len(sig)} of {len(ok)}")
    up = [r for r in ok if r.get("deg_slope", -1) > 0]
    print(f"  networks whose deficit *grows* with L (positive slope, the wrong "
          f"sign entirely): {len(up)}")

    # Where the signal actually is: not in the slope but in the level.  If the
    # deficit were n/kappa^L it would be a geometric decay through 1; instead it
    # is a large and nearly constant factor.
    print("\n        the deficit's level, ln(true_3 / null_3)")
    summarise("vs degree-preserving rewiring", [r.get("deg_r3") for r in ok])
    summarise("vs block-preserving rewiring", [r.get("sbm_r3") for r in ok])
    print("        (in ratio terms, the median network has this many times more "
          "triangles")
    med = np.nanmedian([r.get("deg_r3", np.nan) for r in ok])
    print(f"        than its degree-preserving rewiring: {np.exp(med):.0f}x)")

    # ---------------------------------------------------------- TEST 3
    print("\nTEST 3  where the crossover falls (networks with L* inside 3..8 only)")
    vis = [r for r in ok if 3 <= r.get("lstar_operational", np.nan) <= 8]
    print(f"        {len(vis)} networks are in the visible window")
    print(f"  {'network':<40} {'n':>8} {'kappa':>7} {'obs':>7} {'(a) ln n/ln k':>14} "
          f"{'gamma':>6} {'(b) headline':>13}")
    for r in sorted(vis, key=lambda r: r["lstar_operational"]):
        b = r.get("lstar_headline", np.nan)
        print(f"  {r['network']:<40} {r['n']:>8.0f} {r['kap_rank1']:>7.2f} "
              f"{r.get('deg_cross', np.nan):>7.2f} {r['lstar_operational']:>14.2f} "
              f"{r.get('gamma', np.nan):>6.2f} "
              f"{b if np.isfinite(b) else float('inf'):>13.2f}")
    inrange = [r for r in vis if 3 <= r.get("deg_cross", np.nan) <= r["lmax_reached"]]
    print(f"  crossovers that actually fall inside the measured range of L, "
          f"rather than being extrapolated: {len(inrange)} of {len(vis)}")
    ndef = sum(1 for r in vis if not np.isfinite(r.get("lstar_headline", np.nan)))
    print(f"  visible networks with no finite (b) at all, because the fitted "
          f"gamma is not below 3: {ndef} of {len(vis)}")
    for tag, lab in (("lstar_operational", "(a) ln n / ln kappa"),
                     ("lstar_headline", "(b) (gamma-1)/(3-gamma)")):
        err = [r.get("deg_cross", np.nan) - r.get(tag, np.nan) for r in vis]
        summarise(f"observed crossover minus {lab}", err, target="0")

    # ---------------------------------------------------------- TEST 4
    print("\nTEST 4  the scaling relation kappa ~ n^{(3-gamma)/(gamma-1)}")
    print("        regress ln kappa on ln n within bands of similar gamma;")
    print("        the slope should be (3-gamma)/(gamma-1) at the band centre")
    # this needs only degrees, so it runs on every scouted network rather than
    # only the ones whose cycles could be counted
    try:
        pool = load(PATH.replace("crossover_results", "crossover_candidates"))
        print(f"        (using all {len(pool)} scouted networks -- gamma and "
              f"kappa need only degrees)")
    except FileNotFoundError:
        pool = rows
    for r in pool:
        r.setdefault("kap_rank1", r.get("kap", np.nan))
        if not np.isfinite(r.get("kap_rank1", np.nan)):
            r["kap_rank1"] = r.get("kap", np.nan)
    pl = [r for r in pool if 2.0 < r.get("gamma", np.nan) < 3.0
          and r.get("gamma_ntail", 0) >= 100 and r.get("gamma_ks", 1) < 0.10]
    print(f"        {len(pl)} networks have a credible power-law tail "
          f"(2 < gamma < 3, >=100 in the tail, KS < 0.10)")
    for lo, hi in ((2.0, 2.4), (2.4, 2.7), (2.7, 3.0)):
        band = [r for r in pl if lo <= r["gamma"] < hi]
        if len(band) < 4:
            print(f"  gamma in [{lo}, {hi}): {len(band)} networks -- too few to regress")
            continue
        x = np.log([r["n"] for r in band])
        y = np.log([r["kap_rank1"] for r in band])
        A = np.vstack([x, np.ones_like(x)]).T
        slope = np.linalg.lstsq(A, y, rcond=None)[0][0]
        resid = y - A @ np.linalg.lstsq(A, y, rcond=None)[0]
        se = np.sqrt((resid ** 2).sum() / (len(x) - 2)
                     * np.linalg.inv(A.T @ A)[0, 0])
        gmid = np.median([r["gamma"] for r in band])
        pred = (3 - gmid) / (gmid - 1)
        print(f"  gamma in [{lo}, {hi}): {len(band):>3} networks, median gamma "
              f"{gmid:.2f}  measured {slope:.3f} +- {se:.3f}   predicted {pred:.3f}")

    # ---------------------------------------------------------- verdict
    print("\nVERDICT")
    v1 = np.nanmedian([r.get("valK_slope", np.nan) for r in ok])
    v2 = np.nanmedian([r.get("deg_slope", np.nan) / r.get("minus_ln_kap1", np.nan)
                       for r in ok])
    tm = abs(v1) < 0.15
    print(f"  transfer-matrix theory (corrected kappa): "
          f"{'holds' if tm else 'fails'} -- median slope {v1:+.3f} against 0")
    print(f"  locality deficit slope = -ln kappa: "
          f"{'holds' if abs(v2 - 1) < 0.25 else 'fails'} -- "
          f"median ratio {v2:.3f} against 1")


if __name__ == "__main__":
    main()
