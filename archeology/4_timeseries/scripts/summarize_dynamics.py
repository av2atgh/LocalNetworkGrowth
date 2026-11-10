"""Summarise dynamics_thr*.pkl: observed vs geo (no-memory) and stationary (Chaco) nulls.
usage: gt-python -I summarize_dynamics.py <pkl>"""
import sys, pickle, numpy as np
o = pickle.load(open(sys.argv[1], 'rb'))
keys = ['logOR_persist', 'beta_fwd', 'beta_bwd', 'gamma_sim']
print(f"{'pair':18s} {'n':>4s} " + " ".join(f"{k:>24s}" for k in keys) + "   fwd-bwd [stat-null p]")
agg = {k: [] for k in ['fb_obs', 'fb_stat', 'per_geo_z']}
for r in o:
    ob = r['obs']; line = f"{r['ds']} {r['p0']}-{r['p1']:<9} {ob['n']:4d} "
    for k in keys:
        g = np.array([x[k] for x in r.get('geo', []) if x and np.isfinite(x.get(k, np.nan))])
        p = ((g >= ob[k]).sum() + 1) / (len(g) + 1) if len(g) else np.nan
        line += f"{ob[k]:8.2f} (geo {g.mean() if len(g) else np.nan:6.2f} p{p:5.2f})"
    fb = ob['beta_fwd'] - ob['beta_bwd']
    st = np.array([x['beta_fwd'] - x['beta_bwd'] for x in r.get('stat', []) if x])
    ps = ((st >= fb).sum() + 1) / (len(st) + 1) if len(st) else np.nan
    line += f"   {fb:6.3f} [{st.mean() if len(st) else np.nan:6.3f} p{ps:5.2f}]"
    print(line)
    if ob['n'] >= 30:
        agg['fb_obs'].append(fb)
        if len(st): agg['fb_stat'].append(ps)
fb = np.array(agg['fb_obs'])
print(f"\npairs with n>=30: {len(fb)}; fwd-bwd > 0 in {np.sum(fb > 0)}/{len(fb)}; mean {fb.mean():.3f}")
if agg['fb_stat']: print("stationary-null p values (Chaco, n>=30):", np.round(agg['fb_stat'], 2))
