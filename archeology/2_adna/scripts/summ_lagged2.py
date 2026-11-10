import sys, json, glob
print("| run | dyads | edges | β_nf | z/p decay | z/p strat | z/p swap (null mean) | β_hf | z/p voronoi | z/p hstrat |")
print("|---|---|---|---|---|---|---|---|---|---|")
for f in sorted(glob.glob(f"{sys.argv[1]}/lagged2_R*.json")):
    d = json.load(open(f)); N = d["nulls"]
    zp = lambda k: f'{N[k]["z"]:.1f} / {N[k]["p"]:.3f}' if k in N else "-"
    print(f'| {d["tag"]} | {d["n_dyads"]} | {d["n_edges"]} | {d["beta_nf(G_lag->sameCulture)"]:.2f} | {zp("decay")} | {zp("strat")} | '
          f'{zp("swap")} ({N["swap"]["null_mean"]:.2f}) | {d["beta_hf(C_lag->IBD)"]:.2f} | {zp("voronoi")} | {zp("hstrat")} |')
