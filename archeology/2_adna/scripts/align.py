"""Alignment of IBD communities (nested DC-SBM) with culture labels, locality, time, geography.
usage: gt-python -I align.py <results_dir> <region> <min_sum12_cM> <n_null>"""
import sys, json, numpy as np, pandas as pd, graph_tool.all as gt
gt.openmp_set_num_threads(1)
sys.path.insert(0, sys.argv[0].rsplit("/",1)[0]); from common import *
RES, REG, TH, NN = sys.argv[1], sys.argv[2], float(sys.argv[3]), int(sys.argv[4])
rng = np.random.default_rng(1)
s = pd.read_csv(f"{RES}/{REG}_individuals.tsv", sep="\t").reset_index(drop=True)
e = pd.read_csv(f"{RES}/{REG}_ibd12_pairs.tsv", sep="\t"); e = e[e.sum12 >= TH]
idx = dict(zip(s.iid, s.index))
g = gt.Graph(directed=False); g.add_vertex(len(s))
g.add_edge_list(np.c_[e.iid1.map(idx).values, e.iid2.map(idx).values])
best = None
for seed in range(8):
    gt.seed_rng(seed); np.random.seed(seed)
    st = gt.minimize_nested_blockmodel_dl(g, state_args=dict(deg_corr=True))
    for _ in range(3): st.multiflip_mcmc_sweep(niter=10, beta=np.inf)
    if best is None or st.entropy() < best.entropy(): best = st
levels = best.get_levels()
deg = g.get_total_degrees(g.get_vertices())
keep = (deg > 0)
s["deg"] = deg
for l in range(min(3, len(levels))):
    s[f"sbm{l}"] = best.project_level(l).get_blocks().a
s["cult_bin"] = s.culture + "@" + s.tbin.astype(str)
named = keep & s.is_culture.values
S = s[named].copy()
Kt = S.groupby("tbin").culture.nunique().to_dict()
out = {"region": REG, "min_sum12_cM": TH, "n_nodes": int(len(s)), "n_edges": int(g.num_edges()),
       "n_nonisolated": int(keep.sum()), "n_named_nonisolated": int(named.sum()),
       "DL": float(best.entropy()), "blocks_per_level": [int(len(set(best.project_level(l).get_blocks().a[keep]))) for l in range(min(4,len(levels)))]}
attrs = {"culture": S.culture, "culture@bin": S.cult_bin, "locality": S["loc"], "tbin": S.tbin, "period": S.period}
for l in range(min(3, len(levels))):
    b = S[f"sbm{l}"].values
    row = {k: {"nmi": nmi(b, v), "ari": ari(b, v), "K": int(pd.Series(v).nunique())} for k, v in attrs.items()}
    # spatial null: random contiguous partitions within each bin with K_t matched to cultures per bin
    nul = np.array([[nmi(b, lab), ari(b, lab)] for lab in (voronoi_labels(S, Kt, rng) for _ in range(NN))])
    row["voronoi_null"] = {"nmi_mean": nul[:,0].mean(), "nmi_q95": np.quantile(nul[:,0], .95), "ari_mean": nul[:,1].mean(),
                           "ari_q95": np.quantile(nul[:,1], .95),
                           "p_nmi(culture@bin)": float(((nul[:,0] >= row["culture@bin"]["nmi"]).sum()+1)/(NN+1)),
                           "p_ari(culture@bin)": float(((nul[:,1] >= row["culture@bin"]["ari"]).sum()+1)/(NN+1))}
    # label-shuffle null within bin (destroys space too): culture permuted among individuals in same bin
    sh = []
    for _ in range(NN):
        perm = S.groupby("tbin").culture.transform(lambda x: rng.permutation(x.values))
        sh.append(nmi(b, perm + "@" + S.tbin.astype(str)))
    row["withinbin_shuffle_null_nmi_mean"] = float(np.mean(sh))
    out[f"level{l}"] = row
s.to_csv(f"{RES}/{REG}_sbm_th{int(TH)}.tsv", sep="\t", index=False)
json.dump(out, open(f"{RES}/{REG}_align_th{int(TH)}.json", "w"), indent=1, default=float)
print(json.dumps(out, indent=1, default=lambda x: round(float(x), 3)))
