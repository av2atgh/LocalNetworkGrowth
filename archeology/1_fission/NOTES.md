# 1_fission: group fissions as natural experiments

Status 2026-11-10: one case analysed (Ngogo chimpanzees) plus one calibration case (karate club).
The Amboseli baboon data are public but the automated download was refused (HTTP 403); someone has to fetch them by hand.
All other cases need a data request (table below). No one has been contacted.

## Bottom line (neutral reading)

- **Ngogo (n = 126 active nodes in the giant component for 2011-2014; 90 of them have a fission label).** The pre-fission network predicts
  the fission line well. The spectral cut gets 86% accuracy (ARI 0.50, perm p < 0.001). The LOO neighbour vote gets 0.93 balanced accuracy.
  Sex and age do no better than chance (LOO balanced accuracy 0.48, below the majority baseline).
  **Space predicts it perfectly**: the median GPS position over 2011-2014 gives LOO accuracy 1.00, and every year from 2011 on is at 0.92-1.00.
  The network adds nothing once space is in the model. This is the spatial confound flagged in the README, in its strongest form.
  The data cannot separate tie → co-location from co-location → tie.
- The fission line cuts across sex, as any split of a mixed-sex group into two complete social units must.
  The spectral cut separates the two groups within males (balanced accuracy 0.94 on the male-only subgraph, n = 46) and within females (0.92, n = 25).
  So sex and age homophily is close to impossible as a fission mechanism here. The homophily alternatives worth testing are
  **kinship (matriline/pedigree)** and **space**. Kinship is not public for Ngogo. Space is, and it wins (in a snapshot sense).
- **Method note.** On the full graph, the minimum-description-length DC-SBM with B = 2 does *not* recover the fission line (ARI ≈ 0).
  It recovers **males vs females**, which is the sampling design: males are focal subjects and females enter mostly through party records.
  With B free it finds 5 blocks (ARI 0.29). On the labelled subgraph it finds B = 3 (ARI 0.49).
  The spectral and modularity cuts on log weights do much better.
  Inside each sex (n = 41-47) the binary DC-SBM finds B = 1. That comes from binarising a dense weighted graph (male density 0.70), not from a scale effect.
- **Karate club (n = 34)**: the modularity cut makes 1 error and the Fiedler cut 2. MDL DC-SBM prefers B = 1, and with B = 2 forced it gets ARI ≈ 0.
  At this size the outcome depends on the method; it is not evidence for or against the 80-100 node threshold.
- **Scale:** Ngogo (126 active, ~200 total) is above the book's r_κ ~ 80-100. The karate club and Gombe (~30 adults) are below it.
  Amboseli groups (~40-100 per group) straddle it.

## Cases

| Case | Citation (verified) | Network data | Attributes | Time res. | Public? | Size | Status |
|---|---|---|---|---|---|---|---|
| Ngogo chimpanzees, split 2015-2018 | Sandel, He, Ren, ..., Mitani, *Science* 392(6794):216-220 (2026), doi:10.1126/science.adz4944 | Learned weighted networks combining grooming, proximity and party (219 nodes, monthly 1998-2022); binary yearly networks; 77-focal-male proximity network | Sex (from list membership plus GPS); age (GPS file, mature individuals only); **no rank, no kinship** | monthly | Yes. Zenodo 10.5281/zenodo.19210783 (CC-BY-4.0, code MIT), 18626723 (CC-BY-4.0), 18603419 (GPS subsample); Dryad 10.5061/dryad.sf7m0cgkg (CC0; API download needs a token, browser download works) | 219 individuals; 142 labelled post-fission (88/54) | **in hand, analysed** |
| Amboseli baboons, 7 fissions | Lerch, Abbott, Archie & Alberts, *Proc R Soc B* 288:20212060 (2021), doi:10.1098/rspb.2021.2060; Van Horn et al., *Behav Ecol Sociobiol* 61:1823 (2007) (4 fissions, 81 females, kin vs partners) | Monthly adult grooming for the 2 yr before and after each fission; monthly ranks | Ranks; birthdates (repository v6, 2025); pedigree only through Babase | monthly | Yes. Duke RDR doi:10.7924/r4bz67v25 (v6, 2025-04-10). Automated fetch got 403 and was not circumvented | ~7 fissions, groups ~40-100 | **download by hand** |
| Gombe Kasakela/Kahama, 1970-72 | Feldblum, Manfredi, Gilby & Pusey, *AJPA* 166:730-744 (2018), doi:10.1002/ajpa.23462 | Association, grooming and ranging digitised from Goodall's checksheets | Rank, age, sex, maternity (partial) | monthly/annual | No (JGI Research Center, Duke) | ~30 adults; below r_κ | request |
| Cayo Santiago rhesus | Widdig et al., *Mol Ecol* 15:3825 (2006), doi:10.1111/j.1365-294X.2006.03039.x (groups R, S splits, maternal + paternal kinship); Larson, Ruiz-Lambides, Platt & Brent, *Anim Behav* 136:185 (2018) (mass eviction, grooming networks) | Grooming/proximity (Brent lab, 2010s); census | Genetic pedigree, matriline, rank, age | focal, daily | No (CPRC / Brent, Platt, Higham labs) | groups 100-300 | request; best pedigree + network case |
| Affenberg Landskron Japanese macaques, 2020 fission | Univ. Vienna thesis plus PMC10078252 methods paper (not fully verified) | Grooming/proximity around the split | Matrilines (likely), rank | daily | Not found | ~160 | request (lower priority) |
| Yanomamö village fissions | Chagnon 1975, 1979; genealogies archived at Univ. Michigan (deposit status not confirmed) | Genealogy (kinship network), marriage alliances, village rosters | Lineage, age, sex | census snapshots | Not found public | villages 40-300 | request; IRB-sensitive |
| Ache village splits (2) | Walker & Hill, *Hum Nat* 25:465-475 (2014) | Genealogy, residence | age, sex, lineage | snapshots | No (Hill, Walker) | villages ~100-200 | request; lineal-effect index ≈ 0 reported |
| Hutterite colony fissions | Olsen, *Am Anthropol* (1987) (title seen in a reference list, not checked); Mange (1964) fission model | Genealogy only (no contact network) | age, sex, lineage | per-fission | No (pedigree e.g. Ober lab Hutterite study) | colonies ~100-150 at fission | request; kinship-only test |
| Zachary karate club (calibration) | Zachary, *J Anthropol Res* 33:452 (1977) | Interaction graph | none | snapshot | Yes (networkx) | 34 | analysed |

## Ngogo: what was done

Data: `data/ngogo/` comes from the Zenodo deposits (untrusted, read-only).
- `learned/L/` holds the learned graphs (`full_array_{year}_{month}.npy`, 219×219), the names, and the post-fission labels (`src/combined_membership_labels.npy`).
- `net/` holds the binary yearly graphs and the R/IPython code.
- `space/` holds `gps_qualified.csv` (163 individuals, Sex, Age_at_sampling, 2011-2023).

The Dryad files duplicate these. Dryad also has `population_snapshots.csv` and `patrol-data-quarterly.csv`, which were not fetched because the API needs a token.

**Outcome caveat.** The labels are themselves network-defined. They are the majority connected-component label over 2018-2022 of the post-split network (authors' `create_combined_labels.py`).
So "network predicts the split" partly measures the persistence of network structure.
Individuals who died or disappeared before 2018 have no label; 36 such nodes are in the 2011-2014 giant component.

**Attributes.**
- Coded nodes (the first 97) are males.
- Full-name nodes get sex from the GPS file when present. The "non_focal_female" list contains some males, so it is not trusted for sex.
- Age = window midpoint − birth year from the GPS file.
- Male two-letter codes were mapped to GPS names heuristically (`scripts/map_codes.py`): letters match, plus Jaccard of active years, then Hungarian assignment. Only pairs with year-Jaccard ≥ 0.5 are used (≈ 55 males); the map is in `results/ngogo_codemap.csv`.
- Rank and kinship are not available.

**Analyses** (`scripts/ngogo_fission.py`). Window 2011-2014 (pre-polarisation per the paper), log(1 + weight), giant component.
- Unsupervised cuts (all evaluated nodes, n = 90; permutation null over 1000 label shuffles):

| Method | acc (null mean) | ARI | p |
|---|---|---|---|
| Fiedler cut (normalised Laplacian) | 0.856 (0.56) | 0.496 | <0.001 |
| Leading modularity eigenvector | 0.833 (0.54) | 0.438 | <0.001 |
| Louvain (Q = 0.115, 3 comms; one is 27/29 of group 1) | – | 0.333 | <0.001 |
| DC-SBM B=2 binary / weighted | 0.511 | -0.025 / -0.021 | 1.0 (it splits males vs females) |
| DC-SBM B free (B=5) | – | 0.292 | <0.001 |

- Supervised leave-one-out logistic regression on the common subset with sex, age and GPS all known (n = 67, 21 in group 1, majority baseline 0.687). The AUC null comes from 200 label permutations:

| Predictors | LOO acc | bal acc | AUC | perm p |
|---|---|---|---|---|
| network Fiedler coordinate | 0.731 | 0.584 | 0.908 | 0.005 |
| network neighbour vote (label of focal node hidden) | 0.955 | 0.929 | 0.972 | 0.005 |
| sex + age (+age², sex×age) | 0.657 | 0.478 | 0.519 | 0.15 |
| space (median GPS x, y) | **1.000** | **1.000** | 1.000 | 0.005 |
| space + network | 1.000 | 1.000 | 1.000 | |
| attr + network | 0.746 | 0.595 | 0.666 | |
| attr + space + network | 0.985 | 0.976 | 1.000 | |

  The LOO logistic fit on a single feature has the usual anti-correlation artefact, which is why the null-mean AUC for the Fiedler fit is 0.11. The p-values are computed against that same LOO null.

- Descriptive: group 1 (n = 29) vs group 0 (n = 61). Male fraction 0.41 vs 0.59; mean age 20.4 vs 25.2 years; group 1 sits about 1.9 km further west.
  The ex-post attribute differences come from the demography of the two halves and do not predict individuals.
- Within-sex spectral cut on sex-specific subgraphs (`ngogo_spectral_bysex.py`): males n = 46, balanced accuracy 0.94; females n = 25, balanced accuracy 0.92; p < 0.001 for both.
- Within-sex DC-SBM (`ngogo_sbm_bysex.py`): B = 1 for males, and B = 1 for females when B is free; nested SBM gives B = 1 at all levels. Binarisation of dense weighted graphs explains this (see above).
- Year by year (`ngogo_temporal.py`, `results/ngogo_temporal.csv`), how well each year predicts the 2018-2022 membership:
  - Network neighbour vote: balanced accuracy 0.80-0.92 from 2007-2008 onward; 2003 is 0.80, and 1998-2002 (n ≤ 16) is at chance.
  - Single-year Fiedler cut: noisy from 2006 to 2013 (0.52-0.88), because some years it picks another axis. It is ≥ 0.96 from 2014.
  - Space: 1.00, 1.00, 0.92, 0.99, 0.99, 0.99, 1.00 for 2011-2017.
  - Sex + age: 0.42-0.54 in every year.
  - So the line was spatially *and* socially present at least 4-7 years before the 2015 polarisation. The authors' `cluster_membership_by_year.csv` shows West/Central clusters already in 1998.
  - GPS data begin in 2011, so we cannot tell whether spatial separation preceded social separation.

Reading:
- Among the variables in hand, the fission line follows the pre-fission network and space, not sex or age.
- That does not decide network-driven vs homophily-driven. Ngogo's two halves are long-standing spatial neighbourhoods. The literature on male subgroups (Mitani & Amsler 2003) and female neighbourhoods (Wakefield 2008) is cited from memory and not checked here.
- Kinship, the one attribute-like variable that could carry homophily, is missing.

## Data-request table (nothing sent)

| Case | Holder | Ask for |
|---|---|---|
| Ngogo | Ngogo Chimpanzee Project: A. Sandel (UT Austin), J. Mitani (Michigan), K. Langergraber (ASU), D. Watts (Yale); network-analysis contact J. Ren (UCLA, listed in Zenodo README) | (1) code↔name key for 97 males; (2) birth dates for all 219; (3) maternity/paternity (genetic pedigree) and matriline IDs; (4) yearly male dominance ranks; (5) raw `gps-mature-chimps.csv` (stated "available upon request") for 2008-2010 coverage. Goal: kinship-as-attribute vs kinship-as-network test, and spatial vs social temporal order |
| Amboseli | Alberts, Archie, Tung (Babase); Lerch (Duke) | Manual download of the Duke RDR v6 data; from Babase: maternal and paternal pedigree relatedness, ages, post-fission group IDs for the 7 fissions, home-range or GPS centroid per group-month (spatial null) |
| Gombe | Pusey, Gilby, Feldblum; JGI Research Center (Duke) | 1967-1975 association, grooming and ranging matrices, maternity, rank, Kasakela/Kahama membership |
| Cayo Santiago | Caribbean Primate Research Center (Ruiz-Lambides); Brent (Exeter), Platt (Penn), Higham (NYU); Widdig (Leipzig) | Pre-fission grooming/proximity for any fissioning group, genetic pedigree, ranks, post-fission membership (incl. the Larson 2018 eviction group) |
| Affenberg Landskron | Univ. Vienna group behind the 2020 fission study | Grooming/proximity 2019-2020, matrilines, post-fission membership |
| Yanomamö | Chagnon archive (Univ. Michigan; custodians to identify); co-authors of Chagnon et al. PNAS 2017 | Anonymised genealogy plus pre/post village rosters for documented fissions; marriage-alliance links. Ethics restrictions likely |
| Ache | K. Hill, R. Walker | Genealogy plus residence rosters for the 2 splits in Walker & Hill 2014 |
| Hutterites | Hutterite pedigree holders (e.g. Ober lab, U Chicago); Olsen 1987 | Colony of residence before/after each branching, with pedigree. Kinship-only test (no contact network) |

## Next steps

1. Download the Amboseli Duke RDR v6 by hand into `data/amboseli/`. Then run the same pipeline per fission on the pre-fission grooming networks:
   spectral, modularity and DC-SBM cuts vs rank, age and (if present) maternal-kin IDs. Seven fissions give a small meta-analysis, and group sizes straddle r_κ.
2. Ngogo kinship test once a pedigree is available. Score kinship two ways: as an attribute (matriline ID, a classifier) and as a network (relatedness matrix, then the same cuts as for the social network).
3. Spatial null for Ngogo. Social-network prediction conditional on spatial overlap: residualise edge weights on pairwise home-range overlap (Bhattacharyya affinity, computable from the `gps_qualified.csv` subsample), then redo the cuts.
4. Reruns on the shared Mac: one process at a time, `nice -n 15`, `OMP_NUM_THREADS=1` and the other BLAS thread variables set to 1, graph-tool pinned to one thread (`gt.openmp_set_num_threads(1)` is already in the scripts). Wait while the 1-minute load is above 10.

## Queued runs (compute freeze; nothing running)

All runs above finished before the freeze. These are reruns or extensions only. Run them one at a time from `archeology/1_fission/`.
Prefix every command with
`nice -n 15 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 ~/anaconda3/envs/gt/bin/python -I`.

| # | Command (after prefix) | Purpose | Expected runtime (1 thread) |
|---|---|---|---|
| 1 | `scripts/ngogo_fission.py data/ngogo results/ngogo_codemap.csv results 2011 2014` | reproduce main table | ~1 min (took 46 s) |
| 2 | `scripts/ngogo_fission.py data/ngogo results/ngogo_codemap.csv results 2012 2014` | window sensitivity | ~1 min |
| 3 | `scripts/ngogo_temporal.py data/ngogo results/ngogo_codemap.csv results/ngogo_temporal.csv` | year-by-year table | ~1-2 min |
| 4 | `scripts/ngogo_sbm_bysex.py data/ngogo results/ngogo_nodes_2011_2014.csv` | DC-SBM within sex | ~1-2 min |
| 5 | (to write) Amboseli pipeline, after the manual download | 7 fissions | ~5 min |
| 6 | (to write) Ngogo spatial-overlap residualised cuts (next step 3) | spatial null | ~5 min (pairwise KDE overlap on ~160 individuals) |

## Files

- `scripts/`: `map_codes.py`, `ngogo_fission.py`, `ngogo_temporal.py`, `ngogo_sbm_bysex.py`, `ngogo_spectral_bysex.py`, `karate_check.py`, `sklearn_free.py` (ARI). Run with `~/anaconda3/envs/gt/bin/python -I`.
- `results/`: `ngogo_results_2011_2014.json`, `ngogo_nodes_2011_2014.csv`, `ngogo_temporal.csv`, `ngogo_codemap.csv`, `ngogo_sbm_bysex.txt`, `ngogo_spectral_bysex.txt`, `karate_check.txt`.
- `data/ngogo/` (369 MB, Zenodo).

## Spatial confound resolved (2026-11-10, `scripts/ngogo_space_resid.py`)
Individuals with ≥30 GPS fixes in 2011–2014 (n = 95 full layer, 83 grooming layer; 67 labelled).
Home-range overlap = Bhattacharyya affinity of Gaussian KDEs (h = 300 m, 100 m grid); also centroid distance.
Outputs: `results/ngogo_space_resid_2011_2014.json` (full learned layer) and `results/ngogo_space_resid_grooming_2011_2014.json` (`LAYER=grooming`).
- Space explains 40% (full) or 19% (grooming) of the variance in log tie weight.
- **Matched test.** Pairs are matched on overlap decile × distance decile × sex pair (MM/MF/FF). Within these cells, are ties to future groupmates stronger than ties across the future line? Node bootstrap, 500–1000 resamples.
  - Full layer: −0.30 log-weight units (95% CI −0.70 to 0.01).
  - Grooming: −0.04 (CI −0.14 to 0.03). Raw same-minus-different is +0.10 for grooming.
  - Without sex-pair matching: −0.87 (full) and −0.07 (grooming). This comes from male composition.
- **Verdict.** At equal spatial overlap the network gives no extra weight to future groupmates (grooming CI includes 0; full layer leans negative). In 2011–14 the network predicts the fission because it tracks space. The fission is therefore spatial/residential, not trait-based (sex, age at chance) and not network-beyond-space.
- The residual-vote statistic (first version, kept in the json) is ill-posed: space separates the groups, so regressing w on space removes the group contrast. Use the matched test.
- Open: whether social separation preceded spatial separation (the network signal goes back to 2007–08; GPS starts in 2011). This needs the raw 2008–10 GPS, available on request.
