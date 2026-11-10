# 3_foragers — attributes vs network mechanisms in small-scale-society networks

Status 2026-11-10: data search done; 3 usable public network datasets in hand;
two analyses run (Arang Dak, Agta); Hadza and a corrected Arang Dak rerun are **queued**
(compute freeze). No fully public forager network with time stamps, so no temporal-order test yet.

## 1. Datasets

| Dataset | Citation (verified) | Content | n | Public / licence | Status here |
|---|---|---|---|---|---|
| **Arang Dak support network** (Mayangna/Miskito *horticulturalists*, Nicaragua) | Koster 2018 R Soc Open Sci 5:172159, doi:10.1098/rsos.172159; data figshare 10.6084/m9.figshare.6056963 | Directed "who provides tangible support to you monthly" (recipient report `y`, donor report `y.donor.oriented`); sex, age, BMI, household (32), household wealth, consanguineal r, affinal r, godparent tie, inter-household distance (m), avg inter-household r. **Within-household dyads not in file** (183 pairs). April 2013 snapshot | 108 adults (all ≥18 in a 279-person community), 5594 between-hh dyads | CC BY 4.0 | `data/mayangna_koster2018/` — analysed |
| **Agta sharing-game gift network** (Philippines) | Smith et al. 2018 Evol Hum Behav, "A friend in need is a friend indeed"; data figshare 10.6084/m9.figshare.6960998 | Directed ego→alter gift (sharing game), kinship category (Spouse, PK, DK, SPK/PKS, SDK/OA, none), age, sex, household proximity (1–4), "coop" (game giving), need, camp | 272 individuals, 18 camps of 7–39, 1312 directed dyads (complete only in small camps) | CC BY 4.0 | `data/agta_smith2019_friendinneed/` — analysed |
| **Hadza GPS proximity + friendship** (Tanzania) | Fedurek et al. 2024 Hunter Gatherer Research, "The relationship between perceived friendship and proximity networks among Tanzanian Hadza"; figshare 10.6084/m9.figshare.24989361 | Weighted GPS proximity matrix and friendship nominations, 3 camps. **No attributes** in these files | 33, 38, 10 | CC BY 4.0 | `data/hadza_fedurek2024/` — analysis queued |
| Agta/BaYaka children close-proximity | Chaudhary et al. 2024 Evol Hum Sci, doi:10.1017/ehs.2024.1; OSF osf.io/n4b9e | Only child→caregiver proportions by kin category; not a full network | 29 + 17 focal children | OSF (no licence stated) | `data/agta_bayaka_children_chaudhary2024/` — not usable for this test |
| Hadza camp networks + kinship (Fedurek) | Fedurek et al. 2021 Behav Ecol, doi:10.1093/beheco/arab110; Dryad 10.5061/dryad.nk98sf7v6 | Same 3 camps as above **plus kinship and hunting-reputation matrices**, 2 m / 5 m and in/out-of-camp proximity | ~33/38/10 men | CC0 | **public, not downloaded**: Dryad API now needs a bearer token; browser download by the user works |
| Agta camp stability / food sharing | Smith et al. 2016 R Soc Open Sci, doi:10.1098/rsos.160131; Dryad 10.5061/dryad.pf67c (`Smith_Data.xlsx`, 2.2 MB) | Agta sharing data with camp stability | ? | CC0 | public, not downloaded (Dryad token) |
| BaYaka gift-game popularity | Chaudhary et al. 2015 R Soc Open Sci, doi:10.1098/rsos.150054; Dryad 10.5061/dryad.0dk3n (13 kB) | gift nominations/popularity, polygyny | ? | CC0 | public, not downloaded (Dryad token) |
| BaYaka women's subsistence networks | Jang et al. 2024 Sci Adv, doi:10.1126/sciadv.adj2543; Dryad 10.5061/dryad.70rxwdc4t (`PublicVersion.RData`) | 230 days focal follows of women's foraging groups (STRAND models) | ~community | CC0 | public, not downloaded (Dryad token) |
| Tsimane mothers' support (scan samples) | Seabright, Alami et al. 2022 Phil Trans B, doi:10.1098/rstb.2021.0442; Dryad 10.25349/D9KK7B | scan-sample co-presence of mothers/children, patrilocal vs not | ? | CC0 | public, not downloaded (Dryad token) |

Search notes: Dryad, figshare, OSF APIs and web search (2026-11-10). No public data for Migliano et al. 2017
NHB / 2020 Sci Adv mote networks, Dyble et al. 2015/2016, Apicella et al. 2012, Smith et al. 2018 (Hadza
multi-year), Hill et al. 2011. Hadza synthetic (synthpop) datasets on figshare (Fedurek 2022–2024) are
synthetic node-level tables — not usable. STRAND bundled data are Colombian (not forager).

**User action that unlocks four more datasets without contacting anyone:** download the four Dryad
records above in a browser (or provide a Dryad API token) into `data/<name>/`. The Fedurek Dryad kinship
matrices would let the Hadza camps get an attribute (kinship) model.

## 2. Methods (scripts in `scripts/`, run with `python -I`)

- `common.py`: L2 logistic regression (standardised), AUC, log-loss, NMI/ARI, permutation test,
  DC-SBM fitting (graph-tool, best of k runs), fixed-partition description length, DC-SBM
  plug-in edge scores λ_ij = k_i k_j e_rs/(e_r e_s), neighbour features (CN, Adamic-Adar,
  Jaccard, log degrees with own edge left out).
- Held-out dyad prediction, random K-fold over dyads; network features recomputed from training
  ties only; SBM refitted in each fold.
- Feature blocks: **demo** (sex, age, wealth), **kin** (r, affinal r, godparent / kin category),
  **space** (inter-household distance / household proximity), **degree** (popularity / activity),
  **closure** (CN, AA, Jaccard; reciprocity and transitive/cyclic 2-paths for directed),
  **SBM** (inferred DC-SBM block score), and attribute-block DC-SBMs (partition fixed to
  household, sex×age, or spatial clusters).
- Communities: flat and nested DC-SBM on the full graph vs household, sex, age tercile,
  sex×age, spatial clusters (average-linkage on log inter-household distance, k = B),
  and blocks of the kinship graph (r ≥ 0.125). NMI with label-permutation null (1000), ARI;
  description length of the free SBM vs SBM with partition fixed to each attribute.

## 3. Results

### 3a. Arang Dak (n = 108; just above the book's ~80–100 detectability scale)
Undirected tie = any of the four reports; density 0.42 between households.
(v1 run; training degree features included the dyad's own edge, a mild leak in favour of
degree. Fixed in `common.py`; rerun queued. The SBM/community part is unaffected.)

Held-out AUC (5-fold × 2), tie = union, within-household pairs NOT added as edges / added:

| model | AUC (A) | AUC (B) |
|---|---|---|
| demo | 0.647 | 0.647 |
| kin | 0.707 | 0.707 |
| space (log distance) | 0.538 | 0.538 |
| attr = demo+kin+space | 0.747 | 0.747 |
| degree only | 0.781 | 0.780 |
| net: degree+closure | 0.793 | 0.823 |
| net: degree+SBM | 0.783 | 0.816 |
| net: degree+closure+SBM | 0.794 | 0.829 |
| attr+degree | 0.823 | 0.823 |
| all | 0.827 | 0.843 |

Recipient-only ties give the same ordering (attr 0.727, degree 0.775, all 0.826/0.842).
- Popularity/activity (degree) is the largest single block. Beyond degree, attributes add more
  (+0.04 AUC, kinship carries most of it) than closure/SBM do in variant A (+0.01). In variant B
  (co-residents counted as ties) closure adds as much as attributes, but there closure partly
  re-encodes household membership (j co-resides with i's supporters), so it is not clean
  evidence for a network mechanism.
- Spatial distance alone barely predicts (AUC 0.54). Space is not the driver here.
- **Community structure is weak.** Free DC-SBM: B = 3 (A) / 4 (B); nested hierarchy collapses
  to 1 block at level 1. DL free SBM 4241 vs one block 4255 nats (A): Δ = 14 nats. Fixed
  attribute partitions compress **worse than a single block** (household 4750, sex×age 4481,
  spatial 4306, kin blocks 4474).
- The weak blocks align with **kinship-graph blocks** (NMI 0.27 vs null 0.06, ARI 0.19,
  p < 0.001) and households (NMI 0.27 vs 0.16 null, but ARI 0.03: blocks are unions of
  related households), not with sex (NMI ≈ 0) and not with space (NMI = null). Variant B: age
  enters weakly (NMI 0.10 vs 0.02).
- Reading: in this 108-person community, ties are mostly degree plus dyadic kinship; meso-scale
  communities are barely detectable, and what exists follows the genealogy. Kinship is itself a
  network (genealogy), so "kin" here is not cleanly "attribute homophily". Sex/age homophily is
  weak. Consistent with the n ≈ 100 threshold (structure marginal), but one snapshot cannot
  separate the hypotheses.

### 3b. Agta sharing game (18 camps, 7–39 people)
Held-out AUC (5-fold × 5 over directed dyads; leave-own-tie-out degree), `results/agta_log_v2.txt`:

| model | AUC |
|---|---|
| demo (sex, age) | 0.56 |
| kin category | 0.57 |
| household proximity | 0.54 |
| need | 0.48 |
| exogenous attributes (demo+kin+prox+need) | 0.59 |
| degree only | 0.77 |
| net: degree+reciprocity+closure | 0.78 |
| exog attr + degree | 0.79 |
| exog attr + degree + reciprocity + closure | 0.79 |
| with the game's "coop" variables | 0.86–0.89 |

- The "coop" variables are **endogenous**: `coop.ego` correlates ρ = 0.83 with the number of gifts
  the ego makes in the same game, so coop and coop-similarity (Apicella-style cooperation
  homophily) cannot be used as exogenous predictors here. Excluded from the comparison.
- Exogenous attributes predict almost nothing (0.59 within camps; leave-one-camp-out 0.50).
  Giver/receiver heterogeneity (degree) dominates; reciprocity and closure add ≈ 0.01.
- Camps are 7–39 people, far below the community-detection scale; no SBM run. Neither homophily
  nor closure organises these within-camp gift choices.

### 3c. Hadza (Fedurek): queued (n = 33, 38, 10; network-only: friendship ~ proximity vs friendship closure; SBM of proximity vs friendship).

### Temporal / camp-composition tests
Not possible with public data. The right dataset is Smith et al. 2018 Curr Biol (Hadza, 2010–2016,
56 camps, 383 people, PGG cooperation, kinship): they report that similarity in cooperation does
**not** predict future co-residence and that cooperation converges to current campmates
(tie → similarity). That is their analysis, not ours; data on request.

## 4. Queued runs (single process; run sequentially; check `uptime` first, 1-min load < 10)

```sh
cd /Users/avazquez/av2atg/LocalNetworkGrowth/archeology/3_foragers
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
# Q1  Arang Dak rerun with leave-own-edge-out degrees (~10-15 min single-threaded; v1 took ~4 min on ~4.5 cores)
nice -n 15 ~/anaconda3/envs/gt/bin/python -I scripts/koster_analysis.py data/mayangna_koster2018/rsos172159_si_002.csv results > results/koster_log_v2.txt 2>&1
# Q2  Hadza proximity vs friendship (~1-3 min)
nice -n 15 ~/anaconda3/envs/gt/bin/python -I scripts/hadza_analysis.py data/hadza_fedurek2024 results > results/hadza_log.txt 2>&1
```
`common.py` sets the BLAS/OMP env vars and `gt.openmp_set_num_threads(1)`.

## 5. Data requests (not contacted)

| Data | Holder(s) | Ask for | Why |
|---|---|---|---|
| Agta & BaYaka mote proximity networks (Migliano et al. 2017 NHB; Migliano et al. 2020 Sci Adv multicamp; Dyble et al. 2015 Science, 2016 Curr Biol) | A. B. Migliano (Zurich), L. Vinicius, M. Dyble, A. E. Page, D. Smith, N. Chaudhary; UCL Hunter-Gatherer Resilience Project (R. Mace) | Time-stamped dyadic contacts (2-min resolution), camp membership through time (incl. inter-camp visits), genealogy / pairwise r, sex, age, household | Only forager networks with ~200 (Agta) / 132 (BaYaka) nodes and time stamps: above the n ≈ 80–100 scale; temporal-order and camp-turnover tests |
| Hadza campmate + honey-gift networks (Apicella et al. 2012 Nature) and multi-year residence (Smith et al. 2018 Curr Biol) | C. Apicella (UPenn), K. M. Smith; F. Marlowe | Campmate/gift nominations (205 adults), camp membership 2010/13/14/16, PGG contributions, kinship (2010, 2016), sex, age, GPS of camps | Whole-population Hadza network (≈ 200+): SBM vs attributes at scale; residential change gives the temporal test |
| Hill et al. 2011 Science co-residence, 32 foraging societies | K. Hill (ASU), R. S. Walker (Missouri) | Band rosters with kin relation to ego | Composition null: what fraction of co-residents are kin vs expected from network models |
| Tsimane networks (friendship, politics, food sharing) | C. von Rueden (Richmond), M. Gurven (UCSB), THLHP | Village-level networks + genealogy + sex/age + household location | Larger villages (100s): scale test, spatial null |
| Mayangna longitudinal (Koster) | J. Koster (MPI-EVA / Cincinnati) | Later waves of Arang Dak support network; within-household pairs | Temporal order at n = 108 |
| ENDOW multi-site networks | C. T. Ross, J. Koster, D. Redhead et al. (MPI-EVA) | Forager/horticulturalist community networks with attributes | Many communities of varied n: test the scale prediction across sizes |
| Ju/'hoansi hxaro (Wiessner) | P. Wiessner | Hxaro partner lists, genealogy, camp | Long-range exchange; spatial null |

## 6. Next steps
1. Run Q1, Q2.
2. User downloads the four Dryad records; add Fedurek kinship to Hadza (attribute vs network at n ≈ 35),
   Chaudhary 2015 BaYaka gift network, Smith 2016 Agta.
3. Arang Dak: model directed ties with a social-relations / latent-space model (STRAND-like:
   sender/receiver effects + attributes + latent blocks) and check whether residual blocks remain
   after kinship; within-household pairs need a censoring model rather than 0/1 imputation.
4. Treat kinship as a network: test whether support ties follow genealogical distance (path length
   in the pedigree) beyond r, i.e. locality on the genealogy.
5. Pursue the data requests (Agta/BaYaka motes, Hadza multi-year) — the only routes to the
   temporal-order and n > 100 forager tests.

## Kinship as a network (2026-11-10, `scripts/koster_kinnet.py` → `results/koster_kinnet.json`)
Kin-network features:
- common kin (r ≥ 0.125);
- Σ_k r_ik r_kj, consanguineal and affinal;
- kin-graph path length, with co-residents linked.

Support closure is split by intermediary: kin or co-resident of i or j, versus unrelated to both and living in neither household.

Held-out AUC (5-fold × 5, same folds):
| model | AUC |
|---|---|
| attributes + degree | 0.824 |
| attributes + kin network + degree | 0.836 (+0.012, better in 25/25 folds) |
| + closure via kin | +0.0006 |
| + closure via non-kin | +0.0007 |
| + both | +0.003 (21/25 folds) |
| attributes + degree + both closures, no kin network | 0.830 |

Reading:
- Support follows the genealogy *as a network*: common kin and genealogical distance predict support beyond pairwise r.
- Closure in the support network beyond the genealogy is negligible.
- The "network" that shapes Arang Dak is the kinship network. Kinship is grown by descent and marriage, local rules, but it is also the classic homophily variable. So this supports "communities follow the genealogical network" rather than "support ties self-organise by closure".
