# 2_adna — IBD kinship networks vs archaeological culture labels

Started 2026-11-10. Question (README): do genetic-tie communities align with archaeological culture
beyond space and time, and which comes first, the ties or the shared culture?
Status: data in hand; alignment analyses done for Central Europe. Temporal (cross-lagged) script
written but **not yet run** because of the compute freeze (see *Queued runs*).

## 1. Sources (verified 2026-11-10)

| Source | Citation / location | What it holds | Status |
|---|---|---|---|
| AADR v66.p1 | Mallick et al., *Sci. Data* 11:182 (2024); Harvard Dataverse doi:10.7910/DVN/FFIDCW (v14.0, released 2026-06-08) | `.anno` metadata for 23,089 rows: dates (BP mean/sd), Group ID, locality, lat/lon, publication, **Family relations** column (format `1.0d:id1:id2`, `2.5d:…`) | **Downloaded** `data/aadr/v66.p1_1240K.aadr.PUB.anno` (13 MB, md5 matches the Dataverse md5sum file). Genotypes (1.9–12 GB) skipped. |
| ancIBD IBD segments | Ringbauer H, Huang Y, Akbari A, Mallick S, Olalde I, Patterson N, Reich D, "Accurate detection of identity-by-descent segments in human ancient DNA", *Nat. Genet.* 56:143–151 (2024), doi:10.1038/s41588-023-01582-w; data: Zenodo doi:10.5281/zenodo.8417049 | All IBD segments ≥6 cM (as released) among 4,248 published Eurasian individuals (AADR v54.1 IDs); 952,655 segments | **Downloaded** `data/ancibd_zenodo8417049/ibd220.ibd.v54.1.pub.tsv` (108 MB; zip md5 d5aa2761… matches Zenodo). Supp. Tables (xlsx: Table 1 = metadata for all 4,248, incl. locality, lat/lon, age, AADR cluster label; Table 2 = pairwise IBD summaries) downloaded and converted to TSV in `supp_tables/`. |
| No larger public ancIBD release | Zenodo searches for "ancIBD", "IBD segments ancient" etc. (sub-agent, 2026-11-10) | Only other ancIBD output on Zenodo: 10848927 (Amvrakia, small) | GitHub, Edmond and Dataverse not searched |
| BRONZESITES_IBERIA 1.0 | Peres M (2026), Zenodo doi:10.5281/zenodo.22935094 (CC-BY-4.0); cited as the database behind Rosas-Casals, Risch, Peres, Delgado-Raack, "An Approach from Networks Science to Bronze Age Archaeology in Southeast Iberia", *J. Archaeol. Method Theory* 34:30 (2027), doi:10.1007/s10816-026-09837-6 (bib key rosascasals2027) | Public xlsx has only 1,445 rows of site name, municipality, province, quality of research, bibliography. **No coordinates, no chronology, no 12-category economic matrix.** | **Downloaded** `data/bronzesites_iberia/` (108 kB). The paper's Data Availability says "No datasets were generated or analysed". Springer ESM not checked: link.springer.com refuses automated fetch, and I did not try to get round the bot check. |
| Gnecchi-Ruscone et al. 2024 | "Network of large pedigrees reveals social practices of Avar communities", *Nature* 629:376–383, doi:10.1038/s41586-024-07312-4 | ST4: KIN relatedness plus ancIBD-based analyses; ST1/ST2: archaeological and anthropological metadata; ENA PRJEB72021; Poseidon package | Public supplement (not downloaded). Medieval, so outside the prehistoric focus. |
| Fowler et al. 2022 | "A high-resolution picture of kinship practices in an Early Neolithic tomb", *Nature* 601:584–587, doi:10.1038/s41586-021-04241-4 | ST4/5: pairwise r and degrees; ST6: IBD windows; pedigree with 27 of 35 individuals; ENA PRJEB46958 | Public supplement (not downloaded); n=35, one tomb |
| Mittnik et al. 2019 | "Kinship-based social inequality in Bronze Age Europe", *Science* 366:731–734, doi:10.1126/science.aax6219 | Edmond doi:10.17617/3.2Y holds genotypes only; ENA PRJEB34400 | Kinship and grave-good tables are in the Science supplement (**unverified**, paywalled) |
| Rivollat et al. 2023 | "Extensive pedigrees reveal the social organization of a Neolithic community", *Nature* 620:600–606, doi:10.1038/s41586-023-06350-8 | Zenodo 7224898: `IBD_degrees.zip` (4.3 MB) and others; ST10 has per-pair ancIBD summaries; ENA PRJEB61818 | Public; most complete pedigree release (not downloaded yet) |
| Sikora et al. 2017 | *Science* 358:659–662, doi:10.1126/science.aao1807 | n=4 Sunghir individuals; ENA PRJEB22592 | Not useful for networks |
| Villalba-Mouco et al. 2021 | "Genomic transformation and social organization during the Copper Age–Bronze Age transition in southern Iberia", *Sci. Adv.* 7:eabi7038, doi:10.1126/sciadv.abi7038 | Tables S1.5–S1.7: PWMR/READ/lcMLkin kinship; S1.1: site, culture, dates; ENA PRJEB46907 | Public; the La Almoloya pedigree was deferred to a separate study |
| Olalde et al. 2019 | "The genomic history of the Iberian Peninsula over the past 8000 years", *Science* 363:1230–1234, doi:10.1126/science.aav4040 | Relatives flagged in the supplement (**unverified**); per-individual metadata are in the AADR | Use the AADR Family relations column instead |

AADR v66 Family relations: 3,747 of 19,118 dated ancient rows carry an entry; 99 of 594 Iberian individuals older than 2000 BP. These are only close (1st- to 3rd-degree), mostly within-site relatives, so they describe pedigrees rather than between-site ties.

## 2. Construction (`scripts/prep.py`)

- Regions are lat/lon boxes, keeping individuals older than 2000 BP: **CE** = Central Europe 45–55°N, 8–24°E (705 individuals, 203 localities); **BI** = Britain/Ireland (414); **IB** = Iberia (147). These counts are within the ancIBD 4,248 set.
- Tie: at least one IBD segment ≥12 cM, with the total of ≥12 cM segments at least TH (TH=12 by default). CE has 4,431 tied pairs. The released table uses ancIBD's own cutoffs; I applied no further false-positive filter.
- Culture: a keyword map over the AADR v54.1 Group ID (LBK, Lengyel, Baalberge, GlobularAmphora, CordedWare, BellBeaker, Unetice, Maros, Knoviz, LaTene, …; 37 rules).
  - Site names are never used as culture.
  - Genetic qualifiers are stripped before matching: `_o*`, low/high EEF, kin tags, `possible.contam`.
  - In CE, 83% of individuals get a named culture (35 cultures; Unetice 116, BellBeaker 67, LaTene 64, CordedWare 47, …).
- **In BI and IB the AADR labels are period labels** (England_MIA, Spain_C, …). Only 11% and 2% of individuals get a named culture there. Those regions cannot test culture and are left for locality/geography-only checks.
- Time bins are 500 years (BP), with a coarse period N/C/EBA/MBA/LBA/IA.

Descriptive results for all 4,248 individuals, ≥12 cM, 20,505 tied pairs:
- Only 14% of tied pairs are within the same locality; the median distance between tied individuals is 570 km.
- Within CE, the share within-locality, within-bin and same-culture rises steeply with tie strength:

| Total ≥12 cM | Same locality | Same culture |
|---|---|---|
| 12–20 cM (n≈3,000, single short segment) | 10% | 32% |
| 20–40 cM | 23% | 60% |
| 40–100 cM | 43% | 90% |
| >100 cM | 74–98% | 98% |

So the ≥12 cM network has two layers: close kin within the site, and a broad between-site layer of single shared segments (population-level, ancestors roughly 5–15 generations back). The README's caveat that "IBD is mostly within-site" holds for strong ties only.

## 3. Analyses and results (CE, TH = 12 cM)

### 3a. Nested DC-SBM partition vs attributes (`scripts/align.py`)

Results file: `results/CE_align_th12.json`. Best of 8 runs: nested DC-SBM with 10/3/2 blocks. 605 nodes are non-isolated, 509 of them with a named culture.

| Level-0 partition (10 blocks) vs | NMI | ARI |
|---|---|---|
| culture | 0.571 | 0.294 |
| culture×bin (45 groups) | 0.550 | 0.277 |
| locality (141) | 0.511 | 0.175 |
| 500-yr bin | 0.508 | 0.280 |
| period | 0.518 | 0.349 |
| **spatial null**: random contiguous Voronoi partitions per bin, K_t = number of cultures in that bin (200 draws) | 0.517 (q95 0.533) | 0.249 (q95 0.281) |
| culture shuffled within bin (spatial structure destroyed) | 0.447 | — |

- At level 0, culture×bin beats the spatial null with p=0.005 for NMI and p=0.065 for ARI.
- At level 1, NMI is 0.422 vs 0.383 and ARI 0.203 vs 0.147, both p=0.005.
- The excess over a random contiguous partition is real but small: ΔNMI ≈ 0.03–0.04.
- Most of the IBD block structure is explained by time plus spatial contiguity.

### 3b. Dyadic test: same culture → IBD tie, conditioned on space and time (`scripts/dyad.py`)

Results files: `results/CE_dyad_th12_*.json` and `results/CE_dyad_rates_*.tsv`.
- Dyads: 20,270 pairs of named-culture individuals in the same 500-yr bin but at different localities; 1,258 of them are tied.
- Statistic: Mantel–Haenszel odds ratio of a tie given same culture.

| Strata | OR(same culture) | Spatial-null OR: mean (q95) | p |
|---|---|---|---|
| bin × distance decile | **2.60** | 1.34 (1.88) | 0.004 |
| bin × distance decile × Δt (<100, 100–250, >250 yr) × same-study | **2.67** | 1.24 (1.71) | 0.002 |

Raw tie rates, different culture vs same culture, by distance band:

| Distance | Different culture | Same culture |
|---|---|---|
| 0–50 km | 0.074 | 0.117 |
| 50–100 km | 0.040 | 0.154 |
| 100–200 km | 0.054 | 0.063 |
| 200–400 km | 0.060 | 0.080 |
| 400–800 km | 0.023 | 0.089 |

Ties concentrate within culture labels beyond what a random spatially contiguous partition of the same granularity produces. The same-study stratum controls for batch and imputation effects.

This is a **symmetric association**. It is equally predicted by:
- homophily: similar groups exchange mates;
- network-driven formation: groups linked by kin and mating come to share material culture;
- label construction: archaeological horizons such as Bell Beaker or Corded Ware are defined partly by burial rite, which travels with people.

It cannot discriminate between them on its own.

### 3c. Description length: is there structure beyond culture blocks?

DC-SBM description length on the named-culture subgraph (lower is better):

| Partition | DL (nats) |
|---|---|
| culture×bin, planted (52 blocks) | 16,329 |
| random Voronoi, same K: mean (best of 200) | 16,473 (16,275) |
| locality (161 blocks) | 20,076 |
| inferred flat DC-SBM, 7–8 blocks | ≈13,000–13,050 |

- The inferred partition compresses the graph by ~3,300 nats more than the culture partition.
- The culture partition is only marginally better than random contiguous partitions, and some random partitions beat it.
- So there is substantial IBD block structure that culture labels do not capture. It is coarse, and probably macro-region × era.
- One earlier `minimize_blockmodel_dl` run gave 69 blocks and DL 14,098, so the optimiser is unstable. Repeat with many seeds before quoting the block count.

### 3d. Temporal order (cross-lagged): NOT RUN

`scripts/lagged.py` is written but untested.
- Design: for dyads at bin t′, the "past" of each member is the set of named individuals in the previous 500-yr bin within R km.
- Network-first model: sameCulture(t′) ~ G_lag + C_lag + log d + bin FE.
- Homophily-first model: IBD(t′) ~ C_lag + G_lag + log d + bin FE.
- Both are tested against spatial nulls that preserve distance structure:
  - G_lag is recomputed from per-bin distance-decay random graphs;
  - C_lag is recomputed from random Voronoi partitions with K_t matched.
- A grid-cell panel was rejected for power: only 3–11 cells (1°×1.5°) are sampled in two consecutive bins, which pools to about 126 cell dyads.
- Caveat for interpretation: cross-lagged comparisons are biased toward the better-measured variable, and IBD and culture labels differ in measurement error and base rate (tie rate ~6%, same-culture ~43%). Calibrate with the 0_power simulations before reading an asymmetry.

## 4. Confounds (explicit)

1. **Culture labels are archaeologists' constructs, filtered through AADR group naming.** Some AADR groups are split on genetic grounds; I stripped the genetic qualifiers, but residual circularity is possible. Burial rite defines several horizons and moves with people.
2. **Sampling.** Samples come from cemeteries chosen by the projects that sequenced them. Same-study pairs share coverage and imputation quality, which drives IBD detection power (handled with a stratum, not removed). Unetice and Bell Beaker are oversampled in Bohemia.
3. **What IBD measures.** IBD ≥12 cM marks common ancestors a few to ~15 generations back. It records genealogy plus population size and bottlenecks, not contact at time t. Between-site single-segment sharing in Corded Ware / Bell Beaker partly reflects the post-steppe bottleneck, not ties.
4. **Time.** Cultures are period-specific, so "same culture" carries finer time information than a 500-yr bin. Δt strata address this only partly, because many dates are contextual.
5. **Space.** All nulls are random contiguous partitions or distance strata. A barrier-aware null (rivers, mountains) was not used.

## 5. Queued runs (compute freeze 2026-11-10: none started)

Run sequentially from `archeology/2_adna/` and check `uptime` first (1-min load < 10). `scripts/run.sh` sets OMP/BLAS threads to 1, applies `nice -n 15` and uses `python -I`. graph-tool is single-threaded in align/dyad. Runtimes were estimated on this Mac under load ~20; one process each.

| # | Command | Purpose | Est. time |
|---|---|---|---|
| Q1 | `scripts/run.sh scripts/lagged.py results CE 100 2` | smoke test of the untested lagged script | < 1 min |
| Q2 | `scripts/run.sh scripts/lagged.py results CE 100 200` | temporal order, R=100 km → `results/CE_lagged_R100.json` | 5–15 min |
| Q3 | `scripts/run.sh scripts/lagged.py results CE 200 200` | sensitivity, R=200 km | 10–25 min |
| Q4 | `scripts/run.sh scripts/dyad.py results CE 30 500 bin_d_dt_study` | strong ties only (≥30 cM) | ~1–2 min |
| Q5 | `scripts/run.sh scripts/align.py results CE 30 200` | SBM alignment, strong ties | ~1–2 min |
| Q6 | `scripts/run.sh scripts/align.py results CE 12 200` | re-run with 1 thread (earlier run used default threads); check block-count stability | ~1–2 min |

`prep.py` needs no rerun. The xlsx conversion uses `~/anaconda3/bin/python3 -I scripts/xlsx2tsv.py` (openpyxl), because the gt env has no openpyxl.

## 6. Data requests (not contacted)

| Holder (role) | Ask |
|---|---|
| BRONZESITES / CORNERSTONES team (UAB Prehistory; corresponding authors of rosascasals2027) | The site × 12-economic-category binary matrix, with site coordinates and EBA/MBA/EMBA assignment, and settlement form. **Resolved 2026-11-10:** the ESM (100 sites × 12 categories, phases, form, coordinates) was obtained by workstream 4: `../4_timeseries/data/rosascasals2027/MOESM1_ESM.csv`. No request needed. |
| ancIBD developers (MPI-EVA / Reich lab) | ancIBD calls on a newer AADR (v62–v66, ≫4,248 individuals), especially Iberia and Central Europe. Current Iberian coverage in the release is 147 individuals older than 2000 BP. |
| MPI-EVA / UAB ASOME (Villalba-Mouco 2021 corresponding authors) | La Almoloya / El Argar pedigree and grave-goods coding per individual, or the follow-up kinship paper if published. |
| Mittnik 2019 corresponding authors (MPI-SHH/EVA) | Pairwise kinship plus grave-good status coding for the Lech valley farmsteads, if the Science supplement lacks it. |

## 7. Next steps

1. Run Q1–Q6.
2. Per-bin breakdown of the dyadic OR. Is the same-culture excess driven by the Corded Ware / Bell Beaker horizons (mobile, burial-rite defined) or general?
3. Replace culture labels with an independent material-culture attribute where one exists. The Iberian matrix (pending request) is the cleanest homophily network by construction, but aDNA overlap with the BRONZESITES sites needs checking (Motilla del Azuer / Acequión).
4. Add the Gurgy (Zenodo 7224898) and Avar pedigrees as within-community kinship networks for the 1_fission-style "fission follows pre-fission network" question.
5. Hand the observed CE numbers (n≈500, tie rate 6%, OR≈2.6) to 0_power. Can a cross-lagged design at this n distinguish the two mechanisms?

**2026-11-10 fix (main session):** the Q1 smoke test gave NaN betas because `lagged.py` took
`np.log(D)` of dyads at distinct localities with identical coordinates (D=0). Changed it to `np.log1p(D)`,
which matches the distance-decay null. Smoke test now runs (nf_z≈5.7, hf_z≈0.3 with only 2 nulls, so not meaningful).
Q2 rerun → `results/lagged_CE_R100.log`; Q3 in `../run_queue.sh` uses the fixed script.

### Q2 result (CE, R=100 km, 200 nulls; 9,572 dyads, 416 individuals) — 2026-11-10
| direction | observed β | null mean ± sd | z | p |
|---|---|---|---|---|
| network-first: past IBD density → same culture now | +0.71 | −1.03 ± 0.56 | 3.1 | 0.005 |
| homophily-first: past culture sharing → IBD tie now | −0.03 | 0.00 ± 0.19 | −0.2 | 0.53 |

Asymmetric, in the network-first direction. Do not over-read it yet:
- The nf null mean is strongly negative, so the z is partly a shifted null. Check why: distance-decay graphs vs the observed G_lag correlation with log d.
- AADR group labels are partly genetic, which makes "IBD → later label" partly circular. Re-run with labels from archaeology only.
- Demic turnover (e.g. Corded Ware/Bell Beaker arriving with steppe ancestry) gives IBD → culture through migration. That is network-first in the broad sense, not local-rule community formation. Stratify by bin and drop the transition bins.
- The cross-lagged design is biased toward the better-measured variable (IBD ≫ culture labels), which favours the nf direction.
- Next checks: R=200 (Q3), only 12–20 cM ties, leave-one-transition-out, labels with genetic names removed.

### Q2 robustness (2026-11-10): `scripts/lagged2.py`, R=100 km, 100 nulls per null type

Commands: `scripts/robust_queue.sh`. Outputs: `results/lagged2_R*.json|log`; summary table in `results/lagged2_summary.md` (made by `scripts/summ_lagged2.py`). Each run takes about 1 min, single-threaded and niced. R0 reproduces Q2 exactly (β_nf=0.71, β_hf=−0.03).

**Null models.**
- *nf (network-first) nulls*, applied to G_lag:
  - `decay` = the Q2 null;
  - `strat` = G_lag permuted among dyads within strata of bin × log-d decile × C_lag quartile × neighbourhood overlap. This preserves G_lag's correlation with log d and with C_lag.
  - `swap` = per-bin double-edge swaps of the past-bin IBD graph. It preserves every individual's degree and each edge's length class (same locality, 0–50, 50–100, 100–200, 200–400, 400–800, >800 km).
- *hf (homophily-first) nulls*, applied to C_lag:
  - `voronoi` = the Q2 null;
  - `hstrat` = C_lag permuted within bin × log-d × G_lag × overlap strata.
- `--noself` removes self-pairs (x,x) from the lag variables where past neighbourhoods overlap (62% of dyads). R1 onward use it.

| run | dyads | IBD edges | β_nf | decay z / p | strat z / p | swap z / p (null mean) | β_hf | voronoi z / p | hstrat z / p |
|---|---|---|---|---|---|---|---|---|---|
| R0 base (= Q2) | 9572 | 3549 | 0.71 | 2.9 / 0.02 | 4.2 / 0.01 | −2.1 / 0.96 (1.43) | −0.03 | 0.0 / 0.52 | −0.4 / 0.65 |
| R1 no self-pairs | 9555 | 3549 | 0.75 | 2.8 / 0.01 | 3.0 / 0.01 | −1.7 / 0.96 (1.34) | −0.04 | 0.1 / 0.43 | −0.2 / 0.58 |
| R2 archaeological labels only (30 individuals with `_o*`/EEF/numbered genetic sub-cluster labels dropped) | 8412 | 3309 | **0.24** | 2.9 / 0.02 | 7.9 / 0.01 | −1.5 / 0.90 (1.15) | 0.14 | 0.3 / 0.41 | 1.3 / 0.12 |
| R3 drop t′=7000 (LBK onset) | 9512 | 3549 | 0.73 | 2.8 / 0.01 | 3.2 / 0.01 | −2.0 / 0.96 | −0.06 | −0.3 / 0.61 | −0.6 / 0.74 |
| R4 drop t′=4500 (CW/GA arrival) | 9196 | 3549 | 0.73 | 2.8 / 0.01 | 2.0 / 0.04 | −2.0 / 0.97 | −0.05 | 0.0 / 0.43 | 0.2 / 0.42 |
| R5 drop t′=4000 (BB + early Únětice) | 6852 | 3549 | 2.50 | 1.6 / 0.07 | 18.7 / 0.01 | −1.5 / 0.88 (6.03) | 0.23 | −0.4 / 0.61 | 0.1 / 0.42 |
| R6 drop t′=3500 (Únětice/Maros, past = BB/CW/early Únětice) | 5885 | 3549 | **−0.36** | 0.2 / 0.27 | −1.3 / 0.91 | −1.3 / 0.90 | 0.10 | 1.2 / 0.09 | 1.8 / 0.06 |
| R7 drop all four transitions | 2780 | 3549 | 0.63 | 1.0 / 0.12 | 4.6 / 0.01 | −0.4 / 0.65 (0.67) | 0.53 | 1.9 / 0.03 | 0.4 / 0.42 |
| R8 weak ties only (12–20 cM total) | 9555 | 2452 | 1.17 | 1.2 / 0.12 | 0.2 / 0.41 | −0.8 / 0.80 (1.58) | −0.04 | 0.0 / 0.52 | 0.7 / 0.25 |
| R9 arch labels + drop all + weak ties | 2331 | 2264 | 0.11 | 0.6 / 0.22 | 3.4 / 0.09 | −0.1 / 0.63 | 0.57 | 2.0 / 0.04 | 0.9 / 0.21 |

(z values in R5 and R9 are inflated by near-separation in the logistic fits; null sd are tiny or null means very large. Read p there, not z.)

**Diagnosis of the shifted decay null (null mean −1.0).**
- In a decay-null graph, G_lag is essentially a smooth function of geometry: the mean of p(d_xy, same-locality) over the past pairs.
- Its correlation with log d (−0.48) matches the observed one (−0.45). Alone (+bin FE) it predicts future same culture with β=+0.78.
- Once log d is in the model, its residual is pure neighbourhood composition (the share of same-locality past pairs, overlap). That composition is negatively associated with future same culture, so β flips to −0.76.
- So the decay null is not centred at zero. Observed β exceeds it because observed G_lag carries idiosyncratic variance that the smooth null lacks, not necessarily because the wiring is informative.
- Self-pairs are not the cause: R1 changes little. Still, corr(G_lag, C_lag) is −0.4 within overlapping dyads vs +0.5 without overlap, so the overlap geometry matters throughout.

**Better nulls.**
- *Against `strat`* (G_lag's joint distribution with distance, C_lag and overlap preserved):
  - R1 gives β 0.75 vs 0.62 ± 0.04, p=0.01. That is significant but small: about 0.1 sd units of the 0.75 survive.
  - The excess vanishes for weak ties (R8, p=0.41) and with t′=3500 dropped (R6, p=0.91).
- *Against `swap`* (degrees and edge-length classes preserved), the observed β is *below* the null mean in every run (p=0.63–0.97):
  - Who is linked to whom adds nothing beyond how many IBD partners each past individual has and how long the edges are.
  - Degree here mixes kin-cluster size with sampling and coverage (large cemeteries of one culture).

**Verdict: the network-first asymmetry does not survive.**
- It depends on a mis-centred null.
- It fails a degree- and length-preserving null.
- It sits in one transition (BB/CW/early Únětice → Únětice/Maros at ~3500–4000 BP); removing that bin gives β_nf=−0.36.
- It disappears for weak between-site ties.
- It is fragile to dropping 5% of individuals with genetically qualified labels (0.75 → 0.24).

Homophily-first is null on the full data. It is weakly positive only against the Voronoi null once all transitions are dropped (R7 p=0.03, R9 p=0.04), and not against the stratified permutation (p=0.42, 0.21). At this sample size (≈550 individuals, 2–10k dyads, 11 bin pairs) the cross-lagged dyadic design does not identify temporal order in either direction. The symmetric association (§3b, OR≈2.6 beyond spatial null) stands.

Possible next steps (not run):
- Per-bin β with bootstrap over localities.
- A swap null that also preserves locality-pair counts at coarse resolution.
- The same design on a larger IBD release, once requested (§6).
