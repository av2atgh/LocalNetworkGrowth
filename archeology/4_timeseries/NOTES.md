# 4_timeseries: archaeological interaction time series

Started 2026-11-10. Status: **data in hand, pipeline written, compute frozen by the coordinator.**
Only lightweight results so far. The main runs are queued below.

## What these data can and cannot test

All the networks here are **similarity (co-occurrence) networks**. An edge means two assemblages
look alike: Brainerd–Robinson (BR) similarity of ware proportions, or a count of shared economic
categories. Edges are therefore **homophily by construction**. A thresholded BR graph is a random
geometric graph in composition space. It shows communities whenever compositions cluster, and
clustered compositions are, by definition, a latent shared attribute. So a cross-sectional
"communities beyond the spatial null" result would only show that ceramic composition clusters
beyond geography. It would not separate homophily from network-driven formation.

Any leverage comes from the dynamics, and the dynamics are themselves compromised:

- **Apportioning overlap.** The Chaco and SWSN slices come from assemblages apportioned across
  25- or 50-year intervals using ware/type date ranges (Roberts et al. 2012; UPDA, Ortman 2016).
  The same sherds feed adjacent slices, so persistence between consecutive slices is partly
  mechanical. Lag-2 comparisons and a stationary-resampling null (below) are the controls. Neither
  removes the effect completely.
- **Change in composition partly reflects the type chronology.** It is not purely a change in
  behaviour.
- **Coordinates are jittered for site protection**: 7–10 km in the Chaco attribute file, an
  undocumented amount in CriticalScales, and rounded to 10 km in SWSN_2013. That is fine for a
  regional distance-decay null and too coarse for anything local.

## Datasets

Citations were checked against Crossref on 2026-11-10.

| Name / dir | Citation | Content | Status |
|---|---|---|---|
| `data/SWSN_2013/` (github.com/mpeeples2008/SWSN_2013) | Mills, Clark, Peeples, Haas, Roberts, Hill, Huntley, Borck, Breiger, Clauset, Shackley 2013, *Transformation of social networks in the late pre-Hispanic US Southwest*, PNAS 110(15):5785–5790, doi:10.1073/pnas.1219966110 | BR similarity matrices for 5 × 50-yr slices AD 1200–1450. n = 279, 312, 173, 142, 44 sites; 191/102/129/38 sites shared between consecutive slices. `attributes.csv` holds 514 sites with macro (23) and micro (65) region, UTM rounded to 10 km, and room counts per period | **in hand**. No raw counts, so no resampling null is possible |
| `data/chaco_timeavg/` (github.com/cocoemily/time-averaged-networks, `Data/Chaco/AllCeramics.csv`) | Ceramics from Mills, Peeples, Aragon, Bellorado, Clark, Giomi, Windes 2018, *Evaluating Chaco migration scenarios using dynamic social network analysis*, Antiquity 92(364):922–939, doi:10.15184/aqy.2018.86. Repository of Daems, Coco, Gillreath-Brown, Kafetzaki 2023, *The effects of time-averaging on archaeological networks*, JAMT 31:473–506, doi:10.1007/s10816-023-09608-7 | Apportioned ware counts: 280 sites × 28 wares × 25-yr intervals, AD 800–1300 | **in hand**. Best series for the growth question: n rises from 24 to 185 sites, then falls to 37 |
| `data/Archaeological-Network-Uncertainty/` (github.com/mpeeples2008) | Peeples & Brughmans CAA 2017 workshop | `AD1050attr.csv`: 223 Chacoan great-house/great-kiva sites with macro group (20), GH/GK counts and jittered UTM | **in hand**. Used for Chaco coordinates and attributes (209 of the 280 sites match) |
| `data/CriticalScales/` (github.com/mpeeples2008/CriticalScales) | Peeples & Bischoff 2023, *Archaeological networks, community detection, and critical scales of interaction in the U.S. Southwest/Mexican Northwest*, J. Anthropol. Archaeol. 70:101511, doi:10.1016/j.jaa.2023.101511 | Raw type counts: 5,763 sites, 12.8 M sherds, type date ranges, jittered coordinates. The paper builds 9 × 50-yr slices (AD 1000–1450) via UPDA and runs distance-constrained Louvain | **in hand, not yet processed** (needs a UPDA port). **Directly relevant prior art**: community detection in these networks as a function of spatial scale |
| `data/rosascasals2027/MOESM1_ESM.csv` | Rosas-Casals, Peres, Delgado-Raack, Risch 2027, *An approach from networks science to Bronze Age archaeology in Southeast Iberia*, JAMT 34:30, doi:10.1007/s10816-026-09837-6 | ESM 1: 100 sites × 12 binary economic categories, plus form (FH/FL/UH/UL), chronology (EBA 20 / MBA 45 / EMBA 35 persisting), lat/lon and size | **in hand, re-analysed (light)**. The paper's own Data Availability statement says "No datasets were generated or analysed", yet the ESM CSV exists. No data request needed |
| `data/bronzesites_iberia/` | Peres 2026, BRONZESITES_IBERIA 1.0, CORA/Zenodo doi:10.5281/zenodo.22935094, CC-BY-4.0 | The public xlsx is only a gazetteer: name, municipality, province, research quality and bibliography for 1,445 sites. No coordinates, chronology or categories | not usable for networks. A fuller version would need a request; none made |
| ArchNetSci repo (not downloaded) | Brughmans & Peeples 2023, *Network Science in Archaeology* (CUP), companion github.com/mpeeples2008/ArchNetSci | `SanPedro_nets.Rdata` (3 slices AD 1250–1400, small), Cibola, AD1050/AD1300 single slices | optional |
| time-averaged-networks repo (not downloaded) | Daems et al. 2023 (above) | ICRATES Roman tableware: 275 E. Mediterranean sites, time slices (Bes 2015; Bes et al. 2019). Prignano Etruria: 5 slices, but edges are roads, not similarity | candidate for a second series. ICRATES `CATALOGUE.csv` is 7 MB |
| not public / not found | Coward 2013 (in Knappett ed., *Network Analysis in Archaeology*, OUP, 247–280); Golitko & Feinman 2015 JAMT 22:206–247 (doi:10.1007/s10816-014-9211-1; ESM contents unverified); Gjesfjeld 2015 JAMT 22:182–205 (tDAR 25979, login) | | not pursued; no contact made |

## Pipeline (`scripts/`; all Python with `-I`, graph-tool env)

- `swsn_load.py`, `build_slices.py` → `results/slices.pkl`. Chaco is aggregated to 50-yr slices
  (pairs of 25-yr columns), keeping sites with ≥10 sherds that have attributes, with BR computed
  on ware proportions. SWSN uses the published matrices. Chaco site counts per slice: AD 800:5,
  850:24, 900:41, 950:57, 1000:141, 1050:181, 1100:185, 1150:128, 1200:74, 1250:37.
- `common.py`: BR similarity, threshold (0.75, the Mills et al. convention), and a DC-SBM fit on
  the LCC (graph-tool `minimize_blockmodel_dl`, best of k runs). Detection means B ≥ 2.
  Σ = (DL_{B=1} − DL_best)/E. Nulls, all on the same site positions:
  - `dist`: non-parametric distance decay, the empirical edge probability per distance-quantile bin;
  - `attr`: block densities by region;
  - `dist_attr`: distance bin × same region;
  - `config`: degree-preserving rewiring, the book's control.
- `scale_detect.py`, question (i). Part A: each slice as observed, against 20 samples from each
  null. Part B: in slices with n ≥ 140, nested subsets of size 10–130. The spatially contiguous
  kNN ball around a random site is the growth analogue; random subsets are the control. Observed
  versus the nulls. Checkpointed per slice.
- `dynamics.py`, questions (ii) and (iii). Pair-level analysis of consecutive slices at lags 1
  and 2:
  - persistence: logit same_comm(t+1) ~ same_comm(t) + distance deciles + same region;
  - forward convergence: Δs ~ s(t) + same_comm(t) + controls;
  - backward convergence, time-reversed: −Δs ~ s(t+1) + same_comm(t+1) + controls. Forward minus
    backward is the time-asymmetry test. Under a stationary truth plus sampling noise the two are
    equal in expectation;
  - similarity → later co-membership: logit same_comm(t+1) ~ same_comm(t) + z[s(t)].

  CIs come from a node bootstrap. Two nulls. *geo*: independent `dist_attr` graphs at t and t+1;
  this gives the persistence that fixed geography produces with no memory, and its β values are
  on a binary scale, so they are not comparable. *stat*: Chaco only; both slices are redrawn
  multinomially from the pooled composition, which keeps community structure static and adds
  sampling noise only. Partition self-consistency is checked with two independent fits.
  **This script is written but has not been executed.**
- `iberia.py`: Rosas-Casals re-analysis (below).
- `summarize_scale.py`: turns the scale output into tables.

## Results so far

### (i) Scale: observed slices only

The 2-worker run was killed by the coordinator over machine load. These numbers come from its
log. **The null ensembles were lost**, so nothing below is yet compared with a null.

| slice | n sites | n LCC | B | Σ |
|---|---|---|---|---|
| Chaco 850 | 24 | 20 | 2 | 0.150 |
| Chaco 900 | 41 | 21 | 1 | 0 |
| Chaco 950 | 57 | 34 | 2 | 0.263 |
| Chaco 1000 | 141 | — | (not reached) | |
| Chaco 1050 | 181 | 128 | 4 | 0.064 |
| Chaco 1100 | 185 | 174 | 7 | 0.474 |
| Chaco 1150 | 128 | 60 | 4 | 0.124 |
| Chaco 1200 | 74 | 39 | 2 | 0.030 |
| Chaco 1250 | 37 | 14 | 1 | 0 |
| SWSN 1200 | 279 | 109 | 9 | 0.569 |
| SWSN 1250 | 312 | 125 | 11 | 0.627 |
| SWSN 1300 | 173 | 45 | 2 | 0.046 |
| SWSN 1350 | 142 | 66 | 3 | 0.009 |
| SWSN 1400 | 44 | 26 | 1 | 0 |

Reading, provisional:

- B rises with the size of the LCC. B = 1 at LCC sizes 14, 21 and 26; B ≥ 2 from about 34 upward;
  B ≈ 7–11 only at LCC ≥ 109. The one exception is Chaco 850 (B = 2 at n = 20).
- Chaco is a growth-then-decline series, and B tracks n on both arms. That is the shape the book
  predicts for size-limited detectability. It is equally what any clustered latent-similarity
  graph gives, because SBM resolution grows with n. Without the nulls this does not discriminate.
- The thresholded graphs are fragmented. Only 109 of the 279 SWSN AD 1200 sites are in the LCC,
  so the "n" relevant to detection is the LCC size, not the site count.
- One pilot draw (anecdotal, n = 1) on Chaco AD 1100: observed B = 7, Σ = 0.47; `dist` null
  B = 4, Σ = 0.05; `config` null B = 1. That suggests structure beyond distance decay and beyond
  degrees, as expected for a composition-similarity graph.

### Rosas-Casals Iberia (light run: numpy only, no SBM) — `results/iberia_light.txt`

- Paper densities reproduced: density of shared-category edges (w > 0) is 0.947 (EBA, n = 20),
  0.921 (MBA, n = 45) and 0.943 (EMBA, n = 35). The paper reports about 0.93.
- **No structure beyond the margins.** Hold each site's number of categories and each category's
  prevalence fixed (curveball null, 200 draws). The variance of pairwise shared-category counts is
  then indistinguishable from the null in every set: p = 0.68 (EBA), 0.17 (MBA), 0.36 (EMBA),
  0.30 (EBA+EMBA), 0.25 (MBA+EMBA), 0.25 (all 100). With 12 binary categories and density 0.93,
  the data cannot carry community structure. This agrees with the paper's "close to random graph"
  conclusion.
- **Space is weak.** The Mantel correlation of shared categories against distance is
  r = −0.20 (p = 0.001) for MBA and −0.05 (p = 0.26) for EBA. Across all 100 sites it is −0.08
  (p = 0.006).
- **Settlement form matters only in MBA.** Same-form pairs share +0.43 more categories
  (permutation p = 0.015). This permutation does not hold the margins fixed. For EBA the value is
  −0.11 (p = 0.74).
- Relative to the book: EBA (20), MBA (45) and the persisting set (35) all lie below the
  r_κ ≈ 80–100 detectability scale. So "no communities" is the expected outcome under either
  mechanism and does not discriminate between them. The ESM has one category profile per site, not
  one per period, so no dynamic test is possible with it.

### (ii) and (iii): not run (compute freeze)

## Queued runs (single process, run sequentially)

Prefix every command with
`cd ~/av2atg/LocalNetworkGrowth/archeology/4_timeseries/scripts && OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 nice -n 15`.
Check `uptime` first and wait while the 1-min load is above 10. Both main scripts checkpoint and
resume: `results/scale_parts_thr*/` and `results/dyn_parts_thr*/`.

1. `THR=0.75 ~/anaconda3/envs/gt/bin/python -I scale_detect.py > ../results/scale_thr0.75.log 2>&1 && ~/anaconda3/envs/gt/bin/python -I summarize_scale.py 0.75`
   Expected runtime: about 15–25 min single-core. 13 slices × (obs + 4 × 20 nulls), plus subsets
   for the 6 slices with n ≥ 140. The largest fits take about 1–2 s.
2. `~/anaconda3/envs/gt/bin/python -I iberia.py`
   The full version, with DC-SBM and 50 curveball nulls per set. About 1–3 min.
3. `THR=0.75 ~/anaconda3/envs/gt/bin/python -I dynamics.py > ../results/dynamics_thr0.75.log 2>&1`
   Expected runtime: about 1–1.5 h. There are 22 slice pairs (Chaco lag 1: 8, lag 2: 7; SWSN lag 1:
   4, lag 2: 3). Each pair needs 3 partitions, 40 geo-null partitions, 40 stationary-null
   partitions (Chaco only) and 200 bootstrap regressions. **Untested.** Run it first on a single
   pair as a smoke test, e.g. by temporarily slicing `jobs[:1]`. No summarizer has been written
   for this output yet.
4. Sensitivity: rerun item 1 with `THR=0.85`, about 15–25 min.

## Next steps

- Once items 1–3 are done: compare P(B ≥ 2) and Σ against n for the observed data and for the
  `dist`, `dist_attr` and `config` nulls, using kNN-ball subsets (the growth analogue) against
  random subsets.
  - If the observed detection threshold n* matches the `dist_attr` null, space and region explain
    the scale signature.
  - If observed Σ exceeds the nulls only above about 80–100 sites, that is consistent with the
    book. It is still not diagnostic, because composition similarity is itself a latent attribute.
- The dynamic tests carry the real weight here:
  - forward minus backward convergence, measured against the stationary-resampling null (Chaco);
  - persistence OR against the geo null and at lag 2 against lag 1, to separate apportioning
    overlap from memory.
- CriticalScales: port UPDA (Ortman 2016; Peeples' `UniformProbabilityDensityAnalysis` repo) to
  Python and build the 9-slice AD 1000–1450 networks over about 5,800 sites. That gives the largest
  n-range and independent apportioning. It is memory-heavy (the authors quote 16+ GB for the full
  R pipeline), so it needs care on this machine.
- ICRATES as a non-Southwest replication.
- Iberia: no further leverage without per-period category profiles. BRONZESITES_IBERIA would need
  its full version, and no request has been made.

## Dynamics results (2026-11-10, main session)
Run: `NNULL=200 THR=0.75 dynamics.py` (~3 min). Summary: `results/dynamics_thr0.75_null200_summary.txt`
(`scripts/summarize_dynamics.py`). The earlier 20-null run is `results/dynamics_thr0.75_summary.txt`.
Added `NJOBS` (smoke test) and `NNULL` env options to dynamics.py; non-default NNULL writes to separate files.
- Persistence of co-membership: not above the geo (no-memory) null in most pairs, so geography explains it.
- beta_fwd (same community at t → similarity gain by t+1, controlling for s(t)): above the geo null at p ≤ 0.02 in 6 of 10 pairs with n ≥ 30 (corrected from "7 of 10" on recount). beta_bwd (time-reversed) ≈ 0 or negative everywhere.
- fwd − bwd > 0 in 9 of 10 pairs with n ≥ 30. Against the Chaco stationary null: p = .01, .09, .02, .03, .50, .07, .03, .02. Lag-2 pairs, which avoid apportioning overlap, are included: 1000–1100 .07, 1050–1150 .03, 1100–1200 .02.
- gamma (similarity at t → co-membership at t+1): also above the geo null in Chaco (p = 0.00). Negative in Chaco 1100–1200 and SWSN 1300–1350.
- Reading: a time arrow exists, with convergence inside communities going forward and not backward. This is the "tie → later similarity" signature that the power study says only network-driven/contagion produces. Similarity → later co-membership is ALSO present, so the two directions coexist. Caveats: communities are clusters of similarity, not interaction ties, so a cluster-level latent tradition evolving jointly would produce the same pattern; slice pairs overlap and are not independent; there is no multiple-testing correction; under Holm over the 8 Chaco stationary-null p values none survives (smallest adjusted 0.08).

## Joint null and contact test (2026-11-10, `scripts/dynamics_joint.py`, `NNULL=500`)
Output: `results/dynamics_joint_null500.json` (+ `.log`).
- **Joint stationary null.** Every Chaco slice (950–1200) is redrawn from each site's composition pooled over all slices, all slices are re-partitioned, and all 8 pairs are evaluated on the same draw. Pair dependence is therefore preserved.
- Mean fwd−bwd over 8 pairs: 0.120 vs null 0.008 [−0.04, 0.06], **p < 0.002** (above all 500 draws).
  - Lag-1 pairs: 0.156, p < 0.002.
  - Lag-2 pairs, which share no apportioned sherds: 0.084, **p = 0.010**.
  - This replaces the per-pair tests that failed Holm.
- **Contact test.** Add same_t × z(log d) to the forward model; contact predicts a negative interaction. Observed 0.001 vs null 0.007, p = 0.19. No distance decay of within-community convergence, so it is community-wide, not between adjacent sites. Static tradition: excluded. Spatially local contact: excluded. A regional tradition that evolves cluster by cluster: still not excluded.
