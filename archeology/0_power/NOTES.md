# 0_power: which statistics can tell network-driven from homophily-driven communities?

Status 2026-11-10: **done (v1)**. Synthetic study, 15 260 networks. Each one is fitted, re-grown 3 more times for the history test, and run through a simulated fission. About 40 min on 2 workers.

## Bottom line

1. **Statistics that take the observed attribute as given work at every size, even n = 20.**
   These are the dyadic logit (tie ~ same-attribute + common neighbours + log-degrees + distance)
   and the description-length comparison against the attribute block model. Both have AUC ≥ 0.99
   for LS/DS λ=0 vs λ=1 at n = 20 and power ≥ 0.97 at α = 0.05. They recover λ
   (Spearman 0.72–0.98, combined CV RMSE 0.17 at n = 20 down to 0.03 at n = 500).
2. **Statistics that first need the SBM to find communities are weak at every size.**
   These are NMI/ARI(SBM, attr), path dependence and the pre-split bipartition. The reason is not
   only that small groups sit below r_κ. Homophily communities at ⟨k⟩ ≈ 4 sit at the MDL detection
   threshold (see "Why homophily is hard for the SBM"), so P(B>1) for the homophily models stays
   at 0.1–0.55 from n = 45 to n = 500 and never approaches 1.
3. **Contagion breaks every snapshot statistic built on attributes.** If attributes are copied
   along ties (network-driven world), snapshot alignment reads as *more* homophilous than true
   homophily with exogenous drift: AUC drops to 0.00–0.08 at n ≥ 100, the reverse of the
   prediction. Three things survive:
   - the closure term of the logit (|β_CN|, logit ratio);
   - the temporal statistics;
   - the description-length *beyond* attributes, at n ≥ 100.
4. **At density 0.9 (n = 20–45 similarity networks), the SBM never finds more than one block**
   (P(B>1) = 0 in every model). Only the dyadic logit keeps power: AUC 0.85–1.00 for b_same and
   the logit ratio.
5. **Space mimics homophily twice.** Pure distance decay with spatially sorted attributes gives
   "attributes explain the structure" (expl_dc AUC 0.97–1.00 vs LS at n ≥ 45). It also gives
   history-independent partitions (path-dependence AUC 0.76 at n = 200, 0.91 at n = 500). Putting
   distance in the logit removes the false homophily: b_same AUC is 0.53–0.61 for pure space vs LS.
6. **Main caveat.** In this design the closure term separates the classes because the homophily
   alternative has no transitivity beyond the observed attribute. Latent homophily on finer,
   unobserved types does generate closure. With 8 latent types and only type mod 2 observed,
   β_CN rises from about 0 to 0.67 (n = 45) and 0.34 (n = 200), against 1.4 and 1.1 for LS. AUC
   is still 1.0, but the margin depends on how fine the latent structure is. **Closure beyond the
   observed attributes is evidence for locality *or* for homophily on something unmeasured; no
   snapshot statistic separates these.** This is the Shalizi–Thomas problem again.

## Design (and why)

`models.py` holds the generators and `stats.py` the statistics. `run.py` is the runner
(resumable; appends to `data/results.csv`). `analyze.py` produces the power tables and figures.

- **Mixing knob λ** (share of links placed by homophily), with K = 2 balanced attributes (a
  moiety-like binary, which keeps the binary fission cut clean):
  - **LS(n, l=1)** (= triadic closure, Ch. 3/6): link 1 goes to a uniform anchor and link 2 to a
    random neighbour of the anchor. Each link is a homophily draw instead with probability λ.
    λ = 0 is exactly the book's LS(n, 1).
  - **DS(n, q=0.5)** (Ch. 6): the whole newcomer is a homophily node (2 draws) with probability λ.
  - **Homophily draw:** with probability h = 0.8 the draw is restricted to the newcomer's type,
    and the target is chosen ∝ (k+1) (PA × homophily, as in Karimi et al. 2018). With K = 2 this
    gives 90 % within-type homophily ties. The PA factor keeps degree heterogeneity comparable,
    so λ changes *locality*, not the degree distribution.
- **Controls and other worlds:**
  - **ASBM:** static attribute SBM with ⟨k⟩ = 4 and the same within-type fraction.
  - **SP:** positions uniform in the unit square; attribute = side of x = ½, 10 % flipped; each of
    2 links drawn ∝ exp(−d/0.1), or by homophily with probability λ.
  - **BA (m = 2):** no locality and no types, the book's non-local control.
- **Contagion variant (LSc, DSc):** after every growth step, 2 random nodes update their
  attribute. With probability 1−λ they copy a random neighbour (network-driven similarity); with
  probability λ they redraw it uniformly (exogenous drift). Homophily draws use the *current*
  types. A newcomer's type is exogenous at birth.
- **Dense:** the sparse network is grown, then the 90 % of pairs with the smallest graph distance
  are kept, ties broken at random. This is a stylised "similarity network" with the same
  transform for every mechanism.
- **Sizes:** sparse n = 20, 35, 45, 100, 200, 500; dense n = 20, 35, 45.
- **Replicates:** 60 per condition; 100 for the λ=0 rows at n = 20 and part of n = 35. I cut this
  from the planned 100 after the load warning. CIs are bootstrap (400 resamples).
- **Sensitivity at n = 45, 100, 200 (LS, DS, ASBM):** h ∈ {0.5, 0.95}; K = 4; latent K = 8 with
  type mod 2 observed.
- **Detector:** flat DC-SBM (`minimize_blockmodel_dl`, best of 2), plus a non-degree-corrected fit.
  I did not use the nested model: for (b) the attribute model and the free fit must be the same
  model class.

### Statistics (README table → implementation)

| | statistic | predicted larger under |
|---|---|---|
| a | NMI and ARI between the DC-SBM partition and the attribute | homophily |
| b | `beyond` = (S_attr − S_free)/E. S_attr is the attribute block model with the partition given as data (no partition DL); S_free is the fitted SBM, which pays for its partition. `expl` = (S_1 − S_attr)/E. Both DC and NDC. | beyond: network; expl: homophily |
| c | standardized dyadic logit: tie ~ same + CN + log(k_i−A_ij)+log(k_j−A_ij) + distance. Reported as b_same, \|b_CN\|, and the ratio b_same/(\|b_same\|+\|b_CN\|) | ratio: homophily |
| d | mean pairwise ARI of the SBM partitions of 4 histories with the same population (attribute, position, arrival order) and a different structural RNG. NaN when both partitions are trivial. | homophily |
| e | fission: two top-degree leaders, sides spread by neighbour majority, 5 relaxation sweeps. Reported as ARI(cut, attr), ARI(cut, forced-B=2 SBM of the pre-split network), and their difference. The cascade is network-mediated in every world, a conservative choice. | homophily |
| f | `conv` = P(same at n \| tied at t1=0.75n, different at t1) − the same for untied pairs (tie → similarity). `sel` = same-type share of ties formed after t1, *at formation*, minus the newcomer's same-type baseline (similarity → tie). | conv: network; sel: homophily |

AUC is computed in the predicted direction, so AUC < 0.5 means the statistic points the wrong
way. Power is the one-sided rejection rate at α = 0.05. Under null = network (λ=0) it is `pow_h`;
under null = homophily (λ=1) it is `pow_n`. Both are in `data/power.csv`.

## Statistic × size → power (main setting: h = 0.8, K = 2, λ = 0 vs λ = 1)

AUC for LS / DS. Brackets give `pow_h` for LS at α = 0.05.

| statistic | n=20 | 35 | 45 | 100 | 200 | 500 | dense 20 | 35 | 45 |
|---|---|---|---|---|---|---|---|---|---|
| a ARI(SBM, attr) | .50/.56 [0] | .56/.60 | .66/.64 [.12] | .71/.58 [.27] | .72/.65 [.25] | .74/.75 [.40] | .50 | .50 | .51 |
| a NMI(SBM, attr) | wrong way: .50→.25 (biased up by large B under LS/DS) | | | | | | .50 | .50 | .50 |
| b expl_dc / beyond_dc | .99/.99 [.97] | 1/1 | 1/1 [1] | 1/1 | 1/1 | 1/1 | .55/.62 [.07] | .78/.58 | .76/.65 [.32] |
| c b_same | 1/1 [1] | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 | .96/.96 [.85] | 1/.99 | 1/1 [1] |
| c logit ratio | 1/1 [1] | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 | .91/.85 [.33] | .96/.97 [.88] | .98/1 [.93] |
| c \|b_CN\| | 1/1 [1] | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 | .54/.40 | .60/.71 | .62/.86 |
| d ARI across histories | .50/.54 [0] | .55/.52 | .63/.58 [.02] | .62/.44 [.17] | .65/.47 [.18] | .72/.52 [.40] | n/a (B=1) | n/a | .50 |
| e ARI(cut, attr) | .71/.78 [.60] | .85/.83 | .92/.93 [.78] | .93/.93 [.83] | .96/.90 [.90] | .97/.94 [.87] | .54/.43 | .44/.47 | .49/.50 |
| e ARI(cut,attr) − ARI(cut,topo) | .84/.59 [.25] | .77/.67 | .82/.64 | .78/.63 | .80/.78 | .74/.74 | ≈.5 | ≈.5 | ≈.5 |

Contagion world (LSc / DSc: network + copying vs homophily + drift; λ = 0 vs 1), AUC:

| statistic | 20 | 35 | 45 | 100 | 200 | 500 |
|---|---|---|---|---|---|---|
| f conv (tie → similarity) | .71/.64 | .77/.75 | .81/.70 | .93/.77 | 1.0/.91 | 1.0/.93 |
| f sel (similarity → tie) | .98/.99 | 1/.99 | 1/1 | 1/1 | 1/1 | 1/1 |
| c logit ratio | .69/.84 | .82/.85 | .90/.92 | 1/.98 | 1/1 | 1/1 |
| b beyond_dc | .35/.70 | .22/.79 | .31/.84 | .92/1 | 1/1 | 1/1 |
| b expl_dc (**confounded**) | .35/.67 | .18/.60 | .20/.58 | .08/.50 | .01/.28 | .00/.33 |
| c b_same (**confounded**) | .32/.42 | .27/.43 | .25/.55 | .20/.51 | .05/.36 | .00/.57 |
| a ARI(SBM, attr) (**confounded**) | .50/.56 | .52/.54 | .45/.47 | .27/.48 | .05/.25 | .07/.09 |
| d, e | ≈ .4–.8, no consistent signal | | | | | |

Other contrasts (`data/power.csv`):

- **Mixture detection, LS λ=0 vs λ=0.5:** b and c stay at AUC ≥ 0.94 from n = 35 (0.75–0.98 at
  n = 20). The SBM-based statistics do not detect a 50 % homophily share.
- **Sensitivity of the SBM-based statistics to h:**
  - At h = 0.5, ARI(SBM, attr) is about 0.6 and path dependence about 0.5 at every n.
  - At h = 0.95, both reach 0.96–1.0 from n = 45.
  - At K = 4 they become usable at n = 200 (0.87–0.95).
  - b and c are at about 1.0 in every sensitivity set.
- **Four-way mechanism classification** (network / homophily / space / BA; softmax on all snapshot
  statistics, 5-fold CV; `data/confusion.csv`):
  - Balanced accuracy is 0.85, 0.90, 0.95 at n = 20, 35, 45 and 1.00 from n = 100.
  - Dense: 0.61, 0.76, 0.85.
  - At small n the residual error is network vs BA (locality vs no locality), not network vs
    homophily.
  - This is in-model accuracy: train and test come from the same generators.

### Why homophily is hard for the SBM (analytic check of the simulations)

K balanced types with a within-type fraction p of the edges give an evidence per edge over one
block of KL = p ln(pK) + (1−p) ln[(1−p)K/(K−1)]. The partition costs about ln K per node. The
detection condition is therefore

  (⟨k⟩/2) · KL > ln K  (independent of n).

For K = 2 and ⟨k⟩ = 4:
- p = 0.9 (h = 0.8) gives 0.74 vs 0.69, marginal. The simulations show P(B>1) = 0.27–0.55 for
  ASBM and 0.12–0.42 for LS λ=1 at n = 45–500.
- p = 0.75 (h = 0.5) gives 0.26 < 0.69: simulated P(B>1) = 0 at n = 45, 100, 200.
- p = 0.975 (h = 0.95) gives 1.15: simulated P(B>1) = 0.85–0.98 at n = 45 and 1.0 at n ≥ 100.

Local-rule communities instead gain evidence as n grows: P(B>1) for LS λ=0 is 0, 0.23, 0.57, 1, 1,
1 at n = 20…500, consistent with r_κ ≈ 81. At small n, "the SBM finds no communities" says nothing
about either mechanism. At large n, it is evidence *against* local rules but not against moderate
homophily.

A side observation: equal mixtures destroy SBM-detectable structure. For LS at λ = 0.5,
P(B>1) ≤ 0.12 even at n = 500, against 1.0 for λ = 0 and 0.42 for λ = 1. For LS at λ = 0.75 it
is ≤ 0.02. DS is more robust: 0.95 at λ = 0.5 and n = 500.

## Useless or misleading, and where

- **NMI(SBM, attr):** wrong direction at every size. It is inflated whenever the SBM returns many
  blocks, which is exactly what local rules produce. Use ARI, or better the DL statistics.
- **Path dependence (d):** AUC 0.5–0.7 at h = 0.8 at every n. It needs strong homophily (h ≥ 0.95)
  or K ≥ 4 with n ≥ 200, and space passes it as "homophily". It also needs replicate histories,
  which data rarely give.
- **Any SBM-partition statistic** (a, d, the topological half of e) at n ≤ 45 sparse, and at every
  dense size: the SBM returns B = 1.
- **Attribute snapshot statistics (a, b-expl, c-b_same)** whenever attributes can spread along
  ties. Under contagion they reverse sign.
- **expl_dc / b_same without a distance term** whenever attributes are spatially sorted.
- **fis_diff:** weaker than fis_attr alone (0.6–0.85). Its topological half is noisy at small n.

## Recommendations to workstreams

- **1_fission** (n ≈ 20–200): ARI(cut, attr) has AUC 0.71 at n = 20 and 0.92 at n ≥ 45. It is the
  usable fission statistic, *provided* the attribute is measured before the split. Pair it with a
  dyadic logit on the pre-fission ties. Do not lean on the SBM bipartition of the pre-split network
  below n ≈ 80 (B = 1 for most local-rule networks of that size). Treat kinship as network, not
  attribute: it is a genealogy and spreads like contagion.
- **2_adna:** this is where temporal order is identifiable.
  - `sel` (attribute similarity *at tie formation* above baseline) has AUC ≥ 0.98 even at n = 20.
  - `conv` (convergence after the tie) needs n ≥ 100 for AUC > 0.9.
  - Both need dated ties and dated attributes. `sel` is confounded if newcomers adopt the host's
    culture on arrival; with coarse dating this looks identical to homophily.
  - Never use snapshot attribute–community alignment as homophily evidence: under contagion it
    reverses.
- **3_foragers** (sparse, n ≈ 50–300):
  - The workhorse is the dyadic logit with same-attribute (several attributes) + CN + degree +
    camp/geographic distance, plus the attribute-block DL comparison. Both are near-perfect in-model
    at these sizes.
  - Report β_CN as "closure beyond the measured attributes". It discriminates locality from
    homophily on *measured* traits only. Latent homophily produces it too, so add every attribute
    available (age, sex, kin, residence) before reading it as locality.
- **4_timeseries** (dense similarity networks, n = 20–45):
  - SBM-based tests are dead at density 0.9.
  - Use the dyadic logit with a distance term (AUC 0.85–1.00), not community alignment.
  - The spatial null is mandatory: without distance, pure space makes attributes look explanatory
    (AUC up to 1.0).
  - Material-culture similarity networks are homophily networks by construction, so the tie is not
    independent of the attribute. Use them as the attribute or the outcome, not both.
- **All workstreams:**
  - Pre-register the direction of each statistic.
  - Report AUC-type effect sizes against simulated λ=0 and λ=1 references grown at the *observed*
    n, ⟨k⟩ and density with `models.py`. The numbers above hold for ⟨k⟩ ≈ 4 and h = 0.8.

## Caveats

- **In-model results.** Every class is one of my generators. Real alternatives can include
  homophily with closure, homophily on latent attributes, or space with PA, and these narrow the
  margins. The latent K = 8 run shows the CN margin shrinking but not closing.
- **Homophily strength.** One homophily strength is the main setting (h = 0.8 ⇒ 90 % within-type
  ties), checked against 0.5 and 0.95. Mean degree ≈ 4 throughout, except DS, whose degree grows.
- **Dense networks** come from a distance-threshold transform. This is a stand-in for real
  similarity thresholding, not a model of it.
- **Fission** is a network-mediated cascade in every world. If homophily-world members choose a
  side by attribute directly, fission statistics get *more* power than reported here.
- **Contagion world.** The update rate (2 per step) and the wave t1 = 0.75n are fixed and not
  scanned.
- **Bug fixed during the run.** In `run.py` the history index shadowed the homophily strength `h`,
  so the main network of every λ>0 / ASBM condition was grown with h = 0. Those rows were deleted
  and re-run after the fix; λ=0 and BA rows were unaffected and kept. The buggy table is in
  `scratch/results_buggy_h.csv` (gitignored) for the record.

## Files

- `models.py`, `stats.py`, `run.py`, `analyze.py`. Run with `OMP_NUM_THREADS=1 nice -n 15
  ~/anaconda3/envs/gt/bin/python run.py --reps 60 --workers 2`, then `analyze.py`.
- `data/results.csv`: one row per replicate, all statistics.
- `data/power.csv`: contrast × n × stat → AUC, CI, pow_h, pow_n, NaN fraction.
- `data/lambda.csv`: Spearman ρ(stat, λ) and the combined CV RMSE.
- `data/confusion.csv`: 4-class confusion matrices.
- `analyze.py` also writes exploratory figures to `figs/` (removed; the manuscript figures are in `../figs_ms`).

## Next steps (if wanted)

- Add a homophily-with-closure alternative (homophily anchor + LS walk) and finer latent types.
  This is the real stress test for the CN statistic.
- Scan ⟨k⟩ (forager networks often have ⟨k⟩ ≈ 8–15). The MDL bound predicts that the SBM-based
  statistics become usable for homophily at higher degree.
- Calibrate against each workstream's actual n, ⟨k⟩, density and K once the data are in hand.
