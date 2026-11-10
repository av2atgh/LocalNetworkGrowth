# Archeology: are early communities network-driven or homophily-driven?

Started 2026-11-10. Book claim (Ch. 6, `communities.tex`): communities emerge from
**locality** of growth rules (LS, DS), not from node types; triangles are not required.
The traditional view is that homophily drives community formation.
Hypothesis under test: in early/small-scale societies, network-driven formation dominated.

**Neutral stance.** Modelling arguments that homophily is ancient (e.g. McElreath,
Boyd & Richerson 2003) are hypotheses, not evidence. Neither mechanism is the prior.

## Discriminating predictions

| Prediction | Homophily | Network-driven |
|---|---|---|
| Attributes predict community labels | yes | weakly / not |
| Structure beyond attribute blocks | no | yes |
| Path dependence (same population, different history) | same partition | different partitions |
| Fission line | follows attributes | follows pre-fission network |
| Temporal order | similarity -> tie | tie -> similarity |

Scale prediction from the book: local-rule communities become detectable only above
the Ramsey community number, r_kappa ~ 80-100 nodes (SBM, eps=0.05) for LS/DS.
Below that, communities in a small group are not expected from local rules alone.

Identification caveats: homophily vs contagion confounded in snapshots
(Shalizi & Thomas 2011); latent attributes; space is the dominant prehistoric driver
(always use a spatial null); kinship is a genealogy (network), not a plain attribute;
material-culture similarity networks are homophily networks by construction.

## Workstreams (run in parallel)

| Dir | Option | Data |
|---|---|---|
| `0_power/` | Synthetic power study: which statistics recover the network/homophily mixing at realistic n | simulation (book models + attribute SBM) |
| `1_fission/` | Fission natural experiments | Ngogo, Gombe, Amboseli baboons, Yanomamö, Hutterites |
| `2_adna/` | aDNA IBD networks vs material culture, temporal order | AADR, ancIBD output; Iberian Bronze Age (rosascasals2027) etc. |
| `3_foragers/` | Forager contact networks; attribute vs closure models | Agta, BaYaka, Hadza |
| `4_timeseries/` | Archaeological interaction time series, dynamic signatures vs spatial null | SWSN / cyberSW; Rosas-Casals Iberia |

Each subdirectory keeps a `NOTES.md` (what was found, what was run, status, next steps).
Downloaded data live in `<dir>/data/` and are treated as untrusted (scripts outside, `python -I`).
graph-tool: `~/anaconda3/envs/gt/bin/python` (2.97).

## Status (2026-11-10, end of day 1)

Details in each `NOTES.md`. Queue: `run_queue.sh`, log `queue.log`.

- **0_power.** The dyadic logit and the description-length gain from attributes discriminate even at n=20. SBM-based alignment and path dependence are weak at every n. P(B>1) for LS is 0, .23, .57, 1 at n=20, 35, 45, ≥100 (r_kappa ≈ 81 confirmed). Contagion inverts every snapshot statistic, so only temporal order identifies. Closure beyond the measured attributes = locality OR latent homophily.
- **1_fission (Ngogo, n=126).** The pre-split network predicts the split: spectral accuracy .86 (.90 for the 2012–14 window), against .56 for the null. Sex and age are at chance, and the cut runs across sex. Median GPS predicts it perfectly, so space and the network are confounded. Kinship is not public. Amboseli needs a manual download.
- **2_adna (CE, ~550 individuals).** Same culture goes with IBD ties: odds ratio ~2.6 above the spatial null, which is symmetric and does not discriminate. The temporal-order asymmetry (network-first z=3.1) did NOT survive robustness: degree-preserving swap nulls, archaeology-only labels, and the signal sits in one transition (Únětice). Not identifiable at this sampling.
- **3_foragers.** Arang Dak (n=108), leak-fixed: degree+closure AUC .80–.83, attr+degree .82, attr+degree+closure .85. Closure and attributes are complementary; the weak SBM blocks follow the kinship network. Agta camps are too small. Hadza (no attributes): proximity predicts friendship (.75–.78), and friend-closure adds to degree (.49 → .70). Four Dryad sets need a manual download.
- **4_timeseries.**
  - Distance-decay nulls already produce B≥2 at full slice size, so space explains the existence of communities. Observed Σ exceeds the spatial null in some slices (Chaco 1100: .47 vs .04).
  - In spatially contiguous (kNN) growth subsets, observed P(B≥2) reaches .6–.8 at 30–65 sites, against 90–130 for the spatial nulls. In random subsets the observed data and the nulls are similar. Similarity networks, so latent ceramic attributes cannot be excluded.
  - Iberia: B=1 for every phase (dense, n ≤ 80, as the power study predicts); weak distance decay; MBA form homophily +.43 (p=.008).
  - Dynamics (200 nulls): convergence inside communities runs forward in time (fwd−bwd > 0 in 9/10 pairs; Chaco stationary null p ≤ .03 in 5/8, incl. lag-2 pairs), above the geography null. This is the tie→similarity arrow. Similarity→later co-membership is also present. Latent cluster-level tradition not excluded; no single pair survives Holm (smallest adjusted p 0.08), but the joint stationary null over all 8 Chaco pairs gives p < 0.002 (lag-2 only p = 0.01); no distance decay within communities (contact test p = 0.19). Ngogo: at matched space and sex pair, grooming is no stronger within future groups (−0.04, CI −0.14 to 0.03), so the fission is spatial. Arang Dak: kin-network structure adds +0.012 AUC beyond r; support closure beyond the genealogy is negligible. Write-up: `manuscript.tex` (J. R. Soc. Interface).
