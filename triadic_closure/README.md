# Triadic closure with arrival, deletion, closure and opening

Code and data for `manuscript.tex`, "Phase diagram of networks grown with triadic closure rules".

## Programs

- `tc.c` — the simulation. `cc -O3 -o tc tc.c`, then
  `./tc MODE PARAM N SEED OUTPREFIX [PARAM2 [PARAM3]]`.
  Modes `l` (LS(n,1)), `w`, `p`, `c`, `f` (closure rules) and `r` (per-triad rates).
  PARAM is the closure probability per event u₊, PARAM2 the opening probability u₋,
  PARAM3 the deletion probability α₋/Z.
- `trclust.c` — edge-connected triangle clusters (P_△, P_•, P_{•–•}).
  `cc -O3 -o trclust trclust.c`, then `./trclust EDGES N`.

## Scans

| script | what | output |
|---|---|---|
| `scan.py`, `runs_ts.sh`, `runs_onset.sh`, `onset.py` | rules without opening; onset of condensation | `data/scan.csv`, `data/onset.csv`, `data/ts/` |
| `scan_rm.py`, `runs_rm_ts.sh`, `cross.py`, `phase.py` | rule (w) with opening; tree transition | `data/scan_rm.csv`, `data/cross.csv`, `data/phase.csv` |
| `opgrid.py`, `bnd.py` | triangle clusters across opening and deletion; boundary | `data/opgrid.csv`, `data/bnd.csv` |
| `delgrid.py` | deletion without opening | `data/delgrid.csv` |
| `rulesclust.py` | triangle clusters of each rule | `data/rulesclust.csv` |
| `comm.py` | communities (run with graph-tool) | `data/comm.csv` |

`mkfigs.py` draws the figures from `data/`.
