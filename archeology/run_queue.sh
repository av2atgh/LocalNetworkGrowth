#!/bin/zsh
# Sequential queue for the archeology workstreams (2026-11-10).
# One process at a time, single-threaded, nice 15; waits while 1-min load > 8.
# A failing job is logged and skipped; the queue keeps going.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
A=~/av2atg/LocalNetworkGrowth/archeology
PY=~/anaconda3/envs/gt/bin/python
LOG=$A/queue.log

wait_load() {
  while :; do
    l=$(sysctl -n vm.loadavg | awk '{print $2}')
    (( l < 8 )) && return
    echo "$(date +%T) load $l, waiting" >> $LOG; sleep 60
  done
}
job() {  # job <dir> <label> <command...>
  local d=$1 lab=$2; shift 2
  wait_load
  echo "$(date +%T) START $lab" >> $LOG
  ( cd $A/$d && nice -n 15 "$@" ) >> $LOG 2>&1
  echo "$(date +%T) END $lab rc=$?" >> $LOG
}

# 1. aDNA temporal order (the identifiable test per 0_power)
job 2_adna adna-Q1-smoke  scripts/run.sh scripts/lagged.py results CE 100 2
job 2_adna adna-Q2-R100   scripts/run.sh scripts/lagged.py results CE 100 200
# 2. foragers
job 3_foragers for-Q1-arangdak zsh -c "$PY -I scripts/koster_analysis.py data/mayangna_koster2018/rsos172159_si_002.csv results > results/koster_log_v2.txt 2>&1"
job 3_foragers for-Q2-hadza    zsh -c "$PY -I scripts/hadza_analysis.py data/hadza_fedurek2024 results > results/hadza_log.txt 2>&1"
# 3. fission reruns
job 1_fission fis-1 $PY -I scripts/ngogo_fission.py data/ngogo results/ngogo_codemap.csv results 2011 2014
job 1_fission fis-2 $PY -I scripts/ngogo_fission.py data/ngogo results/ngogo_codemap.csv results 2012 2014
job 1_fission fis-3 $PY -I scripts/ngogo_temporal.py data/ngogo results/ngogo_codemap.csv results/ngogo_temporal.csv
job 1_fission fis-4 $PY -I scripts/ngogo_sbm_bysex.py data/ngogo results/ngogo_nodes_2011_2014.csv
# 4. time series: scale with nulls, full Iberia
job 4_timeseries/scripts ts-scale075 zsh -c "THR=0.75 $PY -I scale_detect.py > ../results/scale_thr0.75.log 2>&1 && $PY -I summarize_scale.py 0.75"
job 4_timeseries/scripts ts-iberia   zsh -c "$PY -I iberia.py > ../results/iberia_full.txt 2>&1"
# 5. aDNA sensitivity
job 2_adna adna-Q3-R200   scripts/run.sh scripts/lagged.py results CE 200 200
job 2_adna adna-Q4        scripts/run.sh scripts/dyad.py results CE 30 500 bin_d_dt_study
job 2_adna adna-Q5        scripts/run.sh scripts/align.py results CE 30 200
job 2_adna adna-Q6        scripts/run.sh scripts/align.py results CE 12 200
# 6. time-series sensitivity (dynamics.py needs a smoke test first; not queued here)
job 4_timeseries/scripts ts-scale085 zsh -c "THR=0.85 $PY -I scale_detect.py > ../results/scale_thr0.85.log 2>&1 && $PY -I summarize_scale.py 0.85"
echo "$(date +%T) QUEUE DONE" >> $LOG
