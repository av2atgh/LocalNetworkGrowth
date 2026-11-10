#!/bin/zsh
# usage: scripts/run.sh script.py args...   (single-threaded, niced; from 2_adna/)
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
exec nice -n 15 ~/anaconda3/envs/gt/bin/python -I "$@"
