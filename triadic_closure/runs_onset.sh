#!/bin/sh
# Seed-to-seed statistics of the condensation onset; time series only.
cd "$(dirname "$0")"
export TSONLY=1
for u in 0.25 0.2 0.15 0.12 0.1 0.08; do for s in 2 3 4 5 6; do
  ./tc w $u 3000000 $s data/onset/w_u${u}_n3e6_s$s; done; echo $u done; done
