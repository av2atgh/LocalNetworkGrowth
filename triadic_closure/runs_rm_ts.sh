#!/bin/sh
# (rm = 0.25 was run to 1.3e6 and stopped: the dense core makes removals slow.)
# Trajectories with triangle opening along <k> = 4 (beta_+ = alpha + beta_-); time series only.
cd "$(dirname "$0")"
export TSONLY=1
for rm in 4 2 1 0.5; do
  up=$(echo "(1+$rm)/(2+2*$rm)" | bc -l); um=$(echo "$rm/(2+2*$rm)" | bc -l)
  ./tc w $up 10000000 1 data/ts/w_k4_rm${rm}_n1e7_s1 $um; echo w $rm done
done
