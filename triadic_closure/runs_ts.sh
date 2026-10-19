#!/bin/sh
# Long single trajectories for the condensation transition; time series only.
cd "$(dirname "$0")"
export TSONLY=1
for b in 0.02 0.04 0.06 0.07 0.08 0.10 0.15; do ./tc r $b 1000000 1 data/ts/r_b${b}_n1e6_s1; done
for u in 0.05 0.1 0.15 0.2 0.3 0.4 0.5; do ./tc w $u 10000000 1 data/ts/w_u${u}_n1e7_s1; echo w $u done; done
./tc p 0.5 10000000 1 data/ts/p_u0.5_n1e7_s1
for m in c f l; do ./tc $m 0.5 10000000 1 data/ts/${m}_u0.5_n1e7_s1; done
