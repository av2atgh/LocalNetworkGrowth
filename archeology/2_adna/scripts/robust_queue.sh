#!/bin/zsh
# Robustness runs for the cross-lagged test (sequential, single process; waits while load > 8). From 2_adna/.
cd ${0:A:h}/..
w() { while (( $(sysctl -n vm.loadavg | awk '{printf "%d", $2}') > 8 )); do sleep 30; done; }
r() { w; echo "$(date +%T) $1"; scripts/run.sh scripts/lagged2.py results "$@" > results/lagged2_$1.log 2>&1; }
r R0_base                --nn 100
r R1_noself              --nn 100 --noself
r R2_arch                --nn 100 --noself --labels arch
r R3_drop7000            --nn 100 --noself --drop 7000
r R4_drop4500            --nn 100 --noself --drop 4500
r R5_drop4000            --nn 100 --noself --drop 4000
r R6_drop3500            --nn 100 --noself --drop 3500
r R7_dropall             --nn 100 --noself --drop 7000 4500 4000 3500
r R8_weak                --nn 100 --noself --tie 12-20
r R9_combined            --nn 100 --noself --labels arch --drop 7000 4500 4000 3500 --tie 12-20
echo "$(date +%T) done"
