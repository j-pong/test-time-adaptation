deltas=("1.4e-4")
for delta in ${deltas[*]}; do
    bash benchmark.sh "ssa" "d2v" "1 2 3 4" "SSA.EPS $delta SSA.DUAL_KF True" DKF_eps${delta}_realtime g1
done