deltas=("1.0e-11" "2.0e-11" "3.0e-11" "4.0e-11")
for delta in ${deltas[*]}; do
    bash benchmark.sh "ssa" "d2v" "1" "SSA.EPS $delta SSA.DUAL_KF True" DKF_eps${delta}_realtime g1
done