deltas=("1.0" "1.5" "2" "3")
for delta in ${deltas[*]}; do
    bash benchmark.sh "ssa" "d2v" "1 2 3 4" "SSA.SS $delta SSA.DUAL_KF True" DKF_SS${delta} g1
done