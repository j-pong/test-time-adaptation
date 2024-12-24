deltas=("0.8e-12" "1e-12" "1.2e-12" "1.4e-12" "1.6e-12")
for delta in ${deltas[*]}; do
    bash benchmark.sh "ssa" "d2v" "1 2 3 4" "SSA.EPS $delta SSA.DUAL_KF True" DKF_eps${delta} g1
done