deltas=("0.02" "0.03")
for delta in ${deltas[*]}; do
    bash benchmark.sh "ssa" "d2v" "1 2 3 4" "SSA.KAPPA_2 $delta " DKF_${delta} a1
done