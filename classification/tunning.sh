deltas=("0.025" "0.03" "0.035" "0.04")
for delta in ${deltas[*]}; do
    bash benchmark.sh "ssa" "d2v" "1 2 3 4" "SSA.KAPPA_2 $delta " DKF_${delta} g1
done