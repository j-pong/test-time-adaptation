deltas=("2.5e-04")
for delta in ${deltas[*]}; do
    bash benchmark.sh "roid cmf ssa" "vit_b_16 swin_b" "1 2 3 4" "OPTIM.LR $delta" DKF_LR${delta}_model g1
done