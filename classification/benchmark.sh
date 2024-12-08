export ROOT_PATH=$PWD
# dataset
export settings=(reset_each_shift continual correlated)
export dataset=(imagenet_c imagenet_d)

NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
GPUS=($(seq 0 $((NUM_GPUS-1))))

methods=("$1")
architectures=("$2") # (resnet50 vit_b_16 swin_b d2v)
seeds=($3)

options=("$4")
tag="$5"

host_name="$6"

for setting in ${settings[*]}; do
	if [ "$setting" = "correlated" ] || [ "$setting" = "mixed_correlated" ]; then
		deltas=("0.0" "0.1")
	else
		deltas=("1.0")
	fi
	for delta in ${deltas[*]}; do
	for method in ${methods[*]}; do
	if [ "$method" = "sar" ]; then
		MIXED_PRECISION=False
	else
		MIXED_PRECISION=True
	fi
	# ================== #
	# start default loop #
	# ================== #
	for ds in ${dataset[*]}; do
		if [ "$ds" = "imagenet_c" ]; then
			# Default Dataset
			imagenet_c=("cfgs/imagenet_c/${method}.yaml")
			for arch in ${architectures[*]}; do
				if [ "$arch" = "d2v" ]; then
					lr=1.0e-5
				else
					lr=2.5e-4
				fi
				(
				trap 'kill 0' SIGINT; \
				for seed in ${seeds[*]}; do
					for var in "${imagenet_c[@]}"; do
						save_dir="./output/${setting}/${arch}/${delta}_${method}_${ds}_seed${seed}"
						rm -rf $save_dir
						CUDA_VISIBLE_DEVICES=${GPUS[i % ${NUM_GPUS}]} python test_time.py --cfg $var \
							SETTING $setting RNG_SEED $seed TEST.DELTA_DIRICHLET $delta \
							MODEL.ARCH $arch OPTIM.LR $lr MIXED_PRECISION $MIXED_PRECISION \
							SAVE_DIR $save_dir $options & \
						i=$((i + 1))
					done
				done
				wait
				)
			done
		else
			imagenet_others=("cfgs/imagenet_others/${method}.yaml")
			for arch in ${architectures[*]}; do
				if [ "$arch" = "d2v" ]; then
					lr=1.0e-5
				else
					lr=2.5e-4
				fi
				(
				trap 'kill 0' SIGINT; \
				for seed in ${seeds[*]}; do
					for var in "${imagenet_others[@]}"; do
						save_dir="./output/${setting}/${arch}/${delta}_${method}_${ds}_seed${seed}"
						rm -rf $save_dir
						CUDA_VISIBLE_DEVICES=${GPUS[i % ${NUM_GPUS}]} python test_time.py --cfg $var \
							SETTING $setting RNG_SEED $seed TEST.DELTA_DIRICHLET $delta \
							MODEL.ARCH $arch CORRUPTION.DATASET $ds OPTIM.LR $lr MIXED_PRECISION $MIXED_PRECISION \
							SAVE_DIR $save_dir $options & \
						i=$((i + 1))
					done
				done
				wait
				)
			done
		fi
		res_file=./output/${setting}/${setting}_${method}_${ds}.res
		python summary_results.py --root_path $PWD \
			--setting ${setting} --dataset ${ds} \
			--method ${method} --models ${architectures[*]} --tag ${ds} \
			--seeds ${seeds[*]} --deltas $delta \
			> ${res_file}
		python $ROOT_PATH/../../dmail.py \
			--subject "[${host_name}] PIKA ${method}_${setting}${delta}_${ds}_${tag}" \
			--body "$(cat ${res_file})"
	done
	# ================ #
	# end default loop #
	# ================ #
	done
	done
done