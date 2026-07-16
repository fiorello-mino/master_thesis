# !/bin/bash

python3 /home/fiorello/CRANE_bc/train_bc.py \
	--device  'cuda:0' \
	--padding 'circular' 'reflect' \
	--size 128 \
	--seed 666 \
	--epochs 500 \
	--nocrop \
	--bias \
	--lr 5e-5 \
	--batch 3 \
	--weightd 0e-5 \
	--train_set 'train_set.txt' \
	--valid_set 'valid_set.txt' \
	--id 'coeffE1e-3_hl3_reload_random' \
	--logfreq 1 \
	--kernel_size 5 \
	--hidden 3 \
	--channels 16 \
	--divergence \
	--nproc 4 \
	--num_params 0 \
	--subseq_min 1 \
	--subseq_max 49 \
	--reflection \
	--noise_reg 0.0 \
	--massW 0.0 \
	--coeffE 1e-3 \
	--coeffG 0e-2 \
        --eps 0.024739583333333334 \
	--dx 0.014960629921259843 \
	--ramp \
	--ramp_length 48 \
	--reload_model '/home/fiorello/master_thesis/machine_learning/train_pores/train_logs/random/model/epoch_483.pt'
	#--conservative \
	#--reload_model '/home/fiorello/master_thesis/machine_learning/train_pores/train_logs/bc_y2/model/epoch_20.pt'
	#--dual \
	#--rotation90 \
