# !/bin/bash

python3 /home/fiorello/CRANE_bc/train.py \
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
	--id 'bc_y' \
	--logfreq 1 \
	--kernel_size 5 \
	--hidden 2 \
	--channels 16 \
	--nproc 4 \
	--num_params 0 \
	--subseq_min 1 \
	--subseq_max 49 \
	--reflection \
	--noise_reg 0 \
	--rotation90 \
	--divergence \
	--dual \
	--massW 2.0 \
	--coeffE 1e-1 \
	--coeffG 1e-5 \
        --eps 0.078125 \
	--dx 0.015625 \
	--ramp \
	--ramp_length 48 
	#--reload_model '/home/fiorello/master_thesis/machine_learning/train_mod/train_logs/coeffE1e-1/model/epoch_476.pt'
