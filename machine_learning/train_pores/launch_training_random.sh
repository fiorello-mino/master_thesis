# !/bin/bash

python3 /home/fiorello/CRANE_bc/train_bc.py \
	--device  'cuda:0' \
	--padding 'circular' 'circular' \
	--size 64 \
	--seed 666 \
	--epochs 500 \
	--nocrop \
	--bias \
	--lr 5e-5 \
	--batch 3 \
	--weightd 0e-5 \
	--train_set 'train_set_64.txt' \
	--valid_set 'valid_set_64.txt' \
	--id 'random' \
	--logfreq 1 \
	--kernel_size 5 \
	--hidden 3 \
	--channels 16 \
	--nproc 4 \
	--num_params 0 \
	--subseq_min 1 \
	--subseq_max 49 \
	--reflection \
	--noise_reg 0.0125 \
	--dual \
	--divergence \
	--rotation90 \
	--massW 0.0 \
	--coeffE 0e-1 \
	--coeffG 0.0 \
        --eps 0.078125 \
	--dx 0.015873015873 \
	--ramp \
	--ramp_length 48
	#--reload_model '/home/fiorello/master_thesis/machine_learning/train_mod/train_logs/coeffE1e-1/model/epoch_476.pt'
	#--reload_model '/home/fiorello/master_thesis/machine_learning/train_pores/train_logs/bc_y2/model/epoch_20.pt'
