# !/bin/bash

python3 /home/fiorello/CRANE/train.py \
	--device  'cuda:0' \
	--padding 'circular' \
	--size 64 \
	--seed 666 \
	--epochs 500 \
	--nocrop \
	--bias \
	--lr 5e-5 \
	--batch 3 \
	--weightd 0e-4 \
	--train_set 'training_set_from_start.txt' \
	--valid_set 'validation_set_from_start.txt' \
	--id 'test_lr_5e-5_hl_3_from_start' \
	--logfreq 1 \
	--kernel_size 5 \
	--hidden 3 \
	--channels 16 \
	--nproc 4 \
	--num_params 0 \
	--subseq_min 1 \
	--subseq_max 99 \
	--ramp \
	--ramp_length 98 \
	--reflection \
	--noise_reg 0.0125 \
	--rotation90 \
	--divergence \
	--dual
	#--reload_model '/home/fiorello/master_thesis/machine_learning/train/train_logs/test_lr_5e-5_hl_3/model/epoch_487.pt' 
