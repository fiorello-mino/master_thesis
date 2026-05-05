# !/bin/bash

python3 /home/fiorello/CRANE/train.py \
	--device  'cuda:0' \
	--padding 'circular' \
	--size 64 \
	--seed 666 \
	--epochs 500 \
	--nocrop \
	--bias \
	--lr 1e-4 \
	--batch 3 \
	--weightd 0e-4 \
	--train_set 'training_set.txt' \
	--valid_set 'validation_set.txt' \
	--id 'test' \
	--logfreq 1 \
	--kernel_size 5 \
	--hidden 2 \
	--channels 16 \
	--nproc 4 \
	--num_params 0 \
	--subseq_min 1 \
	--subseq_max 49 \
	--ramp \
	--ramp_length 48 \
	--reflection \
	--noise_reg 0.0125 \
	--rotation90 \
	--divergence \
	--dual
	#--reload_model 'models/epoch_361.pt' 
