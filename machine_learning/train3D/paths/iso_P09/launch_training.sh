# !/bin/bash

python3 /home/fiorello/CRANE/train.py \
	--device  'cuda:0' \
	--threeD \
	--padding  'reflect' \
	--seed 666 \
	--epochs 500 \
	--nocrop \
	--bias \
	--lr 5e-5 \
	--batch 1 \
	--weightd 0e-5 \
	--train_set 'paths.txt' \
	--valid_set 'paths.txt' \
	--id 'prova3D' \
	--logfreq 1 \
	--kernel_size 5 \
	--hidden 2 \
	--channels 16 \
	--nproc 4 \
	--divergence \
	--num_params 0 \
	--subseq_min 1 \
	--subseq_max 19 \
	--reflection \
	--noise_reg 0.0 \
	--massW 0.0 \
	--ramp \
	--ramp_length 18 \
	#--conservative \
	#--reload_model '/home/fiorello/master_thesis/machine_learning/train_pores/train_logs/bc_y2/model/epoch_20.pt'
	#--dual \
	#--rotation90 \
