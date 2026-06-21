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
	--weightd 2e-5 \
	--train_set 'training_set_64_2_from_10.txt' \
	--valid_set 'validation_set_64_2_from_10.txt' \
	--id 'prova_ram' \
	--logfreq 1 \
	--kernel_size 7 \
	--hidden 3 \
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
	--ramp_length 48 \
	--ramp 
	#--reload_model '/home/fiorello/master_thesis/machine_learning/train/train_logs/lr_5e-5_hl_3_2_from_10/model/epoch_279.pt'
