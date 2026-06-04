# !/bin/bash

python3 /home/fiorello/CRANE/train.py \
	--device  'cuda:0' \
	--padding 'circular' \
	--size 64 \
	--seed 666 \
	--epochs 300 \
	--nocrop \
	--bias \
	--lr 1e-4 \
	--batch 4 \
	--weightd 2e-5 \
	--train_set 'training_set_64_2_from_10.txt' \
	--valid_set 'validation_set_64_2_from_10.txt' \
	--id 'lr1e-4_b4_k5_hl2_ch16_seq20_ramp5_wd2e-5' \
	--logfreq 1 \
	--kernel_size 5 \
	--hidden 2 \
	--channels 16 \
	--nproc 4 \
	--num_params 0 \
	--subseq_min 5 \
	--subseq_max 20 \
	--reflection \
	--noise_reg 0 \
	--rotation90 \
	--divergence \
	--dual \
	--ramp_length 5 \
	--ramp 
	#--reload_model '/home/fiorello/master_thesis/machine_learning/train/train_logs/lr_5e-5_hl_3_2_from_10/model/epoch_279.pt'
