# !/bin/bash

python3 /home/fiorello/CRANE/train.py \
	--device  'cuda:0' \
	--padding 'circular' \
	--size 64 \
	--seed 666 \
	--epochs 400 \
	--nocrop \
	--bias \
	--lr 5e-5 \
	--batch 8 \
	--weightd 1e-5 \
	--train_set 'training_set_64_2_from_10.txt' \
	--valid_set 'validation_set_64_2_from_10.txt' \
	--id 'lr5e-5_b8_k5_hl2_ch16_seq20_ramp5_wd1e-5' \
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
