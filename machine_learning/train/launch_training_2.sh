# !/bin/bash

python3 /home/fiorello/CRANE/train.py \
	--device  'cuda:0' \
	--padding 'circular' \
	--size 64 \
	--seed 666 \
	--epochs 400 \
	--nocrop \
	--bias \
	--lr 8e-4 \
	--batch 8 \
	--weightd 1e-5 \
	--train_set 'training_set_64_2_from_10.txt' \
	--valid_set 'validation_set_64_2_from_10.txt' \
	--id 'lr_8e-4_b_8_hl_4_ch40_seq40' \
	--logfreq 1 \
	--kernel_size 5 \
	--hidden 4 \
	--channels 40 \
	--nproc 4 \
	--num_params 0 \
	--subseq_min 1 \
	--subseq_max 40 \
	--reflection \
	--noise_reg 0.0 \
	--rotation90 \
	--divergence \
	--dual \
	--ramp \
	--ramp_length 20
	#--reload_model '/home/fiorello/master_thesis/machine_learning/train/train_logs/lr_5e-5_hl_3_2_from_10/model/epoch_279.pt'
