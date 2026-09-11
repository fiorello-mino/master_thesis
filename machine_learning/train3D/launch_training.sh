# !/bin/bash

python3 /home/fiorello/CRANE/train.py \
	--device  'cuda:0' \
	--threeD \
	--vtk \
	--padding  'reflect' \
	--seed 666 \
	--epochs 500 \
	--nocrop \
	--bias \
	--lr 5e-5 \
	--batch 1 \
	--weightd 0e-5 \
	--train_set 'train_set.txt' \
	--valid_set 'valid_set.txt' \
	--id 'E1e-1' \
	--logfreq 1 \
	--kernel_size 5 \
	--hidden 3 \
	--channels 16 \
	--nproc 4 \
	--divergence \
	--num_params 0 \
	--subseq_min 1 \
	--subseq_max 19 \
	--reflection \
	--noise_reg 0.025 \
	--massW 0.0 \
	--ramp \
	--ramp_length 18 \
	--epsilon 0.1 \
	--dt 5e-3 \
	--dx 0.025 \
	--dy 0.025 \
	--dz 0.025 \
        --w_energy 0.1 \
	--w_bounds 0.0	
	#--conservative \
	#--reload_model '/home/fiorello/master_thesis/machine_learning/train_pores/train_logs/bc_y2/model/epoch_20.pt'
	#--dual \
	#--rotation90 \
