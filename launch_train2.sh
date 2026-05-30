#!/usr/bin/env bash

# ================== CONFIGURAZIONE ==================

PYTHON=python3
TRAIN_SCRIPT="/home/fiorello/CRANE/train.py"

DEVICE="cuda:0"
RUN_ID="lr_5e-5_hl_3_tr_5"

TRAINSET="/home/fiorello/master_thesis/machine_learning/train/training_set_64_from_5.txt"
VALIDSET="/home/fiorello/master_thesis/machine_learning/train/validation_set_64_from_5.txt"

EPOCHS=500
BATCH=3
LR=5e-5
WEIGHTD=0e-4
SEED=666

SIZE=64
PADDING="circular"
HIDDEN=3
CHANNELS=16
KERNELSIZE=5

NPROC=4
NUMPARAMS=0
SUBSEQMIN=1
SUBSEQMAX=49
NOISEREG=0.0125
RAMPLENGTH=48

# ================== LANCIO TRAINING ==================

$PYTHON "$TRAIN_SCRIPT" \
    --device "$DEVICE" \
    --padding "$PADDING" \
    --size "$SIZE" \
    --seed "$SEED" \
    --epochs "$EPOCHS" \
    --nocrop \
    --bias \
    --lr "$LR" \
    --batch "$BATCH" \
    --weightd "$WEIGHTD" \
    --trainset "$TRAINSET" \
    --validset "$VALIDSET" \
    --id "$RUN_ID" \
    --logfreq 1 \
    --kernelsize "$KERNELSIZE" \
    --hidden "$HIDDEN" \
    --channels "$CHANNELS" \
    --nproc "$NPROC" \
    --numparams "$NUMPARAMS" \
    --subseqmin "$SUBSEQMIN" \
    --subseqmax "$SUBSEQMAX" \
    --reflection \
    --noisereg "$NOISEREG" \
    --rotation90 \
    --divergence \
    --dual \
    --ramp \
    --ramplength "$RAMPLENGTH"
#   --reloadmodel "/home/fiorello/master_thesis/machine_learning/train/train_logs/lr_5e-5_hl_3_2_from_10/model/epoch_279.pt"