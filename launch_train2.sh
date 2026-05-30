#!/usr/bin/env bash

# =============== CONFIG ==================

PYTHON=python3
TRAIN_SCRIPT="/home/fiorello/CRANE/train.py"

DEVICE="cuda:0"

TRAINSET="/home/fiorello/master_thesis/machine_learning/train/training_set_64_from_5.txt"
VALIDSET="/home/fiorello/master_thesis/machine_learning/train/validation_set_64_from_5.txt"

# iperparametri principali
EPOCHS=400         # 300–500, scegli tu; qui metto 400 come via di mezzo
BATCH=6
LR=5e-4
WEIGHTD=1e-5
SEED=666

SIZE=64
PADDING="circular"
HIDDEN=3           # hidden layers ConvGRU
CHANNELS=32        # channels per layer
KERNELSIZE=5

NPROC=4
NUMPARAMS=0
SUBSEQMIN=1
SUBSEQMAX=64
NOISEREG=0.01      # ~0.01–0.0125
RAMPLENGTH=50
STARTRAMP=0

RUN_ID="CH_h${HIDDEN}_c${CHANNELS}_b${BATCH}_lr${LR}_seq${SUBSEQMAX}"

# =============== RUN TRAINING ===============

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
    --ramplength "$RAMPLENGTH" \
    --startramp "$STARTRAMP"
#   --reloadmodel "/path/to/old/model.pt"