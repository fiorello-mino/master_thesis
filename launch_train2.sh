#!/usr/bin/env bash

TRAIN_SET="/home/fiorello/master_thesis/machine_learning/train/training_set_64_from_5.txt"
VALID_SET="/home/fiorello/master_thesis/machine_learning/train/validation_set_64_from_5.txt"

DEVICE="cuda"
RUN_ID="train_normal"

EPOCHS=100
BATCH=1
LR=5e-4
HIDDEN=2
CHANNELS=35
KERNELSIZE=3
SUBSEQMIN=1
SUBSEQMAX=99
NPROC=1
MASSW=2.0
DROPOUTPROB=0.25

python train.py \
  --epochs "$EPOCHS" \
  --batch "$BATCH" \
  --device "$DEVICE" \
  --train_set "$TRAIN_SET" \
  --valid_set "$VALID_SET" \
  --id "$RUN_ID" \
  --lr "$LR" \
  --hidden "$HIDDEN" \
  --channels "$CHANNELS" \
  --kernelsize "$KERNELSIZE" \
  --subseqmin "$SUBSEQMIN" \
  --subseqmax "$SUBSEQMAX" \
  --nproc "$NPROC" \
  --massW "$MASSW" \
  --padding circular \
  --size 64 \
  --dropout \
  --dropoutprob "$DROPOUTPROB"