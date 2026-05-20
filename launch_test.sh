#!/bin/sh

MODEL_PATH="/home/fiorello/master_thesis/machine_learning/train/train_logs/test_lr_5e-5/model/epoch_480.pt"
SEQUENCE_TABLE="/data/fiorello/external_test_128_random/testing_set.txt"
OUTPUT_FOLDER="/data/fiorello/external_test_128_random/test_lr_5e-5"
SCRIPT_PATH="./test_script.py"

IMG_SIZE=64
MIN_SEQ=1
HIDDEN_UNITS=2
INPUT_CHANNELS=1
OUTPUT_CHANNELS=1
HIDDEN_CHANNELS=16
KERNEL_SIZE=5
PADDING_MODE="circular"
NUM_PARAMS=0

DELTA_PNG=1
NUM_PNG=100
NUM_NPY=100
NUM_VTK=0
NUM_EVO=25000

OVERWRITE="delete"
LOG_LEVEL="INFO"

EPSILON=1.0
DX=1.0

python "$SCRIPT_PATH" \
  --model-path "$MODEL_PATH" \
  --sequence-table "$SEQUENCE_TABLE" \
  --output-folder "$OUTPUT_FOLDER" \
  --img-size "$IMG_SIZE" \
  --use-cuda \
  --delta-png "$DELTA_PNG" \
  --min-seq "$MIN_SEQ" \
  --hidden-units "$HIDDEN_UNITS" \
  --input-channels "$INPUT_CHANNELS" \
  --output-channels "$OUTPUT_CHANNELS" \
  --hidden-channels "$HIDDEN_CHANNELS" \
  --kernel-size "$KERNEL_SIZE" \
  --padding-mode "$PADDING_MODE" \
  --bias \
  --divergence \
  --num-params "$NUM_PARAMS" \
  --num-png "$NUM_PNG" \
  --num-vtk "$NUM_VTK" \
  --num-npy "$NUM_NPY" \
  --num-evo "$NUM_EVO" \
  --overwrite "$OVERWRITE" \
  --log-level "$LOG_LEVEL" \
  --epsilon "$EPSILON" \
  --dx "$DX"