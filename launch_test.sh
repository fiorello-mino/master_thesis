#!/bin/sh

LOG_DIR_NAME="lr_5e-5_hl_3_tr_5"
SEQUENCE_TABLE="/data/fiorello/ext_test_64/testing_set_from_5_half_total_time.txt"
OUTPUT_FOLDER="/data/fiorello/ext_test_64/lr_5e-5_hl_3_tr_5"
SCRIPT_PATH="./test_script.py"

IMG_SIZE=64
MIN_SEQ=1
HIDDEN_UNITS=3
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

EPSILON=0.078125
DX=0.015625

python "$SCRIPT_PATH" \
  --log-dir-name "$LOG_DIR_NAME" \
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
