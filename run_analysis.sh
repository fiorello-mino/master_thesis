#!/bin/sh
set -e

# 1. test esterno del miglior modello ottenuto dal training
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
DT=1e-6
STEPS_PER_SAVE=100000
STARTING_FRAME=10

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
  --dx "$DX" \
  --dt "$DT" \
  --steps_per_save "$STEPS_PER_SAVE" \
  --starting_frame "$STARTING_FRAME" 


# 2. Calcolo in post-process della mediana degli errori relativi e dei quartili
python energy_errors_median.py \
  --root_dir "$OUTPUT_FOLDER" \
  --dt "$DT" \
  --steps_per_save "$STEPS_PER_SAVE" \
  --starting_frame "$STARTING_FRAME"

# 3. Confronto tra i risultati di questo modello e il miglior modello precedente
OUTPUT_FOLDER_BEST="/data/fiorello/ext_test_64/lr5e-5_hl3_2_tr10"
ERRORS_BEST="$OUTPUT_FOLDER_BEST/errors.txt"
ERRORS_NEW="$OUTPUT_FOLDER/errors.txt"
MEDIAN_ERROR_BEST="$OUTPUT_FOLDER_BEST/median_energy_error.txt"
MEDIAN_ERROR_NEW="$OUTPUT_FOLDER/median_energy_error.txt"
LABEL_BEST="lr5e-5_hl3_2_tr10"
LABEL_NEW="lr5e-5_hl3_k7"
OUT_DIR_PLOT="/data/fiorello/plots/prec_vs_new_2/"

mkdir -p $OUT_DIR_PLOT

gnuplot -e "errorsA='$ERRORS_BEST'; 
            errorsB='$ERRORS_NEW'; 
            medianA='$MEDIAN_ERROR_BEST'; 
            medianB='$MEDIAN_ERROR_NEW'; 
            labelA='$LABEL_BEST';
            labelB='$LABEL_NEW';
            outdir='$OUT_DIR_PLOT'" compare_models.gnu
