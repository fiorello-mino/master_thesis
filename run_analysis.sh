#!/bin/sh
set -e

# 1. Test esterno del miglior modello ottenuto dal training
LOG_DIR_NAME="coeffE0"
SEQUENCE_TABLE="test_set.txt"
OUTPUT_FOLDER="/data/fiorello/ext_test_64/lr_5e-5_hl_3_tr_5"
SCRIPT_PATH="./test_script.py"

NUM_PNG=100
NUM_NPY=100
NUM_EVO=25000

DT=1e-6
STEPS_PER_SAVE=100000
STARTING_FRAME=10

OVERWRITE="replace"

python "$SCRIPT_PATH" \
  --log-dir-name "$LOG_DIR_NAME" \
  --sequence-table "$SEQUENCE_TABLE" \
  --output-folder "$OUTPUT_FOLDER" \
  --num-png "$NUM_PNG" \
  --num-npy "$NUM_NPY" \
  --num-evo "$NUM_EVO" \
  --dt "$DT" \
  --steps-per-save "$STEPS_PER_SAVE" \
  --starting-frame "$STARTING_FRAME" \
  --overwrite "$OVERWRITE"

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
LABEL_NEW="$LOG_DIR_NAME"
OUT_DIR_PLOT="/data/fiorello/plots/prec_vs_new_2/"

mkdir -p "$OUT_DIR_PLOT"

gnuplot -e "errorsA='$ERRORS_BEST'; 
            errorsB='$ERRORS_NEW'; 
            medianA='$MEDIAN_ERROR_BEST'; 
            medianB='$MEDIAN_ERROR_NEW'; 
            labelA='$LABEL_BEST';
            labelB='$LABEL_NEW';
            outdir='$OUT_DIR_PLOT'" compare_models.gnu