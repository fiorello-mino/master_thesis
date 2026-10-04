#!/bin/sh
set -e

OUTPUT_FOLDER="/scratch/fiorello/test3D/square/train1/fv_c24_mse"
OUTPUT_FOLDER_BEST="/scratch/fiorello/test3D/square/train1/fv_mse"

LABEL_NEW="fv ch24 mse"
LABEL_BEST="fv ch16 mse"

ERRORS_NEW="$OUTPUT_FOLDER/errors.txt"
MEDIAN_NEW="$OUTPUT_FOLDER/medians.txt"

ERRORS_BEST="$OUTPUT_FOLDER_BEST/errors.txt"
MEDIAN_BEST="$OUTPUT_FOLDER_BEST/medians.txt"

OUT_DIR_PLOT="/scratch/fiorello/test3D/square/plots/fv ch24 mse *vs* fv ch16 mse/"

mkdir -p "$OUT_DIR_PLOT"

gnuplot -e "
errorsA='$ERRORS_BEST';
errorsB='$ERRORS_NEW';

medianA='$MEDIAN_BEST';
medianB='$MEDIAN_NEW';

labelA='$LABEL_BEST';
labelB='$LABEL_NEW';

outdir='$OUT_DIR_PLOT';
" compare_models2.gnu

