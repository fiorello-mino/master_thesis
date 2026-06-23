#!/bin/bash
#SBATCH --job-name=surf_array
#SBATCH --output=logs/surf_%A_%a.out
#SBATCH --error=logs/surf_%A_%a.err
#SBATCH --array=0-199%40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --nodelist=moseg3,moseg4

INPUT_DIR=/home/fiorello/init_files

ID=$(printf "%03d" ${SLURM_ARRAY_TASK_ID})
INPUT_FILE=${INPUT_DIR}/${ID}.dat

bin/surf "${INPUT_FILE}"