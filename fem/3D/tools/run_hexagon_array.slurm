#!/bin/bash
#SBATCH --job-name=surf_hexagon
#SBATCH --output=logs/surf_hexagon_%A_%a.out
#SBATCH --error=logs/surf_hexagon_%A_%a.err
#SBATCH --nodelist=moseg3,moseg4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --array=0-13%14

set -euo pipefail

mkdir -p logs

# Ordine deterministico: task 0 usa il primo file alfabeticamente,
# task 13 usa l'ultimo dei 14 file.
mapfile -t DAT_FILES < <(find init/hexagon -maxdepth 1 -type f -name "*.dat" | sort)

if [[ "${#DAT_FILES[@]}" -ne 14 ]]; then
    echo "ERRORE: attesi 14 file .dat in init/hexagon, trovati ${#DAT_FILES[@]}."
    exit 1
fi

DAT_FILE="${DAT_FILES[$SLURM_ARRAY_TASK_ID]}"
SIM_NAME="$(basename "${DAT_FILE}" .dat)"

echo "=================================================="
echo "Job ID:       ${SLURM_JOB_ID}"
echo "Array task:   ${SLURM_ARRAY_TASK_ID}"
echo "Node:         $(hostname)"
echo "Simulation:   ${SIM_NAME}"
echo "Input file:   ${DAT_FILE}"
echo "S