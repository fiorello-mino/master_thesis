#!/bin/bash
#SBATCH --job-name=surf_square
#SBATCH --output=logs/surf_square_%A_%a.out
#SBATCH --error=logs/surf_square_%A_%a.err
#SBATCH --nodes=moseg3,moseg4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --array=0-97

set -euo pipefail

# Lo script deve essere sottomesso dalla directory del progetto:
#   ./bin/surf
#   ./init/square/*.dat

mkdir -p logs

# Legge in modo ordinato tutti i 98 file .dat
mapfile -t DAT_FILES < <(find init/square -maxdepth 1 -type f -name "*.dat" | sort)

N_FILES=${#DAT_FILES[@]}

if [[ "${N_FILES}" -ne 98 ]]; then
    echo "ERRORE: attesi 98 file .dat in init/square, trovati ${N_FILES}."
    exit 1
fi

DAT_FILE="${DAT_FILES[$SLURM_ARRAY_TASK_ID]}"
SIM_NAME="$(basename "${DAT_FILE}" .dat)"

echo "=============================================="
echo "SLURM job ID:        ${SLURM_JOB_ID}"
echo "Array task ID:       ${SLURM_ARRAY_TASK_ID}"
echo "Simulation:          ${SIM_NAME}"
echo "Input file:          ${DAT_FILE}"
echo "Host:                $(hostname)"
echo "Start:               $(date)"
echo "=============================================="

# Lancia la simulazione.
# DAT_FILE è già, ad esempio, init/square/<sim_name>.dat
bin/surf "${DAT_FILE}"

echo "=============================================="
echo "Finished: ${SIM_NAME}"
echo "End:      $(date)"
echo "=============================================="