#!/bin/bash
#SBATCH --job-name=surf_cone
#SBATCH --output=logs/surf_cone_%A_%a.out
#SBATCH --error=logs/surf_cone_%A_%a.err
#SBATCH --nodelist=moseg3,moseg4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --array=0-15%16

set -euo pipefail

# Crea la directory dei log se non esiste.
mkdir -p logs

# Costruisce una lista deterministica dei 16 file init:
# task 0 = primo file in ordine alfabetico,
# task 15 = ultimo file in ordine alfabetico.
mapfile -t DAT_FILES < <(
    find init/cone -maxdepth 1 -type f -name "*.dat" | sort
)

# Controllo di sicurezza.
if [[ "${#DAT_FILES[@]}" -ne 16 ]]; then
    echo "ERRORE: attesi 16 file .dat in init/cone, trovati ${#DAT_FILES[@]}."
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
echo "Start:        $(date)"
echo "=================================================="

bin/surf "${DAT_FILE}"

echo "=================================================="
echo "Completed: ${SIM_NAME}"
echo "End:       $(date)"
echo "=================================================="