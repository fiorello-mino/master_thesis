#!/bin/bash

INPUT_DIR=/home/fiorello/init_files

for i in $(seq 0 199); do
    ID=$(printf "%03d" "$i")
    FILE="${INPUT_DIR}/${ID}.dat"
    sbatch --nodelist=moseg3,moseg4 --wrap="./bin/surf ${FILE}"
done