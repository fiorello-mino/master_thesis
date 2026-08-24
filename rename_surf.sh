#!/usr/bin/env bash

set -u

BASE="."
FIRST=0
LAST=149

for folder_number in $(seq "$FIRST" "$LAST"); do
    folder=$(printf '%03d' "$folder_number")
    folder_path="$BASE/$folder"

    if [[ ! -d "$folder_path" ]]; then
        echo "[MANCANTE] Cartella $folder"
        continue
    fi

    found=0

    for i in $(seq 0 200); do
        integer=$((i / 10))
        decimal=$((i % 10))

        old_name="surf_${integer}.${decimal}.npy"
        new_name=$(printf '%03d.npy' "$i")

        old_path="$folder_path/$old_name"
        new_path="$folder_path/$new_name"

        if [[ -f "$old_path" ]]; then
            found=1

            if [[ -e "$new_path" ]]; then
                echo "[SKIP] $folder/$new_name esiste già"
            else
                mv -- "$old_path" "$new_path"
                echo "[OK]   $folder/$old_name -> $folder/$new_name"
            fi
        fi
    done

    if [[ "$found" -eq 0 ]]; then
        echo "[INFO] $folder non contiene file surf_*.npy"
    fi
done