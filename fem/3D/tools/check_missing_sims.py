#!/usr/bin/env python3
"""
Script per trovare simulazioni presenti in /archive/roberto/poresAMDIS/square/
ma assenti in /data/fiorello/poresAMDIS/

Struttura attesa:
  /archive/roberto/poresAMDIS/square/iso_P0X/<sim_name>/...
  /data/fiorello/poresAMDIS/iso_P0X/<sim_name>/...
"""

import os
from pathlib import Path

ARCHIVE_BASE = Path("/archive/roberto/poresAMDIS/square")
DATA_BASE = Path("/data/fiorello/poresAMDIS")

def main():
    if not ARCHIVE_BASE.exists():
        print(f"ERRORE: {ARCHIVE_BASE} non esiste o non è accessibile")
        return
    if not DATA_BASE.exists():
        print(f"ERRORE: {DATA_BASE} non esiste o non è accessibile")
        return

    # Trova tutte le cartelle pitch (iso_P06, iso_P07, ...)
    pitch_folders = sorted([
        d for d in os.listdir(ARCHIVE_BASE)
        if d.startswith("iso_P0") and (ARCHIVE_BASE / d).is_dir()
    ])

    print(f"Trovate {len(pitch_folders)} cartelle pitch: {pitch_folders}\n")

    missing = {}

    for pitch in pitch_folders:
        archive_pitch_path = ARCHIVE_BASE / pitch
        data_pitch_path = DATA_BASE / pitch

        # Simulazioni nell'archivio per questo pitch
        archive_sims = set([
            d for d in os.listdir(archive_pitch_path)
            if (archive_pitch_path / d).is_dir()
        ])

        # Se la cartella pitch non esiste in /data, tutte mancano
        if not data_pitch_path.exists():
            missing[pitch] = archive_sims
            print(f"{pitch}: cartella PITCH ASSENTE in {DATA_BASE} ({len(archive_sims)} simulazioni mancanti)")
            continue

        # Simulazioni presenti in /data per questo pitch
        data_sims = set([
            d for d in os.listdir(data_pitch_path)
            if (data_pitch_path / d).is_dir()
        ])

        # Calcola differenza
        missing_in_pitch = archive_sims - data_sims

        if missing_in_pitch:
            missing[pitch] = missing_in_pitch
            print(f"{pitch}: {len(missing_in_pitch)} simulazioni mancanti")
        else:
            print(f"{pitch}: OK (tutte presenti)")

    # Stampa riepilogo dettagliato
    print("\n" + "=" * 60)
    print("RIEPILOGO SIMULAZIONI MANCANTI")
    print("=" * 60)

    total_missing = 0
    for pitch in sorted(missing.keys()):
        sims = missing[pitch]
        print(f"\n{pitch} ({len(sims)} mancanti):")
        for sim in sorted(sims):
            print(f"  - {sim}")
        total_missing += len(sims)

    print(f"\nTOTALE: {total_missing} simulazioni mancanti")

    # Opzionale: salva lista in un file
    output_file = Path("missing_simulations.txt")
    with output_file.open("w") as f:
        f.write("SIMULAZIONI MANCANTI\n")
        f.write("=" * 60 + "\n\n")
        for pitch in sorted(missing.keys()):
            f.write(f"{pitch}:\n")
            for sim in sorted(missing[pitch]):
                f.write(f"  - {sim}\n")
            f.write("\n")
        f.write(f"TOTALE: {total_missing}\n")
    print(f"\nLista salvata in: {output_file.resolve()}")


if __name__ == "__main__":
    main()