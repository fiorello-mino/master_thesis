#!/usr/bin/env python3
"""
Script per copiare i file .dat e .3d dalle simulazioni mancanti
in /data/fiorello/pores3D/data_train/square/init/

Struttura:
  /archive/roberto/poresAMDIS/square/iso_P0X/<sim_name>/<sim_name>.dat
  /archive/roberto/poresAMDIS/square/iso_P0X/<sim_name>/<qualcosa>.3d
  -> copia in /data/fiorello/pores3D/data_train/square/init/
"""

import os
import shutil
from pathlib import Path

ARCHIVE_BASE = Path("/archive/roberto/poresAMDIS/square")
DATA_BASE = Path("/data/fiorello/poresAMDIS")
INIT_DIR = Path("/data/fiorello/pores3D/data_train/square/init")

def find_3d_file(sim_path):
    """Trova l'unico file .3d nella cartella della simulazione"""
    files_3d = list(sim_path.glob("*.3d"))
    if len(files_3d) == 0:
        return None
    if len(files_3d) > 1:
        print(f"ATTENZIONE: trovati {len(files_3d)} file .3d in {sim_path}, uso il primo")
    return files_3d[0]

def main():
    if not ARCHIVE_BASE.exists():
        print(f"ERRORE: {ARCHIVE_BASE} non esiste o non è accessibile")
        return
    if not DATA_BASE.exists():
        print(f"ERRORE: {DATA_BASE} non esiste o non è accessibile")
        return

    # Crea cartella init se non esiste
    INIT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Cartella init: {INIT_DIR}\n")

    # Trova tutte le cartelle pitch (iso_P06, iso_P07, ...)
    pitch_folders = sorted([
        d for d in os.listdir(ARCHIVE_BASE)
        if d.startswith("iso_P0") and (ARCHIVE_BASE / d).is_dir()
    ])

    print(f"Trovate {len(pitch_folders)} cartelle pitch: {pitch_folders}\n")

    # Raccogli tutte le simulazioni mancanti con il loro percorso completo
    missing_sims = []  # lista di (pitch, sim_name, archive_path)

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
            for sim in archive_sims:
                missing_sims.append((pitch, sim, archive_pitch_path / sim))
            continue

        # Simulazioni presenti in /data per questo pitch
        data_sims = set([
            d for d in os.listdir(data_pitch_path)
            if (data_pitch_path / d).is_dir()
        ])

        # Calcola differenza
        missing_in_pitch = archive_sims - data_sims
        for sim in missing_in_pitch:
            missing_sims.append((pitch, sim, archive_pitch_path / sim))

    print(f"Totale simulazioni mancanti: {len(missing_sims)}\n")

    # Copia i file .dat e .3d
    copied_dat = 0
    copied_3d = 0
    errors = []

    for pitch, sim_name, sim_path in missing_sims:
        # File .dat (stesso nome della simulazione)
        dat_file = sim_path / f"{sim_name}.dat"
        
        # File .3d (unico file con estensione .3d nella cartella)
        macro_file = find_3d_file(sim_path)

        # Copia .dat se esiste
        if dat_file.exists():
            dest_dat = INIT_DIR / f"{sim_name}.dat"
            shutil.copy2(dat_file, dest_dat)
            copied_dat += 1
            print(f"Copiato: {dat_file.name} -> {INIT_DIR}")
        else:
            errors.append(f"MANCANTE .dat: {dat_file}")

        # Copia .3d se esiste
        if macro_file is not None:
            dest_3d = INIT_DIR / macro_file.name  # mantieni nome originale del file .3d
            shutil.copy2(macro_file, dest_3d)
            copied_3d += 1
            print(f"Copiato: {macro_file.name} -> {INIT_DIR}")
        else:
            errors.append(f"MANCANTE .3d in: {sim_path}")

    # Riepilogo
    print("\n" + "=" * 60)
    print("RIEPILOGO COPIA")
    print("=" * 60)
    print(f"File .dat copiati: {copied_dat}")
    print(f"File .3d copiati: {copied_3d}")

    if errors:
        print(f"\nERRORI/AVVISI ({len(errors)}):")
        for err in errors:
            print(f"  - {err}")

    # Lista file copiati
    all_files = sorted(list(INIT_DIR.glob("*.dat")) + list(INIT_DIR.glob("*.3d")))
    print(f"\nTotale file in {INIT_DIR}: {len(all_files)}")

    # Salva log
    log_file = Path("copy_log.txt")
    with log_file.open("w") as f:
        f.write("LOG COPIA FILE .dat e .3d\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"File .dat copiati: {copied_dat}\n")
        f.write(f"File .3d copiati: {copied_3d}\n\n")
        if errors:
            f.write(f"ERRORI ({len(errors)}):\n")
            for err in errors:
                f.write(f"  {err}\n")
    print(f"\nLog salvato in: {log_file.resolve()}")


if __name__ == "__main__":
    main()