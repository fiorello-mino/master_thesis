#!/usr/bin/env python3
"""
Copia i file init delle simulazioni hexagon.

Struttura attesa:
  /archive/roberto/poresAMDIS/hexagon/
      isoHex_R0.2_H1.0_P0.8/
          iso2_R0.2_H1.0_P0.8H.dat

Destinazione:
  /data/fiorello/pores3D/data_train/hexagon/init/
      iso2_R0.2_H1.0_P0.8H.dat
"""

from pathlib import Path
import shutil

SOURCE_DIR = Path("/archive/roberto/poresAMDIS/hexagon")
DEST_DIR = Path("/data/fiorello/pores3D/data_train/hexagon/init")

SOURCE_PREFIX = "isoHex"
INIT_PREFIX = "iso2"


def init_name_from_sim_dir(sim_dir_name: str) -> str:
    """
    Trasforma il nome della cartella nel nome base dell'init.

    Esempio:
      isoHex_R0.2_H1.0_P0.8
      -> iso2_R0.2_H1.0_P0.8H
    """
    if not sim_dir_name.startswith(SOURCE_PREFIX):
        raise ValueError(
            f"La cartella non inizia con '{SOURCE_PREFIX}': {sim_dir_name}"
        )

    # Rimpiazza solo il prefisso iniziale:
    # isoHex... -> iso2...
    init_name = INIT_PREFIX + sim_dir_name[len(SOURCE_PREFIX):]

    # La H finale identifica la geometria hexagon e deve restare nel file .dat.
    return f"{init_name}H"


def main():
    if not SOURCE_DIR.is_dir():
        raise SystemExit(
            f"ERRORE: directory sorgente non trovata o non accessibile:\n"
            f"  {SOURCE_DIR}"
        )

    DEST_DIR.mkdir(parents=True, exist_ok=True)

    # Prende solo le cartelle della simulazione isoHex...
    sim_dirs = sorted(
        path for path in SOURCE_DIR.iterdir()
        if path.is_dir() and path.name.startswith(SOURCE_PREFIX)
    )

    if not sim_dirs:
        raise SystemExit(
            f"ERRORE: nessuna directory con prefisso '{SOURCE_PREFIX}' trovata in:\n"
            f"  {SOURCE_DIR}"
        )

    print(f"Cartelle simulazione isoHex trovate: {len(sim_dirs)}")
    print(f"Destinazione: {DEST_DIR}\n")

    copied = []
    missing = []
    overwritten = []

    for sim_dir in sim_dirs:
        sim_folder_name = sim_dir.name

        # es. isoHex_R0.2_H1.0_P0.8 -> iso2_R0.2_H1.0_P0.8H
        init_name = init_name_from_sim_dir(sim_folder_name)

        source_dat = sim_dir / f"{init_name}.dat"
        dest_dat = DEST_DIR / f"{init_name}.dat"

        if not source_dat.is_file():
            missing.append(source_dat)
            print(f"MANCANTE: {source_dat}")
            continue

        if dest_dat.exists():
            overwritten.append(dest_dat.name)

        shutil.copy2(source_dat, dest_dat)
        copied.append(dest_dat.name)

        print(f"COPIATO: {source_dat.name} -> {dest_dat}")

    print("\n" + "=" * 60)
    print("RIEPILOGO")
    print("=" * 60)
    print(f"File .dat copiati: {len(copied)}")
    print(f"File già presenti e sovrascritti: {len(overwritten)}")
    print(f"File init attesi ma non trovati: {len(missing)}")

    if missing:
        print("\nInit mancanti:")
        for source_dat in missing:
            print(f"  - {source_dat}")


if __name__ == "__main__":
    main()