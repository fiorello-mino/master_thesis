#!/usr/bin/env python3
"""
Copia gli init e i file .3d delle simulazioni hexagon.

Struttura attesa:
  /archive/roberto/poresAMDIS/hexagon/
      isoHex_R0.2_H1.0_P0.8/
          iso2_R0.2_H1.0_P0.8H.dat
          <uno o più file>.3d

Destinazione piatta:
  /data/fiorello/pores3D/data_train/hexagon/init/
      iso2_R0.2_H1.0_P0.8H.dat
      <uno o più file>.3d
"""

from pathlib import Path
import shutil

SOURCE_DIR = Path("/archive/roberto/poresAMDIS/hexagon")
DEST_DIR = Path("/data/fiorello/pores3D/data_train/hexagon/init")

SOURCE_PREFIX = "isoHex"
INIT_PREFIX = "iso2"


def init_name_from_sim_dir(sim_dir_name: str) -> str:
    """
    Esempio:
      isoHex_R0.2_H1.0_P0.8
      -> iso2_R0.2_H1.0_P0.8H
    """
    if not sim_dir_name.startswith(SOURCE_PREFIX):
        raise ValueError(
            f"La cartella non inizia con '{SOURCE_PREFIX}': {sim_dir_name}"
        )

    init_name = INIT_PREFIX + sim_dir_name[len(SOURCE_PREFIX):]

    # La H finale identifica hexagon nel nome dell'init.
    return f"{init_name}H"


def copy_file(source: Path, destination: Path, copied: list, overwritten: list):
    """Copia un file e registra se è stato sovrascritto."""
    if destination.exists():
        overwritten.append(destination.name)

    shutil.copy2(source, destination)
    copied.append(destination.name)


def main():
    if not SOURCE_DIR.is_dir():
        raise SystemExit(
            f"ERRORE: directory sorgente non trovata o non accessibile:\n"
            f"  {SOURCE_DIR}"
        )

    DEST_DIR.mkdir(parents=True, exist_ok=True)

    sim_dirs = sorted(
        path for path in SOURCE_DIR.iterdir()
        if path.is_dir() and path.name.startswith(SOURCE_PREFIX)
    )

    if not sim_dirs:
        raise SystemExit(
            f"ERRORE: nessuna cartella con prefisso '{SOURCE_PREFIX}' trovata in:\n"
            f"  {SOURCE_DIR}"
        )

    print(f"Cartelle simulazione isoHex trovate: {len(sim_dirs)}")
    print(f"Destinazione: {DEST_DIR}\n")

    copied_dat = []
    copied_3d = []
    overwritten = []
    missing_dat = []
    missing_3d = []

    for sim_dir in sim_dirs:
        sim_folder_name = sim_dir.name
        init_name = init_name_from_sim_dir(sim_folder_name)

        print(f"\nSimulazione: {sim_folder_name}")

        # Copia il file init .dat:
        # isoHex_... -> iso2_...H.dat
        source_dat = sim_dir / f"{init_name}.dat"
        dest_dat = DEST_DIR / f"{init_name}.dat"

        if source_dat.is_file():
            copy_file(source_dat, dest_dat, copied_dat, overwritten)
            print(f"  DAT copiato: {source_dat.name}")
        else:
            missing_dat.append(source_dat)
            print(f"  DAT MANCANTE: {source_dat.name}")

        # Copia tutti i file .3d direttamente nella cartella della simulazione.
        files_3d = sorted(
            path for path in sim_dir.iterdir()
            if path.is_file() and path.suffix.lower() == ".3d"
        )

        if not files_3d:
            missing_3d.append(sim_dir)
            print("  Nessun file .3d trovato")
            continue

        for source_3d in files_3d:
            dest_3d = DEST_DIR / source_3d.name
            copy_file(source_3d, dest_3d, copied_3d, overwritten)
            print(f"  3D copiato:  {source_3d.name}")

    print("\n" + "=" * 65)
    print("RIEPILOGO COPIA HEXAGON")
    print("=" * 65)
    print(f"File .dat copiati: {len(copied_dat)}")
    print(f"File .3d copiati:  {len(copied_3d)}")
    print(f"File sovrascritti: {len(overwritten)}")
    print(f"Init .dat mancanti: {len(missing_dat)}")
    print(f"Simulazioni senza file .3d: {len(missing_3d)}")

    if missing_dat:
        print("\nFile .dat non trovati:")
        for source_dat in missing_dat:
            print(f"  - {source_dat}")

    if missing_3d:
        print("\nCartelle senza file .3d:")
        for sim_dir in missing_3d:
            print(f"  - {sim_dir}")

    log_file = DEST_DIR / "copy_hexagon_init_and_3d.log"
    with log_file.open("w", encoding="utf-8") as log:
        log.write("COPIA INIT E FILE .3D - HEXAGON\n")
        log.write("=" * 65 + "\n")
        log.write(f"Sorgente: {SOURCE_DIR}\n")
        log.write(f"Destinazione: {DEST_DIR}\n\n")
        log.write(f"DAT copiati: {len(copied_dat)}\n")
        log.write(f"3D copiati: {len(copied_3d)}\n")
        log.write(f"Sovrascritti: {len(overwritten)}\n")
        log.write(f"DAT mancanti: {len(missing_dat)}\n")
        log.write(f"Directory senza .3d: {len(missing_3d)}\n\n")

        if copied_dat:
            log.write("FILE .DAT COPIATI:\n")
            for name in copied_dat:
                log.write(f"{name}\n")

        if copied_3d:
            log.write("\nFILE .3D COPIATI:\n")
            for name in copied_3d:
                log.write(f"{name}\n")

        if missing_dat:
            log.write("\nFILE .DAT MANCANTI:\n")
            for path in missing_dat:
                log.write(f"{path}\n")

        if missing_3d:
            log.write("\nCARTELLE SENZA .3D:\n")
            for path in missing_3d:
                log.write(f"{path}\n")

    print(f"\nLog salvato in: {log_file}")


if __name__ == "__main__":
    main()