#!/usr/bin/env python3
"""
Copia gli init file dalle simulazioni hexagon.

Struttura attesa:
  /archive/roberto/poresAMDIS/hexagon/
      isoHex_R0.2_H1.0_P0.8/
          iso2_R0.2_H1.0_P0.8.dat

Destinazione:
  /data/fiorello/pores3D/data_train/hexagon/init/
      iso2_R0.2_H1.0_P0.8.dat
"""

from pathlib import Path
import shutil

SOURCE_DIR = Path("/archive/roberto/poresAMDIS/hexagon")
DEST_DIR = Path("/data/fiorello/pores3D/data_train/hexagon/init")

SOURCE_PREFIX = "isoHex"
INIT_PREFIX = "iso2"


def main():
    if not SOURCE_DIR.is_dir():
        raise SystemExit(
            f"ERRORE: directory sorgente non trovata o non accessibile:\n"
            f"  {SOURCE_DIR}"
        )

    DEST_DIR.mkdir(parents=True, exist_ok=True)

    # Cerca solo cartelle simulazione isoHex...
    sim_dirs = sorted(
        path for path in SOURCE_DIR.iterdir()
        if path.is_dir() and path.name.startswith(SOURCE_PREFIX)
    )

    if not sim_dirs:
        raise SystemExit(
            f"ERRORE: nessuna directory con prefisso '{SOURCE_PREFIX}' in:\n"
            f"  {SOURCE_DIR}"
        )

    print(f"Cartelle simulazione trovate: {len(sim_dirs)}")
    print(f"Destinazione: {DEST_DIR}\n")

    copied = []
    missing = []
    overwritten = []

    for sim_dir in sim_dirs:
        source_sim_name = sim_dir.name

        # isoHex_R0.2_H1.0_P0.8 -> iso2_R0.2_H1.0_P0.8
        init_name = source_sim_name.replace(
            SOURCE_PREFIX,
            INIT_PREFIX,
            1
        )

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

        print(f"COPIATO: {source_dat} -> {dest_dat}")

    print("\n" + "=" * 60)
    print("RIEPILOGO")
    print("=" * 60)
    print(f"File .dat copiati: {len(copied)}")
    print(f"File già esistenti e sovrascritti: {len(overwritten)}")
    print(f"Init attesi ma non trovati: {len(missing)}")

    log_file = DEST_DIR / "copy_hexagon_init.log"
    with log_file.open("w", encoding="utf-8") as log:
        log.write("COPIA INIT HEXAGON\n")
        log.write("=" * 60 + "\n")
        log.write(f"Sorgente: {SOURCE_DIR}\n")
        log.write(f"Destinazione: {DEST_DIR}\n")
        log.write(f"Copiati: {len(copied)}\n")
        log.write(f"Sovrascritti: {len(overwritten)}\n")
        log.write(f"Mancanti: {len(missing)}\n\n")

        if copied:
            log.write("FILE COPIATI:\n")
            for name in copied:
                log.write(f"{name}\n")

        if missing:
            log.write("\nFILE MANCANTI:\n")
            for path in missing:
                log.write(f"{path}\n")

    print(f"\nLog scritto in: {log_file}")


if __name__ == "__main__":
    main()