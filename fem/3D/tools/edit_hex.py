#!/usr/bin/env python3
"""
Copia i file init <sim_name>.dat dalle simulazioni in:

    /archive/roberto/poresAMDIS/hexagon/<sim_name>/<sim_name>.dat

verso:

    /data/fiorello/pores3D/data_train/hexagon/init/<sim_name>.dat

Non ci sono cartelle di pitch intermedie.
"""

from pathlib import Path
import shutil

SOURCE_DIR = Path("/archive/roberto/poresAMDIS/hexagon")
DEST_DIR = Path("/data/fiorello/pores3D/data_train/hexagon/init")


def main():
    if not SOURCE_DIR.is_dir():
        raise SystemExit(
            f"ERRORE: directory sorgente non trovata o non accessibile:\n"
            f"  {SOURCE_DIR}"
        )

    # Crea la cartella destinazione se non esiste.
    DEST_DIR.mkdir(parents=True, exist_ok=True)

    # Considera solo directory direttamente dentro hexagon.
    sim_dirs = sorted(path for path in SOURCE_DIR.iterdir() if path.is_dir())

    if not sim_dirs:
        raise SystemExit(f"ERRORE: nessuna cartella simulazione trovata in {SOURCE_DIR}")

    print(f"Cartelle simulazione trovate: {len(sim_dirs)}")
    print(f"Destinazione: {DEST_DIR}\n")

    copied = []
    missing = []
    overwritten = []

    for sim_dir in sim_dirs:
        sim_name = sim_dir.name
        source_dat = sim_dir / f"{sim_name}.dat"
        dest_dat = DEST_DIR / f"{sim_name}.dat"

        if not source_dat.is_file():
            missing.append(str(source_dat))
            print(f"MANCANTE: {source_dat}")
            continue

        if dest_dat.exists():
            overwritten.append(dest_dat.name)

        # copy2 conserva timestamp e metadati quando possibile.
        shutil.copy2(source_dat, dest_dat)
        copied.append(dest_dat.name)
        print(f"COPIATO: {source_dat.name}")

    print("\n" + "=" * 60)
    print("RIEPILOGO")
    print("=" * 60)
    print(f"File .dat copiati: {len(copied)}")
    print(f"File sovrascritti: {len(overwritten)}")
    print(f"Init mancanti: {len(missing)}")

    if missing:
        print("\nFile attesi ma non trovati:")
        for filename in missing:
            print(f"  - {filename}")

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
            log.writelines(f"{name}\n" for name in copied)

        if missing:
            log.write("\nFILE MANCANTI:\n")
            log.writelines(f"{path}\n" for path in missing)

    print(f"\nLog scritto in: {log_file}")


if __name__ == "__main__":
    main()