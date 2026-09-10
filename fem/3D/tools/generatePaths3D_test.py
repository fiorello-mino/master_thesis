from __future__ import annotations

from pathlib import Path


# ============================================================
# CONFIGURAZIONE
# ============================================================

# Cartella che contiene i nuovi NPY convertiti.
ROOT_DIR = Path("/data/fiorello/pores3D/dataset_ext_test/square_reflected")

# File TXT generato: un path assoluto per ogni riga.
OUTPUT_TXT = ROOT_DIR / "test_set.txt"

# Cerca i frame NPY del campo phi.
FRAME_GLOB = "surf_*.npy"

# True: scrive path assoluti, ad esempio:
# /data/fiorello/test_pores3D/iso_P06/.../surf_0.000000.npy
#
# False: scrive path relativi a ROOT_DIR, ad esempio:
# iso_P06/.../surf_0.000000.npy
WRITE_ABSOLUTE_PATHS = True


# ============================================================
# FUNZIONI
# ============================================================


def main() -> None:
    if not ROOT_DIR.is_dir():
        raise FileNotFoundError(
            f"Directory root non trovata: {ROOT_DIR}"
        )

    npy_files = sorted(
        path
        for path in ROOT_DIR.rglob(FRAME_GLOB)
    )

    if not npy_files:
        raise FileNotFoundError(
            f"Nessun file '{FRAME_GLOB}' trovato sotto {ROOT_DIR}"
        )

    OUTPUT_TXT.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_TXT.open("w", encoding="utf-8") as file:
        for npy_path in npy_files:
            path_to_write = (
                npy_path.resolve()
                if WRITE_ABSOLUTE_PATHS
                else npy_path.relative_to(ROOT_DIR)
            )

            file.write(f"{path_to_write}\n")

    print("=" * 78)
    print("GENERAZIONE TXT CON PATH DEI FILE NPY")
    print("=" * 78)
    print(f"Root directory : {ROOT_DIR}")
    print(f"File NPY trovati: {len(npy_files)}")
    print(f"Path assoluti  : {WRITE_ABSOLUTE_PATHS}")
    print(f"File TXT       : {OUTPUT_TXT}")


if __name__ == "__main__":
    main()
