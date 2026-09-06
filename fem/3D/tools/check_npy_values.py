from __future__ import annotations

from pathlib import Path

import numpy as np


# ============================================================
# CONFIGURAZIONE
# ============================================================

# Root che contiene iso_P06, iso_P07, ... e le cartelle delle simulazioni.
ROOT_DIR = Path("/data/fiorello/poresAMDIS")

# Cerca i veri frame del campo phi.
FRAME_GLOB = "surf_*.npy"

# Esclude le maschere create dal converter PyVista/VTK.
VTK_MASK_SUFFIX = "_vtk_fallback_mask.npy"

# Tolleranza numerica: valori come -1e-8 o 1.00000001 possono derivare
# da arrotondamento floating point. Metti 0.0 se vuoi il controllo rigoroso.
TOL = 1e-3

# Report degli NPY con valori fuori dal range [0, 1].
REPORT_PATH = ROOT_DIR / "npy_values_outside_0_1.txt"


# ============================================================
# FUNZIONI
# ============================================================

def is_data_frame(path: Path) -> bool:
    """Riconosce i veri surf_<time>.npy ed esclude le fallback mask."""
    return (
        path.is_file()
        and path.name.startswith("surf_")
        and path.suffix == ".npy"
        and not path.name.endswith(VTK_MASK_SUFFIX)
    )


def format_location(
    npy_path: Path | None,
    index: tuple[int, ...] | None,
) -> str:
    """Restituisce path relativo e indice del voxel che contiene un estremo."""
    if npy_path is None or index is None:
        return "non disponibile"

    relative_path = npy_path.relative_to(ROOT_DIR)
    return f"{relative_path} | voxel index={index}"


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    if not ROOT_DIR.is_dir():
        raise FileNotFoundError(f"Root directory non trovata: {ROOT_DIR}")

    npy_files = sorted(
        path
        for path in ROOT_DIR.rglob(FRAME_GLOB)
        if is_data_frame(path)
    )

    if not npy_files:
        raise FileNotFoundError(
            f"Nessun file '{FRAME_GLOB}' trovato sotto {ROOT_DIR}"
        )

    total_files = 0
    valid_files = 0
    invalid_files = 0
    unreadable_files = 0

    # Estremi globali e posizione del primo voxel che li realizza.
    global_min = np.inf
    global_max = -np.inf

    global_min_path: Path | None = None
    global_max_path: Path | None = None

    global_min_index: tuple[int, ...] | None = None
    global_max_index: tuple[int, ...] | None = None

    report_lines = [
        f"ROOT_DIR = {ROOT_DIR}",
        f"Range richiesto = [0, 1] con tolleranza TOL={TOL}",
        "",
    ]

    print("=" * 80)
    print("CONTROLLO RANGE DEI FILE NPY")
    print("=" * 80)
    print(f"Root directory : {ROOT_DIR}")
    print(f"File trovati   : {len(npy_files)}")
    print("Range richiesto: [0, 1]")
    print(f"Tolleranza     : {TOL}")

    for file_number, npy_path in enumerate(npy_files, start=1):
        total_files += 1

        try:
            # mmap_mode='r' evita una copia preliminare in RAM.
            values = np.load(
                npy_path,
                mmap_mode="r",
                allow_pickle=False,
            )

            if values.ndim != 3:
                raise ValueError(
                    f"array non 3D, shape={values.shape}"
                )

            finite_mask = np.isfinite(values)
            n_nonfinite = int((~finite_mask).sum())

            if n_nonfinite > 0:
                invalid_files += 1

                message = (
                    f"{npy_path}: NaN/Inf={n_nonfinite}/{values.size}"
                )

                print(
                    f"[{file_number:05d}/{len(npy_files):05d}] "
                    f"INVALID | {message}"
                )

                report_lines.append(message)
                continue

            # Estremi del singolo file.
            file_min = float(values.min())
            file_max = float(values.max())

            # Salva anche l'indice (x_index, y_index, z_index)
            # del primo voxel che contiene il valore minimo/massimo del file.
            if file_min < global_min:
                global_min = file_min
                global_min_path = npy_path
                global_min_index = tuple(
                    int(i)
                    for i in np.unravel_index(
                        int(np.argmin(values)),
                        values.shape,
                    )
                )

            if file_max > global_max:
                global_max = file_max
                global_max_path = npy_path
                global_max_index = tuple(
                    int(i)
                    for i in np.unravel_index(
                        int(np.argmax(values)),
                        values.shape,
                    )
                )

            # Con TOL=1e-6 accetta [-1e-6, 1+1e-6].
            is_valid = (
                file_min >= -TOL
                and file_max <= 1.0 + TOL
            )

            relative_path = npy_path.relative_to(ROOT_DIR)

            if is_valid:
                valid_files += 1

                print(
                    f"[{file_number:05d}/{len(npy_files):05d}] "
                    f"OK      | {relative_path} | "
                    f"min={file_min:.8f}, max={file_max:.8f}"
                )
            else:
                invalid_files += 1

                n_below = int((values < -TOL).sum())
                n_above = int((values > 1.0 + TOL).sum())

                message = (
                    f"{relative_path} | "
                    f"min={file_min:.10f}, max={file_max:.10f} | "
                    f"valori < 0: {n_below} | valori > 1: {n_above}"
                )

                print(
                    f"[{file_number:05d}/{len(npy_files):05d}] "
                    f"INVALID | {message}"
                )

                report_lines.append(message)

        except Exception as exc:
            unreadable_files += 1
            invalid_files += 1

            message = f"{npy_path}: {type(exc).__name__}: {exc}"

            print(
                f"[{file_number:05d}/{len(npy_files):05d}] "
                f"ERROR   | {message}"
            )

            report_lines.append(message)

    min_location = format_location(global_min_path, global_min_index)
    max_location = format_location(global_max_path, global_max_index)

    report_lines.extend(
        [
            "",
            "RIEPILOGO",
            f"File totali = {total_files}",
            f"File validi = {valid_files}",
            f"File invalidi = {invalid_files}",
            f"File non leggibili = {unreadable_files}",
            f"Minimo globale = {global_min}",
            f"Posizione minimo globale = {min_location}",
            f"Massimo globale = {global_max}",
            f"Posizione massimo globale = {max_location}",
        ]
    )

    REPORT_PATH.write_text(
        "\n".join(report_lines) + "\n",
        encoding="utf-8",
    )

    print("\n" + "=" * 80)
    print("RIEPILOGO")
    print("=" * 80)
    print(f"File totali       : {total_files}")
    print(f"File validi       : {valid_files}")
    print(f"File invalidi     : {invalid_files}")
    print(f"File non leggibili: {unreadable_files}")

    print(f"\nMinimo globale    : {global_min:.10f}")
    print(f"Dove si trova     : {min_location}")

    print(f"\nMassimo globale   : {global_max:.10f}")
    print(f"Dove si trova     : {max_location}")

    print(f"\nReport            : {REPORT_PATH}")


if __name__ == "__main__":
    main()
