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
TOL = 1e-6

# Se True, controlla tutti i file e stampa tutti quelli non validi.
# Non elimina e non modifica alcun file.
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

    global_min = np.inf
    global_max = -np.inf

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
    print(f"Range richiesto: [0, 1]")
    print(f"Tolleranza     : {TOL}")

    for index, npy_path in enumerate(npy_files, start=1):
        total_files += 1

        try:
            # mmap_mode evita copie inutili in RAM durante il caricamento.
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
            n_invalid_finite = int((~finite_mask).sum())

            if n_invalid_finite > 0:
                invalid_files += 1

                message = (
                    f"{npy_path}: NaN/Inf={n_invalid_finite}/{values.size}"
                )

                print(
                    f"[{index:05d}/{len(npy_files):05d}] INVALID | {message}"
                )
                report_lines.append(message)
                continue

            value_min = float(values.min())
            value_max = float(values.max())

            global_min = min(global_min, value_min)
            global_max = max(global_max, value_max)

            # Con TOL=1e-6 accetta [-1e-6, 1+1e-6].
            is_valid = (
                value_min >= -TOL
                and value_max <= 1.0 + TOL
            )

            relative_path = npy_path.relative_to(ROOT_DIR)

            if is_valid:
                valid_files += 1

                print(
                    f"[{index:05d}/{len(npy_files):05d}] OK      | "
                    f"{relative_path} | "
                    f"min={value_min:.8f}, max={value_max:.8f}"
                )
            else:
                invalid_files += 1

                n_below = int((values < -TOL).sum())
                n_above = int((values > 1.0 + TOL).sum())

                message = (
                    f"{relative_path} | "
                    f"min={value_min:.10f}, max={value_max:.10f} | "
                    f"valori < 0: {n_below} | valori > 1: {n_above}"
                )

                print(
                    f"[{index:05d}/{len(npy_files):05d}] INVALID | {message}"
                )
                report_lines.append(message)

        except Exception as exc:
            unreadable_files += 1
            invalid_files += 1

            message = (
                f"{npy_path}: {type(exc).__name__}: {exc}"
            )

            print(
                f"[{index:05d}/{len(npy_files):05d}] ERROR   | {message}"
            )
            report_lines.append(message)

    report_lines.extend(
        [
            "",
            "RIEPILOGO",
            f"File totali = {total_files}",
            f"File validi = {valid_files}",
            f"File invalidi = {invalid_files}",
            f"File non leggibili = {unreadable_files}",
            f"Minimo globale = {global_min}",
            f"Massimo globale = {global_max}",
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
    print(f"Minimo globale    : {global_min:.10f}")
    print(f"Massimo globale   : {global_max:.10f}")
    print(f"Report            : {REPORT_PATH}")


if __name__ == "__main__":
    main()
