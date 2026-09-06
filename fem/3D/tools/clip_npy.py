from __future__ import annotations

from pathlib import Path

import numpy as np


# ============================================================
# CONFIGURAZIONE
# ============================================================

# Root che contiene iso_P06, iso_P07, ... e le cartelle simulazione.
ROOT_DIR = Path("/data/fiorello/poresAMDIS")

# Cerca i veri frame phi.
FRAME_GLOB = "surf_*.npy"

# Esclude le maschere create dal converter PyVista/VTK.
VTK_MASK_SUFFIX = "_vtk_fallback_mask.npy"

# Un file viene riportato se, PRIMA del clipping, contiene almeno un valore
# al di fuori dell'intervallo [-TOL, 1 + TOL].
# Con 1e-1 il range tollerato è [-0.1, 1.1].
TOL = 1e-1

# Se True, salva una copia di sicurezza accanto a ogni file modificato:
#     surf_0.100000.npy -> surf_0.100000_before_clip.npy
# Le copie sono escluse dalla scansione tramite BACKUP_SUFFIX.
MAKE_BACKUP = False
BACKUP_SUFFIX = "_before_clip.npy"

# Report finale con tutti i file fuori tolleranza, prima del clipping.
REPORT_PATH = ROOT_DIR / "npy_files_outside_tol_before_clipping.txt"


# ============================================================
# FUNZIONI
# ============================================================

def is_data_frame(path: Path) -> bool:
    """Riconosce surf_<time>.npy ed esclude maschere e backup."""
    return (
        path.is_file()
        and path.name.startswith("surf_")
        and path.suffix == ".npy"
        and not path.name.endswith(VTK_MASK_SUFFIX)
        and not path.name.endswith(BACKUP_SUFFIX)
    )


def first_index(mask: np.ndarray) -> tuple[int, ...] | None:
    """Restituisce l'indice del primo True nella mask, oppure None."""
    flat_indices = np.flatnonzero(mask)

    if flat_indices.size == 0:
        return None

    return tuple(
        int(i)
        for i in np.unravel_index(int(flat_indices[0]), mask.shape)
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    if not ROOT_DIR.is_dir():
        raise FileNotFoundError(
            f"Root directory non trovata: {ROOT_DIR}"
        )

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
    clipped_files = 0
    files_outside_tolerance = 0
    unchanged_files = 0
    unreadable_files = 0
    nonfinite_files = 0

    total_values_clipped_below = 0
    total_values_clipped_above = 0

    global_min_before = np.inf
    global_max_before = -np.inf
    global_min_after = np.inf
    global_max_after = -np.inf

    report_lines = [
        "CLIPPING NPY DEL CAMPO phi",
        f"ROOT_DIR = {ROOT_DIR}",
        "Clipping applicato = np.clip(values, 0.0, 1.0)",
        f"TOL = {TOL}",
        f"Range di tolleranza per il report = [{-TOL}, {1.0 + TOL}]",
        f"Backup abilitato = {MAKE_BACKUP}",
        "",
        "FILE FUORI TOLLERANZA PRIMA DEL CLIPPING",
        "",
    ]

    print("=" * 88)
    print("CLIPPING DEI FILE NPY NEL RANGE [0, 1]")
    print("=" * 88)
    print(f"Root directory         : {ROOT_DIR}")
    print(f"File trovati           : {len(npy_files)}")
    print("Clipping               : [0, 1]")
    print(f"Tolleranza per report  : {TOL}")
    print(f"Range tollerato report : [{-TOL}, {1.0 + TOL}]")
    print(f"Backup                 : {MAKE_BACKUP}")

    for number, npy_path in enumerate(npy_files, start=1):
        total_files += 1
        relative_path = npy_path.relative_to(ROOT_DIR)

        try:
            # Per modificare e risalvare serve caricare l'array in memoria.
            values = np.load(npy_path, allow_pickle=False)

            if values.ndim != 3:
                raise ValueError(f"array non 3D, shape={values.shape}")

            finite_mask = np.isfinite(values)
            n_nonfinite = int((~finite_mask).sum())

            # NaN/Inf non vengono alterati: il report li segnala esplicitamente.
            if n_nonfinite > 0:
                nonfinite_files += 1
                unreadable_files += 1

                message = (
                    f"{relative_path} | NaN/Inf={n_nonfinite}/{values.size} | "
                    "NON MODIFICATO"
                )

                print(
                    f"[{number:05d}/{len(npy_files):05d}] "
                    f"NONFINITE | {message}"
                )
                report_lines.append(message)
                continue

            value_min_before = float(values.min())
            value_max_before = float(values.max())

            global_min_before = min(global_min_before, value_min_before)
            global_max_before = max(global_max_before, value_max_before)

            below_zero = values < 0.0
            above_one = values > 1.0

            n_below_zero = int(below_zero.sum())
            n_above_one = int(above_one.sum())
            n_values_clipped = n_below_zero + n_above_one

            # Questo controllo decide solo se il file entra nel TXT.
            outside_tolerance = (
                value_min_before < -TOL
                or value_max_before > 1.0 + TOL
            )

            if outside_tolerance:
                files_outside_tolerance += 1

                below_tolerance = values < -TOL
                above_tolerance = values > 1.0 + TOL

                n_below_tolerance = int(below_tolerance.sum())
                n_above_tolerance = int(above_tolerance.sum())

                index_below_tolerance = first_index(below_tolerance)
                index_above_tolerance = first_index(above_tolerance)

                report_lines.extend(
                    [
                        f"File: {relative_path}",
                        f"  Shape: {values.shape}",
                        f"  Min prima del clipping: {value_min_before:.10f}",
                        f"  Max prima del clipping: {value_max_before:.10f}",
                        f"  Valori < {-TOL:.10f}: {n_below_tolerance}",
                        f"  Primo indice < {-TOL:.10f}: {index_below_tolerance}",
                        f"  Valori > {1.0 + TOL:.10f}: {n_above_tolerance}",
                        f"  Primo indice > {1.0 + TOL:.10f}: {index_above_tolerance}",
                        f"  Valori clipppati a 0: {n_below_zero}",
                        f"  Valori clippati a 1: {n_above_one}",
                        "",
                    ]
                )

            # Modifica soltanto i file che contengono valori strettamente
            # minori di 0 o maggiori di 1.
            if n_values_clipped > 0:
                if MAKE_BACKUP:
                    backup_path = npy_path.with_name(
                        npy_path.stem + BACKUP_SUFFIX
                    )

                    if not backup_path.exists():
                        np.save(backup_path, values)
                    else:
                        print(
                            f"[{number:05d}/{len(npy_files):05d}] "
                            f"ATTENZIONE | backup già presente: "
                            f"{backup_path.relative_to(ROOT_DIR)}"
                        )

                clipped_values = np.clip(values, 0.0, 1.0)
                np.save(npy_path, clipped_values)

                value_min_after = float(clipped_values.min())
                value_max_after = float(clipped_values.max())

                global_min_after = min(global_min_after, value_min_after)
                global_max_after = max(global_max_after, value_max_after)

                clipped_files += 1
                total_values_clipped_below += n_below_zero
                total_values_clipped_above += n_above_one

                status = "FUORI TOL" if outside_tolerance else "CLIPPATO"

                print(
                    f"[{number:05d}/{len(npy_files):05d}] {status:10s} | "
                    f"{relative_path} | "
                    f"prima=[{value_min_before:.6f}, {value_max_before:.6f}] | "
                    f"clippati: sotto={n_below_zero}, sopra={n_above_one}"
                )
            else:
                global_min_after = min(global_min_after, value_min_before)
                global_max_after = max(global_max_after, value_max_before)

                unchanged_files += 1

                status = "FUORI TOL" if outside_tolerance else "OK"

                print(
                    f"[{number:05d}/{len(npy_files):05d}] {status:10s} | "
                    f"{relative_path} | "
                    f"gia' in [0, 1] | "
                    f"min={value_min_before:.6f}, max={value_max_before:.6f}"
                )

        except Exception as exc:
            unreadable_files += 1

            message = (
                f"{relative_path} | {type(exc).__name__}: {exc} | "
                "NON MODIFICATO"
            )

            print(
                f"[{number:05d}/{len(npy_files):05d}] ERROR      | {message}"
            )
            report_lines.append(message)

    report_lines.extend(
        [
            "RIEPILOGO",
            f"File totali = {total_files}",
            f"File modificati dal clipping = {clipped_files}",
            f"File gia' in [0, 1] = {unchanged_files}",
            f"File fuori tolleranza TOL={TOL} = {files_outside_tolerance}",
            f"File con NaN/Inf = {nonfinite_files}",
            f"File non leggibili/con errore = {unreadable_files}",
            f"Valori totali clippati a 0 = {total_values_clipped_below}",
            f"Valori totali clippati a 1 = {total_values_clipped_above}",
            f"Minimo globale prima del clipping = {global_min_before}",
            f"Massimo globale prima del clipping = {global_max_before}",
            f"Minimo globale dopo il clipping = {global_min_after}",
            f"Massimo globale dopo il clipping = {global_max_after}",
        ]
    )

    REPORT_PATH.write_text(
        "\n".join(report_lines) + "\n",
        encoding="utf-8",
    )

    print("\n" + "=" * 88)
    print("RIEPILOGO")
    print("=" * 88)
    print(f"File totali                 : {total_files}")
    print(f"File modificati             : {clipped_files}")
    print(f"File gia' in [0, 1]         : {unchanged_files}")
    print(f"File fuori tolleranza 1e-1  : {files_outside_tolerance}")
    print(f"File con NaN/Inf            : {nonfinite_files}")
    print(f"File non leggibili/errori   : {unreadable_files}")
    print(f"Valori clippati a 0         : {total_values_clipped_below}")
    print(f"Valori clippati a 1         : {total_values_clipped_above}")
    print(f"Min globale prima           : {global_min_before:.10f}")
    print(f"Max globale prima           : {global_max_before:.10f}")
    print(f"Min globale dopo            : {global_min_after:.10f}")
    print(f"Max globale dopo            : {global_max_after:.10f}")
    print(f"Report                      : {REPORT_PATH}")


if __name__ == "__main__":
    main()
