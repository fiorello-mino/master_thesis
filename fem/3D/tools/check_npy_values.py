from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np


# ============================================================
# CONFIGURAZIONE
# ============================================================

# Root che contiene iso_P06, iso_P07, ... e i folder simulazione.
ROOT_DIR = Path("/data/fiorello/poresAMDIS")

# Cerca soltanto i frame phi.
FRAME_GLOB = "surf_*.npy"

# Esclude le mask del fallback PyVista/VTK.
VTK_MASK_SUFFIX = "_vtk_fallback_mask.npy"

# Range fisico atteso:
# phi ∈ [0, 1]
#
# Accetta piccoli overshoot numerici:
# phi ∈ [-TOL, 1 + TOL].
TOL = 1e-3

# SICUREZZA:
# False -> analizza e scrive il report, NON elimina nulla.
# True  -> elimina ricorsivamente le simulazioni non valide.
DELETE_INVALID_SIMULATIONS = False

# Se True, anche un file illeggibile, con NaN o con Inf rende
# invalida l'intera simulazione.
DELETE_ON_UNREADABLE_OR_NONFINITE = True

# Report creato sempre prima dell'eventuale eliminazione.
REPORT_PATH = ROOT_DIR / "simulations_outside_phi_tolerance.txt"


# ============================================================
# FUNZIONI
# ============================================================

def is_data_frame(path: Path) -> bool:
    """
    Include:
        surf_0.000000.npy

    Esclude:
        surf_0.000000_vtk_fallback_mask.npy
    """
    return (
        path.is_file()
        and path.name.startswith("surf_")
        and path.suffix == ".npy"
        and not path.name.endswith(VTK_MASK_SUFFIX)
    )


def find_simulation_folders(root_dir: Path) -> list[Path]:
    """
    Trova tutte le cartelle che contengono direttamente almeno
    un frame phi valido per nome.
    """
    return sorted(
        {
            npy_path.parent
            for npy_path in root_dir.rglob(FRAME_GLOB)
            if is_data_frame(npy_path)
        }
    )


def get_data_frames(folder: Path) -> list[Path]:
    """Restituisce i soli frame phi, escludendo le mask VTK."""
    return sorted(
        path
        for path in folder.glob(FRAME_GLOB)
        if is_data_frame(path)
    )


def inspect_frame(
    npy_path: Path,
) -> tuple[bool, str, float | None, float | None]:
    """
    Controlla un frame.

    Restituisce:
        invalid, reason, value_min, value_max

    invalid=True se:
    - file non leggibile;
    - array non 3D;
    - NaN o Inf;
    - min < -TOL;
    - max > 1 + TOL.
    """
    try:
        values = np.load(
            npy_path,
            mmap_mode="r",
            allow_pickle=False,
        )

        if values.ndim != 3:
            return (
                True,
                f"array non 3D: shape={values.shape}",
                None,
                None,
            )

        nonfinite_mask = ~np.isfinite(values)
        n_nonfinite = int(nonfinite_mask.sum())

        if n_nonfinite > 0:
            return (
                True,
                f"NaN/Inf={n_nonfinite}/{values.size}",
                None,
                None,
            )

        value_min = float(values.min())
        value_max = float(values.max())

        if value_min < -TOL or value_max > 1.0 + TOL:
            return (
                True,
                (
                    f"range fuori tolleranza: "
                    f"min={value_min:.10f}, max={value_max:.10f}, "
                    f"range ammesso=[{-TOL:.10f}, {1.0 + TOL:.10f}]"
                ),
                value_min,
                value_max,
            )

        return False, "OK", value_min, value_max

    except Exception as exc:
        return (
            True,
            f"{type(exc).__name__}: {exc}",
            None,
            None,
        )


def inspect_simulation(
    folder: Path,
) -> tuple[bool, list[str], int, float | None, float | None]:
    """
    Controlla tutti i frame di una simulazione.

    Appena trova un frame non valido, può interrompersi:
    basta un solo frame problematico per eliminare la traiettoria.

    Restituisce:
        invalid_simulation,
        problems,
        n_frames,
        folder_min,
        folder_max
    """
    frames = get_data_frames(folder)

    if not frames:
        return (
            True,
            ["Nessun frame phi trovato"],
            0,
            None,
            None,
        )

    folder_min = np.inf
    folder_max = -np.inf

    for npy_path in frames:
        invalid, reason, value_min, value_max = inspect_frame(
            npy_path
        )

        if invalid:
            return (
                True,
                [
                    f"{npy_path.name}: {reason}"
                ],
                len(frames),
                None if np.isinf(folder_min) else float(folder_min),
                None if np.isinf(folder_max) else float(folder_max),
            )

        assert value_min is not None
        assert value_max is not None

        folder_min = min(folder_min, value_min)
        folder_max = max(folder_max, value_max)

    return (
        False,
        [],
        len(frames),
        float(folder_min),
        float(folder_max),
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    if not ROOT_DIR.is_dir():
        raise FileNotFoundError(
            f"Root directory non trovata: {ROOT_DIR}"
        )

    simulation_folders = find_simulation_folders(ROOT_DIR)

    if not simulation_folders:
        raise FileNotFoundError(
            f"Nessun file '{FRAME_GLOB}' trovato sotto {ROOT_DIR}"
        )

    invalid_simulations: list[
        tuple[Path, list[str], int, float | None, float | None]
    ] = []

    valid_simulations = 0
    total_frames_checked = 0

    print("=" * 84)
    print("CONTROLLO TRAIETTORIE: RANGE phi IN [0, 1]")
    print("=" * 84)
    print(f"Root directory : {ROOT_DIR}")
    print(f"Tolleranza     : {TOL}")
    print(f"Range ammesso  : [{-TOL}, {1.0 + TOL}]")
    print(
        "Eliminazione   : "
        f"{DELETE_INVALID_SIMULATIONS}"
    )
    print(f"Simulazioni    : {len(simulation_folders)}")

    for index, folder in enumerate(simulation_folders, start=1):
        relative_folder = folder.relative_to(ROOT_DIR)

        (
            invalid,
            problems,
            n_frames,
            folder_min,
            folder_max,
        ) = inspect_simulation(folder)

        total_frames_checked += n_frames

        if invalid:
            invalid_simulations.append(
                (
                    folder,
                    problems,
                    n_frames,
                    folder_min,
                    folder_max,
                )
            )

            print(
                f"[{index:03d}/{len(simulation_folders):03d}] "
                f"INVALID | {relative_folder} | "
                f"{n_frames} frame"
            )

            for problem in problems:
                print(f"    {problem}")

        else:
            valid_simulations += 1

            print(
                f"[{index:03d}/{len(simulation_folders):03d}] "
                f"OK      | {relative_folder} | "
                f"{n_frames} frame | "
                f"min={folder_min:.8f}, max={folder_max:.8f}"
            )

    # Report completo: viene scritto prima dell'eventuale delete.
    report_lines = [
        f"ROOT_DIR = {ROOT_DIR}",
        f"TOL = {TOL}",
        f"RANGE AMMESSO = [{-TOL}, {1.0 + TOL}]",
        f"SIMULAZIONI TOTALI = {len(simulation_folders)}",
        f"SIMULAZIONI VALIDE = {valid_simulations}",
        f"SIMULAZIONI INVALIDE = {len(invalid_simulations)}",
        "",
    ]

    for (
        folder,
        problems,
        n_frames,
        folder_min,
        folder_max,
    ) in invalid_simulations:
        report_lines.append(str(folder))
        report_lines.append(f"  Frame trovati: {n_frames}")

        if folder_min is not None:
            report_lines.append(f"  Min prima dell'errore: {folder_min}")

        if folder_max is not None:
            report_lines.append(f"  Max prima dell'errore: {folder_max}")

        for problem in problems:
            report_lines.append(f"  Problema: {problem}")

        report_lines.append("")

    REPORT_PATH.write_text(
        "\n".join(report_lines) + "\n",
        encoding="utf-8",
    )

    print("\n" + "=" * 84)
    print("RIEPILOGO")
    print("=" * 84)
    print(f"Simulazioni totali      : {len(simulation_folders)}")
    print(f"Simulazioni valide      : {valid_simulations}")
    print(f"Simulazioni invalide    : {len(invalid_simulations)}")
    print(f"Frame controllati       : {total_frames_checked}")
    print(f"Report                  : {REPORT_PATH}")

    if not invalid_simulations:
        print("Nessuna simulazione fuori dalla tolleranza.")
        return

    if not DELETE_INVALID_SIMULATIONS:
        print("\nModalità sicura: non è stata eliminata nessuna cartella.")
        print("Controlla il report; se è corretto, modifica:")
        print("DELETE_INVALID_SIMULATIONS = True")
        print("e rilancia lo script.")
        return

    print("\nELIMINAZIONE SIMULAZIONI INVALIDE")

    deleted = 0

    for folder, problems, n_frames, _, _ in invalid_simulations:
        print(
            f"Elimino: {folder} | "
            f"{n_frames} frame | "
            f"{problems[0]}"
        )

        shutil.rmtree(folder)
        deleted += 1

    print(
        f"\nEliminate {deleted} simulazioni fuori dalla "
        f"tolleranza {TOL}."
    )


if __name__ == "__main__":
    main()