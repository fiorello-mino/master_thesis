from __future__ import annotations

import shutil
from pathlib import Path


# ============================================================
# CONFIGURAZIONE
# ============================================================

# Root principale contenente iso_P06, iso_P07, ..., iso_P10.
ROOT_DIR = Path("/data/fiorello/poresAMDIS")

# Cerca i file dei frame della simulazione.
FRAME_GLOB = "surf_*.npy"

# Esclude le maschere diagnostiche PyVista/VTK.
VTK_MASK_SUFFIX = "_vtk_fallback_mask.npy"

# Numero minimo di frame che una simulazione deve contenere per essere mantenuta.
# Le cartelle con n_frames < N_FRAMES sono considerate invalide.
N_FRAMES = 50

# SICUREZZA:
# False -> stampa e crea solo il report, NON elimina nulla.
# True  -> elimina ricorsivamente le cartelle con meno di N_FRAMES.
DELETE_SHORT_SIMULATIONS = False

# Report scritto nella root principale, prima dell'eventuale eliminazione.
REPORT_PATH = ROOT_DIR / "simulations_below_minimum_frames.txt"


# ============================================================
# FUNZIONI
# ============================================================

def is_data_frame(path: Path) -> bool:
    """
    Riconosce soltanto i veri frame phi.

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
    """Trova tutte le cartelle che contengono direttamente almeno un frame."""
    return sorted(
        {
            npy_path.parent
            for npy_path in root_dir.rglob(FRAME_GLOB)
            if is_data_frame(npy_path)
        }
    )


def get_data_frames(simulation_folder: Path) -> list[Path]:
    """Restituisce tutti i frame phi della simulazione."""
    return sorted(
        path
        for path in simulation_folder.glob(FRAME_GLOB)
        if is_data_frame(path)
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    if not ROOT_DIR.is_dir():
        raise FileNotFoundError(
            f"Root directory non trovata: {ROOT_DIR}"
        )

    if N_FRAMES <= 0:
        raise ValueError(
            f"N_FRAMES deve essere maggiore di zero, trovato {N_FRAMES}"
        )

    simulation_folders = find_simulation_folders(ROOT_DIR)

    if not simulation_folders:
        raise FileNotFoundError(
            f"Nessun frame '{FRAME_GLOB}' trovato sotto {ROOT_DIR}"
        )

    short_simulations: list[tuple[Path, int]] = []
    total_frames = 0

    print("=" * 78)
    print("CONTROLLO LUNGHEZZA TRAIETTORIE 3D")
    print("=" * 78)
    print(f"Root directory              : {ROOT_DIR}")
    print(f"Numero simulazioni          : {len(simulation_folders)}")
    print(f"Numero minimo frame richiesto: {N_FRAMES}")
    print(f"Eliminazione attiva         : {DELETE_SHORT_SIMULATIONS}")

    for index, simulation_folder in enumerate(simulation_folders, start=1):
        n_frames = len(get_data_frames(simulation_folder))
        total_frames += n_frames

        relative_folder = simulation_folder.relative_to(ROOT_DIR)

        if n_frames < N_FRAMES:
            short_simulations.append((simulation_folder, n_frames))

            print(
                f"[{index:03d}/{len(simulation_folders):03d}] "
                f"SHORT   {relative_folder} -> {n_frames} frame "
                f"(< {N_FRAMES})"
            )
        else:
            print(
                f"[{index:03d}/{len(simulation_folders):03d}] "
                f"KEEP    {relative_folder} -> {n_frames} frame"
            )

    short_simulations.sort(key=lambda item: item[1])

    # Scrive sempre il report prima di qualunque delete.
    report_lines = [
        f"ROOT_DIR = {ROOT_DIR}",
        f"N_FRAMES = {N_FRAMES}",
        f"SIMULAZIONI TOTALI = {len(simulation_folders)}",
        f"SIMULAZIONI DA ELIMINARE = {len(short_simulations)}",
        "",
    ]

    for folder, n_frames in short_simulations:
        report_lines.append(
            f"{folder} -> {n_frames} frame (< {N_FRAMES})"
        )

    REPORT_PATH.write_text(
        "\n".join(report_lines) + "\n",
        encoding="utf-8",
    )

    print("\n" + "=" * 78)
    print("RIEPILOGO")
    print("=" * 78)
    print(f"Simulazioni totali         : {len(simulation_folders)}")
    print(f"Frame totali               : {total_frames}")
    print(f"Minimo frame richiesto     : {N_FRAMES}")
    print(f"Simulazioni sotto soglia   : {len(short_simulations)}")
    print(f"Report                     : {REPORT_PATH}")

    if not short_simulations:
        print("Nessuna cartella da eliminare.")
        return

    if not DELETE_SHORT_SIMULATIONS:
        print("\nModalità sicura: nessuna cartella è stata eliminata.")
        print("Controlla il report; poi modifica:")
        print("DELETE_SHORT_SIMULATIONS = True")
        print("e riesegui lo script per eliminare le simulazioni sotto soglia.")
        return

    print("\nELIMINAZIONE DELLE SIMULAZIONI SOTTO SOGLIA")

    deleted = 0

    for folder, n_frames in short_simulations:
        print(f"Elimino: {folder} -> {n_frames} frame")
        shutil.rmtree(folder)
        deleted += 1

    print(f"\nEliminate {deleted} simulazioni con meno di {N_FRAMES} frame.")


if __name__ == "__main__":
    main()
