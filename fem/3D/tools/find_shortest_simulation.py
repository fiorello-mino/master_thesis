from __future__ import annotations

from pathlib import Path


# ============================================================
# CONFIGURAZIONE
# ============================================================

ROOT_DIR = Path("/data/fiorello/poresAMDIS")

FRAME_GLOB = "surf_*.npy"


# ============================================================
# FUNZIONI
# ============================================================

def is_data_frame(path: Path) -> bool:
    """
    Restituisce True solo per i veri frame phi.

    Include:
        surf_0.000000.npy

    """
    return (
        path.is_file()
        and path.name.startswith("surf_")
        and path.suffix == ".npy"
    )


def find_simulation_folders(root_dir: Path) -> list[Path]:
    """
    Trova tutte le cartelle che contengono direttamente almeno un frame .npy.

    Esempio di struttura attesa:

        ROOT_DIR/
        ├── iso_P06/
        │   ├── iso2_R0.2_H1.0_P0.6/
        │   │   ├── surf_0.000000.npy
        │   │   └── ...
        │   └── ...
        ├── iso_P07/
        └── iso_P10/
    """
    return sorted(
        {
            npy_path.parent
            for npy_path in root_dir.rglob(FRAME_GLOB)
            if is_data_frame(npy_path)
        }
    )


def get_data_frames(simulation_folder: Path) -> list[Path]:
    """Restituisce in ordine lessicografico i frame phi di una simulazione."""
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

    simulation_folders = find_simulation_folders(ROOT_DIR)

    if not simulation_folders:
        raise FileNotFoundError(
            f"Nessun frame '{FRAME_GLOB}' trovato sotto {ROOT_DIR}"
        )

    shortest_folder: Path | None = None
    shortest_length: int | None = None

    longest_folder: Path | None = None
    longest_length: int | None = None

    total_frames = 0
    frame_counts: list[int] = []

    print("=" * 78)
    print("CONTEGGIO TRAIETTORIE 3D")
    print("=" * 78)
    print(f"Root directory : {ROOT_DIR}")
    print(f"Simulazioni    : {len(simulation_folders)}")
    print()

    for index, simulation_folder in enumerate(simulation_folders, start=1):
        frames = get_data_frames(simulation_folder)
        n_frames = len(frames)

        relative_folder = simulation_folder.relative_to(ROOT_DIR)

        print(
            f"[{index:03d}/{len(simulation_folders):03d}] "
            f"{relative_folder} -> {n_frames} frame"
        )

        total_frames += n_frames
        frame_counts.append(n_frames)

        if shortest_length is None or n_frames < shortest_length:
            shortest_length = n_frames
            shortest_folder = simulation_folder

        if longest_length is None or n_frames > longest_length:
            longest_length = n_frames
            longest_folder = simulation_folder

    assert shortest_folder is not None
    assert shortest_length is not None
    assert longest_folder is not None
    assert longest_length is not None

    mean_frames = total_frames / len(simulation_folders)

    print("\n" + "=" * 78)
    print("RIEPILOGO")
    print("=" * 78)
    print(f"Numero simulazioni       : {len(simulation_folders)}")
    print(f"Numero totale frame      : {total_frames}")
    print(f"Media frame/simulazione  : {mean_frames:.2f}")
    print(f"\nSimulazione più breve   : {shortest_length} frame")
    print(f"Path                     : {shortest_folder}")
    print(f"\nSimulazione più lunga   : {longest_length} frame")
    print(f"Path                     : {longest_folder}")


if __name__ == "__main__":
    main()
