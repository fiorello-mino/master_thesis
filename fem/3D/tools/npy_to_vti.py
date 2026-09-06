from __future__ import annotations

from pathlib import Path

import numpy as np
import pyvista as pv


# ============================================================
# CONFIGURAZIONE
# ============================================================

# Cartella di UNA singola simulazione.
SIMULATION_DIR = Path(
    "/data/fiorello/poresAMDIS/"
    "iso_P09/"
    "iso2_R0.2_H1.0_P0.9"
)

# Cartella dove salvare i file VTK/XML.
# Puoi usare una sottocartella per non mischiare NPY e VTI.
OUTPUT_DIR = SIMULATION_DIR / "vtk"

# Campo mostrato in ParaView.
FIELD_NAME = "phi"

# Numero massimo di frame da convertire.
#
# None -> converte tutti gli NPY della simulazione.
# 3    -> converte solo primo, centrale e ultimo frame.
MAX_FRAMES = None

# Esclude i file mask creati durante il fallback PyVista/VTK.
VTK_MASK_SUFFIX = "_vtk_fallback_mask.npy"


# ============================================================
# FUNZIONI
# ============================================================

def is_data_frame(path: Path) -> bool:
    """
    Accetta i frame del campo phi:

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


def select_frames(frame_paths: list[Path]) -> list[Path]:
    """
    Se MAX_FRAMES=None restituisce tutti i frame.

    Se MAX_FRAMES=3 restituisce primo, centrale e ultimo.
    Per altri valori seleziona frame circa equidistanti.
    """
    if MAX_FRAMES is None or MAX_FRAMES >= len(frame_paths):
        return frame_paths

    if MAX_FRAMES <= 0:
        raise ValueError("MAX_FRAMES deve essere None oppure maggiore di zero")

    indices = np.linspace(
        0,
        len(frame_paths) - 1,
        MAX_FRAMES,
        dtype=int,
    )

    # Rimuove eventuali indici duplicati per dataset molto piccoli.
    indices = np.unique(indices)

    return [frame_paths[index] for index in indices]


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    if not SIMULATION_DIR.is_dir():
        raise FileNotFoundError(
            f"Cartella simulazione non trovata: {SIMULATION_DIR}"
        )

    metadata_path = SIMULATION_DIR / "grid_metadata.npz"

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Metadata non trovato: {metadata_path}"
        )

    metadata = np.load(metadata_path, allow_pickle=False)

    xi = metadata["xi"]
    yi = metadata["yi"]
    zi = metadata["zi"]

    nx = len(xi)
    ny = len(yi)
    nz = len(zi)

    if nx < 2 or ny < 2 or nz < 2:
        raise ValueError(
            f"Griglia troppo piccola: {(nx, ny, nz)}"
        )

    dx = float(xi[1] - xi[0])
    dy = float(yi[1] - yi[0])
    dz = float(zi[1] - zi[0])

    origin = (
        float(xi[0]),
        float(yi[0]),
        float(zi[0]),
    )

    spacing = (dx, dy, dz)
    dimensions = (nx, ny, nz)

    frame_paths = sorted(
        path
        for path in SIMULATION_DIR.glob("surf_*.npy")
        if is_data_frame(path)
    )

    if not frame_paths:
        raise FileNotFoundError(
            f"Nessun frame surf_*.npy trovato in {SIMULATION_DIR}"
        )

    selected_frames = select_frames(frame_paths)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("CONVERSIONE NPY -> VTK IMAGE DATA")
    print("=" * 78)
    print(f"Simulazione     : {SIMULATION_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Frame trovati   : {len(frame_paths)}")
    print(f"Frame selezionati: {len(selected_frames)}")
    print(f"Shape           : {dimensions}")
    print(f"Origin          : {origin}")
    print(f"Spacing         : {spacing}")

    for index, npy_path in enumerate(selected_frames, start=1):
        phi = np.load(npy_path, allow_pickle=False)

        if phi.shape != dimensions:
            raise ValueError(
                f"Shape errata in {npy_path.name}: "
                f"{phi.shape}, attesa {dimensions}"
            )

        if not np.all(np.isfinite(phi)):
            raise ValueError(
                f"{npy_path.name} contiene NaN o Inf"
            )

        # ImageData rappresenta una griglia regolare VTK.
        image = pv.ImageData(
            dimensions=dimensions,
            spacing=spacing,
            origin=origin,
        )

        # Importante:
        # VTK/PyVista usa ordinamento Fortran per associare il vettore
        # 1D ai punti della griglia (x varia più velocemente).
        #
        # Il tuo NPY ha assi (x, y, z), quindi flatten(order='F')
        # mantiene correttamente phi[i, j, k] in (xi[i], yi[j], zi[k]).
        image.point_data[FIELD_NAME] = phi.ravel(order="F").astype(
            np.float32
        )

        output_path = OUTPUT_DIR / f"{npy_path.stem}.vti"

        image.save(output_path)

        print(
            f"[{index:04d}/{len(selected_frames):04d}] "
            f"Salvato: {output_path.name} | "
            f"min={phi.min():.6f}, max={phi.max():.6f}"
        )

    print("\nConversione completata.")
    print(f"Apri in ParaView i file .vti in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()