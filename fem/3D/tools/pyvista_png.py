from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# CONFIGURAZIONE
# ============================================================

# Root contenente le cartelle convertite con resizeFolder.py/PyVista.
INPUT_ROOT = Path("/data/fiorello/iso_P09")

# Root separata dove salvare i PNG; mantiene la struttura delle sottocartelle.
OUTPUT_ROOT = Path("/data/fiorello/iso_P09_png_yz_x0")

# Cerca soltanto i veri frame phi, non le mask VTK.
DATA_GLOB = "surf_*.npy"
VTK_MASK_SUFFIX = "_vtk_fallback_mask.npy"

# Valore fisico della slice richiesta.
X_TARGET = 0.0

# Colormap e range di phi. Modifica vmin/vmax se il tuo phi non è in [-1, 1].
CMAP = "coolwarm"
VMIN = -1.0
VMAX = 1.0

# Per una visualizzazione fedele ai voxel isotropi.
INTERPOLATION = "nearest"
DPI = 250


# ============================================================
# FUNZIONI
# ============================================================

def is_data_frame(path: Path) -> bool:
    """True soltanto per surf_<time>.npy, escludendo le fallback mask."""
    return (
        path.is_file()
        and path.name.startswith("surf_")
        and path.suffix == ".npy"
        and not path.name.endswith(VTK_MASK_SUFFIX)
    )


def find_trajectory_folders(root: Path) -> list[Path]:
    """Trova le cartelle che contengono direttamente almeno un frame phi."""
    return sorted(
        {
            path.parent
            for path in root.rglob(DATA_GLOB)
            if is_data_frame(path)
        }
    )


def select_initial_middle_final(frame_paths: list[Path]) -> list[tuple[str, Path]]:
    """
    Seleziona primo, centrale e ultimo frame senza duplicati.

    Con un solo frame salva solo 'initial'.
    Con due frame salva 'initial' e 'final'.
    """
    n_frames = len(frame_paths)

    if n_frames == 0:
        return []

    selected_indices = [
        ("initial", 0),
        ("middle", n_frames // 2),
        ("final", n_frames - 1),
    ]

    selected: list[tuple[str, Path]] = []
    seen_indices: set[int] = set()

    for label, index in selected_indices:
        if index not in seen_indices:
            selected.append((label, frame_paths[index]))
            seen_indices.add(index)

    return selected


def save_yz_png(
    frame_path: Path,
    metadata_path: Path,
    out_path: Path,
    x_target: float,
    label: str,
) -> tuple[int, float]:
    """
    Salva la vista YZ a x vicino a x_target.

    Ritorna:
        x_index, x_coordinate_effettiva
    """
    grid = np.load(frame_path)
    metadata = np.load(metadata_path, allow_pickle=False)

    xi = metadata["xi"]
    yi = metadata["yi"]
    zi = metadata["zi"]

    expected_shape = (len(xi), len(yi), len(zi))

    if grid.shape != expected_shape:
        raise ValueError(
            f"Shape incompatibile per {frame_path.name}: "
            f"NPY={grid.shape}, coordinate={expected_shape}"
        )

    if grid.ndim != 3:
        raise ValueError(
            f"{frame_path.name} non è 3D: shape={grid.shape}"
        )

    # Prende il nodo della griglia x più vicino a X_TARGET=0.
    x_index = int(np.argmin(np.abs(xi - x_target)))
    x_value = float(xi[x_index])

    # grid: (Nx, Ny, Nz)
    # fissando x: (Ny, Nz)
    slice_yz = grid[x_index, :, :]

    # imshow interpreta la prima dimensione come verticale.
    # Trasponendo (Ny, Nz) -> (Nz, Ny):
    #   orizzontale = y
    #   verticale   = z
    image_yz = slice_yz.T

    extent = [
        float(yi[0]),
        float(yi[-1]),
        float(zi[0]),
        float(zi[-1]),
    ]

    fig, ax = plt.subplots(figsize=(7, 10))

    im = ax.imshow(
        image_yz,
        origin="lower",
        extent=extent,
        aspect="equal",
        interpolation=INTERPOLATION,
        cmap=CMAP,
        vmin=VMIN,
        vmax=VMAX,
    )

    colorbar = fig.colorbar(im, ax=ax)
    colorbar.set_label("phi")

    ax.set_xlabel("y")
    ax.set_ylabel("z")
    ax.set_title(
        f"{label} | {frame_path.stem} | x={x_value:.6f}"
    )

    # Segna z=0 e z=-height, se height è disponibile nei metadata.
    if "height" in metadata.files:
        height = float(metadata["height"])

        ax.axhline(
            0.0,
            color="black",
            linestyle="--",
            linewidth=0.8,
            alpha=0.6,
        )

        ax.axhline(
            -height,
            color="black",
            linestyle="--",
            linewidth=0.8,
            alpha=0.6,
        )

    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)

    return x_index, x_value


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    if not INPUT_ROOT.is_dir():
        raise FileNotFoundError(f"Input root non trovata: {INPUT_ROOT}")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    folders = find_trajectory_folders(INPUT_ROOT)

    if not folders:
        raise FileNotFoundError(
            f"Nessun frame '{DATA_GLOB}' trovato sotto {INPUT_ROOT}"
        )

    print(f"Input root : {INPUT_ROOT}")
    print(f"Output root: {OUTPUT_ROOT}")
    print(f"Piano      : YZ a x vicino a {X_TARGET}")
    print(f"Cartelle   : {len(folders)}")

    total_png = 0
    errors: list[str] = []

    for folder_index, folder in enumerate(folders, start=1):
        try:
            frame_paths = sorted(
                path
                for path in folder.glob(DATA_GLOB)
                if is_data_frame(path)
            )

            if not frame_paths:
                continue

            metadata_path = folder / "grid_metadata.npz"
            if not metadata_path.exists():
                raise FileNotFoundError(
                    f"Metadata mancante: {metadata_path}"
                )

            relative_folder = folder.relative_to(INPUT_ROOT)
            out_folder = OUTPUT_ROOT / relative_folder

            selected_frames = select_initial_middle_final(frame_paths)

            print("\n" + "=" * 78)
            print(f"[{folder_index:03d}/{len(folders):03d}] {relative_folder}")
            print(f"Frame disponibili: {len(frame_paths)}")

            for label, frame_path in selected_frames:
                out_path = out_folder / f"{label}_{frame_path.stem}_yz_x0.png"

                x_index, x_value = save_yz_png(
                    frame_path=frame_path,
                    metadata_path=metadata_path,
                    out_path=out_path,
                    x_target=X_TARGET,
                    label=label,
                )

                print(
                    f"  {label:7s}: {frame_path.name} -> {out_path.name} "
                    f"(x-index={x_index}, x={x_value:.8f})"
                )

                total_png += 1

        except Exception as exc:
            message = f"{folder}: {type(exc).__name__}: {exc}"
            errors.append(message)
            print(f"[ERROR] {message}")

    print("\n" + "=" * 78)
    print("RIEPILOGO")
    print(f"PNG creati : {total_png}")
    print(f"Errori     : {len(errors)}")

    if errors:
        error_log = OUTPUT_ROOT / "png_generation_errors.log"
        error_log.write_text("\n".join(errors) + "\n", encoding="utf-8")
        print(f"Log errori: {error_log}")


if __name__ == "__main__":
    main()
