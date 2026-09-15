from __future__ import annotations


import re
import time
from pathlib import Path

import numpy as np
import pyvista as pv
from scipy.interpolate import NearestNDInterpolator


# ============================================================
# CONFIGURAZIONE
# ============================================================

# Root che contiene le cartelle iso2_R..._H..._P...
INPUT_ROOT = Path(
    "/home/fiorello/mnt/data_train3D/hexagon/iso_P08"
)

# Root che riceve gli NPY.
OUTPUT_ROOT = Path(
    "/data/fiorello/pores3D/data/hexagon/iso_P08"
)

# Campo scalare nel point_data del VTU.
FIELD_NAME = "phi"

# Rigenera i file già esistenti.
OVERWRITE = True

# Salva una maschera diagnostica:
# 0 = campionamento VTK riuscito nella cella FEM
# 1 = punto non contenuto in una cella FEM, quindi fallback nearest
SAVE_FALLBACK_MASK = True

# Geometria fisica nominale del dominio esagonale.
LX = 0.4
LY = 0.69282005
EPS = 0.1

# Risoluzione nominale della griglia.
#
# x e z useranno esattamente VOXEL_SIZE.
# y userà il passo più vicino a VOXEL_SIZE che preserva LY esatto.
VOXEL_SIZE = 0.025

# Tolleranza per le coordinate floating-point del VTU.
GEOMETRY_TOL = 1e-6

VTU_GLOB = "surf_*.vtu"
ERROR_LOG_NAME = "conversion_errors.log"


# ============================================================
# PARSING E GEOMETRIA
# ============================================================

def extract_height(folder_name: str) -> float:
    """
    Esempio:
        iso2_R0.2_H1.0_P0.9 -> 1.0
    """
    match = re.search(
        r"(?:^|_)H(-?\d+(?:\.\d+)?)",
        folder_name,
    )

    if match is None:
        raise ValueError(
            f"Impossibile estrarre H dal nome cartella: {folder_name}"
        )

    return float(match.group(1))


def make_grid_preserving_hexagonal_domain(
    height: float,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Costruisce la griglia di campionamento.

    Dominio laterale:
        x in [x_min, x_min + LX]
        y in [y_min, y_min + LY]

    Regione verticale:
        z in [-height - 2*EPS, 2*EPS]

    Spacing:
        dx = VOXEL_SIZE
        dz = VOXEL_SIZE
        dy = LY / round(LY / VOXEL_SIZE)

    Per LX=0.4 e VOXEL_SIZE=0.025:
        Nx = 17
        dx = 0.025

    Per LY=0.69282005 e VOXEL_SIZE=0.025:
        Ny = 29
        dy = 0.69282005 / 28 ≈ 0.0247435732

    Gli endpoint sono inclusi e la geometria fisica originale
    del reticolo esagonale viene preservata.
    """
    mesh_lx = x_max - x_min
    mesh_ly = y_max - y_min

    if not np.isclose(
        mesh_lx,
        LX,
        rtol=0.0,
        atol=GEOMETRY_TOL,
    ):
        raise ValueError(
            f"Estensione x inattesa: {mesh_lx:.12g}; "
            f"attesa LX={LX:.12g}"
        )

    if not np.isclose(
        mesh_ly,
        LY,
        rtol=0.0,
        atol=GEOMETRY_TOL,
    ):
        raise ValueError(
            f"Estensione y inattesa: {mesh_ly:.12g}; "
            f"attesa LY={LY:.12g}"
        )

    z_start = -height - 2.0 * EPS
    z_end = 2.0 * EPS
    z_extent = z_end - z_start

    # X: spacing esattamente pari a VOXEL_SIZE.
    n_intervals_x = int(round(LX / VOXEL_SIZE))

    if not np.isclose(
        n_intervals_x * VOXEL_SIZE,
        LX,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError(
            f"LX={LX} non è multiplo esatto di voxel={VOXEL_SIZE}"
        )

    # Z: spacing esattamente pari a VOXEL_SIZE.
    n_intervals_z = int(round(z_extent / VOXEL_SIZE))

    if not np.isclose(
        n_intervals_z * VOXEL_SIZE,
        z_extent,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError(
            f"H={height}: estensione z={z_extent} non compatibile "
            f"con voxel={VOXEL_SIZE}"
        )

    # Y: preserva l'estensione esatta LY, con un numero di intervalli
    # scelto vicino a LY / VOXEL_SIZE.
    n_intervals_y = int(round(LY / VOXEL_SIZE))

    if n_intervals_y <= 0:
        raise ValueError(
            f"Numero intervalli y non valido: {n_intervals_y}"
        )

    nx = n_intervals_x + 1
    ny = n_intervals_y + 1
    nz = n_intervals_z + 1

    xi = np.linspace(
        x_min,
        x_min + LX,
        nx,
        dtype=np.float64,
    )

    yi = np.linspace(
        y_min,
        y_min + LY,
        ny,
        dtype=np.float64,
    )

    zi = np.linspace(
        z_start,
        z_end,
        nz,
        dtype=np.float64,
    )

    dx = float(xi[1] - xi[0])
    dy = float(yi[1] - yi[0])
    dz = float(zi[1] - zi[0])

    if not np.isclose(dx, VOXEL_SIZE, rtol=0.0, atol=1e-12):
        raise RuntimeError(
            f"Spacing x inatteso: dx={dx}, atteso={VOXEL_SIZE}"
        )

    if not np.isclose(dz, VOXEL_SIZE, rtol=0.0, atol=1e-12):
        raise RuntimeError(
            f"Spacing z inatteso: dz={dz}, atteso={VOXEL_SIZE}"
        )

    if not np.isclose(
        yi[-1] - yi[0],
        LY,
        rtol=0.0,
        atol=1e-12,
    ):
        raise RuntimeError(
            f"Estensione y errata: {yi[-1] - yi[0]}, attesa={LY}"
        )

    return xi, yi, zi


def make_query_points(
    xi: np.ndarray,
    yi: np.ndarray,
    zi: np.ndarray,
) -> np.ndarray:
    """
    Costruisce i punti query con ordine NumPy C.

    Dopo il sampling, i valori vengono ricostruiti tramite:

        values.reshape(nx, ny, nz)

    Assi dell'array NPY:
        axis 0 -> x
        axis 1 -> y
        axis 2 -> z
    """
    x_grid, y_grid, z_grid = np.meshgrid(
        xi,
        yi,
        zi,
        indexing="ij",
    )

    return np.column_stack(
        (
            x_grid.ravel(order="C"),
            y_grid.ravel(order="C"),
            z_grid.ravel(order="C"),
        )
    )


def output_path_for(vtu_path: Path) -> Path:
    """Mantiene la struttura relativa e sostituisce .vtu con .npy."""
    relative_path = vtu_path.relative_to(INPUT_ROOT)

    return (OUTPUT_ROOT / relative_path).with_suffix(".npy")


# ============================================================
# CAMPIONAMENTO VTK/PYVISTA
# ============================================================

def sample_mesh_with_pyvista(
    mesh: pv.DataSet,
    query_points: np.ndarray,
    output_shape: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Campiona FIELD_NAME sulla griglia usando le celle FEM.

    Ritorna:
        grid:
            float32, shape (nx, ny, nz)

        fallback_mask:
            uint8, shape (nx, ny, nz)

    Convenzione fallback_mask:
        0 = VTK ha trovato una cella FEM contenente il punto
        1 = VTK non ha trovato una cella; usato nearest-node fallback
    """
    if FIELD_NAME not in mesh.point_data:
        raise KeyError(
            f"Campo '{FIELD_NAME}' assente. "
            f"Disponibili: {list(mesh.point_data.keys())}"
        )

    target_points = pv.PolyData(query_points)

    # target.sample(source) trasferisce il campo della mesh sorgente
    # ai punti target tramite interpolazione nella cella FEM.
    sampled_target = target_points.sample(
        mesh,
        tolerance=GEOMETRY_TOL,
        pass_cell_data=False,
        pass_point_data=True,
        pass_field_data=False,
        mark_blank=True,
        locator="static_cell",
    )

    if FIELD_NAME not in sampled_target.point_data:
        raise RuntimeError(
            f"Campo '{FIELD_NAME}' assente dopo sample(). "
            f"Disponibili: {list(sampled_target.point_data.keys())}"
        )

    if "vtkValidPointMask" not in sampled_target.point_data:
        raise RuntimeError(
            "vtkValidPointMask assente dopo sample()."
        )

    sampled_values = np.asarray(
        sampled_target.point_data[FIELD_NAME]
    ).squeeze()

    valid_mask = np.asarray(
        sampled_target.point_data["vtkValidPointMask"]
    ).astype(bool)

    if sampled_values.ndim != 1:
        raise ValueError(
            "Campo campionato non scalare: "
            f"shape={sampled_values.shape}"
        )

    if sampled_values.size != query_points.shape[0]:
        raise RuntimeError(
            f"Numero valori campionati={sampled_values.size}, "
            f"attesi={query_points.shape[0]}"
        )

    if valid_mask.size != query_points.shape[0]:
        raise RuntimeError(
            f"Numero valid mask={valid_mask.size}, "
            f"attesi={query_points.shape[0]}"
        )

    fallback_mask_1d = ~valid_mask

    source_points = np.asarray(
        mesh.points[:, :3],
        dtype=np.float64,
    )

    source_values = np.asarray(
        mesh.point_data[FIELD_NAME]
    ).squeeze()

    if source_values.ndim != 1:
        raise ValueError(
            f"Campo sorgente non scalare: shape={source_values.shape}"
        )

    if source_values.size != source_points.shape[0]:
        raise ValueError(
            f"Campo sorgente con {source_values.size} valori, "
            f"ma mesh con {source_points.shape[0]} punti"
        )

    if not np.isfinite(source_values).all():
        n_nonfinite = int((~np.isfinite(source_values)).sum())

        raise ValueError(
            f"Il campo sorgente contiene {n_nonfinite} NaN/Inf"
        )

    # Costruito solo se serve, perché può essere relativamente costoso.
    nearest: NearestNDInterpolator | None = None

    def nearest_values(points: np.ndarray) -> np.ndarray:
        nonlocal nearest

        if nearest is None:
            nearest = NearestNDInterpolator(
                source_points,
                source_values,
            )

        return np.asarray(nearest(points))

    # Evita che VTK lasci valori zero/artificiali fuori dal dominio FEM.
    if fallback_mask_1d.any():
        sampled_values = sampled_values.astype(
            np.float64,
            copy=True,
        )

        sampled_values[fallback_mask_1d] = nearest_values(
            query_points[fallback_mask_1d]
        )

    # Protezione ulteriore contro NaN/Inf imprevisti.
    nonfinite_mask = ~np.isfinite(sampled_values)

    if nonfinite_mask.any():
        sampled_values = sampled_values.astype(
            np.float64,
            copy=True,
        )

        sampled_values[nonfinite_mask] = nearest_values(
            query_points[nonfinite_mask]
        )

        fallback_mask_1d |= nonfinite_mask

    if not np.isfinite(sampled_values).all():
        n_nonfinite = int((~np.isfinite(sampled_values)).sum())

        raise RuntimeError(
            f"Restano {n_nonfinite} valori NaN/Inf dopo il fallback"
        )

    grid = sampled_values.astype(
        np.float32,
        copy=False,
    ).reshape(output_shape, order="C")

    fallback_mask = fallback_mask_1d.astype(
        np.uint8,
        copy=False,
    ).reshape(output_shape, order="C")

    return grid, fallback_mask


def convert_one_vtu(
    vtu_path: Path,
    out_path: Path,
    query_points: np.ndarray,
    output_shape: tuple[int, int, int],
) -> tuple[int, int, float, float]:
    """
    Converte un VTU in NPY.

    Ritorna:
        n_fallback
        n_total
        value_min
        value_max
    """
    mesh = pv.read(vtu_path)

    grid, fallback_mask = sample_mesh_with_pyvista(
        mesh=mesh,
        query_points=query_points,
        output_shape=output_shape,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, grid)

    if SAVE_FALLBACK_MASK:
        mask_path = out_path.with_name(
            f"{out_path.stem}_vtk_fallback_mask.npy"
        )

        np.save(mask_path, fallback_mask)

    return (
        int(fallback_mask.sum()),
        int(fallback_mask.size),
        float(grid.min()),
        float(grid.max()),
    )


# ============================================================
# PROCESSAMENTO DI UNA SIMULAZIONE
# ============================================================

def process_folder(folder_path: Path) -> tuple[int, int]:
    """
    Processa tutti i surf_*.vtu presenti nella simulazione.

    Costruisce una sola griglia per folder. Tutti i frame della
    stessa traiettoria avranno la stessa shape e gli stessi assi.
    """
    vtu_files = sorted(folder_path.glob(VTU_GLOB))

    if not vtu_files:
        return 0, 0

    height = extract_height(folder_path.name)

    # Primo frame usato solo per estrarre geometria e creare la griglia.
    first_mesh = pv.read(vtu_files[0])

    first_points = np.asarray(
        first_mesh.points[:, :3],
        dtype=np.float64,
    )

    x_min, y_min, z_min = first_points.min(axis=0)
    x_max, y_max, z_max = first_points.max(axis=0)

    xi, yi, zi = make_grid_preserving_hexagonal_domain(
        height=height,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
    )

    z_start = float(zi[0])
    z_end = float(zi[-1])

    if (
        z_start < z_min - GEOMETRY_TOL
        or z_end > z_max + GEOMETRY_TOL
    ):
        raise ValueError(
            f"Crop z richiesto=[{z_start:.8f}, {z_end:.8f}] "
            f"fuori dai bounds VTU=[{z_min:.8f}, {z_max:.8f}]"
        )

    query_points = make_query_points(
        xi=xi,
        yi=yi,
        zi=zi,
    )

    output_shape = (
        len(xi),
        len(yi),
        len(zi),
    )

    output_folder = OUTPUT_ROOT / folder_path.relative_to(
        INPUT_ROOT
    )

    output_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata_path = output_folder / "grid_metadata.npz"

    if OVERWRITE or not metadata_path.exists():
        np.savez(
            metadata_path,
            xi=xi,
            yi=yi,
            zi=zi,
            shape=np.array(output_shape, dtype=np.int32),
            height=np.float64(height),
            eps=np.float64(EPS),
            voxel_size_nominal=np.float64(VOXEL_SIZE),
            dx=np.float64(xi[1] - xi[0]),
            dy=np.float64(yi[1] - yi[0]),
            dz=np.float64(zi[1] - zi[0]),
            lx=np.float64(LX),
            ly=np.float64(LY),
            sampling_method=np.array(
                "pyvista_vtk_cell_sample"
            ),
            fallback_method=np.array(
                "nearest_node"
            ),
        )

    dx = float(xi[1] - xi[0])
    dy = float(yi[1] - yi[0])
    dz = float(zi[1] - zi[0])

    print("\n" + "=" * 86)
    print(f"Folder input : {folder_path}")
    print(f"Folder output: {output_folder}")
    print(f"Height       : {height}")
    print(f"Frame trovati: {len(vtu_files)}")
    print(f"Shape output : {output_shape}")
    print(
        "Spacing      : "
        f"dx={dx:.10f}, dy={dy:.10f}, dz={dz:.10f}"
    )
    print(
        "Regione x    : "
        f"[{xi[0]:.10f}, {xi[-1]:.10f}]"
    )
    print(
        "Regione y    : "
        f"[{yi[0]:.10f}, {yi[-1]:.10f}]"
    )
    print(
        "Regione z    : "
        f"[{z_start:.10f}, {z_end:.10f}]"
    )

    converted = 0
    skipped = 0

    for index, vtu_path in enumerate(vtu_files, start=1):
        out_path = output_path_for(vtu_path)

        if out_path.exists() and not OVERWRITE:
            print(
                f"[{index:04d}/{len(vtu_files):04d}] "
                f"SKIP {out_path.name}"
            )

            skipped += 1
            continue

        start_time = time.perf_counter()

        (
            n_fallback,
            n_total,
            value_min,
            value_max,
        ) = convert_one_vtu(
            vtu_path=vtu_path,
            out_path=out_path,
            query_points=query_points,
            output_shape=output_shape,
        )

        elapsed = time.perf_counter() - start_time
        fallback_percent = 100.0 * n_fallback / n_total

        print(
            f"[{index:04d}/{len(vtu_files):04d}] "
            f"DONE {out_path.name} | "
            f"min={value_min:.7f}, max={value_max:.7f} | "
            f"fallback={n_fallback}/{n_total} "
            f"({fallback_percent:.4f}%) | "
            f"{elapsed:.2f} s"
        )

        converted += 1

    return converted, skipped


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    if not INPUT_ROOT.is_dir():
        raise FileNotFoundError(
            f"Input root non trovata: {INPUT_ROOT}"
        )

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    folders = sorted(
        {
            vtu_path.parent
            for vtu_path in INPUT_ROOT.rglob(VTU_GLOB)
        }
    )

    if not folders:
        raise FileNotFoundError(
            f"Nessun file '{VTU_GLOB}' trovato sotto {INPUT_ROOT}"
        )

    print("=" * 86)
    print("CONVERSIONE VTU -> NPY: RETICOLO ESAGONALE")
    print("=" * 86)
    print(f"Input root        : {INPUT_ROOT}")
    print(f"Output root       : {OUTPUT_ROOT}")
    print(f"Campo             : {FIELD_NAME}")
    print("Sampling          : PyVista/VTK cell-based sample")
    print("Fallback          : nearest node")
    print(f"LX                : {LX}")
    print(f"LY                : {LY}")
    print(f"EPS               : {EPS}")
    print(f"Voxel nominale    : {VOXEL_SIZE}")
    print(f"Overwrite         : {OVERWRITE}")
    print(f"Salva fallback mask: {SAVE_FALLBACK_MASK}")
    print(f"Cartelle trovate  : {len(folders)}")

    total_converted = 0
    total_skipped = 0
    errors: list[str] = []

    global_start_time = time.perf_counter()

    for folder_index, folder_path in enumerate(
        folders,
        start=1,
    ):
        print(
            f"\n######## CARTELLA "
            f"{folder_index}/{len(folders)} ########"
        )

        try:
            converted, skipped = process_folder(folder_path)

            total_converted += converted
            total_skipped += skipped

        except Exception as exc:
            message = (
                f"{folder_path}: "
                f"{type(exc).__name__}: {exc}"
            )

            errors.append(message)

            print(f"[ERROR] {message}")

    elapsed = time.perf_counter() - global_start_time

    print("\n" + "=" * 86)
    print("RIEPILOGO")
    print("=" * 86)
    print(f"Convertiti  : {total_converted}")
    print(f"Saltati     : {total_skipped}")
    print(f"Falliti     : {len(errors)}")
    print(f"Tempo totale: {elapsed:.2f} s")

    if errors:
        error_log = OUTPUT_ROOT / ERROR_LOG_NAME

        error_log.write_text(
            "\n".join(errors) + "\n",
            encoding="utf-8",
        )

        print(f"Log errori  : {error_log}")


if __name__ == "__main__":
    main()