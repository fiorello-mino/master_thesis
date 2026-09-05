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
INPUT_ROOT = Path("/archive/roberto/poresAMDIS/iso_P09")

# Root che riceve gli NPY.
OUTPUT_ROOT = Path("/data/fiorello/iso_P09_pyvista")

# Campo scalare nel point_data del VTU.
FIELD_NAME = "phi"

# Rigenera i file già esistenti.
# Metti True almeno una volta per sostituire gli NPY prodotti
# in precedenza con scipy.griddata.
OVERWRITE = True

# Salva una maschera diagnostica:
# 0 = campionamento VTK riuscito nella cella FEM
# 1 = punto non contenuto in una cella FEM, quindi fallback nearest
SAVE_FALLBACK_MASK = True

# Geometria fisica nominale.
LX = 0.45
LY = 0.45
EPS = 0.1

# Passo isotropo della griglia NPY.
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
    match = re.search(r"(?:^|_)H(-?\d+(?:\.\d+)?)", folder_name)

    if match is None:
        raise ValueError(
            f"Impossibile estrarre H dal nome cartella: {folder_name}"
        )

    return float(match.group(1))


def make_isotropic_grid(
    height: float,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Costruisce una griglia isotropa con endpoint inclusi.

    Regione z salvata:
        [-height - 2*EPS, 2*EPS]

    Per H=1.0 ed EPS=0.1:
        [-1.2, 0.2]

    Restituisce xi, yi, zi con:
        dx = dy = dz = VOXEL_SIZE
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
            f"attesa {LX:.12g}"
        )

    if not np.isclose(
        mesh_ly,
        LY,
        rtol=0.0,
        atol=GEOMETRY_TOL,
    ):
        raise ValueError(
            f"Estensione y inattesa: {mesh_ly:.12g}; "
            f"attesa {LY:.12g}"
        )

    z_start = -height - 2.0 * EPS
    z_end = 2.0 * EPS
    crop_height = z_end - z_start

    n_intervals_x = int(round(LX / VOXEL_SIZE))
    n_intervals_y = int(round(LY / VOXEL_SIZE))
    n_intervals_z = int(round(crop_height / VOXEL_SIZE))

    if not np.isclose(
        n_intervals_x * VOXEL_SIZE,
        LX,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError(
            f"LX={LX} non è multiplo esatto di {VOXEL_SIZE}"
        )

    if not np.isclose(
        n_intervals_y * VOXEL_SIZE,
        LY,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError(
            f"LY={LY} non è multiplo esatto di {VOXEL_SIZE}"
        )

    if not np.isclose(
        n_intervals_z * VOXEL_SIZE,
        crop_height,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError(
            f"H={height}: estensione z={crop_height} non compatibile "
            f"con voxel={VOXEL_SIZE}"
        )

    nx = n_intervals_x + 1
    ny = n_intervals_y + 1
    nz = n_intervals_z + 1

    # Estremi nominali: non propagano piccoli errori float del VTU.
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

    dx = xi[1] - xi[0]
    dy = yi[1] - yi[0]
    dz = zi[1] - zi[0]

    if not (
        np.isclose(dx, VOXEL_SIZE, atol=1e-12)
        and np.isclose(dy, VOXEL_SIZE, atol=1e-12)
        and np.isclose(dz, VOXEL_SIZE, atol=1e-12)
    ):
        raise RuntimeError(
            f"Griglia non isotropa: dx={dx}, dy={dy}, dz={dz}"
        )

    return xi, yi, zi


def make_query_points(
    xi: np.ndarray,
    yi: np.ndarray,
    zi: np.ndarray,
) -> np.ndarray:
    """
    Costruisce punti con ordine NumPy C compatibile con:

        values.reshape(nx, ny, nz)

    Assi NPY:
        0 -> x
        1 -> y
        2 -> z
    """
    X, Y, Z = np.meshgrid(
        xi,
        yi,
        zi,
        indexing="ij",
    )

    return np.column_stack(
        (
            X.ravel(),
            Y.ravel(),
            Z.ravel(),
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
    Campiona FIELD_NAME sulla griglia di query usando le celle FEM.

    Restituisce:
        grid          -> float32, shape (nx, ny, nz)
        fallback_mask -> uint8,  shape (nx, ny, nz)

    fallback_mask:
        0 = VTK ha trovato una cella FEM contenente il punto
        1 = VTK non ha trovato una cella; applicato nearest fallback
    """
    if FIELD_NAME not in mesh.point_data:
        raise KeyError(
            f"Campo '{FIELD_NAME}' assente. "
            f"Disponibili: {list(mesh.point_data.keys())}"
        )

    # PolyData mantiene esattamente l'ordine di query_points;
    # questo evita ambiguità di ordinamento VTK/NumPy.
    target_points = pv.PolyData(query_points)

    # sample(source_mesh): trasferisce i dati del source mesh
    # ai punti del target, interpolando nella cella contenente il punto.
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
            f"Il campo '{FIELD_NAME}' non è presente dopo sample(). "
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
            f"Campo campionato non scalare: shape={sampled_values.shape}"
        )

    if len(sampled_values) != len(query_points):
        raise RuntimeError(
            f"Numero valori campionati={len(sampled_values)}, "
            f"attesi={len(query_points)}"
        )

    if len(valid_mask) != len(query_points):
        raise RuntimeError(
            f"Numero valid mask={len(valid_mask)}, "
            f"attesi={len(query_points)}"
        )

    # I punti non validi non devono diventare phi=0.
    # Applica nearest ai soli punti fuori dalle celle FEM.
    fallback_mask_1d = ~valid_mask

    if fallback_mask_1d.any():
        source_points = np.asarray(
            mesh.points[:, :3],
            dtype=np.float64,
        )

        source_values = np.asarray(
            mesh.point_data[FIELD_NAME]
        ).squeeze()

        nearest = NearestNDInterpolator(
            source_points,
            source_values,
        )

        sampled_values[fallback_mask_1d] = nearest(
            query_points[fallback_mask_1d]
        )

    # Protezione ulteriore nel caso di valori non finiti inattesi.
    invalid_mask = ~np.isfinite(sampled_values)

    if invalid_mask.any():
        source_points = np.asarray(
            mesh.points[:, :3],
            dtype=np.float64,
        )

        source_values = np.asarray(
            mesh.point_data[FIELD_NAME]
        ).squeeze()

        nearest = NearestNDInterpolator(
            source_points,
            source_values,
        )

        sampled_values[invalid_mask] = nearest(
            query_points[invalid_mask]
        )

        fallback_mask_1d |= invalid_mask

    if not np.all(np.isfinite(sampled_values)):
        n_invalid = int((~np.isfinite(sampled_values)).sum())

        raise RuntimeError(
            f"Restano {n_invalid} valori NaN/Inf dopo fallback nearest"
        )

    grid = sampled_values.astype(np.float32).reshape(output_shape)

    fallback_mask = fallback_mask_1d.astype(np.uint8).reshape(
        output_shape
    )

    return grid, fallback_mask


def convert_one_vtu(
    vtu_path: Path,
    out_path: Path,
    query_points: np.ndarray,
    output_shape: tuple[int, int, int],
) -> tuple[int, int]:
    """
    Legge un VTU, lo campiona con PyVista/VTK e salva l'NPY.

    Ritorna:
        n_fallback, n_total
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

    n_fallback = int(fallback_mask.sum())
    n_total = int(fallback_mask.size)

    return n_fallback, n_total


# ============================================================
# PROCESSAMENTO DI UNA CARTELLA
# ============================================================

def process_folder(folder_path: Path) -> tuple[int, int]:
    """
    Processa tutti i surf_*.vtu in una cartella.

    Una sola griglia xi/yi/zi viene costruita per folder:
    tutti i frame della traiettoria avranno stessa shape.
    """
    vtu_files = sorted(folder_path.glob(VTU_GLOB))

    if not vtu_files:
        return 0, 0

    height = extract_height(folder_path.name)

    # Primo frame: soltanto per estremi geometrici e griglia comune.
    first_mesh = pv.read(vtu_files[0])

    first_points = np.asarray(
        first_mesh.points[:, :3],
        dtype=np.float64,
    )

    x_min, y_min, z_min = first_points.min(axis=0)
    x_max, y_max, z_max = first_points.max(axis=0)

    xi, yi, zi = make_isotropic_grid(
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
            f"Crop richiesto z=[{z_start:.8f}, {z_end:.8f}] "
            f"fuori dal dominio VTU z=[{z_min:.8f}, {z_max:.8f}]"
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

    output_folder = OUTPUT_ROOT / folder_path.relative_to(INPUT_ROOT)

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
            voxel_size=np.float64(VOXEL_SIZE),
            dx=np.float64(xi[1] - xi[0]),
            dy=np.float64(yi[1] - yi[0]),
            dz=np.float64(zi[1] - zi[0]),
            sampling_method=np.array("pyvista_vtk_cell_sample"),
            fallback_method=np.array("nearest_node"),
        )

    print("\n" + "=" * 78)
    print(f"Folder input : {folder_path}")
    print(f"Folder output: {output_folder}")
    print(f"Height       : {height}")
    print(f"Frame trovati: {len(vtu_files)}")
    print(f"Shape output : {output_shape}")
    print(
        "Spacing      : "
        f"dx={xi[1] - xi[0]:.8f}, "
        f"dy={yi[1] - yi[0]:.8f}, "
        f"dz={zi[1] - zi[0]:.8f}"
    )
    print(f"Regione z    : [{z_start:.8f}, {z_end:.8f}]")

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

        t0 = time.perf_counter()

        n_fallback, n_total = convert_one_vtu(
            vtu_path=vtu_path,
            out_path=out_path,
            query_points=query_points,
            output_shape=output_shape,
        )

        elapsed = time.perf_counter() - t0
        fallback_percent = 100.0 * n_fallback / n_total

        print(
            f"[{index:04d}/{len(vtu_files):04d}] "
            f"DONE {out_path.name} | "
            f"VTK->nearest fallback: "
            f"{n_fallback}/{n_total} "
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

    print(f"Input root    : {INPUT_ROOT}")
    print(f"Output root   : {OUTPUT_ROOT}")
    print(f"Campo         : {FIELD_NAME}")
    print("Sampling      : PyVista/VTK cell-based sample")
    print("Fallback      : nearest node")
    print(f"Voxel size    : {VOXEL_SIZE}")
    print(f"Overwrite     : {OVERWRITE}")
    print(f"Cartelle      : {len(folders)}")

    total_converted = 0
    total_skipped = 0
    errors: list[str] = []

    global_start = time.perf_counter()

    for folder_index, folder_path in enumerate(folders, start=1):
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

    elapsed = time.perf_counter() - global_start

    print("\n" + "=" * 78)
    print("RIEPILOGO")
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
