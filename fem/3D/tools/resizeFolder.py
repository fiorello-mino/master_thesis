from __future__ import annotations

import re
import time
from pathlib import Path

import numpy as np
import pyvista as pv
from scipy.interpolate import griddata


# ============================================================
# CONFIGURAZIONE
# ============================================================

INPUT_ROOT = Path("/archive/roberto/poreAMDIS/iso_P09")
OUTPUT_ROOT = Path("/data/fiorello/iso_P09")

FIELD_NAME = "phi"
METHOD = "nearest"  # "linear" per output finale; "nearest" per un test rapido
OVERWRITE = False

# Geometria fisica
LX = 0.45
LY = 0.45
EPS = 0.1

# Passo isotropo della griglia finale.
# È compatibile esattamente con LX=LY=0.45 e H=1.0, 1.1, ..., 4.9.
DX_SIZE = 0.0125

# Tutte le cartelle di input sono ricercate ricorsivamente;
# i file da convertire devono chiamarsi surf_*.vtu.
VTU_GLOB = "surf_*.vtu"
ERROR_LOG_NAME = "conversion_errors.log"


# ============================================================
# GEOMETRIA E PATH
# ============================================================

def extract_height(folder_name: str) -> float:
    """Estrae H dal nome della cartella, ad esempio H1.0 -> 1.0."""
    match = re.search(r"(?:^|_)H(-?\d+(?:\.\d+)?)", folder_name)
    if match is None:
        raise ValueError(
            f"Impossibile trovare il parametro H nel nome: {folder_name}"
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
    Costruisce la griglia del crop con passo isotropo DX_SIZE.

    Regione mantenuta:
        x in [x_min, x_max]
        y in [y_min, y_max]
        z in [-height - 2*EPS, 2*EPS]

    La geometria nominale richiede:
        x_max - x_min = LX
        y_max - y_min = LY

    Gli endpoint sono inclusi; quindi N = intervalli + 1.
    """
    mesh_lx = x_max - x_min
    mesh_ly = y_max - y_min

    if not np.isclose(mesh_lx, LX, rtol=0.0, atol=1e-8):
        raise ValueError(
            f"Estensione x del VTU inattesa: {mesh_lx:.12g}; attesa {LX:.12g}"
        )
    if not np.isclose(mesh_ly, LY, rtol=0.0, atol=1e-8):
        raise ValueError(
            f"Estensione y del VTU inattesa: {mesh_ly:.12g}; attesa {LY:.12g}"
        )

    z_start = -height - 2.0 * EPS
    z_end = 2.0 * EPS
    crop_height = z_end - z_start  # height + 4*EPS

    n_intervals_x = int(round(LX / DX_SIZE))
    n_intervals_y = int(round(LY / DX_SIZE))
    n_intervals_z = int(round(crop_height / DX_SIZE))

    # Impedisce arrotondamenti silenziosi: la griglia deve essere isotropa
    # esattamente, non solo circa.
    if not np.isclose(n_intervals_x * DX_SIZE, LX, rtol=0.0, atol=1e-12):
        raise ValueError(f"LX={LX} non è multiplo di DX_SIZE={DX_SIZE}")
    if not np.isclose(n_intervals_y * DX_SIZE, LY, rtol=0.0, atol=1e-12):
        raise ValueError(f"LY={LY} non è multiplo di DX_SIZE={DX_SIZE}")
    if not np.isclose(
        n_intervals_z * DX_SIZE,
        crop_height,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError(
            f"height={height}: crop_height={crop_height} non è multiplo "
            f"di DX_SIZE={DX_SIZE}"
        )

    nx = n_intervals_x + 1
    ny = n_intervals_y + 1
    nz = n_intervals_z + 1

    xi = np.linspace(x_min, x_max, nx, dtype=np.float64)
    yi = np.linspace(y_min, y_max, ny, dtype=np.float64)
    zi = np.linspace(z_start, z_end, nz, dtype=np.float64)

    dx = xi[1] - xi[0]
    dy = yi[1] - yi[0]
    dz = zi[1] - zi[0]

    if not (
        np.isclose(dx, DX_SIZE, rtol=0.0, atol=1e-12)
        and np.isclose(dy, DX_SIZE, rtol=0.0, atol=1e-12)
        and np.isclose(dz, DX_SIZE, rtol=0.0, atol=1e-12)
    ):
        raise RuntimeError(
            f"Griglia non isotropa: dx={dx}, dy={dy}, dz={dz}, "
            f"target={DX_SIZE}"
        )

    return xi, yi, zi


def make_query_points(
    xi: np.ndarray,
    yi: np.ndarray,
    zi: np.ndarray,
) -> np.ndarray:
    """Restituisce i punti della griglia in ordine compatibile con reshape(nx, ny, nz)."""
    X, Y, Z = np.meshgrid(xi, yi, zi, indexing="ij")
    return np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))


def output_path_for(vtu_path: Path) -> Path:
    """Mantiene la struttura relativa a INPUT_ROOT e sostituisce .vtu con .npy."""
    return (OUTPUT_ROOT / vtu_path.relative_to(INPUT_ROOT)).with_suffix(".npy")


# ============================================================
# CONVERSIONE
# ============================================================

def convert_vtu_to_npy(
    vtu_path: Path,
    out_path: Path,
    query_points: np.ndarray,
    output_shape: tuple[int, int, int],
) -> None:
    """Interpola il campo FIELD_NAME del VTU direttamente sulla griglia finale."""
    mesh = pv.read(vtu_path)

    if FIELD_NAME not in mesh.point_data:
        raise KeyError(
            f"Campo '{FIELD_NAME}' non presente in {vtu_path}. "
            f"Disponibili: {list(mesh.point_data.keys())}"
        )

    points = np.asarray(mesh.points[:, :3], dtype=np.float64)
    values = np.asarray(mesh.point_data[FIELD_NAME]).squeeze()

    if values.ndim != 1:
        raise ValueError(
            f"Il campo '{FIELD_NAME}' deve essere scalare, ma ha shape {values.shape}"
        )
    if len(values) != len(points):
        raise ValueError(
            f"Numero di valori ({len(values)}) diverso dal numero di punti ({len(points)})"
        )

    sampled = griddata(
        points,
        values,
        query_points,
        method=METHOD,
        fill_value=np.nan,
    )

    grid = np.nan_to_num(
        sampled,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(np.float32)

    grid = grid.reshape(output_shape)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, grid)


# ============================================================
# PROCESSAMENTO DI UNA CARTELLA / TRAIETTORIA
# ============================================================

def process_folder(folder_path: Path) -> tuple[int, int]:
    """
    Processa tutti i surf_*.vtu contenuti direttamente in una cartella.
    Restituisce (numero_convertiti, numero_saltati).
    """
    vtu_files = sorted(folder_path.glob(VTU_GLOB))
    if not vtu_files:
        return 0, 0

    height = extract_height(folder_path.name)

    # Si legge il primo frame soltanto per ricavare gli estremi fisici in x/y.
    reference_mesh = pv.read(vtu_files[0])
    reference_points = np.asarray(reference_mesh.points[:, :3], dtype=np.float64)

    x_min, y_min, z_min = reference_points.min(axis=0)
    x_max, y_max, z_max = reference_points.max(axis=0)

    xi, yi, zi = make_isotropic_grid(
        height=height,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
    )

    z_start = float(zi[0])
    z_end = float(zi[-1])

    if z_start < z_min - 1e-8 or z_end > z_max + 1e-8:
        raise ValueError(
            f"Regione z richiesta [{z_start}, {z_end}] fuori dal dominio "
            f"del primo VTU [{z_min}, {z_max}]"
        )

    query_points = make_query_points(xi, yi, zi)
    output_shape = (len(xi), len(yi), len(zi))

    output_folder = OUTPUT_ROOT / folder_path.relative_to(INPUT_ROOT)
    output_folder.mkdir(parents=True, exist_ok=True)

    # Metadata: viene creato una sola volta e documenta la geometria fisica.
    metadata_path = output_folder / "grid_metadata.npz"
    if not metadata_path.exists() or OVERWRITE:
        np.savez(
            metadata_path,
            xi=xi,
            yi=yi,
            zi=zi,
            height=np.float64(height),
            eps=np.float64(EPS),
            voxel_size=np.float64(DX_SIZE),
            dx=np.float64(xi[1] - xi[0]),
            dy=np.float64(yi[1] - yi[0]),
            dz=np.float64(zi[1] - zi[0]),
            shape=np.array(output_shape, dtype=np.int32),
        )

    print("\n" + "=" * 72)
    print(f"Folder input : {folder_path}")
    print(f"Folder output: {output_folder}")
    print(f"height       : {height}")
    print(f"shape        : {output_shape}")
    print(
        f"spacing      : dx={xi[1] - xi[0]:.8f}, "
        f"dy={yi[1] - yi[0]:.8f}, dz={zi[1] - zi[0]:.8f}"
    )
    print(f"region z     : [{z_start:.8f}, {z_end:.8f}]")
    print(f"frame trovati: {len(vtu_files)}")

    converted = 0
    skipped = 0

    for index, vtu_path in enumerate(vtu_files, start=1):
        out_path = output_path_for(vtu_path)

        if out_path.exists() and not OVERWRITE:
            print(f"[{index:04d}/{len(vtu_files):04d}] SKIP {out_path.name}")
            skipped += 1
            continue

        t0 = time.perf_counter()
        convert_vtu_to_npy(
            vtu_path=vtu_path,
            out_path=out_path,
            query_points=query_points,
            output_shape=output_shape,
        )
        elapsed = time.perf_counter() - t0

        print(
            f"[{index:04d}/{len(vtu_files):04d}] DONE {out_path.name} "
            f"({elapsed:.2f} s)"
        )
        converted += 1

    return converted, skipped


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    if not INPUT_ROOT.is_dir():
        raise FileNotFoundError(f"INPUT_ROOT non trovata: {INPUT_ROOT}")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    # Trova tutte le directory che contengono direttamente surf_*.vtu.
    folders = sorted({vtu_path.parent for vtu_path in INPUT_ROOT.rglob(VTU_GLOB)})

    if not folders:
        raise FileNotFoundError(
            f"Nessun file '{VTU_GLOB}' trovato sotto {INPUT_ROOT}"
        )

    print(f"Input root : {INPUT_ROOT}")
    print(f"Output root: {OUTPUT_ROOT}")
    print(f"Campo      : {FIELD_NAME}")
    print(f"Metodo     : {METHOD}")
    print(f"Voxel size : {DX_SIZE}")
    print(f"Cartelle   : {len(folders)}")

    total_converted = 0
    total_skipped = 0
    errors: list[str] = []

    global_start = time.perf_counter()

    for folder_index, folder_path in enumerate(folders, start=1):
        print(f"\n######## CARTELLA {folder_index}/{len(folders)} ########")

        try:
            converted, skipped = process_folder(folder_path)
            total_converted += converted
            total_skipped += skipped
        except Exception as exc:
            message = f"{folder_path}: {type(exc).__name__}: {exc}"
            errors.append(message)
            print(f"[ERROR] {message}")

    elapsed = time.perf_counter() - global_start

    print("\n" + "=" * 72)
    print("RIEPILOGO")
    print(f"Convertiti : {total_converted}")
    print(f"Saltati    : {total_skipped}")
    print(f"Falliti    : {len(errors)}")
    print(f"Tempo totale: {elapsed:.2f} s")

    if errors:
        error_log = OUTPUT_ROOT / ERROR_LOG_NAME
        error_log.write_text("\n".join(errors) + "\n", encoding="utf-8")
        print(f"Log errori: {error_log}")


if __name__ == "__main__":
    main()
