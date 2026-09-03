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

INPUT_ROOT = Path("/archive/roberto/poresAMDIS/iso_P09")
OUTPUT_ROOT = Path("/data/fiorello/iso_P09")

FIELD_NAME = "phi"

METHOD = "linear"

OVERWRITE = True

LX = 0.45
LY = 0.45
EPS = 0.1

# Passo isotropo richiesto nella griglia finale.
VOXEL_SIZE = 0.0125

# Tolleranza per le coordinate floating point lette dal VTU.
# Esempio reale: 0.4499999988079 invece di 0.45.
GEOMETRY_TOL = 1e-6

VTU_GLOB = "surf_*.vtu"
ERROR_LOG_NAME = "conversion_errors.log"


# ============================================================
# FUNZIONI AUSILIARIE
# ============================================================

def extract_height(folder_name: str) -> float:
    """
    Estrae l'altezza dal nome della cartella.

    Esempio:
        iso2_R0.2_H1.0_P0.9 -> 1.0
    """
    match = re.search(r"(?:^|_)H(-?\d+(?:\.\d+)?)", folder_name)

    if match is None:
        raise ValueError(
            f"Impossibile estrarre il parametro H dal nome: {folder_name}"
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
    Costruisce una griglia finale isotropa, con endpoint inclusi.

    Regione z salvata:
        [-height - 2*EPS, 2*EPS]

    La sua estensione è:
        height + 4*EPS

    La shape della griglia è:
        (Nx, Ny, Nz)

    e verifica dx = dy = dz = VOXEL_SIZE.
    """
    mesh_lx = x_max - x_min
    mesh_ly = y_max - y_min

    # Il VTU può memorizzare 0.45 come 0.4499999988: è normale.
    if not np.isclose(mesh_lx, LX, rtol=0.0, atol=GEOMETRY_TOL):
        raise ValueError(
            f"Estensione x del VTU inattesa: {mesh_lx:.12g}; "
            f"attesa {LX:.12g}"
        )

    if not np.isclose(mesh_ly, LY, rtol=0.0, atol=GEOMETRY_TOL):
        raise ValueError(
            f"Estensione y del VTU inattesa: {mesh_ly:.12g}; "
            f"attesa {LY:.12g}"
        )

    z_start = -height - 2.0 * EPS
    z_end = 2.0 * EPS
    crop_height = z_end - z_start

    # Numero di intervalli, non di punti. Gli endpoint sono inclusi,
    # quindi il numero di punti finale sarà n_intervals + 1.
    n_intervals_x = int(round(LX / VOXEL_SIZE))
    n_intervals_y = int(round(LY / VOXEL_SIZE))
    n_intervals_z = int(round(crop_height / VOXEL_SIZE))

    # Non accetta una griglia solo approssimativamente isotropa.
    if not np.isclose(
        n_intervals_x * VOXEL_SIZE,
        LX,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError(
            f"LX={LX} non è multiplo esatto di VOXEL_SIZE={VOXEL_SIZE}"
        )

    if not np.isclose(
        n_intervals_y * VOXEL_SIZE,
        LY,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError(
            f"LY={LY} non è multiplo esatto di VOXEL_SIZE={VOXEL_SIZE}"
        )

    if not np.isclose(
        n_intervals_z * VOXEL_SIZE,
        crop_height,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError(
            f"Per height={height}, crop_height={crop_height} non è "
            f"multiplo esatto di VOXEL_SIZE={VOXEL_SIZE}"
        )

    nx = n_intervals_x + 1
    ny = n_intervals_y + 1
    nz = n_intervals_z + 1

    # Usiamo gli estremi nominali LX e LY, non x_max/y_max direttamente,
    # per evitare che l'errore floating point del VTU alteri dx/dy.
    xi = np.linspace(x_min, x_min + LX, nx, dtype=np.float64)
    yi = np.linspace(y_min, y_min + LY, ny, dtype=np.float64)
    zi = np.linspace(z_start, z_end, nz, dtype=np.float64)

    dx = xi[1] - xi[0]
    dy = yi[1] - yi[0]
    dz = zi[1] - zi[0]

    if not (
        np.isclose(dx, VOXEL_SIZE, rtol=0.0, atol=1e-12)
        and np.isclose(dy, VOXEL_SIZE, rtol=0.0, atol=1e-12)
        and np.isclose(dz, VOXEL_SIZE, rtol=0.0, atol=1e-12)
    ):
        raise RuntimeError(
            "La griglia finale non è isotropa: "
            f"dx={dx:.16g}, dy={dy:.16g}, dz={dz:.16g}, "
            f"target={VOXEL_SIZE:.16g}"
        )

    return xi, yi, zi


def make_query_points(
    xi: np.ndarray,
    yi: np.ndarray,
    zi: np.ndarray,
) -> np.ndarray:
    """
    Crea i punti della griglia in un ordine compatibile con:
        values.reshape(len(xi), len(yi), len(zi))
    """
    X, Y, Z = np.meshgrid(xi, yi, zi, indexing="ij")
    return np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))


def output_path_for(vtu_path: Path) -> Path:
    """Replica sotto OUTPUT_ROOT il percorso relativo del file VTU."""
    relative = vtu_path.relative_to(INPUT_ROOT)
    return (OUTPUT_ROOT / relative).with_suffix(".npy")


def convert_one_vtu(
    vtu_path: Path,
    out_path: Path,
    query_points: np.ndarray,
    output_shape: tuple[int, int, int],
) -> None:
    """Legge un VTU, interpola phi e salva l'array (Nx, Ny, Nz) in float32."""
    mesh = pv.read(vtu_path)

    if FIELD_NAME not in mesh.point_data:
        raise KeyError(
            f"Campo '{FIELD_NAME}' non presente in {vtu_path}. "
            f"Campi disponibili: {list(mesh.point_data.keys())}"
        )

    points = np.asarray(mesh.points[:, :3], dtype=np.float64)
    values = np.asarray(mesh.point_data[FIELD_NAME]).squeeze()

    if values.ndim != 1:
        raise ValueError(
            f"Il campo '{FIELD_NAME}' deve essere scalare, "
            f"ma ha shape {values.shape}"
        )

    if len(values) != len(points):
        raise ValueError(
            f"Numero valori ({len(values)}) diverso dal numero di punti "
            f"({len(points)}) in {vtu_path}"
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


def process_folder(folder_path: Path) -> tuple[int, int]:
    """
    Converte tutti i surf_*.vtu presenti direttamente dentro folder_path.
    Ritorna: (numero_convertiti, numero_saltati).
    """
    vtu_files = sorted(folder_path.glob(VTU_GLOB))

    if not vtu_files:
        return 0, 0

    height = extract_height(folder_path.name)

    # Il primo frame serve solo per conoscere origine ed estensione del mesh.
    first_mesh = pv.read(vtu_files[0])
    first_points = np.asarray(first_mesh.points[:, :3], dtype=np.float64)

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

    # Controllo che il crop richiesto sia nel dominio del VTU.
    if z_start < z_min - GEOMETRY_TOL or z_end > z_max + GEOMETRY_TOL:
        raise ValueError(
            f"Regione z richiesta [{z_start:.8f}, {z_end:.8f}] fuori "
            f"dal dominio VTU [{z_min:.8f}, {z_max:.8f}]"
        )

    query_points = make_query_points(xi, yi, zi)
    output_shape = (len(xi), len(yi), len(zi))

    output_folder = OUTPUT_ROOT / folder_path.relative_to(INPUT_ROOT)
    output_folder.mkdir(parents=True, exist_ok=True)

    # Salva le coordinate fisiche e i metadati una volta per traiettoria.
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
        )

    print("\n" + "=" * 72)
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

    for i, vtu_path in enumerate(vtu_files, start=1):
        out_path = output_path_for(vtu_path)

        if out_path.exists() and not OVERWRITE:
            print(f"[{i:04d}/{len(vtu_files):04d}] SKIP {out_path.name}")
            skipped += 1
            continue

        t0 = time.perf_counter()

        convert_one_vtu(
            vtu_path=vtu_path,
            out_path=out_path,
            query_points=query_points,
            output_shape=output_shape,
        )

        elapsed = time.perf_counter() - t0
        print(
            f"[{i:04d}/{len(vtu_files):04d}] DONE {out_path.name} "
            f"({elapsed:.2f} s)"
        )
        converted += 1

    return converted, skipped


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    if not INPUT_ROOT.is_dir():
        raise FileNotFoundError(f"Input root non trovata: {INPUT_ROOT}")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    # Trova tutte le cartelle che contengono almeno un surf_*.vtu.
    folders = sorted({vtu_path.parent for vtu_path in INPUT_ROOT.rglob(VTU_GLOB)})

    if not folders:
        raise FileNotFoundError(
            f"Nessun file '{VTU_GLOB}' trovato sotto {INPUT_ROOT}"
        )

    print(f"Input root : {INPUT_ROOT}")
    print(f"Output root: {OUTPUT_ROOT}")
    print(f"Campo      : {FIELD_NAME}")
    print(f"Metodo     : {METHOD}")
    print(f"Voxel size : {VOXEL_SIZE}")
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
