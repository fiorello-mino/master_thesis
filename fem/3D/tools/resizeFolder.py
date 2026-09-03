from __future__ import annotations

import re
import time
from pathlib import Path

import numpy as np
import pyvista as pv
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator


# ============================================================
# CONFIGURAZIONE
# ============================================================

INPUT_ROOT = Path("/archive/roberto/poreAMDIS/iso_P09")
INPUT_ROOT = Path("/home/fiorello/iso_P09")
OUTPUT_ROOT = Path("/data/fiorello/iso_P09")

FIELD_NAME = "phi"

TOTAL_POINTS: int = 64 * 64 * 64
EPS: float = 0.1

LX: float = 0.45
LY: float = 0.45
LZ: float = 6.0

# La regione mantenuta è:
# [-height - 2*EPS, 0 + 2*EPS]
#
# La sua lunghezza è:
# height + 4*EPS
METHOD = "linear"

# Se True, sovrascrive gli NPY già esistenti.
OVERWRITE = False

# Nome del log degli errori
ERROR_LOG = OUTPUT_ROOT / "conversion_errors.log"


# ============================================================
# FUNZIONI DI SUPPORTO
# ============================================================

def extract_height(folder_name: str) -> float:
    """
    Estrae height dal nome della cartella.

    Esempi:
        iso2_R0.2_H1.0_P0.9 -> 1.0
        iso2_R0.2_H2.7_P0.9 -> 2.7
    """
    match = re.search(r"(?:^|_)H(-?\d+(?:\.\d+)?)", folder_name)

    if match is None:
        raise ValueError(
            f"Impossibile estrarre height dal nome della cartella: {folder_name}"
        )

    return float(match.group(1))


def compute_grid_dimensions(height: float):
    """
    Calcola le dimensioni della griglia per una determinata altezza.
    """
    total_height = height + 4.0 * EPS

    nx = round((TOTAL_POINTS * LX / total_height) ** (1.0 / 3.0))
    ny = nx
    nz = round((TOTAL_POINTS * total_height**2 / LX) ** (1.0 / 3.0))
    nz_tot = round(nz * LZ / total_height)

    return nx, ny, nz, nz_tot, total_height


def build_interpolator(mesh: pv.DataSet, field_name: str, method: str):
    """
    Costruisce l'interpolatore una sola volta per il mesh.

    Questo è utile se una cartella contiene molti frame con la stessa
    griglia geometrica e si vuole riutilizzare la triangolazione.
    """
    pts = np.asarray(mesh.points[:, :3], dtype=np.float64)

    if field_name not in mesh.point_data:
        available = list(mesh.point_data.keys())
        raise KeyError(
            f"Campo '{field_name}' non presente nel VTU. "
            f"Campi disponibili: {available}"
        )

    vals = np.asarray(mesh.point_data[field_name]).squeeze()

    if vals.ndim != 1:
        raise ValueError(
            f"Il campo '{field_name}' deve essere scalare. "
            f"Shape trovata: {vals.shape}"
        )

    if len(vals) != len(pts):
        raise ValueError(
            f"Numero di valori incompatibile con i punti: "
            f"{len(vals)} valori per {len(pts)} punti."
        )

    if method == "linear":
        interpolator = LinearNDInterpolator(
            pts,
            vals,
            fill_value=np.nan,
        )
    elif method == "nearest":
        interpolator = NearestNDInterpolator(
            pts,
            vals,
        )
    else:
        raise ValueError(
            f"Metodo '{method}' non supportato. "
            "Usa 'linear' oppure 'nearest'."
        )

    return interpolator, pts


def convert_single_vtu(
    vtu_path: Path,
    out_path: Path,
    height: float,
    method: str = METHOD,
    overwrite: bool = OVERWRITE,
):
    """
    Converte un singolo VTU in NPY.

    L'array finale ha shape:
        (Nx, Ny, Nz)

    La griglia completa temporanea ha shape:
        (Nx, Ny, Nz_tot)
    """
    if out_path.exists() and not overwrite:
        print(f"[SKIP] Esiste già: {out_path}")
        return "skipped"

    nx, ny, nz, nz_tot, total_height = compute_grid_dimensions(height)

    print(f"\n[VTU] {vtu_path}")
    print(
        f"[GRID] height={height:.6g}, "
        f"total_height={total_height:.6g}, "
        f"shape_full=({nx}, {ny}, {nz_tot}), "
        f"shape_output=({nx}, {ny}, {nz})"
    )

    mesh = pv.read(vtu_path)

    pts = np.asarray(mesh.points[:, :3], dtype=np.float64)

    x_min, y_min, z_min = pts.min(axis=0)
    x_max, y_max, z_max = pts.max(axis=0)

    # Griglia completa del dominio.
    xi = np.linspace(x_min, x_max, nx, dtype=np.float64)
    yi = np.linspace(y_min, y_max, ny, dtype=np.float64)
    zi_full = np.linspace(z_min, z_max, nz_tot, dtype=np.float64)

    X, Y, Z = np.meshgrid(
        xi,
        yi,
        zi_full,
        indexing="ij",
    )

    query_points = np.column_stack(
        (
            X.ravel(),
            Y.ravel(),
            Z.ravel(),
        )
    )

    interpolator, _ = build_interpolator(
        mesh=mesh,
        field_name=FIELD_NAME,
        method=method,
    )

    grid_full = interpolator(query_points)
    grid_full = np.asarray(grid_full, dtype=np.float32)
    grid_full = grid_full.reshape(nx, ny, nz_tot)

    # Regione da mantenere:
    #
    # z in [-height - 2*EPS, 0 + 2*EPS]
    #
    # Esempio con height=1.0 ed EPS=0.1:
    # z in [-1.2, 0.2]
    z_start = -height - 2.0 * EPS
    z_end = 2.0 * EPS

    if z_start < z_min or z_end > z_max:
        raise ValueError(
            f"La regione richiesta non è contenuta nel dominio del VTU.\n"
            f"Dominio z: [{z_min}, {z_max}]\n"
            f"Regione richiesta: [{z_start}, {z_end}]"
        )

    # Trova gli indici più vicini agli estremi fisici.
    z_idx_start = int(np.argmin(np.abs(zi_full - z_start)))
    z_idx_end = int(np.argmin(np.abs(zi_full - z_end)))

    if z_idx_start > z_idx_end:
        z_idx_start, z_idx_end = z_idx_end, z_idx_start

    # Slice inclusiva dell'estremo superiore.
    grid_pore = grid_full[:, :, z_idx_start : z_idx_end + 1]

    # Il numero di punti effettivamente ottenuto può differire di 1
    # a causa dell'arrotondamento sulla griglia completa.
    #
    # Per ottenere sempre esattamente Nz punti, si ricampiona direttamente
    # la regione richiesta con Nz coordinate z.
    if grid_pore.shape[2] != nz:
        zi_crop = np.linspace(
            zi_full[z_idx_start],
            zi_full[z_idx_end],
            nz,
            dtype=np.float64,
        )

        X_crop, Y_crop, Z_crop = np.meshgrid(
            xi,
            yi,
            zi_crop,
            indexing="ij",
        )

        query_crop = np.column_stack(
            (
                X_crop.ravel(),
                Y_crop.ravel(),
                Z_crop.ravel(),
            )
        )

        grid_pore = interpolator(query_crop)
        grid_pore = np.asarray(grid_pore, dtype=np.float32)
        grid_pore = grid_pore.reshape(nx, ny, nz)

    # Sostituisce eventuali NaN fuori dall'inviluppo convesso.
    grid_pore = np.nan_to_num(
        grid_pore,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(np.float32)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, grid_pore)

    print(
        f"[DONE] {out_path} | "
        f"shape={grid_pore.shape} | "
        f"dtype={grid_pore.dtype}"
    )

    return "converted"


def get_output_path(vtu_path: Path) -> Path:
    """
    Mantiene la struttura delle cartelle e cambia .vtu in .npy.
    """
    relative_path = vtu_path.relative_to(INPUT_ROOT)
    return OUTPUT_ROOT / relative_path.with_suffix(".npy")


# ============================================================
# MAIN
# ============================================================

def main():
    if not INPUT_ROOT.exists():
        raise FileNotFoundError(
            f"La root di input non esiste: {INPUT_ROOT}"
        )

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    vtu_files = sorted(INPUT_ROOT.rglob("surf_*.vtu"))

    if not vtu_files:
        raise FileNotFoundError(
            f"Nessun file 'surf_*.vtu' trovato sotto {INPUT_ROOT}"
        )

    print(f"Root input : {INPUT_ROOT}")
    print(f"Root output: {OUTPUT_ROOT}")
    print(f"VTU trovati: {len(vtu_files)}")
    print(f"Campo      : {FIELD_NAME}")
    print(f"Metodo     : {METHOD}")

    converted = 0
    skipped = 0
    failed = 0
    errors = []

    global_start = time.perf_counter()

    for index, vtu_path in enumerate(vtu_files, start=1):
        try:
            # Il nome della cartella immediatamente superiore contiene H...
            folder_name = vtu_path.parent.name
            height = extract_height(folder_name)

            out_path = get_output_path(vtu_path)

            print(f"\n========== {index}/{len(vtu_files)} ==========")

            status = convert_single_vtu(
                vtu_path=vtu_path,
                out_path=out_path,
                height=height,
                method=METHOD,
                overwrite=OVERWRITE,
            )

            if status == "converted":
                converted += 1
            elif status == "skipped":
                skipped += 1

        except Exception as exc:
            failed += 1
            message = f"{vtu_path}: {type(exc).__name__}: {exc}"
            errors.append(message)
            print(f"[ERROR] {message}")

    elapsed = time.perf_counter() - global_start

    print("\n================ RISULTATO ================")
    print(f"Convertiti : {converted}")
    print(f"Saltati    : {skipped}")
    print(f"Falliti    : {failed}")
    print(f"Tempo totale: {elapsed:.2f} s")

    if errors:
        ERROR_LOG.parent.mkdir(parents=True, exist_ok=True)
        ERROR_LOG.write_text(
            "\n".join(errors) + "\n",
            encoding="utf-8",
        )
        print(f"Log errori: {ERROR_LOG}")


if __name__ == "__main__":
    main()