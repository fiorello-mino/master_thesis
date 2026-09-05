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

# Root contenente le cartelle iso2_R..._H..._P...
INPUT_ROOT = Path("/archive/roberto/poresAMDIS/iso_P09")

# Root dei file .npy convertiti.
# La struttura delle sottocartelle viene mantenuta.
OUTPUT_ROOT = Path("/data/fiorello/iso_P09")

# Nome del campo scalare presente nei point_data dei file VTU.
FIELD_NAME = "phi"

# Metodo di interpolazione principale.
PRIMARY_METHOD = "linear"

# Metodo usato solo per i punti dove linear restituisce NaN.
FALLBACK_METHOD = "nearest"

# True: rigenera tutti gli NPY anche se esistono già.
OVERWRITE = True

# Salva per ogni frame una maschera:
# 0 = interpolato con linear
# 1 = linear fallito, valore sostituito con nearest
SAVE_NAN_MASK = True

# Geometria fisica nominale.
LX = 0.45
LY = 0.45
EPS = 0.1

# Voxel isotropo:
# dx = dy = dz = 0.025
VOXEL_SIZE = 0.025

# Tolleranza per piccoli errori floating point del VTU.
# Per esempio: 0.4499999988 invece di 0.45.
GEOMETRY_TOL = 1e-6

# Pattern dei file input.
VTU_GLOB = "surf_*.vtu"

# Nome del file di log per eventuali errori.
ERROR_LOG_NAME = "conversion_errors.log"


# ============================================================
# FUNZIONI AUSILIARIE
# ============================================================

def extract_height(folder_name: str) -> float:
    """
    Estrae il valore H dal nome della cartella.

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
    Crea la griglia isotropa finale con endpoint inclusi.

    Regione z salvata:

        z in [-height - 2*EPS, 2*EPS]

    Esempio per H=1.0 e EPS=0.1:

        z in [-1.2, 0.2]

    La griglia finale soddisfa:

        dx = dy = dz = VOXEL_SIZE
    """
    mesh_lx = x_max - x_min
    mesh_ly = y_max - y_min

    # Il mesh può avere una piccola differenza da LX/LY per effetto
    # della precisione floating point nel VTU.
    if not np.isclose(mesh_lx, LX, rtol=0.0, atol=GEOMETRY_TOL):
        raise ValueError(
            f"Estensione x inattesa nel VTU: {mesh_lx:.12g}; "
            f"attesa {LX:.12g}"
        )

    if not np.isclose(mesh_ly, LY, rtol=0.0, atol=GEOMETRY_TOL):
        raise ValueError(
            f"Estensione y inattesa nel VTU: {mesh_ly:.12g}; "
            f"attesa {LY:.12g}"
        )

    # Regione di z da salvare.
    z_start = -height - 2.0 * EPS
    z_end = 2.0 * EPS
    crop_height = z_end - z_start

    # Numero di intervalli, non di nodi.
    # N punti corrispondono a N-1 intervalli con endpoint inclusi.
    n_intervals_x = int(round(LX / VOXEL_SIZE))
    n_intervals_y = int(round(LY / VOXEL_SIZE))
    n_intervals_z = int(round(crop_height / VOXEL_SIZE))

    # Verifica compatibilità esatta con il passo scelto.
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
            f"height={height}: crop_height={crop_height} non è "
            f"multiplo esatto di VOXEL_SIZE={VOXEL_SIZE}"
        )

    nx = n_intervals_x + 1
    ny = n_intervals_y + 1
    nz = n_intervals_z + 1

    # Si usano gli estremi nominali x_min + LX e y_min + LY,
    # non x_max/y_max direttamente, per non cambiare dx/dy a causa
    # di una differenza floating point microscopica nel VTU.
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
        np.isclose(dx, VOXEL_SIZE, rtol=0.0, atol=1e-12)
        and np.isclose(dy, VOXEL_SIZE, rtol=0.0, atol=1e-12)
        and np.isclose(dz, VOXEL_SIZE, rtol=0.0, atol=1e-12)
    ):
        raise RuntimeError(
            "Griglia non isotropa: "
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
    Crea coordinate (x, y, z) della griglia regolare.

    L'ordine è compatibile con:

        array.reshape(len(xi), len(yi), len(zi))

    Quindi l'NPY salvato avrà assi:

        axis 0 -> x
        axis 1 -> y
        axis 2 -> z
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
    """
    Mantiene lo stesso path relativo rispetto a INPUT_ROOT,
    sostituendo l'estensione .vtu con .npy.
    """
    relative_path = vtu_path.relative_to(INPUT_ROOT)

    return (OUTPUT_ROOT / relative_path).with_suffix(".npy")


# ============================================================
# CONVERSIONE VTU -> NPY
# ============================================================

def convert_one_vtu(
    vtu_path: Path,
    out_path: Path,
    query_points: np.ndarray,
    output_shape: tuple[int, int, int],
) -> tuple[int, int]:
    """
    Converte un singolo VTU.

    Procedura:
    - interpolazione lineare;
    - identificazione di NaN/Inf;
    - nearest-neighbour soltanto nei punti invalidi;
    - salvataggio in float32.

    Ritorna:
        n_fallback, n_total

    dove n_fallback è il numero di voxel in cui è stato necessario
    sostituire linear con nearest.
    """
    mesh = pv.read(vtu_path)

    if FIELD_NAME not in mesh.point_data:
        raise KeyError(
            f"Campo '{FIELD_NAME}' non presente in {vtu_path}. "
            f"Campi disponibili: {list(mesh.point_data.keys())}"
        )

    points = np.asarray(
        mesh.points[:, :3],
        dtype=np.float64,
    )

    values = np.asarray(
        mesh.point_data[FIELD_NAME],
    ).squeeze()

    if values.ndim != 1:
        raise ValueError(
            f"Il campo '{FIELD_NAME}' deve essere scalare, "
            f"ma ha shape {values.shape}"
        )

    if len(values) != len(points):
        raise ValueError(
            f"Numero di valori ({len(values)}) diverso dal numero "
            f"di punti ({len(points)}) in {vtu_path}"
        )

    # Interpolazione principale: lineare.
    #
    # I punti fuori dalla regione interpolabile ricevono NaN.
    sampled = griddata(
        points,
        values,
        query_points,
        method=PRIMARY_METHOD,
        fill_value=np.nan,
    )

    # Include NaN e Inf, anche se tipicamente avrai solo NaN.
    invalid_mask = ~np.isfinite(sampled)

    n_fallback = int(invalid_mask.sum())
    n_total = int(sampled.size)

    # IMPORTANTE:
    #
    # Non usare np.nan_to_num(..., nan=0.0).
    # Per phi, zero può essere l'interfaccia fisica.
    #
    # Nei voxel dove il lineare non è definito, usa nearest.
    if n_fallback > 0:
        fallback_values = griddata(
            points,
            values,
            query_points[invalid_mask],
            method=FALLBACK_METHOD,
        )

        sampled[invalid_mask] = fallback_values

    # Sicurezza: dopo nearest non devono restare valori non finiti.
    if not np.all(np.isfinite(sampled)):
        still_invalid = int((~np.isfinite(sampled)).sum())

        raise RuntimeError(
            f"Dopo il fallback nearest restano "
            f"{still_invalid} valori non finiti"
        )

    grid = sampled.astype(np.float32).reshape(output_shape)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, grid)

    # Maschera diagnostica opzionale:
    #
    # 0 = valore ottenuto con linear
    # 1 = linear non definito, sostituito con nearest
    if SAVE_NAN_MASK:
        nan_mask_path = out_path.with_name(
            f"{out_path.stem}_linear_nan_mask.npy"
        )

        np.save(
            nan_mask_path,
            invalid_mask.astype(np.uint8).reshape(output_shape),
        )

    return n_fallback, n_total


# ============================================================
# ELABORAZIONE DI UNA CARTELLA
# ============================================================

def process_folder(folder_path: Path) -> tuple[int, int]:
    """
    Converte tutti i file surf_*.vtu contenuti direttamente in folder_path.

    Tutti i frame della stessa cartella usano:
    - stessa height;
    - stessa griglia xi, yi, zi;
    - stessa shape NPY;
    - stesso voxel size isotropo.

    Ritorna:
        converted, skipped
    """
    vtu_files = sorted(folder_path.glob(VTU_GLOB))

    if not vtu_files:
        return 0, 0

    height = extract_height(folder_path.name)

    # Primo frame: viene usato soltanto per conoscere gli estremi
    # del dominio e costruire una griglia comune alla traiettoria.
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

    # Verifica che la regione fisica richiesta sia contenuta nel VTU.
    if (
        z_start < z_min - GEOMETRY_TOL
        or z_end > z_max + GEOMETRY_TOL
    ):
        raise ValueError(
            f"Regione z richiesta [{z_start:.8f}, {z_end:.8f}] "
            f"fuori dal dominio VTU "
            f"[{z_min:.8f}, {z_max:.8f}]"
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
    output_folder.mkdir(parents=True, exist_ok=True)

    # Salva coordinate e metadata una volta per cartella.
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
            primary_method=np.array(PRIMARY_METHOD),
            fallback_method=np.array(FALLBACK_METHOD),
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
            f"fallback nearest: "
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

    # Trova tutte le cartelle che contengono direttamente almeno un surf_*.vtu.
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

    print(f"Input root      : {INPUT_ROOT}")
    print(f"Output root     : {OUTPUT_ROOT}")
    print(f"Campo           : {FIELD_NAME}")
    print(f"Metodo primario : {PRIMARY_METHOD}")
    print(f"Metodo fallback : {FALLBACK_METHOD}")
    print(f"Voxel size      : {VOXEL_SIZE}")
    print(f"Overwrite       : {OVERWRITE}")
    print(f"Save NaN mask   : {SAVE_NAN_MASK}")
    print(f"Cartelle        : {len(folders)}")

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
