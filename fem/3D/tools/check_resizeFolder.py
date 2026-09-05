from __future__ import annotations

import re
from pathlib import Path

import numpy as np


# ============================================================
# CONFIGURAZIONE
# ============================================================

# Root con le cartelle originali contenenti surf_*.vtu.
INPUT_ROOT = Path("/archive/roberto/poresAMDIS/iso_P09")

# Root creata da resizeFolder.py, contenente gli NPY convertiti.
OUTPUT_ROOT = Path("/data/fiorello/iso_P09")

# Parametri che devono coincidere con resizeFolder.py.
LX = 0.45
LY = 0.45
EPS = 0.1
VOXEL_SIZE = 0.025

# Tolleranza nei confronti floating point.
TOL = 1e-6

# True: controlla ogni NPY della cartella.
# False: controlla soltanto primo, centrale e ultimo frame per folder.
CHECK_EVERY_FILE = True

VTU_GLOB = "surf_*.vtu"
DATA_NPY_GLOB = "surf_*.npy"
NAN_MASK_SUFFIX = "_linear_nan_mask.npy"
REPORT_NAME = "resize_check_report.txt"


# ============================================================
# PATH E GEOMETRIA
# ============================================================

def extract_height(folder_name: str) -> float:
    """Esempio: iso2_R0.2_H1.0_P0.9 -> 1.0."""
    match = re.search(r"(?:^|_)H(-?\d+(?:\.\d+)?)", folder_name)

    if match is None:
        raise ValueError(
            f"Impossibile estrarre H dal nome della cartella: {folder_name}"
        )

    return float(match.group(1))


def expected_shape(height: float) -> tuple[int, int, int]:
    """
    Shape attesa della griglia isotropa.

    Regione salvata:
        z in [-height - 2*EPS, 2*EPS]

    Endpoint inclusi: N punti = N-1 intervalli + 1.
    """
    crop_height = height + 4.0 * EPS

    nx = int(round(LX / VOXEL_SIZE)) + 1
    ny = int(round(LY / VOXEL_SIZE)) + 1
    nz = int(round(crop_height / VOXEL_SIZE)) + 1

    return nx, ny, nz


def data_npy_path_for(vtu_path: Path) -> Path:
    """Restituisce l'NPY atteso per un VTU, conservando il path relativo."""
    relative_path = vtu_path.relative_to(INPUT_ROOT)
    return (OUTPUT_ROOT / relative_path).with_suffix(".npy")


def is_data_npy(path: Path) -> bool:
    """
    Riconosce solo gli NPY che contengono phi.

    Accetta:
        surf_0.000000.npy

    Esclude:
        surf_0.000000_linear_nan_mask.npy
    """
    return (
        path.is_file()
        and path.name.startswith("surf_")
        and path.suffix == ".npy"
        and not path.name.endswith(NAN_MASK_SUFFIX)
    )


def is_mask_npy(path: Path) -> bool:
    """Riconosce le maschere binarie linear -> nearest."""
    return path.is_file() and path.name.endswith(NAN_MASK_SUFFIX)


def get_files_to_check(data_files: list[Path]) -> list[Path]:
    """Restituisce tutti i file oppure primo/centrale/ultimo."""
    if CHECK_EVERY_FILE or len(data_files) <= 3:
        return data_files

    indices = sorted({0, len(data_files) // 2, len(data_files) - 1})
    return [data_files[index] for index in indices]


# ============================================================
# CONTROLLI METADATA
# ============================================================

def check_metadata(
    metadata_path: Path,
    height: float,
    shape_expected: tuple[int, int, int],
) -> list[str]:
    """Controlla coordinate, shape e isotropia salvate nei metadata."""
    problems: list[str] = []

    if not metadata_path.exists():
        return [f"Metadata mancante: {metadata_path.name}"]

    try:
        metadata = np.load(metadata_path, allow_pickle=False)
    except Exception as exc:
        return [
            f"Impossibile leggere {metadata_path.name}: "
            f"{type(exc).__name__}: {exc}"
        ]

    required_keys = {
        "xi",
        "yi",
        "zi",
        "shape",
        "height",
        "eps",
        "voxel_size",
        "dx",
        "dy",
        "dz",
    }

    missing_keys = required_keys - set(metadata.files)
    if missing_keys:
        return [
            f"Chiavi mancanti in {metadata_path.name}: {sorted(missing_keys)}"
        ]

    xi = metadata["xi"]
    yi = metadata["yi"]
    zi = metadata["zi"]

    stored_shape = tuple(int(value) for value in metadata["shape"])
    coordinate_shape = (len(xi), len(yi), len(zi))

    stored_height = float(metadata["height"])
    stored_eps = float(metadata["eps"])
    stored_voxel_size = float(metadata["voxel_size"])

    stored_dx = float(metadata["dx"])
    stored_dy = float(metadata["dy"])
    stored_dz = float(metadata["dz"])

    if stored_shape != shape_expected:
        problems.append(
            f"Shape metadata={stored_shape}, attesa={shape_expected}"
        )

    if coordinate_shape != shape_expected:
        problems.append(
            f"Lunghezza coordinate={coordinate_shape}, attesa={shape_expected}"
        )

    if not np.isclose(stored_height, height, rtol=0.0, atol=TOL):
        problems.append(
            f"Height metadata={stored_height}, attesa={height}"
        )

    if not np.isclose(stored_eps, EPS, rtol=0.0, atol=TOL):
        problems.append(
            f"EPS metadata={stored_eps}, atteso={EPS}"
        )

    if not np.isclose(
        stored_voxel_size,
        VOXEL_SIZE,
        rtol=0.0,
        atol=TOL,
    ):
        problems.append(
            f"Voxel size metadata={stored_voxel_size}, "
            f"atteso={VOXEL_SIZE}"
        )

    if not np.isclose(stored_dx, VOXEL_SIZE, rtol=0.0, atol=TOL):
        problems.append(f"dx metadata={stored_dx}, atteso={VOXEL_SIZE}")

    if not np.isclose(stored_dy, VOXEL_SIZE, rtol=0.0, atol=TOL):
        problems.append(f"dy metadata={stored_dy}, atteso={VOXEL_SIZE}")

    if not np.isclose(stored_dz, VOXEL_SIZE, rtol=0.0, atol=TOL):
        problems.append(f"dz metadata={stored_dz}, atteso={VOXEL_SIZE}")

    if len(xi) > 1 and not np.allclose(
        np.diff(xi),
        VOXEL_SIZE,
        rtol=0.0,
        atol=TOL,
    ):
        problems.append("xi non ha spaziatura uniforme VOXEL_SIZE")

    if len(yi) > 1 and not np.allclose(
        np.diff(yi),
        VOXEL_SIZE,
        rtol=0.0,
        atol=TOL,
    ):
        problems.append("yi non ha spaziatura uniforme VOXEL_SIZE")

    if len(zi) > 1 and not np.allclose(
        np.diff(zi),
        VOXEL_SIZE,
        rtol=0.0,
        atol=TOL,
    ):
        problems.append("zi non ha spaziatura uniforme VOXEL_SIZE")

    expected_z_start = -height - 2.0 * EPS
    expected_z_end = 2.0 * EPS

    if not np.isclose(zi[0], expected_z_start, rtol=0.0, atol=TOL):
        problems.append(
            f"zi[0]={zi[0]}, atteso={expected_z_start}"
        )

    if not np.isclose(zi[-1], expected_z_end, rtol=0.0, atol=TOL):
        problems.append(
            f"zi[-1]={zi[-1]}, atteso={expected_z_end}"
        )

    return problems


# ============================================================
# CONTROLLI FRAME phi
# ============================================================

def check_data_npy(
    npy_path: Path,
    shape_expected: tuple[int, int, int],
) -> list[str]:
    """Verifica integrità, shape, dtype, NaN/Inf e range del frame phi."""
    problems: list[str] = []

    if not npy_path.exists():
        return [f"NPY mancante: {npy_path.name}"]

    try:
        grid = np.load(
            npy_path,
            mmap_mode="r",
            allow_pickle=False,
        )
    except Exception as exc:
        return [
            f"Impossibile leggere {npy_path.name}: "
            f"{type(exc).__name__}: {exc}"
        ]

    if grid.ndim != 3:
        problems.append(
            f"{npy_path.name}: ndim={grid.ndim}, atteso=3"
        )

    if grid.shape != shape_expected:
        problems.append(
            f"{npy_path.name}: shape={grid.shape}, attesa={shape_expected}"
        )

    if grid.dtype != np.float32:
        problems.append(
            f"{npy_path.name}: dtype={grid.dtype}, atteso=float32"
        )

    # Con voxel=0.025 i file sono piccoli: caricarli per controllare i valori
    # non comporta un problema significativo di memoria.
    values = np.asarray(grid)

    invalid_mask = ~np.isfinite(values)
    n_invalid = int(invalid_mask.sum())

    if n_invalid > 0:
        problems.append(
            f"{npy_path.name}: NaN/Inf trovati={n_invalid}/{values.size}"
        )
        return problems

    value_min = float(values.min())
    value_max = float(values.max())

    # Per phi ci si aspetta un intervallo approssimativamente [-1, 1].
    # Se il tuo campo ha una normalizzazione differente, modifica questa soglia.
    if value_min < -1.05 or value_max > 1.05:
        problems.append(
            f"{npy_path.name}: range phi inatteso "
            f"[min={value_min:.7f}, max={value_max:.7f}]"
        )

    return problems


# ============================================================
# CONTROLLI MASCHERE linear -> nearest
# ============================================================

def mask_path_for_data(data_path: Path) -> Path:
    """Da surf_0.000000.npy ottiene surf_0.000000_linear_nan_mask.npy."""
    return data_path.with_name(
        f"{data_path.stem}_linear_nan_mask.npy"
    )


def check_mask_npy(
    mask_path: Path,
    shape_expected: tuple[int, int, int],
) -> tuple[list[str], int, int]:
    """
    Controlla una mask separatamente dal dato phi.

    La mask deve essere uint8, con valori solo 0 e 1.
    Restituisce: problems, n_fallback, n_total.
    """
    problems: list[str] = []

    if not mask_path.exists():
        return [f"Mask mancante: {mask_path.name}"], 0, 0

    try:
        mask = np.load(mask_path, mmap_mode="r", allow_pickle=False)
    except Exception as exc:
        return [
            f"Impossibile leggere mask {mask_path.name}: "
            f"{type(exc).__name__}: {exc}"
        ], 0, 0

    if mask.ndim != 3:
        problems.append(
            f"{mask_path.name}: ndim={mask.ndim}, atteso=3"
        )

    if mask.shape != shape_expected:
        problems.append(
            f"{mask_path.name}: shape={mask.shape}, attesa={shape_expected}"
        )

    if mask.dtype != np.uint8:
        problems.append(
            f"{mask_path.name}: dtype={mask.dtype}, atteso=uint8"
        )

    mask_values = np.asarray(mask)
    unique_values = np.unique(mask_values)

    if not np.all(np.isin(unique_values, [0, 1])):
        problems.append(
            f"{mask_path.name}: valori non binari trovati: "
            f"{unique_values[:10]}"
        )

    n_fallback = int(mask_values.sum())
    n_total = int(mask_values.size)

    return problems, n_fallback, n_total


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    if not INPUT_ROOT.is_dir():
        raise FileNotFoundError(f"Input root non trovata: {INPUT_ROOT}")

    if not OUTPUT_ROOT.is_dir():
        raise FileNotFoundError(f"Output root non trovata: {OUTPUT_ROOT}")

    input_folders = sorted(
        {
            vtu_path.parent
            for vtu_path in INPUT_ROOT.rglob(VTU_GLOB)
        }
    )

    if not input_folders:
        raise FileNotFoundError(
            f"Nessun file '{VTU_GLOB}' trovato in {INPUT_ROOT}"
        )

    print("=" * 78)
    print("VERIFICA OUTPUT resizeFolder")
    print("=" * 78)
    print(f"Input root      : {INPUT_ROOT}")
    print(f"Output root     : {OUTPUT_ROOT}")
    print(f"Voxel size      : {VOXEL_SIZE}")
    print(f"Controlla tutti : {CHECK_EVERY_FILE}")
    print(f"Cartelle input  : {len(input_folders)}")

    folders_ok = 0
    folders_with_problems = 0

    total_vtu = 0
    total_data_npy = 0
    total_masks = 0
    total_checked_data = 0
    total_checked_masks = 0
    total_fallback = 0
    total_mask_voxels = 0

    report_lines: list[str] = []

    for folder_index, input_folder in enumerate(input_folders, start=1):
        relative_folder = input_folder.relative_to(INPUT_ROOT)
        output_folder = OUTPUT_ROOT / relative_folder

        height = extract_height(input_folder.name)
        shape_expected = expected_shape(height)

        vtu_files = sorted(input_folder.glob(VTU_GLOB))

        if output_folder.exists():
            all_output_npy = sorted(output_folder.glob(DATA_NPY_GLOB))
            data_npy_files = [
                path for path in all_output_npy if is_data_npy(path)
            ]
            mask_npy_files = [
                path for path in all_output_npy if is_mask_npy(path)
            ]
        else:
            data_npy_files = []
            mask_npy_files = []

        total_vtu += len(vtu_files)
        total_data_npy += len(data_npy_files)
        total_masks += len(mask_npy_files)

        folder_problems: list[str] = []

        print("\n" + "-" * 78)
        print(f"[{folder_index:03d}/{len(input_folders):03d}] {input_folder.name}")
        print(f"Height         : {height}")
        print(f"Shape attesa   : {shape_expected}")
        print(f"VTU trovati    : {len(vtu_files)}")
        print(f"NPY phi trovati: {len(data_npy_files)}")
        print(f"Mask trovate   : {len(mask_npy_files)}")

        if not output_folder.is_dir():
            folder_problems.append(
                f"Cartella output mancante: {output_folder}"
            )
        else:
            metadata_path = output_folder / "grid_metadata.npz"
            folder_problems.extend(
                check_metadata(
                    metadata_path=metadata_path,
                    height=height,
                    shape_expected=shape_expected,
                )
            )

        # Per ogni VTU deve esistere un vero NPY phi con stesso stem.
        expected_data_paths = {
            data_npy_path_for(vtu_path)
            for vtu_path in vtu_files
        }
        actual_data_paths = set(data_npy_files)

        missing_data = sorted(expected_data_paths - actual_data_paths)
        extra_data = sorted(actual_data_paths - expected_data_paths)

        if missing_data:
            folder_problems.append(f"NPY phi mancanti: {len(missing_data)}")
            for missing_path in missing_data[:5]:
                folder_problems.append(f"  Manca: {missing_path.name}")
            if len(missing_data) > 5:
                folder_problems.append("  ...")

        if extra_data:
            folder_problems.append(
                f"NPY phi extra/non associati a VTU: {len(extra_data)}"
            )
            for extra_path in extra_data[:5]:
                folder_problems.append(f"  Extra: {extra_path.name}")
            if len(extra_data) > 5:
                folder_problems.append("  ...")

        # Controllo dei frame phi: le mask non entrano qui.
        files_to_check = get_files_to_check(data_npy_files)

        for data_path in files_to_check:
            total_checked_data += 1
            folder_problems.extend(
                check_data_npy(
                    npy_path=data_path,
                    shape_expected=shape_expected,
                )
            )

            # Controllo della mask corrispondente, se desideri verificarla.
            mask_path = mask_path_for_data(data_path)

            if mask_path.exists():
                total_checked_masks += 1

                mask_problems, n_fallback, n_total = check_mask_npy(
                    mask_path=mask_path,
                    shape_expected=shape_expected,
                )

                folder_problems.extend(mask_problems)
                total_fallback += n_fallback
                total_mask_voxels += n_total

        # Se ci sono mask, assicurati anche che non esistano mask orfane.
        expected_mask_paths = {
            mask_path_for_data(data_path)
            for data_path in data_npy_files
        }
        actual_mask_paths = set(mask_npy_files)

        orphan_masks = sorted(actual_mask_paths - expected_mask_paths)
        if orphan_masks:
            folder_problems.append(
                f"Mask orfane/non associate a NPY phi: {len(orphan_masks)}"
            )

        if folder_problems:
            folders_with_problems += 1
            print("STATO: PROBLEMI TROVATI")

            for problem in folder_problems:
                print(f"  - {problem}")
                report_lines.append(f"{input_folder.name}: {problem}")
        else:
            folders_ok += 1
            print(
                "STATO: OK | "
                f"frame phi controllati={len(files_to_check)}"
            )

    print("\n" + "=" * 78)
    print("RIEPILOGO")
    print("=" * 78)
    print(f"Cartelle OK              : {folders_ok}")
    print(f"Cartelle con problemi    : {folders_with_problems}")
    print(f"VTU totali               : {total_vtu}")
    print(f"NPY phi totali           : {total_data_npy}")
    print(f"Mask totali              : {total_masks}")
    print(f"Frame phi controllati    : {total_checked_data}")
    print(f"Mask controllate         : {total_checked_masks}")

    if total_vtu == total_data_npy:
        print("Corrispondenza VTU/NPY   : OK")
    else:
        print(
            "Corrispondenza VTU/NPY   : ATTENZIONE "
            f"({total_vtu} VTU, {total_data_npy} NPY phi)"
        )

    if total_mask_voxels > 0:
        fallback_percent = 100.0 * total_fallback / total_mask_voxels
        print(
            f"Fallback linear->nearest : {total_fallback}/{total_mask_voxels} "
            f"({fallback_percent:.6f}%)"
        )
    else:
        print("Fallback linear->nearest : nessuna mask controllata")

    if report_lines:
        report_path = OUTPUT_ROOT / REPORT_NAME
        report_path.write_text(
            "\n".join(report_lines) + "\n",
            encoding="utf-8",
        )
        print(f"Report problemi          : {report_path}")
    else:
        print("Report problemi          : nessun problema trovato")


if __name__ == "__main__":
    main()
