from __future__ import annotations

import re
from pathlib import Path

import numpy as np


# ============================================================
# CONFIGURAZIONE
# ============================================================

# Root con le cartelle originali contenenti surf_*.vtu.
INPUT_ROOT = Path("/archive/roberto/poresAMDIS/iso_P06")

# Root prodotta dallo script resizeFolder.py basato su PyVista/VTK.
OUTPUT_ROOT = Path("/data/fiorello/poresAMDIS/iso_P06")

# Parametri che devono coincidere con resizeFolder.py.
LX = 0.3
LY = 0.3
EPS = 0.1
VOXEL_SIZE = 0.025

# Tolleranza per confronti floating point.
TOL = 1e-6

# True  -> controlla ogni frame .npy.
# False -> controlla primo, centrale e ultimo frame di ogni cartella.
CHECK_EVERY_FILE = False

VTU_GLOB = "surf_*.vtu"
DATA_NPY_GLOB = "surf_*.npy"

# Suffisso delle mask create dalla versione PyVista/VTK di resizeFolder.py.
# 0 = campionato dalla cella FEM VTK
# 1 = punto esterno/non valido, riempito con nearest-node fallback
VTK_MASK_SUFFIX = "_vtk_fallback_mask.npy"

REPORT_NAME = "resize_pyvista_check_report.txt"


# ============================================================
# PATH, NOMI E SHAPE ATTESE
# ============================================================

def extract_height(folder_name: str) -> float:
    """Esempio: iso2_R0.2_H1.0_P0.9 -> 1.0."""
    match = re.search(r"(?:^|_)H(-?\d+(?:\.\d+)?)", folder_name)

    if match is None:
        raise ValueError(
            f"Impossibile estrarre H dal nome cartella: {folder_name}"
        )

    return float(match.group(1))


def expected_shape(height: float) -> tuple[int, int, int]:
    """
    Shape della griglia isotropa prodotta da resizeFolder.py.

    Regione z:
        [-height - 2*EPS, 2*EPS]

    Endpoint inclusi.
    """
    crop_height = height + 4.0 * EPS

    nx = int(round(LX / VOXEL_SIZE)) + 1
    ny = int(round(LY / VOXEL_SIZE)) + 1
    nz = int(round(crop_height / VOXEL_SIZE)) + 1

    return nx, ny, nz


def expected_data_path(vtu_path: Path) -> Path:
    """Path NPY phi corrispondente a un file VTU."""
    relative_path = vtu_path.relative_to(INPUT_ROOT)
    return (OUTPUT_ROOT / relative_path).with_suffix(".npy")


def expected_mask_path(data_path: Path) -> Path:
    """Path della mask VTK corrispondente a un NPY phi."""
    return data_path.with_name(
        f"{data_path.stem}_vtk_fallback_mask.npy"
    )


def is_data_npy(path: Path) -> bool:
    """
    True solo per gli NPY contenenti phi.

    Include:
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


def is_vtk_mask(path: Path) -> bool:
    """True per le mask uint8 prodotte dal sampling PyVista/VTK."""
    return path.is_file() and path.name.endswith(VTK_MASK_SUFFIX)


def get_files_to_check(data_files: list[Path]) -> list[Path]:
    """Tutti i frame oppure primo/centrale/ultimo, secondo configurazione."""
    if CHECK_EVERY_FILE or len(data_files) <= 3:
        return data_files

    indices = sorted({0, len(data_files) // 2, len(data_files) - 1})
    return [data_files[index] for index in indices]


# ============================================================
# CONTROLLO METADATA
# ============================================================

def check_metadata(
    metadata_path: Path,
    height: float,
    shape_expected: tuple[int, int, int],
) -> list[str]:
    """Verifica coordinate, isotropia e metadati della singola traiettoria."""
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
        "sampling_method",
        "fallback_method",
    }

    missing_keys = required_keys - set(metadata.files)
    if missing_keys:
        return [
            f"Chiavi mancanti in metadata: {sorted(missing_keys)}"
        ]

    xi = metadata["xi"]
    yi = metadata["yi"]
    zi = metadata["zi"]

    stored_shape = tuple(int(value) for value in metadata["shape"])
    coordinate_shape = (len(xi), len(yi), len(zi))

    stored_height = float(metadata["height"])
    stored_eps = float(metadata["eps"])
    stored_voxel = float(metadata["voxel_size"])

    stored_dx = float(metadata["dx"])
    stored_dy = float(metadata["dy"])
    stored_dz = float(metadata["dz"])

    sampling_method = str(metadata["sampling_method"])
    fallback_method = str(metadata["fallback_method"])

    if stored_shape != shape_expected:
        problems.append(
            f"Shape metadata={stored_shape}, attesa={shape_expected}"
        )

    if coordinate_shape != shape_expected:
        problems.append(
            f"Lunghezza coordinate={coordinate_shape}, "
            f"attesa={shape_expected}"
        )

    if not np.isclose(stored_height, height, rtol=0.0, atol=TOL):
        problems.append(
            f"Height metadata={stored_height}, attesa={height}"
        )

    if not np.isclose(stored_eps, EPS, rtol=0.0, atol=TOL):
        problems.append(
            f"EPS metadata={stored_eps}, atteso={EPS}"
        )

    if not np.isclose(stored_voxel, VOXEL_SIZE, rtol=0.0, atol=TOL):
        problems.append(
            f"Voxel metadata={stored_voxel}, atteso={VOXEL_SIZE}"
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

    if sampling_method != "pyvista_vtk_cell_sample":
        problems.append(
            f"sampling_method='{sampling_method}', atteso "
            "'pyvista_vtk_cell_sample'"
        )

    if fallback_method != "nearest_node":
        problems.append(
            f"fallback_method='{fallback_method}', atteso 'nearest_node'"
        )

    return problems


# ============================================================
# CONTROLLO NPY phi
# ============================================================

def check_data_npy(
    data_path: Path,
    shape_expected: tuple[int, int, int],
) -> list[str]:
    """Controlla integrità, shape, dtype, NaN/Inf e range del campo phi."""
    problems: list[str] = []

    if not data_path.exists():
        return [f"NPY phi mancante: {data_path.name}"]

    try:
        grid = np.load(
            data_path,
            mmap_mode="r",
            allow_pickle=False,
        )
    except Exception as exc:
        return [
            f"Impossibile leggere {data_path.name}: "
            f"{type(exc).__name__}: {exc}"
        ]

    if grid.ndim != 3:
        problems.append(f"{data_path.name}: ndim={grid.ndim}, atteso=3")

    if grid.shape != shape_expected:
        problems.append(
            f"{data_path.name}: shape={grid.shape}, "
            f"attesa={shape_expected}"
        )

    if grid.dtype != np.float32:
        problems.append(
            f"{data_path.name}: dtype={grid.dtype}, atteso=float32"
        )

    values = np.asarray(grid)
    invalid_mask = ~np.isfinite(values)

    if invalid_mask.any():
        problems.append(
            f"{data_path.name}: NaN/Inf={int(invalid_mask.sum())}/"
            f"{values.size}"
        )
        return problems

    phi_min = float(values.min())
    phi_max = float(values.max())

    # Modifica soltanto se phi nel tuo solver non sta approssimativamente in [-1, 1].
    if phi_min < -1.05 or phi_max > 1.05:
        problems.append(
            f"{data_path.name}: range phi inatteso "
            f"[min={phi_min:.7f}, max={phi_max:.7f}]"
        )

    return problems


# ============================================================
# CONTROLLO MASK PyVista/VTK
# ============================================================

def check_vtk_mask(
    mask_path: Path,
    shape_expected: tuple[int, int, int],
) -> tuple[list[str], int, int]:
    """
    Verifica mask VTK uint8.

    Convenzione:
        0 = sampling VTK dentro una cella FEM
        1 = sampling VTK non valido, usato fallback nearest-node

    Restituisce:
        problems, n_fallback, n_total
    """
    problems: list[str] = []

    if not mask_path.exists():
        return [f"Mask VTK mancante: {mask_path.name}"], 0, 0

    try:
        mask = np.load(
            mask_path,
            mmap_mode="r",
            allow_pickle=False,
        )
    except Exception as exc:
        return [
            f"Impossibile leggere {mask_path.name}: "
            f"{type(exc).__name__}: {exc}"
        ], 0, 0

    if mask.ndim != 3:
        problems.append(f"{mask_path.name}: ndim={mask.ndim}, atteso=3")

    if mask.shape != shape_expected:
        problems.append(
            f"{mask_path.name}: shape={mask.shape}, "
            f"attesa={shape_expected}"
        )

    if mask.dtype != np.uint8:
        problems.append(
            f"{mask_path.name}: dtype={mask.dtype}, atteso=uint8"
        )

    values = np.asarray(mask)
    unique_values = np.unique(values)

    if not np.all(np.isin(unique_values, [0, 1])):
        problems.append(
            f"{mask_path.name}: valori non binari: {unique_values[:10]}"
        )

    n_fallback = int(values.sum())
    n_total = int(values.size)

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
            f"Nessun file '{VTU_GLOB}' trovato sotto {INPUT_ROOT}"
        )

    print("=" * 80)
    print("VERIFICA DATASET resizeFolder - PyVista/VTK")
    print("=" * 80)
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

        if output_folder.is_dir():
            all_npy = sorted(output_folder.glob(DATA_NPY_GLOB))
            data_files = [path for path in all_npy if is_data_npy(path)]
            mask_files = [path for path in all_npy if is_vtk_mask(path)]
        else:
            data_files = []
            mask_files = []

        total_vtu += len(vtu_files)
        total_data_npy += len(data_files)
        total_masks += len(mask_files)

        folder_problems: list[str] = []

        print("\n" + "-" * 80)
        print(f"[{folder_index:03d}/{len(input_folders):03d}] {input_folder.name}")
        print(f"Height         : {height}")
        print(f"Shape attesa   : {shape_expected}")
        print(f"VTU trovati    : {len(vtu_files)}")
        print(f"NPY phi trovati: {len(data_files)}")
        print(f"Mask VTK       : {len(mask_files)}")

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

        # Ogni VTU deve corrispondere a un NPY phi omonimo.
        expected_data_files = {
            expected_data_path(vtu_path)
            for vtu_path in vtu_files
        }
        actual_data_files = set(data_files)

        missing_data = sorted(expected_data_files - actual_data_files)
        extra_data = sorted(actual_data_files - expected_data_files)

        if missing_data:
            folder_problems.append(
                f"NPY phi mancanti: {len(missing_data)}"
            )
            for data_path in missing_data[:5]:
                folder_problems.append(f"  Manca: {data_path.name}")
            if len(missing_data) > 5:
                folder_problems.append("  ...")

        if extra_data:
            folder_problems.append(
                f"NPY phi extra/non associati a VTU: {len(extra_data)}"
            )
            for data_path in extra_data[:5]:
                folder_problems.append(f"  Extra: {data_path.name}")
            if len(extra_data) > 5:
                folder_problems.append("  ...")

        # Controlla i frame phi e, per ciascuno, la mask associata.
        files_to_check = get_files_to_check(data_files)

        for data_path in files_to_check:
            total_checked_data += 1

            folder_problems.extend(
                check_data_npy(
                    data_path=data_path,
                    shape_expected=shape_expected,
                )
            )

            mask_path = expected_mask_path(data_path)

            if mask_path.exists():
                total_checked_masks += 1

                mask_problems, n_fallback, n_total = check_vtk_mask(
                    mask_path=mask_path,
                    shape_expected=shape_expected,
                )

                folder_problems.extend(mask_problems)
                total_fallback += n_fallback
                total_mask_voxels += n_total
            else:
                folder_problems.append(
                    f"Mask VTK mancante per {data_path.name}: "
                    f"{mask_path.name}"
                )

        # Verifica mask orfane: devono avere un corrispondente frame phi.
        expected_masks = {
            expected_mask_path(data_path)
            for data_path in data_files
        }
        actual_masks = set(mask_files)

        orphan_masks = sorted(actual_masks - expected_masks)
        if orphan_masks:
            folder_problems.append(
                f"Mask VTK orfane: {len(orphan_masks)}"
            )
            for mask_path in orphan_masks[:5]:
                folder_problems.append(f"  Orfana: {mask_path.name}")
            if len(orphan_masks) > 5:
                folder_problems.append("  ...")

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

    print("\n" + "=" * 80)
    print("RIEPILOGO")
    print("=" * 80)
    print(f"Cartelle OK              : {folders_ok}")
    print(f"Cartelle con problemi    : {folders_with_problems}")
    print(f"VTU totali               : {total_vtu}")
    print(f"NPY phi totali           : {total_data_npy}")
    print(f"Mask VTK totali          : {total_masks}")
    print(f"Frame phi controllati    : {total_checked_data}")
    print(f"Mask VTK controllate     : {total_checked_masks}")

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
            f"VTK->nearest fallback    : {total_fallback}/{total_mask_voxels} "
            f"({fallback_percent:.6f}%)"
        )
    else:
        print("VTK->nearest fallback    : nessuna mask controllata")

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
