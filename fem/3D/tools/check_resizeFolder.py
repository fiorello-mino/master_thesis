from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import numpy as np


# ============================================================
# CONFIGURAZIONE
# ============================================================

INPUT_ROOT = Path("/archive/roberto/poresAMDIS/iso_P09")
OUTPUT_ROOT = Path("/data/fiorello/iso_P09")

VTU_GLOB = "surf_*.vtu"
NPY_GLOB = "surf_*.npy"

# Deve essere coerente con resizeFolder.py.
LX = 0.45
LY = 0.45
EPS = 0.1
VOXEL_SIZE = 0.025

# Tolleranza per confronti numerici.
TOL = 1e-6

# True: controlla ogni NPY.
# False: controlla solo primo, centrale e ultimo frame di ogni folder.
CHECK_EVERY_FILE = True


# ============================================================
# FUNZIONI
# ============================================================

def extract_height(folder_name: str) -> float:
    """
    Esempio:
        iso2_R0.2_H1.0_P0.9 -> 1.0
    """
    match = re.search(r"(?:^|_)H(-?\d+(?:\.\d+)?)", folder_name)

    if match is None:
        raise ValueError(
            f"Impossibile estrarre H dal nome della cartella: {folder_name}"
        )

    return float(match.group(1))


def expected_shape(height: float) -> tuple[int, int, int]:
    """
    Shape attesa per la griglia isotropa:

        z in [-height - 2*EPS, 2*EPS]

    con dx=dy=dz=VOXEL_SIZE e endpoint inclusi.
    """
    crop_height = height + 4.0 * EPS

    nx = int(round(LX / VOXEL_SIZE)) + 1
    ny = int(round(LY / VOXEL_SIZE)) + 1
    nz = int(round(crop_height / VOXEL_SIZE)) + 1

    return nx, ny, nz


def get_output_npy_path(vtu_path: Path) -> Path:
    """
    Dato un VTU nella root input, restituisce il path NPY atteso
    nella root output.
    """
    relative = vtu_path.relative_to(INPUT_ROOT)

    return (OUTPUT_ROOT / relative).with_suffix(".npy")


def choose_files_to_check(npy_files: list[Path]) -> list[Path]:
    """
    Se CHECK_EVERY_FILE=False, controlla:
    - primo frame;
    - frame centrale;
    - ultimo frame.
    """
    if CHECK_EVERY_FILE:
        return npy_files

    if len(npy_files) <= 3:
        return npy_files

    indices = sorted(
        {
            0,
            len(npy_files) // 2,
            len(npy_files) - 1,
        }
    )

    return [npy_files[i] for i in indices]


def check_metadata(
    metadata_path: Path,
    height: float,
    shape_expected: tuple[int, int, int],
) -> list[str]:
    """
    Verifica shape, voxel size e isotropia memorizzate nei metadata.
    """
    problems = []

    if not metadata_path.exists():
        return [f"Metadata mancante: {metadata_path.name}"]

    metadata = np.load(metadata_path)

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

    missing = required_keys - set(metadata.files)

    if missing:
        problems.append(
            f"Chiavi mancanti nel metadata: {sorted(missing)}"
        )
        return problems

    xi = metadata["xi"]
    yi = metadata["yi"]
    zi = metadata["zi"]

    saved_shape = tuple(int(x) for x in metadata["shape"])
    coordinates_shape = (len(xi), len(yi), len(zi))

    metadata_height = float(metadata["height"])
    metadata_eps = float(metadata["eps"])
    metadata_voxel_size = float(metadata["voxel_size"])

    dx = float(metadata["dx"])
    dy = float(metadata["dy"])
    dz = float(metadata["dz"])

    if saved_shape != shape_expected:
        problems.append(
            f"Shape metadata errata: {saved_shape}, "
            f"attesa {shape_expected}"
        )

    if coordinates_shape != shape_expected:
        problems.append(
            f"Coordinate metadata errate: {coordinates_shape}, "
            f"attese {shape_expected}"
        )

    if not np.isclose(metadata_height, height, atol=TOL):
        problems.append(
            f"Height metadata={metadata_height}, attesa={height}"
        )

    if not np.isclose(metadata_eps, EPS, atol=TOL):
        problems.append(
            f"EPS metadata={metadata_eps}, atteso={EPS}"
        )

    if not np.isclose(metadata_voxel_size, VOXEL_SIZE, atol=TOL):
        problems.append(
            f"Voxel size metadata={metadata_voxel_size}, "
            f"atteso={VOXEL_SIZE}"
        )

    if not np.isclose(dx, VOXEL_SIZE, atol=TOL):
        problems.append(f"dx={dx}, atteso={VOXEL_SIZE}")

    if not np.isclose(dy, VOXEL_SIZE, atol=TOL):
        problems.append(f"dy={dy}, atteso={VOXEL_SIZE}")

    if not np.isclose(dz, VOXEL_SIZE, atol=TOL):
        problems.append(f"dz={dz}, atteso={VOXEL_SIZE}")

    # Controllo diretto degli array delle coordinate.
    if len(xi) > 1 and not np.allclose(np.diff(xi), VOXEL_SIZE, atol=TOL):
        problems.append("xi non ha passo costante VOXEL_SIZE")

    if len(yi) > 1 and not np.allclose(np.diff(yi), VOXEL_SIZE, atol=TOL):
        problems.append("yi non ha passo costante VOXEL_SIZE")

    if len(zi) > 1 and not np.allclose(np.diff(zi), VOXEL_SIZE, atol=TOL):
        problems.append("zi non ha passo costante VOXEL_SIZE")

    expected_z_start = -height - 2.0 * EPS
    expected_z_end = 2.0 * EPS

    if not np.isclose(zi[0], expected_z_start, atol=TOL):
        problems.append(
            f"zi[0]={zi[0]}, atteso={expected_z_start}"
        )

    if not np.isclose(zi[-1], expected_z_end, atol=TOL):
        problems.append(
            f"zi[-1]={zi[-1]}, atteso={expected_z_end}"
        )

    return problems


def check_npy_file(
    npy_path: Path,
    shape_expected: tuple[int, int, int],
) -> list[str]:
    """
    Controlla che il file:
    - esista;
    - sia leggibile;
    - sia 3D;
    - abbia shape attesa;
    - sia float32;
    - non abbia NaN / Inf;
    - abbia valori ragionevoli per phi.
    """
    problems = []

    if not npy_path.exists():
        return [f"NPY mancante: {npy_path.name}"]

    try:
        # mmap_mode='r': legge header e accede ai dati senza caricare
        # necessariamente l'intero volume in RAM.
        grid = np.load(
            npy_path,
            mmap_mode="r",
            allow_pickle=False,
        )
    except Exception as exc:
        return [
            f"Impossibile caricare {npy_path.name}: "
            f"{type(exc).__name__}: {exc}"
        ]

    if grid.ndim != 3:
        problems.append(
            f"{npy_path.name}: ndim={grid.ndim}, atteso 3"
        )

    if grid.shape != shape_expected:
        problems.append(
            f"{npy_path.name}: shape={grid.shape}, "
            f"attesa={shape_expected}"
        )

    if grid.dtype != np.float32:
        problems.append(
            f"{npy_path.name}: dtype={grid.dtype}, "
            "atteso=float32"
        )

    # Carica il contenuto per controllare NaN, Inf, min e max.
    # Le tue griglie con voxel=0.025 sono piccole, quindi va bene.
    values = np.asarray(grid)

    if not np.all(np.isfinite(values)):
        n_invalid = int((~np.isfinite(values)).sum())

        problems.append(
            f"{npy_path.name}: trovati {n_invalid} NaN/Inf"
        )

    value_min = float(values.min())
    value_max = float(values.max())

    # Adatta questi limiti se il tuo phi non è definito in [-1, 1].
    # Tolleranza piccola per interpolazione e precisione float.
    if value_min < -1.05 or value_max > 1.05:
        problems.append(
            f"{npy_path.name}: phi fuori range atteso "
            f"[min={value_min:.6f}, max={value_max:.6f}]"
        )

    return problems


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    if not INPUT_ROOT.is_dir():
        raise FileNotFoundError(
            f"Input root non trovata: {INPUT_ROOT}"
        )

    if not OUTPUT_ROOT.is_dir():
        raise FileNotFoundError(
            f"Output root non trovata: {OUTPUT_ROOT}"
        )

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
    print("VERIFICA DATASET resizeFolder")
    print("=" * 78)
    print(f"Input root      : {INPUT_ROOT}")
    print(f"Output root     : {OUTPUT_ROOT}")
    print(f"Voxel size      : {VOXEL_SIZE}")
    print(f"Controlla tutto : {CHECK_EVERY_FILE}")
    print(f"Cartelle input  : {len(input_folders)}")

    total_vtu = 0
    total_npy_found = 0
    total_checked = 0
    folders_ok = 0
    folders_with_errors = 0
    all_problems: list[str] = []

    for folder_index, input_folder in enumerate(input_folders, start=1):
        relative_folder = input_folder.relative_to(INPUT_ROOT)
        output_folder = OUTPUT_ROOT / relative_folder

        height = extract_height(input_folder.name)
        shape_expected = expected_shape(height)

        vtu_files = sorted(input_folder.glob(VTU_GLOB))
        npy_files = sorted(output_folder.glob(NPY_GLOB)) if output_folder.exists() else []

        total_vtu += len(vtu_files)
        total_npy_found += len(npy_files)

        folder_problems: list[str] = []

        print("\n" + "-" * 78)
        print(f"[{folder_index:03d}/{len(input_folders):03d}] {input_folder.name}")
        print(f"Height         : {height}")
        print(f"Shape attesa   : {shape_expected}")
        print(f"VTU trovati    : {len(vtu_files)}")
        print(f"NPY trovati    : {len(npy_files)}")

        if not output_folder.exists():
            folder_problems.append(
                f"Cartella output mancante: {output_folder}"
            )
        else:
            metadata_path = output_folder / "grid_metadata.npz"

            metadata_problems = check_metadata(
                metadata_path=metadata_path,
                height=height,
                shape_expected=shape_expected,
            )

            folder_problems.extend(metadata_problems)

        # Ogni VTU deve avere il corrispondente NPY.
        expected_npy_paths = {
            get_output_npy_path(vtu_path)
            for vtu_path in vtu_files
        }

        actual_npy_paths = set(npy_files)

        missing_npy = sorted(expected_npy_paths - actual_npy_paths)
        extra_npy = sorted(actual_npy_paths - expected_npy_paths)

        if missing_npy:
            folder_problems.append(
                f"NPY mancanti: {len(missing_npy)}"
            )

            for path in missing_npy[:5]:
                folder_problems.append(f"  Manca: {path.name}")

            if len(missing_npy) > 5:
                folder_problems.append("  ...")

        if extra_npy:
            folder_problems.append(
                f"NPY extra/non associati a VTU: {len(extra_npy)}"
            )

        # Controlla un sottocampione o tutti gli NPY.
        files_to_check = choose_files_to_check(npy_files)

        for npy_path in files_to_check:
            total_checked += 1

            file_problems = check_npy_file(
                npy_path=npy_path,
                shape_expected=shape_expected,
            )

            folder_problems.extend(file_problems)

        if folder_problems:
            folders_with_errors += 1

            print("STATO: PROBLEMI TROVATI")

            for problem in folder_problems:
                print(f"  - {problem}")

                all_problems.append(
                    f"{input_folder.name}: {problem}"
                )

        else:
            folders_ok += 1
            print(
                "STATO: OK | "
                f"{len(files_to_check)} NPY controllati | "
                "shape, dtype, isotropia e valori validi"
            )

    print("\n" + "=" * 78)
    print("RIEPILOGO")
    print("=" * 78)
    print(f"Cartelle OK            : {folders_ok}")
    print(f"Cartelle con problemi  : {folders_with_errors}")
    print(f"VTU totali             : {total_vtu}")
    print(f"NPY trovati            : {total_npy_found}")
    print(f"File NPY controllati   : {total_checked}")

    if total_vtu == total_npy_found:
        print("Numero VTU/NPY         : OK, coincidono")
    else:
        print(
            "Numero VTU/NPY         : ATTENZIONE, non coincidono "
            f"({total_vtu} VTU vs {total_npy_found} NPY)"
        )

    if all_problems:
        report_path = OUTPUT_ROOT / "resize_check_report.txt"

        report_path.write_text(
            "\n".join(all_problems) + "\n",
            encoding="utf-8",
        )

        print(f"Report problemi        : {report_path}")
    else:
        print("Report problemi        : nessun problema trovato")


if __name__ == "__main__":
    main()