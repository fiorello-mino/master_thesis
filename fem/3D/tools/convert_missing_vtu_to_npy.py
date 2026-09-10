from __future__ import annotations

from pathlib import Path

import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree


# ============================================================
# CONFIGURAZIONE
# ============================================================

# Archivio sorgente: contiene le simulazioni e i VTU originali.
ARCHIVE_ROOT = Path("/archive/roberto/poresAMDIS")

# Dataset NPY gia' esistente: viene usato SOLO per il confronto.
# Lo script non modifica nulla dentro questa cartella.
EXISTING_DATA_ROOT = Path("/data/fiorello/poresAMDIS")

# Destinazione separata: qui verranno salvati tutti i nuovi NPY convertiti.
OUTPUT_ROOT = Path("/data/fiorello/test_pores3D")

# Campo scalare da estrarre dai VTU.
FIELD_NAME = "phi"

# Viene convertito soltanto il frame iniziale di ogni simulazione mancante.
VTU_GLOB = "surf_*.vtu"
INITIAL_VTU_NAME = "surf_0.000000.vtu"
INITIAL_NPY_NAME = "surf_0.000000.npy"

# Spacing della griglia cartesiana uniforme richiesta.
DX = 0.025
DY = 0.025
DZ = 0.025

# Tolleranza passata a PyVista/VTK durante il probe della mesh.
EPS = 0.1

# Assi laterali x e y: da 0.0 a L inclusi.
# Esempio iso_P06: Lx = Ly = 0.6 / 2 = 0.3.
LATERAL_LENGTH = {
    "iso_P06": 0.6 / 2,
    "iso_P07": 0.7 / 2,
    "iso_P08": 0.8 / 2,
    "iso_P09": 0.9 / 2,
    "iso_P10": 1.0 / 2,
}

# Se True, una simulazione e' considerata gia' disponibile in
# EXISTING_DATA_ROOT quando contiene almeno un vero surf_*.npy.
# Se False, serve esattamente surf_0.000000.npy.
PRESENT_IF_ANY_NPY_EXISTS = True

# Sicurezza: non sovrascrive un NPY iniziale gia' creato in OUTPUT_ROOT.
OVERWRITE_EXISTING_INITIAL_NPY = False

# Report del confronto e delle conversioni: salvato in OUTPUT_ROOT.
REPORT_PATH = OUTPUT_ROOT / "missing_simulations_initial_vtu_to_npy_report.txt"


# ============================================================
# RICONOSCIMENTO FILE E CARTELLE
# ============================================================


def is_npy_frame(path: Path) -> bool:
    """Accetta surf_<time>.npy ed esclude le fallback mask VTK."""
    return (
        path.is_file()
        and path.name.startswith("surf_")
        and path.suffix == ".npy"
        and not path.name.endswith("_vtk_fallback_mask.npy")
    )


def is_vtu_frame(path: Path) -> bool:
    """Accetta surf_<time>.vtu."""
    return (
        path.is_file()
        and path.name.startswith("surf_")
        and path.suffix.lower() == ".vtu"
    )


def find_simulation_folders(
    root: Path,
    pattern: str,
    predicate,
) -> set[Path]:
    """Trova le cartelle che contengono direttamente almeno un frame valido."""
    if not root.is_dir():
        return set()

    return {
        path.parent
        for path in root.rglob(pattern)
        if predicate(path)
    }


def relative_simulation_key(root: Path, folder: Path) -> Path:
    """Chiave relativa usata per confrontare una simulazione fra due root."""
    return folder.relative_to(root)


def get_iso_name(simulation_dir: Path) -> str:
    """Trova iso_P06, ..., iso_P10 nelle componenti del path."""
    for part in simulation_dir.parts:
        if part in LATERAL_LENGTH:
            return part

    expected = ", ".join(LATERAL_LENGTH)
    raise ValueError(
        f"Nessuna cartella iso riconosciuta in '{simulation_dir}'. "
        f"Attese: {expected}"
    )


def select_initial_vtu(simulation_dir: Path) -> Path:
    """
    Restituisce surf_0.000000.vtu se esiste; altrimenti seleziona il VTU
    con il tempo minimo nel nome surf_<tempo>.vtu.
    """
    exact_initial = simulation_dir / INITIAL_VTU_NAME

    if exact_initial.is_file():
        return exact_initial

    vtu_files = [
        path
        for path in simulation_dir.glob(VTU_GLOB)
        if is_vtu_frame(path)
    ]

    if not vtu_files:
        raise FileNotFoundError(
            f"Nessun file '{VTU_GLOB}' trovato in {simulation_dir}"
        )

    def parse_vtu_time(path: Path) -> float:
        try:
            return float(path.stem.removeprefix("surf_"))
        except ValueError as exc:
            raise ValueError(f"Nome VTU non valido: {path.name}") from exc

    return min(vtu_files, key=parse_vtu_time)


# ============================================================
# CAMPO SCALARE E GRIGLIA
# ============================================================


def prepare_source_mesh(mesh: pv.DataSet) -> pv.DataSet:
    """
    Assicura che FIELD_NAME sia presente in point_data.

    Se phi e' in cell_data, lo converte in point_data. Se phi non esiste,
    ma esiste un solo campo scalare nei point_data, usa quel campo come phi.
    """
    if FIELD_NAME in mesh.point_data:
        return mesh

    if FIELD_NAME in mesh.cell_data:
        return mesh.cell_data_to_point_data(pass_cell_data=False)

    scalar_candidates: list[str] = []

    for name, array in mesh.point_data.items():
        values = np.asarray(array).squeeze()

        if values.ndim == 1 and values.size == mesh.n_points:
            scalar_candidates.append(name)

    if len(scalar_candidates) == 1:
        source_mesh = mesh.copy(deep=True)
        source_mesh.point_data[FIELD_NAME] = np.asarray(
            source_mesh.point_data[scalar_candidates[0]]
        ).squeeze()
        return source_mesh

    raise KeyError(
        f"Campo '{FIELD_NAME}' non trovato. "
        f"point_data={list(mesh.point_data.keys())}; "
        f"cell_data={list(mesh.cell_data.keys())}"
    )


def get_phi_values(source_mesh: pv.DataSet) -> np.ndarray:
    """Legge e verifica il campo phi memorizzato nei point_data."""
    values = np.asarray(source_mesh.point_data[FIELD_NAME]).squeeze()

    if values.ndim != 1:
        raise ValueError(
            f"Campo '{FIELD_NAME}' non scalare: shape={values.shape}"
        )

    if values.size != source_mesh.n_points:
        raise ValueError(
            f"Campo '{FIELD_NAME}' ha {values.size} valori, ma la mesh ha "
            f"{source_mesh.n_points} punti"
        )

    if not np.isfinite(values).all():
        n_nonfinite = int((~np.isfinite(values)).sum())
        raise ValueError(
            f"Campo '{FIELD_NAME}' contiene {n_nonfinite} valori NaN/Inf"
        )

    return values.astype(np.float64, copy=False)


def axis_from_zero_to_length(length: float, spacing: float) -> np.ndarray:
    """
    Costruisce coordinate da 0 a length inclusi, con passo spacing.

    Esempio: length=0.3 e spacing=0.025
    -> [0.000, 0.025, ..., 0.300], cioe' 13 punti.
    """
    n_intervals = round(length / spacing)

    if not np.isclose(n_intervals * spacing, length):
        raise ValueError(
            f"L={length} non e' un multiplo di spacing={spacing}"
        )

    return np.linspace(
        0.0,
        length,
        n_intervals + 1,
        dtype=np.float64,
    )


def z_axis_from_bounds(mesh: pv.DataSet) -> np.ndarray:
    """Costruisce z dai bounds del VTU con spacing DZ e estremi inclusi."""
    z_min = float(mesh.bounds[4])
    z_max = float(mesh.bounds[5])
    z_extent = z_max - z_min

    if z_extent <= 0.0:
        raise ValueError(
            f"Bounds z non validi: z_min={z_min}, z_max={z_max}"
        )

    n_intervals = round(z_extent / DZ)

    if n_intervals <= 0:
        raise ValueError(
            f"Estensione z={z_extent} troppo piccola per DZ={DZ}"
        )

    if not np.isclose(n_intervals * DZ, z_extent):
        raise ValueError(
            f"Estensione z={z_extent} non multipla di DZ={DZ}; "
            "impossibile costruire una griglia uniforme esatta"
        )

    return np.linspace(
        z_min,
        z_max,
        n_intervals + 1,
        dtype=np.float64,
    )


# ============================================================
# CONVERSIONE VTU -> NPY
# ============================================================


def convert_initial_vtu_to_npy(
    vtu_path: Path,
    output_npy: Path,
    lateral_length: float,
) -> tuple[tuple[int, int, int], float, float, int]:
    """
    Converte un VTU nel campo NPY 3D phi su griglia uniforme.

    L'array salvato ha shape (Nx, Ny, Nz), con:
        x = 0, DX, ..., lateral_length
        y = 0, DY, ..., lateral_length
        z = z_min, DZ, ..., z_max del VTU.

    Per i punti non coperti dal probe VTK, usa il valore del nodo FEM piu'
    vicino. Il conteggio di tali punti e' restituito come fallback.
    """
    raw_mesh = pv.read(vtu_path)
    source_mesh = prepare_source_mesh(raw_mesh)
    source_values = get_phi_values(source_mesh)

    x = axis_from_zero_to_length(lateral_length, DX)
    y = axis_from_zero_to_length(lateral_length, DY)
    z = z_axis_from_bounds(source_mesh)

    nx, ny, nz = len(x), len(y), len(z)

    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    query_points = np.column_stack(
        (
            xx.ravel(order="C"),
            yy.ravel(order="C"),
            zz.ravel(order="C"),
        )
    )

    # Per query_grid.sample(source_mesh), PyVista interpola i point_data della
    # source_mesh nella cella FEM che contiene ciascun punto della query grid.
    query_grid = pv.PolyData(query_points)
    sampled_grid = query_grid.sample(
        source_mesh,
        tolerance=EPS,
        pass_cell_data=False,
        pass_point_data=False,
    )

    sampled_values = np.asarray(
        sampled_grid.point_data[FIELD_NAME]
    ).squeeze()
    valid_mask = np.asarray(
        sampled_grid.point_data["vtkValidPointMask"]
    ).astype(bool)

    if sampled_values.ndim != 1:
        raise ValueError(
            f"Output del sampling non scalare: shape={sampled_values.shape}"
        )

    if sampled_values.size != query_points.shape[0]:
        raise ValueError(
            f"Sampling inatteso: {sampled_values.size} valori per "
            f"{query_points.shape[0]} punti query"
        )

    # Fallback nearest-neighbor per punti esterni/non trovati dal probe.
    fallback_mask = ~valid_mask

    if fallback_mask.any():
        nearest_tree = cKDTree(np.asarray(source_mesh.points))
        _, nearest_indices = nearest_tree.query(
            query_points[fallback_mask],
            k=1,
        )

        sampled_values = sampled_values.astype(np.float64, copy=True)
        sampled_values[fallback_mask] = source_values[nearest_indices]

    values_3d = sampled_values.reshape((nx, ny, nz), order="C")

    output_npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_npy, values_3d)

    return (
        values_3d.shape,
        float(values_3d.min()),
        float(values_3d.max()),
        int(fallback_mask.sum()),
    )


# ============================================================
# MAIN
# ============================================================


def main() -> None:
    if not ARCHIVE_ROOT.is_dir():
        raise FileNotFoundError(
            f"Archivio sorgente non trovato: {ARCHIVE_ROOT}"
        )

    if not EXISTING_DATA_ROOT.is_dir():
        raise FileNotFoundError(
            f"Dataset NPY esistente non trovato: {EXISTING_DATA_ROOT}"
        )

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    archive_folders = find_simulation_folders(
        root=ARCHIVE_ROOT,
        pattern=VTU_GLOB,
        predicate=is_vtu_frame,
    )

    if not archive_folders:
        raise FileNotFoundError(
            f"Nessun file '{VTU_GLOB}' trovato sotto {ARCHIVE_ROOT}"
        )

    if PRESENT_IF_ANY_NPY_EXISTS:
        existing_folders = find_simulation_folders(
            root=EXISTING_DATA_ROOT,
            pattern="surf_*.npy",
            predicate=is_npy_frame,
        )
    else:
        existing_folders = {
            path.parent
            for path in EXISTING_DATA_ROOT.rglob(INITIAL_NPY_NAME)
            if is_npy_frame(path)
        }

    archive_by_key = {
        relative_simulation_key(ARCHIVE_ROOT, folder): folder
        for folder in archive_folders
    }
    archive_keys = set(archive_by_key)

    existing_keys = {
        relative_simulation_key(EXISTING_DATA_ROOT, folder)
        for folder in existing_folders
    }

    # Queste sono le simulazioni che sono nell'archivio VTU ma non nel dataset
    # NPY esistente. Verranno convertite e salvate esclusivamente in OUTPUT_ROOT.
    missing_keys = sorted(archive_keys - existing_keys)
    only_in_existing_keys = sorted(existing_keys - archive_keys)

    report_lines = [
        "CONFRONTO ARCHIVIO VTU E DATASET NPY ESISTENTE",
        f"ARCHIVE_ROOT = {ARCHIVE_ROOT}",
        f"EXISTING_DATA_ROOT = {EXISTING_DATA_ROOT}",
        f"OUTPUT_ROOT = {OUTPUT_ROOT}",
        f"FIELD_NAME = {FIELD_NAME}",
        f"DX = {DX}",
        f"DY = {DY}",
        f"DZ = {DZ}",
        f"EPS = {EPS}",
        f"PRESENT_IF_ANY_NPY_EXISTS = {PRESENT_IF_ANY_NPY_EXISTS}",
        f"Simulazioni archive con VTU = {len(archive_keys)}",
        f"Simulazioni gia' nel dataset NPY = {len(existing_keys)}",
        f"Simulazioni da convertire = {len(missing_keys)}",
        f"Simulazioni solo nel dataset NPY = {len(only_in_existing_keys)}",
        "",
        "SIMULAZIONI DA CONVERTIRE",
    ]

    report_lines.extend(str(key) for key in missing_keys)

    report_lines.extend([
        "",
        "SIMULAZIONI PRESENTI SOLO NEL DATASET NPY ESISTENTE",
    ])
    report_lines.extend(str(key) for key in only_in_existing_keys)

    print("=" * 94)
    print("CONFRONTO SIMULAZIONI E CONVERSIONE DEI SOLI VTU INIZIALI MANCANTI")
    print("=" * 94)
    print(f"Archivio VTU                : {ARCHIVE_ROOT}")
    print(f"Dataset NPY confronto       : {EXISTING_DATA_ROOT}")
    print(f"Output nuovi NPY            : {OUTPUT_ROOT}")
    print(f"Simulazioni in archive      : {len(archive_keys)}")
    print(f"Simulazioni nel dataset     : {len(existing_keys)}")
    print(f"Simulazioni da convertire   : {len(missing_keys)}")
    print(f"Simulazioni solo nel dataset: {len(only_in_existing_keys)}")
    print(f"Griglia                     : dx=dy=dz={DX}")
    print("Assi laterali               : x,y da 0 fino a L inclusi")
    print(f"Tolleranza probe EPS        : {EPS}")

    converted = 0
    skipped_output_already_exists = 0
    failed = 0

    report_lines.extend(["", "CONVERSIONI"])

    for number, key in enumerate(missing_keys, start=1):
        source_simulation_dir = archive_by_key[key]
        destination_simulation_dir = OUTPUT_ROOT / key
        output_npy = destination_simulation_dir / INITIAL_NPY_NAME

        try:
            if output_npy.exists() and not OVERWRITE_EXISTING_INITIAL_NPY:
                skipped_output_already_exists += 1

                message = (
                    f"SKIP | {key} | output gia' esistente: {output_npy}"
                )
                print(f"[{number:04d}/{len(missing_keys):04d}] {message}")
                report_lines.append(message)
                continue

            iso_name = get_iso_name(source_simulation_dir)
            lateral_length = LATERAL_LENGTH[iso_name]
            initial_vtu = select_initial_vtu(source_simulation_dir)

            shape, value_min, value_max, n_fallback = convert_initial_vtu_to_npy(
                vtu_path=initial_vtu,
                output_npy=output_npy,
                lateral_length=lateral_length,
            )

            converted += 1

            message = (
                f"OK | {key} | source={initial_vtu.name} | "
                f"output={output_npy} | Lx=Ly={lateral_length} | "
                f"shape={shape} | min={value_min:.8f} | "
                f"max={value_max:.8f} | fallback={n_fallback}"
            )
            print(f"[{number:04d}/{len(missing_keys):04d}] {message}")
            report_lines.append(message)

        except Exception as exc:
            failed += 1

            message = f"ERROR | {key} | {type(exc).__name__}: {exc}"
            print(f"[{number:04d}/{len(missing_keys):04d}] {message}")
            report_lines.append(message)

    report_lines.extend([
        "",
        "RIEPILOGO",
        f"Convertite = {converted}",
        f"Saltate: output NPY gia' esistente = {skipped_output_already_exists}",
        f"Errori = {failed}",
    ])

    REPORT_PATH.write_text(
        "\n".join(report_lines) + "\n",
        encoding="utf-8",
    )

    print("\n" + "=" * 94)
    print("RIEPILOGO")
    print("=" * 94)
    print(f"Convertite                   : {converted}")
    print(f"Saltate: output gia' presente: {skipped_output_already_exists}")
    print(f"Errori                       : {failed}")
    print(f"Report                       : {REPORT_PATH}")


if __name__ == "__main__":
    main()
