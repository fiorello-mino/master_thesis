#!/usr/bin/env python3
"""Interpolate a VTU point field on a regular 3D grid shared within each simulation folder.

Scalar output: (nx, ny, nz). Vector output: (nx, ny, nz, components).
See README_vtu_to_npy.md for examples and the output convention.
"""

import json
import math
from pathlib import Path
import re
import tempfile
from types import SimpleNamespace


def natural_key(path):
    return [int(part) if part.isdigit() else part.lower()
            for part in re.split(r'(\d+)', str(path))]


# ================= PARAMETRI DA MODIFICARE =================
BASE_DIR = Path('/scratch/fiorello/data3D')
INPUT_PATH = BASE_DIR / 'rectangle'  # File .vtu oppure directory, anche con sottocartelle.
OUTPUT_DIR = BASE_DIR / 'rectangle_npy2'
FIELD = 'phi'                      # Nome del campo nodale nel VTU.
VOXEL_SIZE = 0.025                # Passo uniforme lungo x, y, z, nelle unità della mesh.
EPS = 0.2                        # Regione z salvata: [-H - 2*EPS, 2*EPS].
BOUNDS_TOL = 1e-6                 # Tolleranza assoluta sulle lunghezze, nelle unità della mesh.
# Bounds x/y: primo VTU di ogni cartella; H: nome della cartella (es. isoRec_H1.6_P0607).
DTYPE = 'float32'                  # 'float32' oppure 'float64'.
SLAB_DEPTH = 8                    # Piani z interpolati per blocco.
LIST_FIELDS = False               # True: elenca i campi del primo VTU ed esce.
OVERWRITE = True                 # True: sostituisce gli output esistenti.
# ==========================================================


def configuration():
    if not math.isfinite(VOXEL_SIZE) or VOXEL_SIZE <= 0:
        raise ValueError('VOXEL_SIZE deve essere finito e positivo.')
    if not math.isfinite(EPS) or EPS < 0:
        raise ValueError('EPS deve essere finito e >= 0.')
    if not math.isfinite(BOUNDS_TOL) or not 0 <= BOUNDS_TOL < VOXEL_SIZE / 2:
        raise ValueError('BOUNDS_TOL deve essere finito, >= 0 e < VOXEL_SIZE / 2.')
    if not isinstance(SLAB_DEPTH, int) or isinstance(SLAB_DEPTH, bool) or SLAB_DEPTH < 1:
        raise ValueError('SLAB_DEPTH deve essere un intero >= 1.')
    if DTYPE not in ('float32', 'float64'):
        raise ValueError("DTYPE deve essere 'float32' oppure 'float64'.")
    return SimpleNamespace(input=Path(INPUT_PATH), output=Path(OUTPUT_DIR),
                           field=FIELD, voxel_size=VOXEL_SIZE, dtype=DTYPE,
                           bounds_tol=BOUNDS_TOL,
                           eps=EPS,
                           slab_depth=SLAB_DEPTH, list_fields=LIST_FIELDS,
                           overwrite=OVERWRITE)


def extract_height(folder_name):
    match = re.search(r'(?:^|_)H([+-]?(?:\d+(?:\.\d+)?|\.\d+))(?=_|$)', folder_name)
    if match is None:
        raise ValueError(f'Profondità H non trovata nel nome della cartella: {folder_name}. '
                         'Formato atteso: isoRec_H1.6_P0607.')
    height = float(match.group(1))
    if not math.isfinite(height) or height <= 0:
        raise ValueError(f'Profondità H non valida in {folder_name}: richiesto H > 0.')
    return height


def check_z_coverage(mesh_bounds, crop_bounds, tolerance, source):
    if crop_bounds[4] < mesh_bounds[4] - tolerance or crop_bounds[5] > mesh_bounds[5] + tolerance:
        raise ValueError(f'{source}: regione z richiesta [{crop_bounds[4]:.12g}, '
                         f'{crop_bounds[5]:.12g}] fuori dal dominio VTU '
                         f'[{mesh_bounds[4]:.12g}, {mesh_bounds[5]:.12g}].')


def grid_shape(bounds, voxel_size, bounds_tol=1e-6):
    """Count sample nodes, including both endpoints, at a fixed isotropic step."""
    if len(bounds) != 6 or not all(math.isfinite(v) for v in bounds):
        raise ValueError('BOUNDS deve contenere sei limiti finiti.')
    dimensions = []
    for axis, index in zip('xyz', range(3)):
        length = bounds[2 * index + 1] - bounds[2 * index]
        if length <= 0:
            raise ValueError(f'Asse {axis}: richiesto min < max.')
        intervals = length / voxel_size
        rounded = round(intervals)
        # Compare physical lengths, allowing small VTU coordinate errors.
        error = abs(length - rounded * voxel_size)
        roundoff_tol = 8 * max(math.ulp(length), math.ulp(rounded * voxel_size))
        if rounded < 1 or error > max(bounds_tol, roundoff_tol):
            raise ValueError(f'Asse {axis}: lunghezza {length:.16g} non multipla di '
                             f'VOXEL_SIZE={voxel_size:g}; scarto={error:.12g}, '
                             f'BOUNDS_TOL={bounds_tol:g}. Modificare BOUNDS o VOXEL_SIZE.')
        dimensions.append(rounded + 1)
    return tuple(dimensions)


def read_mesh(path, vtk):
    reader = vtk.vtkXMLUnstructuredGridReader()
    errors = []
    reader.AddObserver('ErrorEvent', lambda *_: errors.append(True))
    reader.SetFileName(str(path))
    reader.Update()
    mesh = reader.GetOutput()
    if errors or reader.GetErrorCode() or not mesh.GetNumberOfPoints() or not mesh.GetNumberOfCells():
        raise ValueError(f'VTU non leggibile o mesh vuota: {path}')
    return mesh


def field_names(attributes):
    return [attributes.GetArrayName(i) for i in range(attributes.GetNumberOfArrays())]


def convert(mesh, source, target, bounds, config, np, vtk, vtk_to_numpy):
    check_z_coverage(mesh.GetBounds(), bounds, config.bounds_tol, source)
    field = mesh.GetPointData().GetArray(config.field)
    if field is None:
        raise ValueError(f'{source}: campo nodale {config.field!r} assente o non numerico. '
                         f'Disponibili: {field_names(mesh.GetPointData())}')
    if field.GetNumberOfTuples() != mesh.GetNumberOfPoints():
        raise ValueError(f'{source}: numero di valori del campo diverso dal numero di nodi.')
    if field.GetDataTypeAsString() not in ('float', 'double'):
        raise ValueError(f'{source}: campo {config.field!r} non floating point; '
                         'l’interpolazione è prevista per campi FEM continui.')
    nx, ny, nz = config.shape
    components = field.GetNumberOfComponents()
    shape = (nx, ny, nz) + ((components,) if components > 1 else ())
    origin = [bounds[2 * i] for i in range(3)]
    spacing = [config.voxel_size] * 3
    targets = [target, target.with_suffix('.mask.npy'), target.with_suffix('.json')]
    if not config.overwrite and any(path.exists() for path in targets):
        raise FileExistsError(f'Output già esistente per {target}; impostare OVERWRITE = True.')
    target.parent.mkdir(parents=True, exist_ok=True)
    # Stage files so an interrupted interpolation does not leave a partial NPY.
    with tempfile.TemporaryDirectory(prefix='.vtu-convert-', dir=target.parent) as staging:
        stage = Path(staging)
        values = np.lib.format.open_memmap(stage / 'values.npy', mode='w+',
                                         dtype=config.dtype, shape=shape)
        mask = np.lib.format.open_memmap(stage / 'mask.npy', mode='w+',
                                       dtype=np.bool_, shape=(nx, ny, nz))
        probe = vtk.vtkProbeFilter()
        probe.SetSourceData(mesh)
        probe.SetValidPointMaskArrayName('__fem_valid_mask__')
        errors = []
        probe.AddObserver('ErrorEvent', lambda *_: errors.append(True))
        valid_count = 0
        nearest_tree = None
        source_values = None
        for start in range(0, nz, config.slab_depth):
            depth = min(config.slab_depth, nz - start)
            grid = vtk.vtkImageData()
            grid.SetDimensions(nx, ny, depth)
            grid.SetOrigin(origin[0], origin[1], origin[2] + start * spacing[2])
            grid.SetSpacing(*spacing)
            probe.SetInputData(grid)
            probe.Update()
            point_data = probe.GetOutput().GetPointData()
            sampled = point_data.GetArray(config.field)
            validity = point_data.GetArray('__fem_valid_mask__')
            if errors or sampled is None or validity is None:
                raise ValueError(f'Interpolazione VTK fallita: {source}')
            slab_shape = (depth, ny, nx) + ((components,) if components > 1 else ())
            slab = vtk_to_numpy(sampled).reshape(slab_shape).astype(config.dtype, copy=True)
            valid = vtk_to_numpy(validity).reshape(depth, ny, nx).astype(bool)
            if not valid.all():
                # Build one nearest-node index per VTU, only when fallback is needed.
                if nearest_tree is None:
                    from scipy.spatial import cKDTree
                    nearest_tree = cKDTree(vtk_to_numpy(mesh.GetPoints().GetData()))
                    source_values = vtk_to_numpy(field).reshape(
                        (mesh.GetNumberOfPoints(), components) if components > 1
                        else (mesh.GetNumberOfPoints(),))
                iz, iy, ix = np.nonzero(~valid)
                query_points = np.column_stack((
                    origin[0] + ix * spacing[0],
                    origin[1] + iy * spacing[1],
                    origin[2] + (start + iz) * spacing[2],
                ))
                _, nearest_indices = nearest_tree.query(query_points)
                slab[~valid] = source_values[nearest_indices]
            # VTK orders points with x varying fastest: (z, y, x).
            # Save explicitly in the requested NumPy order: (x, y, z).
            axes = (2, 1, 0, 3) if components > 1 else (2, 1, 0)
            values[:, :, start:start + depth] = slab.transpose(axes)
            mask[:, :, start:start + depth] = valid.transpose(2, 1, 0)
            valid_count += int(valid.sum())
        values.flush()
        mask.flush()
        del values, mask
        metadata = {
            'source': str(source.resolve()), 'field': config.field,
            'association': 'point', 'shape': list(shape), 'dtype': config.dtype,
            'axis_order': ['x', 'y', 'z'] + (['component'] if components > 1 else []),
            'dimensions_xyz': list(config.shape), 'bounds_xyz': list(bounds),
            'requested_bounds_xyz': list(config.requested_bounds),
            'grid_reference_source': str(config.grid_reference_source),
            'bounds_tolerance': config.bounds_tol,
            'origin_xyz': origin, 'spacing_xyz': spacing,
            'voxel_size': config.voxel_size,
            'height_from_folder': config.height,
            'eps': config.eps,
            'requested_z_crop': [-config.height - 2 * config.eps, 2 * config.eps],
            'sampling': 'nodes, including domain endpoints',
            'interpolation': 'VTK probe using source cell shape functions',
            'fallback_method': 'nearest_source_node',
            'fallback_points': nx * ny * nz - valid_count,
            'mask_semantics': 'True: FEM cell interpolation; False: nearest-node fallback',
            'valid_points': valid_count,
            'total_points': nx * ny * nz, 'mask_file': targets[1].name,
            'mesh_bounds_xyz': list(mesh.GetBounds()),
        }
        (stage / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
        for temporary, destination in zip(
                (stage / 'values.npy', stage / 'mask.npy', stage / 'metadata.json'), targets):
            temporary.replace(destination)
    return shape, valid_count, nx * ny * nz


def main():
    try:
        import numpy as np
        import scipy.spatial  # Required by the nearest-node fallback.
        from vtkmodules.vtkCommonDataModel import vtkImageData
        from vtkmodules.vtkFiltersCore import vtkProbeFilter
        from vtkmodules.vtkIOXML import vtkXMLUnstructuredGridReader
        from vtkmodules.util.numpy_support import vtk_to_numpy
        vtk = SimpleNamespace(vtkImageData=vtkImageData, vtkProbeFilter=vtkProbeFilter,
                              vtkXMLUnstructuredGridReader=vtkXMLUnstructuredGridReader)
    except ImportError as error:
        raise SystemExit('Dipendenze mancanti: eseguire python3 -m pip install numpy scipy vtk') from error
    try:
        config = configuration()
        source = config.input.resolve()
        if source.is_file() and source.suffix.lower() == '.vtu':
            files, root = [source], source.parent
        elif source.is_dir():
            files = sorted(source.rglob('*.vtu'), key=natural_key)
            root = source
        else:
            raise ValueError('Input deve essere un file .vtu o una directory esistente.')
        if not files:
            raise ValueError(f'Nessun file .vtu trovato in {source}.')
        first = read_mesh(files[0], vtk)
        if config.list_fields:
            print(f'File: {files[0]}')
            print(f'PointData: {field_names(first.GetPointData())}')
            print(f'CellData: {field_names(first.GetCellData())}')
            return
        # Check collisions for the entire batch before writing anything.
        targets = [config.output.resolve() / path.relative_to(root).with_suffix('.npy') for path in files]
        all_outputs = [path for target in targets for path in
                       (target, target.with_suffix('.mask.npy'), target.with_suffix('.json'))]
        if len(set(all_outputs)) != len(all_outputs):
            raise ValueError('I nomi dei VTU generano output in conflitto (es. a.vtu e a.mask.vtu).')
        if not config.overwrite:
            for target in targets:
                for path in (target, target.with_suffix('.mask.npy'), target.with_suffix('.json')):
                    if path.exists():
                        raise FileExistsError(f'Output già esistente: {path}; impostare OVERWRITE = True.')
        folders = {}
        for path, target in zip(files, targets):
            folders.setdefault(path.parent, []).append((path, target))
        print(f'File totali: {len(files)} | cartelle: {len(folders)} | '
              f'voxel size={config.voxel_size:g}', flush=True)
        converted = 0
        for folder_index, (folder, frames) in enumerate(folders.items(), start=1):
            reference = frames[0][0]
            first_mesh = first if reference == files[0] else read_mesh(reference, vtk)
            config.grid_reference_source = reference.resolve()
            config.height = extract_height(folder.name)
            config.requested_bounds = (*first_mesh.GetBounds()[:4],
                                       -config.height - 2 * config.eps, 2 * config.eps)
            check_z_coverage(first_mesh.GetBounds(), config.requested_bounds, config.bounds_tol, reference)
            config.shape = grid_shape(config.requested_bounds, config.voxel_size, config.bounds_tol)
            # Keep each lower bound and align its upper bound with the voxel step.
            bounds = list(config.requested_bounds)
            for index in range(3):
                bounds[2 * index + 1] = bounds[2 * index] + (config.shape[index] - 1) * config.voxel_size
            print(f'\n[CARTELLA {folder_index}/{len(folders)}] {folder.relative_to(root)} | '
                  f'frame={len(frames)} | primo frame={reference.name} | '
                  f'H={config.height:g} | EPS={config.eps:g} | '
                  f'griglia xyz={config.shape} | bounds={tuple(bounds)}', flush=True)
            if tuple(bounds) != config.requested_bounds:
                print(f'Bounds originali: {config.requested_bounds}\n'
                      f'Bounds allineati al passo: {tuple(bounds)} | BOUNDS_TOL={config.bounds_tol:g}',
                      flush=True)
            for frame_index, (path, target) in enumerate(frames, start=1):
                relative = path.relative_to(root)
                progress = f'[{converted + 1}/{len(files)}][frame {frame_index}/{len(frames)}]'
                print(f'{progress} Conversione: {relative}', flush=True)
                mesh = first_mesh if frame_index == 1 else read_mesh(path, vtk)
                shape, valid_count, total_count = convert(
                    mesh, path, target, bounds, config, np, vtk, vtk_to_numpy)
                invalid_count = total_count - valid_count
                print(f'{progress} CONVERTITO: {relative} -> {target} | '
                      f'shape={shape} | validi={valid_count / total_count:.2%} | '
                  f'non campionabili={invalid_count}/{total_count} '
                      f'({invalid_count / total_count:.2%}) | nearest={invalid_count}', flush=True)
                converted += 1
            del first_mesh
            if folder_index == 1:
                del first
        print(f'Completato: {len(files)}/{len(files)} VTU convertiti. '
              f'Output: {config.output.resolve()}', flush=True)
    except (OSError, ValueError) as error:
        raise SystemExit(f'Errore: {error}') from error


if __name__ == '__main__':
    main()

