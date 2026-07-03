import numpy as np
import pyvista as pv
from scipy.interpolate import griddata
from pathlib import Path


def rotate_180_spatial(arr, spatial_axes=(0, 1)):
    return np.flip(arr, axis=spatial_axes)


def vtu_to_npy(vtu_path, out_path, field_name, nx=128, ny=128, method="linear"):
    mesh = pv.read(vtu_path)

    pts = mesh.points[:, :2]
    vals = np.asarray(mesh.point_data[field_name]).squeeze()

    x_min, y_min = pts.min(axis=0)
    x_max, y_max = pts.max(axis=0)
    print(x_max - x_min, y_max - y_min)

    xi = np.linspace(x_min, x_max, nx)
    yi = np.linspace(y_min, y_max, ny)
    X, Y = np.meshgrid(xi, yi)

    grid = griddata(pts, vals, (X, Y), method=method)
    grid = np.nan_to_num(grid, nan=0.0).astype(np.float32)

    grid = rotate_180_spatial(grid, spatial_axes=(0, 1))

    np.save(out_path, grid)


def convert_folder(vtu_dir, out_dir, field_name, nx=128, ny=128, method="linear"):
    vtu_dir = Path(vtu_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(vtu_dir.glob("*.vtu"))
    for f in files:
        out_file = out_dir / (f.stem + ".npy")
        vtu_to_npy(f, out_file, field_name, nx=nx, ny=ny, method=method)


def main():
    vtu_root = Path("/scratch/fiorello/mesoEvo_install_seq/dataset_pores")
    npy_root = Path("/scratch/fiorello/prova_vtu")

    for idx in range(2):
        tag = f"{idx:03d}"

        vtu_folder = vtu_root / tag
        npy_folder = npy_root / tag

        if not vtu_folder.exists():
            print(f"Cartella mancante, salto: {vtu_folder}")
            continue

        convert_folder(
            vtu_dir=vtu_folder,
            out_dir=npy_folder,
            field_name="phi",
            nx=128,
            ny=128,
            method="linear"
        )

        print(f"Cartella {vtu_folder} convertita con successo in .npy nella cartella {npy_folder}.")


if __name__ == "__main__":
    main()
