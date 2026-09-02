from pathlib import Path
import numpy as np
import pyvista as pv
from scipy.interpolate import griddata


# === PARAMETRI ===
TOTAL_POINTS: int   = 64*64*64
eps         : float = 0.1

Lx          : float = 0.45
Ly          : float = 0.45
Lz          : float = 6.0

offset      : float = 0.5   # da z=0.5 a z=0
height      : float = 1.0   # poro da z=0 a z=-1
total_height = offset + height + 2*eps

Nx     = round(( TOTAL_POINTS * Lx / total_height ) ** (1/3))
Nz     = round(( TOTAL_POINTS * total_height * total_height / Lx ) ** (1/3))
Nz_tot = round(Nz * Lz / total_height)


def vtu_to_npy_pore(
    vtu_path,
    out_path,
    field_name,
    nx=Nx,
    ny=Nx,
    nz=Nz,
    nz_tot=Nz_tot,
    method="linear",
):
    mesh = pv.read(vtu_path)

    pts = mesh.points[:, :3]
    vals = np.asarray(mesh.point_data[field_name]).squeeze()

    x_min, y_min, z_min = pts.min(axis=0)
    x_max, y_max, z_max = pts.max(axis=0)

    # Griglia completa
    xi = np.linspace(x_min, x_max, nx)
    yi = np.linspace(y_min, y_max, ny)
    zi_full = np.linspace(z_min, z_max, nz_tot)

    X, Y, Z_full = np.meshgrid(xi, yi, zi_full, indexing="ij")
    pts_grid = np.stack([X.ravel(), Y.ravel(), Z_full.ravel()], axis=-1)

    grid_full = griddata(pts, vals, pts_grid, method=method)
    grid_full = np.nan_to_num(grid_full, nan=0.0).astype(np.float32)
    grid_full = grid_full.reshape(nx, ny, nz_tot)

    # Regione da tenere: [-height - 2*eps, 0 + 2*eps]
    z_start = -height - 2*eps   # -1.2
    z_end   =  0.0 + 2*eps      #  0.2

    # Indici discreti su zi_full
    z_idx_start = int(np.round((z_start - z_min) / (z_max - z_min) * (nz_tot - 1)))
    z_idx_end   = int(np.round((z_end   - z_min) / (z_max - z_min) * (nz_tot - 1))) + 1

    grid_pore = grid_full[:, :, z_idx_start:z_idx_end]

    # Sanity check opzionale
    if grid_pore.shape[2] != nz:
        print(
            f"Attenzione: shape z={grid_pore.shape[2]}, nz atteso={nz}. "
            f"z_idx_start={z_idx_start}, z_idx_end={z_idx_end}, "
            f"nz_tot={nz_tot}, z_min={z_min}, z_max={z_max}, "
            f"z_start={z_start}, z_end={z_end}"
        )

    np.save(out_path, grid_pore)
    
    
if __name__ == "__main__":
    
    vtu_path = Path("/archive/roberto/poreAMDIS/iso_P09/iso2_R0.2_H1.0_P0.9/surf_0.000000.vtu")
    out_path = Path("/home/fiorello/output.npy")

    field_name = "phi"  # nome del campo nel VTU, es. "pressure", "phi", ecc.

    vtu_to_npy_pore(
        vtu_path=vtu_path,
        out_path=out_path,
        field_name=field_name,
        method="linear"
    )

    print(f"Salvato: {out_path}")
    print(f"Shape array: {np.load(out_path).shape}")