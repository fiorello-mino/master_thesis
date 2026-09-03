from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def save_yz_at_xmax(
    npy_path,
    out_png,
    vmin=None,
    vmax=None,
    cmap="viridis",
):
    """
    Salva la vista YZ (piano y-z) a x = x_max di un array 3D salvato come .npy.
    
    npy_path: percorso del file .npy (array shape (nx, ny, nz))
    out_png:  percorso del file PNG da salvare
    """
    grid = np.load(npy_path)
    if grid.ndim != 3:
        raise ValueError(f"L'array deve essere 3D, trovato ndim={grid.ndim}")

    nx, ny, nz = grid.shape

    # x = x_max -> ultima slice lungo x
    x_index = nx - 1
    x_index = 0
    slice_yz = grid[x_index, :, :]  # shape (ny, nz)

    if vmin is None:
        vmin = float(grid.min())
    if vmax is None:
        vmax = float(grid.max())

    plt.figure(figsize=(9, 3))
    im = plt.imshow(
        slice_yz,
        origin="lower",
        aspect="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    plt.colorbar(im, label="valore")
    plt.xlabel("y (index)")
    plt.ylabel("z (index)")
    plt.title(f"Vista YZ, x = x_max (idx={x_index})")
    plt.tight_layout()

    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=200)
    plt.close()

    print(f"Salvata vista YZ a x_max: {out_png}")


if __name__ == "__main__":
    npy_path = Path("/data/fiorello/iso_P09/iso2_R0.2_H2.1_P0.9/surf_0.000000.npy")
    out_png  = Path("/home/fiorello/pore_yz_xmax.png")

    save_yz_at_xmax(
        npy_path=npy_path,
        out_png=out_png,
        vmin=0.0,   
        vmax=1.0,   
        cmap="coolwarm",
    )
