from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# MODIFICA QUESTI PATH
# ============================================================

FOLDER = Path(
    "/data/fiorello/iso_P09/"
    "iso2_R0.2_H1.0_P0.9"
)

NPY_PATH = FOLDER / "surf_0.000000.npy"
METADATA_PATH = FOLDER / "grid_metadata.npz"

OUTPUT_PNG = "/home/fiorello/surf_0.000000_yz_xmax.png"


# ============================================================
# CARICAMENTO
# ============================================================

grid = np.load(NPY_PATH)
metadata = np.load(METADATA_PATH)

xi = metadata["xi"]
yi = metadata["yi"]
zi = metadata["zi"]


# ============================================================
# VISTA YZ A x = x_max
# ============================================================

# grid ha shape (Nx, Ny, Nz):
# asse 0 = x
# asse 1 = y
# asse 2 = z

x_index = grid.shape[0] - 1
slice_yz = grid[0, :, :]  # shape (Ny, Nz)

# Trasposizione necessaria:
# imshow vede righe = verticale, colonne = orizzontale.
# slice_yz.T ha shape (Nz, Ny):
# orizzontale = y, verticale = z.
image_yz = slice_yz.T

extent = [
    yi[0],
    yi[-1],
    zi[0],
    zi[-1],
]


# ============================================================
# SALVATAGGIO PNG
# ============================================================

fig, ax = plt.subplots(figsize=(7, 10))

im = ax.imshow(
    image_yz,
    origin="lower",
    extent=extent,
    aspect="equal",
    interpolation="nearest",
    cmap="coolwarm",
    vmin=0,
    vmax=1.0,
)

fig.colorbar(im, ax=ax, label="phi")

ax.set_xlabel("y")
ax.set_ylabel("z")
ax.set_title(f"Vista YZ a x = x_max = {xi[x_index]:.6f}")

fig.tight_layout()

fig.savefig(OUTPUT_PNG, dpi=250)

plt.close(fig)

print(f"PNG salvato: {OUTPUT_PNG}")
