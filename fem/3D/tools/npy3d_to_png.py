from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


npy_path = Path(
    "/data/fiorello/iso_P09/"
    "iso2_R0.2_H1.0_P0.9/"
    "surf_0.000000.npy"
)

metadata_path = Path(
    "/data/fiorello/iso_P09/"
    "iso2_R0.2_H1.0_P0.9/"
    "grid_metadata.npz"
)

output_png = Path(
    "/home/fiorello/pore_yz_xmax_correct.png"
)


grid = np.load(npy_path)
metadata = np.load(metadata_path)

xi = metadata["xi"]
yi = metadata["yi"]
zi = metadata["zi"]

if grid.shape != (len(xi), len(yi), len(zi)):
    raise ValueError(
        f"Shape incompatibile:\n"
        f"NPY: {grid.shape}\n"
        f"Metadata: {(len(xi), len(yi), len(zi))}"
    )

# x=x_max -> ultima slice lungo x.
x_index = grid.shape[0] - 1
slice_yz = grid[x_index, :, :]

print("Grid shape:", grid.shape)
print("x-index:", x_index)
print("x physical:", xi[x_index])
print("phi min/max:", slice_yz.min(), slice_yz.max())
print("zero fraction:", np.mean(slice_yz == 0.0))

# slice_yz shape = (Ny, Nz)
# Trasponendo:
# slice_yz.T shape = (Nz, Ny)
#
# imshow:
# asse orizzontale = y
# asse verticale   = z
image = slice_yz.T

extent = [
    yi[0],
    yi[-1],
    zi[0],
    zi[-1],
]

fig, ax = plt.subplots(figsize=(7, 10))

im = ax.imshow(
    image,
    origin="lower",
    extent=extent,
    aspect="equal",
    cmap="coolwarm",
    vmin=-1.0,
    vmax=1.0,
    interpolation="nearest",
)

colorbar = fig.colorbar(im, ax=ax)
colorbar.set_label("phi")

ax.set_xlabel("y")
ax.set_ylabel("z")
ax.set_title(f"Vista YZ a x = x_max = {xi[x_index]:.6f}")

fig.tight_layout()

output_png.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(output_png, dpi=200)
plt.close(fig)

print(f"Salvato: {output_png}")