from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


# ============================================================
# CONFIGURAZIONE: MODIFICA SOLO QUESTI PATH
# ============================================================

FOLDER = Path(
    "/data/fiorello/iso_P09/"
    "iso2_R0.2_H1.0_P0.9"
)

# Sostituisci con il frame che vuoi controllare.
NPY_PATH = FOLDER / "surf_0.000000.npy"

# Creato dallo script resizeFolder con SAVE_NAN_MASK = True.
MASK_PATH = FOLDER / "surf_0.000000_linear_nan_mask.npy"

# Creato una volta per cartella dallo script di conversione.
METADATA_PATH = FOLDER / "grid_metadata.npz"

# PNG di output.
OUTPUT_PNG = FOLDER / "surf_0.000000_yz_xmax.png"
OUTPUT_MASK_PNG = FOLDER / "surf_0.000000_yz_xmax_fallback_mask.png"

# Colormap e range del campo phi.
CMAP = "coolwarm"
VMIN = -1.0
VMAX = 1.0

# Se True, prende la faccia x=x_max; se False, usa una slice subito interna.
# Per controllare un eventuale problema esclusivamente sul bordo, prova anche False.
USE_XMAX = True


# ============================================================
# LETTURA DATI
# ============================================================

grid = np.load(NPY_PATH)
metadata = np.load(METADATA_PATH)

xi = metadata["xi"]
yi = metadata["yi"]
zi = metadata["zi"]

if grid.ndim != 3:
    raise ValueError(f"L'NPY deve essere 3D; shape trovata: {grid.shape}")

expected_shape = (len(xi), len(yi), len(zi))
if grid.shape != expected_shape:
    raise ValueError(
        "Shape dell'NPY incompatibile con grid_metadata.npz:\n"
        f"  NPY      = {grid.shape}\n"
        f"  metadata = {expected_shape}"
    )

nx, ny, nz = grid.shape

if USE_XMAX:
    x_index = nx - 1
else:
    x_index = nx - 2

x_value = float(xi[x_index])

# Array originale: shape (Nx, Ny, Nz).
# Fissando x ottieni shape (Ny, Nz).
slice_yz = grid[x_index, :, :]

# Per imshow, la prima dimensione è verticale e la seconda orizzontale.
# Trasponendo (Ny, Nz) -> (Nz, Ny), visualizzi:
#   orizzontale = y
#   verticale   = z
image_yz = slice_yz.T

extent = [
    float(yi[0]),
    float(yi[-1]),
    float(zi[0]),
    float(zi[-1]),
]


# ============================================================
# DIAGNOSTICA VALORI
# ============================================================

print("=" * 70)
print("DIAGNOSTICA NPY")
print(f"File NPY       : {NPY_PATH}")
print(f"Shape NPY      : {grid.shape}  (x, y, z)")
print(f"Slice          : YZ a x_index={x_index}, x={x_value:.8f}")
print(f"Intervallo y   : [{yi[0]:.8f}, {yi[-1]:.8f}]")
print(f"Intervallo z   : [{zi[0]:.8f}, {zi[-1]:.8f}]")
print(f"dx             : {xi[1] - xi[0]:.8f}")
print(f"dy             : {yi[1] - yi[0]:.8f}")
print(f"dz             : {zi[1] - zi[0]:.8f}")
print(f"Min/max globale: {grid.min():.8f}, {grid.max():.8f}")
print(f"Min/max YZ     : {slice_yz.min():.8f}, {slice_yz.max():.8f}")
print(f"Frazione phi=0 : {np.mean(slice_yz == 0.0):.4%}")
print(f"Frazione phi<-0.9: {np.mean(slice_yz < -0.9):.4%}")
print(f"Frazione phi> 0.9: {np.mean(slice_yz > 0.9):.4%}")


# ============================================================
# PNG DEL CAMPO phi
# ============================================================

fig, ax = plt.subplots(figsize=(7, 10))

im = ax.imshow(
    image_yz,
    origin="lower",
    extent=extent,
    aspect="equal",
    interpolation="nearest",
    cmap=CMAP,
    vmin=VMIN,
    vmax=VMAX,
)

cbar = fig.colorbar(im, ax=ax)
cbar.set_label("phi")

ax.set_xlabel("y")
ax.set_ylabel("z")
ax.set_title(f"Vista YZ a x = {x_value:.6f}")

# Riferimenti della geometria del poro:
# per H, il poro fisico è approssimativamente tra z=-H e z=0.
if "height" in metadata:
    height = float(metadata["height"])
    ax.axhline(0.0, color="black", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.axhline(-height, color="black", linestyle="--", linewidth=0.8, alpha=0.7)

fig.tight_layout()
fig.savefig(OUTPUT_PNG, dpi=250)
plt.close(fig)

print(f"PNG phi salvato: {OUTPUT_PNG}")


# ============================================================
# PNG MASCHERA DEL FALLBACK linear -> nearest
# ============================================================

if MASK_PATH.exists():
    mask = np.load(MASK_PATH)

    if mask.shape != grid.shape:
        raise ValueError(
            "Shape della mask incompatibile con l'NPY:\n"
            f"  mask = {mask.shape}\n"
            f"  NPY  = {grid.shape}"
        )

    mask_yz = mask[x_index, :, :]
    image_mask_yz = mask_yz.T

    print("=" * 70)
    print("DIAGNOSTICA FALLBACK")
    print(f"File mask      : {MASK_PATH}")
    print(f"Fallback totale: {int(mask.sum())}/{mask.size} ({mask.mean():.4%})")
    print(
        f"Fallback YZ    : {int(mask_yz.sum())}/{mask_yz.size} "
        f"({mask_yz.mean():.4%})"
    )

    # Colori: grigio = linear; giallo = fallback nearest.
    fallback_cmap = ListedColormap(["#222222", "#ffd400"])

    fig, ax = plt.subplots(figsize=(7, 10))

    im_mask = ax.imshow(
        image_mask_yz,
        origin="lower",
        extent=extent,
        aspect="equal",
        interpolation="nearest",
        cmap=fallback_cmap,
        vmin=0,
        vmax=1,
    )

    cbar = fig.colorbar(im_mask, ax=ax, ticks=[0, 1])
    cbar.ax.set_yticklabels(["linear", "fallback nearest"])

    ax.set_xlabel("y")
    ax.set_ylabel("z")
    ax.set_title(f"Maschera fallback YZ a x = {x_value:.6f}")

    if "height" in metadata:
        ax.axhline(0.0, color="white", linestyle="--", linewidth=0.8, alpha=0.8)
        ax.axhline(-height, color="white", linestyle="--", linewidth=0.8, alpha=0.8)

    fig.tight_layout()
    fig.savefig(OUTPUT_MASK_PNG, dpi=250)
    plt.close(fig)

    print(f"PNG mask salvato: {OUTPUT_MASK_PNG}")

else:
    print("=" * 70)
    print("Maschera fallback non trovata.")
    print(f"Attesa qui: {MASK_PATH}")
    print("Controlla che nel resizeFolder sia impostato:")
    print("SAVE_NAN_MASK = True")
