import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

INPUT_DIR = "/data/fiorello/dataset/0004"
OUTPUT_GIF = "/home/fiorello/master_thesis/cahn_hilliard_equation/gif_prova/animation_0004.gif"
TOTAL_DURATION = 10.0  # secondi

os.makedirs(os.path.dirname(OUTPUT_GIF), exist_ok=True)

npy_files = sorted(glob.glob(os.path.join(INPUT_DIR, "[0-9][0-9][0-9][0-9].npy")))

if len(npy_files) != 201:
    raise ValueError(f"Attesi 201 file .npy, trovati {len(npy_files)}")

# scala colori fissa su tutti i frame
vmin = float("inf")
vmax = float("-inf")

for file in npy_files:
    arr = np.load(file)
    vmin = min(vmin, np.min(arr))
    vmax = max(vmax, np.max(arr))

frames = []

for file in npy_files:
    arr = np.load(file)

    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(arr, cmap="RdBu_r", origin="lower", vmin=vmin, vmax=vmax)
    ax.set_title(os.path.basename(file))
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.canvas.draw()

    width, height = fig.canvas.get_width_height()
    rgba = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(height, width, 4)
    rgb = rgba[:, :, :3]

    frames.append(Image.fromarray(rgb))
    plt.close(fig)

duration_ms = int(1000 * TOTAL_DURATION / len(frames))

frames[0].save(
    OUTPUT_GIF,
    save_all=True,
    append_images=frames[1:],
    duration=duration_ms,
    loop=0
)

print(f"GIF salvata in: {OUTPUT_GIF}")
print(f"Numero frame: {len(frames)}")
print(f"Durata totale circa: {TOTAL_DURATION:.2f} s")
print(f"Durata per frame: {duration_ms} ms")
