import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio

INPUT_DIR = "/data/fiorello/dataset/0000"
OUTPUT_GIF = "/home/fiorello/master_thesis/cahn_hilliard_equation/animation_0000.gif"
TOTAL_DURATION = 10.0  # secondi

npy_files = sorted(glob.glob(os.path.join(INPUT_DIR, "[0-9][0-9][0-9][0-9].npy")))

if len(npy_files) != 201:
    raise ValueError(f"Attesi 201 file .npy, trovati {len(npy_files)}")

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
    image = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    image = image[:, :, :3]

    frames.append(image)
    plt.close(fig)

imageio.mimsave(OUTPUT_GIF, frames, duration=TOTAL_DURATION / len(frames))

print(f"GIF salvata in: {OUTPUT_GIF}")
print(f"Numero frame: {len(frames)}")
print(f"Durata totale: {TOTAL_DURATION:.2f} s")
print(f"Durata per frame: {TOTAL_DURATION / len(frames):.5f} s")