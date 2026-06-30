import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def npy_to_png_folder(npy_dir, png_dir, cmap="gray"):
    npy_dir = Path(npy_dir)
    png_dir = Path(png_dir)
    png_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(npy_dir.glob("*.npy"))

    for f in files:
        arr = np.load(f)

        if arr.ndim != 2:
            raise ValueError(f"{f.name} non è 2D, shape trovata: {arr.shape}")

        out_file = png_dir / (f.stem + ".png")
        plt.imsave(out_file, arr, cmap=cmap, vmin=0, vmax=1)
        print(f"Salvato: {out_file}")

if __name__ == "__main__":
    npy_to_png_folder(
        npy_dir="/home/fiorello/dataset_npy/000",
        png_dir="/home/fiorello/master_thesis/fem/png_folder",
        cmap="gray"
    )
