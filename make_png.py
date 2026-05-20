from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def main():
    npy_dir = Path("0000")
    png_dir = Path("0000_png")
    png_dir.mkdir(exist_ok=True)

    for t in range(201):
        npy_path = npy_dir / f"{t:04d}.npy"
        png_path = png_dir / f"{t:04d}.png"

        if not npy_path.exists():
            raise FileNotFoundError(f"File mancante: {npy_path}")

        arr = np.load(npy_path)

        if arr.ndim != 2:
            raise ValueError(f"Mi aspettavo un array 2D in {npy_path}, trovato {arr.shape}")

        plt.figure(figsize=(5, 5))
        plt.imshow(arr, cmap="viridis", vmin=0.0, vmax=1.0)
        plt.axis("off")
        plt.tight_layout(pad=0)
        plt.savefig(png_path, bbox_inches="tight", pad_inches=0)
        plt.close()


if __name__ == "__main__":
    main()