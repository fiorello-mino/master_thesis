from pathlib import Path
import numpy as np

npy_path = Path(
    "/data/fiorello/iso_P09/"
    "iso2_R0.2_H2.2_P0.9/"
    "surf_0.000000.npy"
)

grid = np.load(npy_path)

print("Shape:", grid.shape)
print("Min:", grid.min())
print("Max:", grid.max())

# Vista YZ a x=x_max.
yz = grid[-1, :, :]

print("Min YZ:", yz.min())
print("Max YZ:", yz.max())
print("Numero zeri in YZ:", np.count_nonzero(yz == 0.0))
print("Frazione zeri in YZ:", np.mean(yz == 0.0))

# Verifica per ogni livello z quanto è composto da zeri.
for k in range(yz.shape[1]):
    frac_zero = np.mean(yz[:, k] == 0.0)

    if frac_zero > 0.0:
        print(
            f"z-index={k:02d} | "
            f"frazione zero={100 * frac_zero:6.2f}%"
        )