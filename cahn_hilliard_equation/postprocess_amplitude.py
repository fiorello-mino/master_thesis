# postprocess_amplitude.py

from utils import load_snapshots
import numpy as np
from matplotlib import pyplot as plt
from free_energy import total_free_energy, w_field
from operators import grad_2D_neumann_along_y


times, snapshots = load_snapshots("snapshots")
n_snap, ny, nx = snapshots.shape


dx = 1/64
x = (np.arange(nx) + 0.5) * dx
y = (np.arange(ny) + 0.5) * dx
X, Y = np.meshgrid(x, y)

level = 0.5
amps = np.empty(n_snap)
energies = np.empty(n_snap)

for i in range(n_snap):
    
    phi = snapshots[i]
    
    fig, ax = plt.subplots()
    contList = plt.contour(X, Y, phi, levels=[level])

    # Extract the contour paths
    contours = []
    for paths in contList.allsegs:
        for line in paths:
            contours.append(line)
    
    plt.close(fig)
    
    all_points = np.vstack(contours)

    y_min = all_points[:, 1].min()
    y_max = all_points[:, 1].max()

    amps[i] = 0.5 * (y_max - y_min)
    energies[i] = total_free_energy(phi = phi, epsilon = 5 * 1/64, dx = dx)


plt.figure(figsize=(6, 4))

plt.plot(times, amps, "o-", lw=1, ms=2, color="tab:blue")
plt.xlabel("Tempo")
plt.ylabel("Ampiezza geometrica")
plt.title("Decadimento dell'ampiezza (livello ϕ = 0.5)")

plt.grid(True, which="both", alpha=0.3)

# opzionale: semilog se ti aspetti decadimento esponenziale
# plt.semilogy(times, amps, "o-", lw=1, ms=2, color="tab:blue")

plt.tight_layout()
plt.show()