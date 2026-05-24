# postprocess_amplitude.py

import numpy as np
from matplotlib import pyplot as plt
from cahn_hilliard.utils import load_snapshots, make_ch_gif
from cahn_hilliard.free_energy import total_free_energy, w_field
from cahn_hilliard.operators import grad_2D_neumann_along_y
import cahn_hilliard.parameters as p
from pathlib import Path

def load_snapshots_array(snap_dir="/data/fiorello/run_cosine", first=0, last=100, dt=p.dt):
    snap_dir = Path(snap_dir)

    files = [snap_dir / f"{i:04d}.npy" for i in range(first, last + 1)]

    first_array = np.load(files[0])
    snapshots = np.empty((len(files),) + first_array.shape, dtype=first_array.dtype)

    for k, file in enumerate(files):
        snapshots[k] = np.load(file)

    times = dt * 1_000_000 * np.arange(first, last + 1) 

    return times, snapshots

times, snapshots = load_snapshots_array("/data/fiorello/run_cosine", first=0, last=100, dt=p.dt)
n_snap, ny, nx = snapshots.shape

x = (np.arange(nx) + 0.5) * 1.0/128
y = (np.arange(ny) + 0.5) * 1.0/128
X, Y = np.meshgrid(x, y)

level = 0.5
amps = np.empty(n_snap)
energies = np.empty(n_snap)

for i in range(n_snap):
    phi = snapshots[i]

    fig, ax = plt.subplots()
    contList = plt.contour(X, Y, phi, levels=[level])

    contours = []
    for paths in contList.allsegs:
        for line in paths:
            contours.append(line)

    plt.close(fig)

    all_points = np.vstack(contours)

    y_min = all_points[:, 1].min()
    y_max = all_points[:, 1].max()

    amps[i] = 0.5 * (y_max - y_min)
    energies[i] = total_free_energy(phi=phi, epsilon=5.0/64, dx=1.0/128)

data_out = np.column_stack((times, amps, energies))
np.savetxt(
    "/home/fiorello/master_thesis/fem/amplitude_vs_time_explicit.csv",
    data_out,
    delimiter=",",
    header="time,amplitude,energy",
    comments=""
)

# fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# theory = 1 / 50 * np.exp(-((2 * np.pi) ** 4) * p.M0 * np.array(times))

# ax1 = axes[0]
# ax2 = axes[1]

# ax1.plot(times, amps, "o-", lw=1, ms=2, color="tab:blue", label="Ampiezza numerica")
# ax1.plot(times, theory, "--", lw=1, color="orange",
#          label=r"Ampiezza teorica: $\frac{1}{100} e^{-(2\pi)^4 t}$")
# ax1.set_xlabel("Tempo")
# ax1.set_ylabel("Ampiezza geometrica")
# ax1.set_title("Ampiezza")
# ax1.grid(True, alpha=0.3)
# ax1.legend()

# ax2.plot(times, energies, "o-", lw=1, ms=2, color="tab:red")
# ax2.set_xlabel("Tempo")
# ax2.set_ylabel("Energia")
# ax2.set_title("Energia libera")
# ax2.grid(True, alpha=0.3)

# plt.tight_layout()
# plt.savefig("plot_cosine_amp_64_dt_1e-5.png", dpi=300, bbox_inches="tight")
# plt.show()

# make_ch_gif(
#     snap_dir="snapshots",
#     output_dir="gif_prova",
#     output_name="prova_per_dataset.gif"
# )