# postprocess_amplitude.py

from utils import load_snapshots, make_ch_gif
import numpy as np
from matplotlib import pyplot as plt
from free_energy import total_free_energy, w_field
from operators import grad_2D_neumann_along_y
import params as p


times, snapshots = load_snapshots("snapshots")
n_snap, ny, nx = snapshots.shape


x = (np.arange(p.N) + 0.5) * p.dx
y = (np.arange(p.N) + 0.5) * p.dx
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
    energies[i] = total_free_energy(phi = phi, epsilon = p.epsilon, dx = p.dx)


fig, axes = plt.subplots(1, 2, figsize=(10, 4))


# Curva esponenziale teorica
theory = 1/100 * np.exp(-((2*np.pi)**4) * p.M0 * np.array(times))

ax1 = axes[0]
ax2 = axes[1]

ax1.plot(times, amps, "o-", lw=1, ms=2, color="tab:blue", label="Ampiezza numerica")
ax1.plot(times, theory, "--", lw=1, color="orange", label=r"Ampiezza teorica: $\frac{1}{100} e^{-(2\pi)^4 t}$")
ax1.set_xlabel("Tempo")
ax1.set_ylabel("Ampiezza geometrica")
ax1.set_title("Ampiezza")
ax1.grid(True, alpha=0.3)
ax1.legend()

ax2.plot(times, energies, "o-", lw=1, ms=2, color="tab:red")
ax2.set_xlabel("Tempo")
ax2.set_ylabel("Energia")
ax2.set_title("Energia libera")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("plot_cosine_amp_128_g_phi.png", dpi=300, bbox_inches="tight")

make_ch_gif(
    snap_dir="snapshots",
    output_dir="results",
    output_name="random_init.gif"
)   
