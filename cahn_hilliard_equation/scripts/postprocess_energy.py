# postprocess_energy.py

import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path
from cahn_hilliard.free_energy import total_free_energy, w_field
from cahn_hilliard.operators import grad_2D
import cahn_hilliard.parameters as p

snap_dir = Path("0021")

files = []
for f in snap_dir.glob("*.npy"):
    files.append(f)

files.sort()

snapshots = []
for f in files:
    snapshots.append(np.load(f, allow_pickle=False))

snapshots = np.array(snapshots)

n_snap, ny, nx = snapshots.shape
times = np.empty(n_snap, dtype=float)
energies = np.empty(n_snap, dtype=float)

for i in range(n_snap):
    
    phi = snapshots[i]
    times[i] = i * p.steps_per_save * p.dt
    energies[i] = total_free_energy(phi = phi, epsilon = p.epsilon, dx = p.dx)
    

plt.figure()
plt.plot(times, energies, marker='o', linestyle='-')
plt.xlabel('Time')
plt.ylabel('Free energy')
plt.title('Free energy vs time')
plt.grid(True)
plt.tight_layout()
plt.show()
