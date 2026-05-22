# postprocess_energy.py

import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path
from cahn_hilliard.free_energy import total_free_energy, w_field
from cahn_hilliard.operators import grad_2D
import cahn_hilliard.parameters as p

snap_dir_true = Path("/data/fiorello/dataset_random_fraction_phase1/0021")
snap_dir_pred = Path("/data/fiorello/external_test_random_mean_value/le_5e-5_hl_3_train_from_10/0021/pred_sequence_npy")

# Carico npy true
files_true = []
for f in snap_dir_true.glob("*.npy"):
    if int(f.stem) >= 10:
        files_true.append(f)

files_true.sort()

snapshots_true = []
for f in files_true:
    snapshots_true.append(np.load(f, allow_pickle=False))

snapshots_true = np.array(snapshots_true)

# Carico npy pred
files_pred = []
for f in snap_dir_pred.glob("*.npy"):
    files_pred.append(f)

files_pred.sort()

snapshots_pred = []
for f in files_pred:
    snapshots_pred.append(np.load(f, allow_pickle=False))

snapshots_pred = np.array(snapshots_pred)
    
    
n_snap, ny, nx = snapshots_true.shape
times = np.empty(n_snap, dtype=float)
energies_true = np.empty(n_snap, dtype=float)
energies_pred = np.empty(n_snap, dtype=float)

for i in range(n_snap):
    
    phi_true = snapshots_true[i]
    phi_pred = snapshots_pred[i]
    times[i] = (i+10) * p.steps_per_save * p.dt
    energies_true[i] = total_free_energy(phi = phi_true, epsilon = p.epsilon, dx = p.dx)
    energies_pred[i] = total_free_energy(phi = phi_pred, epsilon = p.epsilon, dx = p.dx)
    

plt.figure(figsize=(7, 5))

plt.plot(times, energies_true, label='True energy', color='blue', linewidth=1.5)
plt.plot(times, energies_pred, label='Pred energy', color='red', linewidth=1.5, linestyle='--')

plt.xlabel('Time')
plt.ylabel('Free energy')
plt.title('True vs Pred free energy')
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.savefig('energy_true_vs_pred.png', dpi=300)
