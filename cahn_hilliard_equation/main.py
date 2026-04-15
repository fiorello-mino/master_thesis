import numpy as np
import matplotlib.pyplot as plt
import time

from initial_conditions import smooth_cosine_interface
from free_energy import w_field
from cahn_hilliard import evolve_ch_const_mob_with_snapshots

t0 = time.perf_counter()

# -----------------------------------------------------
#               Parametri simulazione
# -----------------------------------------------------
N = 64
dx = 1/64
dy = 1/64
epsilon = 5 * dx
n_steps = 10_000_000


# -----------------------------------------------------
#               Condizione iniziale
# -----------------------------------------------------
phi_initial = smooth_cosine_interface(N, dx, epsilon)

# -----------------------------------------------------
#       Evoluzione con salvataggio snapshots
# -----------------------------------------------------
phi_final = evolve_ch_const_mob_with_snapshots(
    phi_init       = phi_initial, 
    dt             = 1e-8, 
    n_steps        = n_steps,
    steps_per_save = 100_000, 
    epsilon        = epsilon,
    M              = 1.0, 
    dx             = dx,
    out_dir        = "snapshots",
)

t1 = time.perf_counter()
print(f"Tempo esecuzione: {t1 - t0:.2f} s")