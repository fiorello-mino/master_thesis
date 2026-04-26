import numpy as np
import matplotlib.pyplot as plt
import time
import params as p

from initial_conditions import smooth_cosine_interface, random_profile
from cahn_hilliard import evolve_ch_const_mob_with_snapshots, evolve_ch_surf_mob_with_snapshots

# import argparse

# def parse_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--out_dir", type=str, required=True)
#     parser.add_argument("--seed", type=int, default=0)
#     return parser.parse_args()

# if __name__ == "__main__":
#     args = parse_args()

t0 = time.perf_counter()

# -----------------------------------------------------
#               Condizione iniziale
# -----------------------------------------------------
phi_initial = smooth_cosine_interface(p.N, p.dx, p.epsilon)
#phi_initial = random_profile(128)

phi_final = np.empty_like(phi_initial)

evolve_ch_surf_mob_with_snapshots(
    phi_init       = phi_initial,
    phi            = phi_final,
    dt             = p.dt,
    n_steps        = p.n_steps,
    steps_per_save = p.steps_per_save,
    epsilon        = p.epsilon,
    M0             = p.M0,
    dx             = p.dx,
    out_dir        = p.out_dir,
    live_plot      = p.live_plot,
    cmap           = "RdBu_r",
)

t1 = time.perf_counter()
print(f"Tempo esecuzione: {t1 - t0:.2f} s")