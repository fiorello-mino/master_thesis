import os
import time
import argparse
import numpy as np
import params as p

from initial_conditions import smooth_cosine_interface, random_profile
from cahn_hilliard import (
    evolve_ch_const_mob_with_snapshots,
    evolve_ch_surf_mob_with_snapshots,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Cartella in cui salvare l'output del run"
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        help="Seed per generare la condizione iniziale"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    t0 = time.perf_counter()

    # -----------------------------------------------------
    # Condizione iniziale
    # -----------------------------------------------------
    # phi_initial = smooth_cosine_interface(p.N, p.dx, p.epsilon)
    # phi_initial = random_profile(p.N)
    phi_initial = 0.5 + 0.05 * (rng.random((p.N, p.N)) - 0.5)

    np.save(os.path.join(args.out_dir, "phi_initial.npy"), phi_initial)

    phi_final = np.empty_like(phi_initial)

    evolve_ch_surf_mob_with_snapshots(
        phi_init=phi_initial,
        phi=phi_final,
        dt=p.dt,
        n_steps=p.n_steps,
        steps_per_save=p.steps_per_save,
        epsilon=p.epsilon,
        M0=p.M0,
        dx=p.dx,
        out_dir=args.out_dir,
        live_plot=False,
        cmap="RdBu_r"
    )

    t1 = time.perf_counter()
    print(f"Tempo esecuzione: {t1 - t0:.2f} s | seed={args.seed} | out_dir={args.out_dir}")


if __name__ == "__main__":
    main()