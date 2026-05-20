import os
import time
import argparse
import numpy as np
import cahn_hilliard.parameters as p

from cahn_hilliard.initial_conditions import smooth_cosine_interface, random_profile
from cahn_hilliard.solver import (
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
    # phi_initial = 0.5 + 0.05 * (rng.random((p.N, p.N)) - 0.5)
    # mean_value = np.random.uniform(0.1, 0.9)
    # phi_initial = mean_value + 0.05 * (rng.random((p.N, p.N)) - 0.5)
    
    #phi_initial = np.zeros((p.N, p.N))

    #h = 32
    #y0 = (p.N - h) // 2
    #y1 = y0 + h

    #phi_initial[y0:y1, :] = 0.5 + 0.05 * (rng.random((h, p.N)) - 0.5)
    
    # mean_value = rng.uniform(0.1, 0.9)

    # mask = rng.random((p.N, p.N)) < mean_value

    # phi_low = rng.uniform(0.35, 0.45, size=(p.N, p.N))
    # phi_high = rng.uniform(0.55, 0.65, size=(p.N, p.N))

    # phi_initial = np.where(mask, phi_high, phi_low)
    
    # -----------------------------------------------------
    # Condizione iniziale: domini random lisci con frazione di fase controllata
    # -----------------------------------------------------
    mean_value = rng.uniform(0.1, 0.9)

    # campo random iniziale
    field = rng.random((p.N, p.N))

    # soglia scelta in modo da avere circa mean_value di fase alta
    threshold = np.quantile(field, 1.0 - mean_value)
    mask = field > threshold

    # valori delle due fasi, uno sotto e uno sopra 0.5
    phi_low = rng.uniform(0.35, 0.45, size=(p.N, p.N))
    phi_high = rng.uniform(0.55, 0.65, size=(p.N, p.N))

    phi_initial = np.where(mask, phi_high, phi_low)
    phi_initial = np.clip(phi_initial, 0.0, 1.0)

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
        live_plot=p.live_plot,
        cmap="RdBu_r"
    )

    t1 = time.perf_counter()
    print(f"Tempo esecuzione: {t1 - t0:.2f} s | seed={args.seed} | out_dir={args.out_dir}")


if __name__ == "__main__":
    main()
