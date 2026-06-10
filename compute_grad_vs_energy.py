from pathlib import Path
import numpy as np
from numba import njit
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dt', type=float, default=1e-6)
    parser.add_argument('--steps_per_save', type=int, default=100_000)
    parser.add_argument('--starting_frame', type=int, default=10)
    return parser.parse_args()


@njit(fastmath=True)
def grad_2D(phi, dx, grad_x, grad_y, x_left, x_right, y_up, y_down):
    ny, nx = phi.shape
    inv_2dx = 1.0 / (2.0 * dx)
    for y in range(ny):
        yu = y_up[y]
        yd = y_down[y]
        for x in range(nx):
            xl = x_left[x]
            xr = x_right[x]
            grad_x[y, x] = (phi[y, xr] - phi[y, xl]) * inv_2dx
            grad_y[y, x] = (phi[yu, x] - phi[yd, x]) * inv_2dx


def build_neighbour_arrays(ny, nx):
    x_left = np.empty(nx, dtype=np.int64)
    x_right = np.empty(nx, dtype=np.int64)
    y_up = np.empty(ny, dtype=np.int64)
    y_down = np.empty(ny, dtype=np.int64)

    for x in range(nx):
        x_left[x] = (x - 1) % nx
        x_right[x] = (x + 1) % nx

    for y in range(ny):
        y_up[y] = (y + 1) % ny
        y_down[y] = (y - 1) % ny

    return x_left, x_right, y_up, y_down


def gradient_energy(phi, dx, gx, gy, x_left, x_right, y_up, y_down):
    grad_2D(phi, dx, gx, gy, x_left, x_right, y_up, y_down)
    return np.sum(gx * gx + gy * gy) * dx * dx


def read_evo_file(path: Path):
    """
    Leggo evo.txt:
    - ogni linea: ... e_true (colonna 9)
    - ritorno lista di energie totali nel tempo
    """
    data = []

    if not path.is_file():
        raise FileNotFoundError(f"File evo.txt non trovato: {path}")

    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            e_true = float(parts[9])
            data.append(e_true)

    return data


def main():
    args = parse_args()

    dataset_dir = Path("/data/fiorello/dataset_64_2")
    test_dir = Path("/data/fiorello/ext_test_64_2/lr5e-5_hl3_2_tr10")
    out_dir = Path("/data/fiorello/grad_vs_energy")
    out_dir.mkdir(parents=True, exist_ok=True)

    dx = 5.0 / 64
    ny, nx = 64, 64
    x_left, x_right, y_up, y_down = build_neighbour_arrays(ny, nx)

    # energy_t sarà una lista di liste:
    # energy_t[t] = [E_grad_run0_t, E_grad_run1_t, ...]
    energy_t = []

    # Ciclo sulle 100 sottocartelle 0000..0099
    for i in range(100):
        npy_dir = dataset_dir / f"{i:04d}"
        evo_path = test_dir / f"{i:04d}" / "evo.txt"

        # 1) Carico evo.txt (energie totali di riferimento)
        evo_energies = read_evo_file(evo_path)

        # 2) Carico tutti i .npy e calco E_grad per ogni frame
        run_energies_grad = []

        for npy_path in sorted(npy_dir.glob("*.npy")):
            phi = np.load(npy_path)
            gx = np.empty_like(phi)
            gy = np.empty_like(phi)

            e_grad = gradient_energy(phi, dx, gx, gy, x_left, x_right, y_up, y_down)
            run_energies_grad.append(e_grad)

        if not run_energies_grad:
            print(f"Nessun file .npy trovato in {npy_dir}, salto")
            continue

        # Se evo_energies e run_energies_grad hanno lunghezze diverse,
        # decidi tu come comportarti (es. tagliare al minimo).
        max_frames = min(len(evo_energies), len(run_energies_grad))

        # Per ogni tempo t, costruiamo una lista di E_grad (da tutte le run)
        # Inizializo energy_t se necessario
        if not energy_t:
            energy_t = [[] for _ in range(max_frames)]

        for t in range(max_frames):
            energy_t[t].append(run_energies_grad[t])

    if not energy_t:
        raise RuntimeError("Nessun dato caricato: controlla dataset_dir e test_dir")

    # Calcolo mediana e percentili per ogni tempo
    median_t = np.array([np.median(v) for v in energy_t], dtype=float)
    p25_t = np.array([np.percentile(v, 25) for v in energy_t], dtype=float)
    p75_t = np.array([np.percentile(v, 75) for v in energy_t], dtype=float)

    out_path = out_dir / "grad_energy_stats.txt"
    with out_path.open("w") as f:
        f.write("# t\tmedian\tp25\tp75\n")
        for t in range(len(median_t)):
            time = (args.starting_frame + t) * args.dt * args.steps_per_save
            f.write(f"{time}\t{median_t[t]}\t{p25_t[t]}\t{p75_t[t]}\n")


if __name__ == "__main__":
    main()