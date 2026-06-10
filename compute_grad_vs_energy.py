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

def main():
    args = parse_args()

    test_dir = Path("/data/fiorello/ext_test_64_2/lr5e-5_hl3_2_tr10")
    out_dir = Path("/data/fiorello/grad_vs_energy")
    out_dir.mkdir(parents=True, exist_ok=True)

    dx = 5.0 / 64
    ny, nx = 64, 64
    x_left, x_right, y_up, y_down = build_neighbour_arrays(ny, nx)

    energy_t = []

    for run_dir in sorted(test_dir.glob("*/")):
        run_energies = []

        for npy_path in sorted(run_dir.glob("*.npy")):
            phi = np.load(npy_path)
            gx = np.empty_like(phi)
            gy = np.empty_like(phi)

            e_grad = gradient_energy(phi, dx, gx, gy, x_left, x_right, y_up, y_down)
            run_energies.append(e_grad)

        if run_energies:
            energy_t.append(run_energies)

    max_len = max(len(run) for run in energy_t)

    energy_by_time = [[] for _ in range(max_len)]
    for run in energy_t:
        for t, e in enumerate(run):
            energy_by_time[t].append(e)

    median_t = np.array([np.median(v) for v in energy_by_time], dtype=float)
    p25_t = np.array([np.percentile(v, 25) for v in energy_by_time], dtype=float)
    p75_t = np.array([np.percentile(v, 75) for v in energy_by_time], dtype=float)

    out_path = out_dir / "grad_energy_stats.txt"
    with out_path.open("w") as f:
        f.write("# t\tmedian\tp25\tp75\n")
        for t in range(len(median_t)):
            time = (args.starting_frame + t) * args.dt * args.steps_per_save
            f.write(f"{time}\t{median_t[t]}\t{p25_t[t]}\t{p75_t[t]}\n")
            
if __name__ == "__main__":
    main()