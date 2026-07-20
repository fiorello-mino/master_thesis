from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt
from numba import njit

PROJECT_ROOT = Path("/home/fiorello/CRANE_bc")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


GLOB_DIR = Path("/data/fiorello/pores/ext_test/ext_test_var_depth")
MODEL_DIR = "coeffE1e-3_coeffG3e-4_hl3_reload_random"
N_FOLDERS = 100
N_NPY = 201
DELTA_PNG = 1

EPSILON = 0.024739583333333336
DX = 0.014960629921259842
DT = 0.1
STEPS_PER_SAVE = 1
STARTING_FRAME = 0
JUMP = 0

THRESHOLD_LOW = 0.1
THRESHOLD_HIGH = 0.9


@njit(fastmath=True)
def grad_2D_neumann_along_y(
    phi: np.ndarray,
    dx: float,
    grad_x: np.ndarray,
    grad_y: np.ndarray
):
    ny, nx = phi.shape
    dx2_inv = 1.0 / (2.0 * dx)

    for y in range(ny):
        for x in range(nx):
            xl = (x - 1) % nx
            xr = (x + 1) % nx

            grad_x[y, x] = (phi[y, xr] - phi[y, xl]) * dx2_inv

            if y == 0 or y == ny - 1:
                grad_y[y, x] = 0.0
            else:
                grad_y[y, x] = (phi[y + 1, x] - phi[y - 1, x]) * dx2_inv


@njit(fastmath=True)
def w_field(phi: np.ndarray, epsilon: float, w: np.ndarray):
    ny, nx = phi.shape
    factor = 18.0 / epsilon

    for y in range(ny):
        for x in range(nx):
            phi_ij = phi[y, x]
            w[y, x] = factor * phi_ij * phi_ij * (1.0 - phi_ij) * (1.0 - phi_ij)


@njit(fastmath=True)
def total_free_energy(phi: np.ndarray, epsilon: float, dx: float) -> float:
    ny, nx = phi.shape
    eps2 = 0.5 * epsilon
    dx2 = dx * dx

    w_local = np.empty_like(phi)
    gx = np.empty_like(phi)
    gy = np.empty_like(phi)

    w_field(phi, epsilon, w_local)
    grad_2D_neumann_along_y(phi, dx, gx, gy)

    total_E = 0.0
    for y in range(ny):
        for x in range(nx):
            grad2 = gx[y, x] * gx[y, x] + gy[y, x] * gy[y, x]
            f_ij = w_local[y, x] + eps2 * grad2
            total_E += f_ij

    return total_E * dx2


@njit(fastmath=True)
def compute_mass(phi: np.ndarray, dx: float) -> float:
    ny, nx = phi.shape
    mass = 0.0

    for y in range(ny):
        for x in range(nx):
            mass += phi[y, x]

    return mass * dx**2


def rotate_180(arr: np.ndarray) -> np.ndarray:
    return np.rot90(arr, 2)


def bin_with_band(
    arr: np.ndarray,
    threshold_low: float = THRESHOLD_LOW,
    threshold_high: float = THRESHOLD_HIGH
) -> np.ndarray:
    out = arr.copy()
    out[arr > threshold_high] = 1
    out[arr < threshold_low] = 0
    return out


def load_true_frame(true_dir: Path, i: int) -> np.ndarray:
    file_path = true_dir / f"surf_{i / 10:.1f}.npy"
    arr = np.load(file_path)
    arr = np.squeeze(arr)

    if arr.ndim != 2:
        raise ValueError(
            f"Atteso array 2D dopo squeeze in {file_path}, trovato shape {arr.shape}"
        )

    arr = rotate_180(arr)
    return arr


def load_pred_bin_frame(pred_dir: Path, i: int) -> np.ndarray:
    file_path = pred_dir / f"snap_{i}.npy"
    arr = np.load(file_path)
    arr = np.squeeze(arr)

    if arr.ndim != 2:
        raise ValueError(
            f"Atteso array 2D dopo squeeze in {file_path}, trovato shape {arr.shape}"
        )

    arr = rotate_180(arr)
    arr = bin_with_band(arr)
    return arr


def save_frame_png(
    frame: np.ndarray,
    out_path: Path,
    cmap: str,
    vmin: float,
    vmax: float
) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(frame, cmap=cmap, vmin=vmin, vmax=vmax, origin="lower")
    ax.axis("off")
    fig.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def process_folder(
    folder: Path,
    jump: int,
    epsilon: float,
    dx: float,
    dt: float,
    steps_per_save: int,
    starting_frame: int,
    delta_png: int,
) -> tuple[float, float, float, float]:
    true_dir = folder / "true_npy"
    pred_dir = folder / "pred_npy"

    pred_bin_dir = folder / "pred_bin_npy"
    pred_png_dir = folder / "pred_bin_png"
    diff_png_dir = folder / "diff_bin_png"

    pred_bin_dir.mkdir(exist_ok=True)
    pred_png_dir.mkdir(exist_ok=True)
    diff_png_dir.mkdir(exist_ok=True)

    evo_path = folder / "evo_bin.txt"

    max_mae = np.nan
    max_mse = np.nan
    sum_mae = 0.0
    sum_mse = 0.0
    count = 0

    with evo_path.open("w") as file_evo:
        file_evo.write(
            "# 1: time | 2: MAE | 3: MSE | 4: avg_True | 5: avg_PredBin | "
            "6: min_True | 7: min_PredBin | 8: max_True | 9: max_PredBin | "
            "10: E_True | 11: E_PredBin | 12: mass_True | 13: mass_PredBin\n"
        )

        for t in range(N_NPY):
            true_2d = load_true_frame(true_dir, t)
            pred_2d_bin = load_pred_bin_frame(pred_dir, t)

            np.save(pred_bin_dir / f"snap_{t}.npy", pred_2d_bin)

            if t % delta_png == 0:
                save_frame_png(
                    frame=pred_2d_bin,
                    out_path=pred_png_dir / f"snap_{t}.png",
                    cmap="RdBu_r",
                    vmin=0.0,
                    vmax=1.0,
                )

                diff_frame = pred_2d_bin - true_2d
                save_frame_png(
                    frame=diff_frame,
                    out_path=diff_png_dir / f"snap_{t}.png",
                    cmap="bwr",
                    vmin=-1.0,
                    vmax=1.0,
                )

            e_true = total_free_energy(true_2d, epsilon, dx)
            mass_true = compute_mass(true_2d, dx)
            time = (t + starting_frame) * dt * steps_per_save

            if t < jump:
                file_evo.write(
                    f"{time}\tnan\tnan\t{true_2d.mean()}\tnan\t"
                    f"{true_2d.min()}\tnan\t{true_2d.max()}\tnan\t"
                    f"{e_true}\tnan\t{mass_true}\tnan\n"
                )
            else:
                diff = pred_2d_bin - true_2d
                mae_t = float(np.abs(diff).mean())
                mse_t = float(np.square(diff).mean())

                e_pred = total_free_energy(pred_2d_bin, epsilon, dx)
                mass_pred = compute_mass(pred_2d_bin, dx)

                file_evo.write(
                    f"{time}\t{mae_t}\t{mse_t}\t"
                    f"{true_2d.mean()}\t{pred_2d_bin.mean()}\t"
                    f"{true_2d.min()}\t{pred_2d_bin.min()}\t"
                    f"{true_2d.max()}\t{pred_2d_bin.max()}\t"
                    f"{e_true}\t{e_pred}\t{mass_true}\t{mass_pred}\n"
                )

                if np.isnan(max_mae) or mae_t > max_mae:
                    max_mae = mae_t
                if np.isnan(max_mse) or mse_t > max_mse:
                    max_mse = mse_t

                sum_mae += mae_t
                sum_mse += mse_t
                count += 1

    overall_mae = np.nan if count == 0 else sum_mae / count
    overall_mse = np.nan if count == 0 else sum_mse / count

    return max_mae, max_mse, overall_mae, overall_mse


def main() -> None:
    root_dir = GLOB_DIR / MODEL_DIR
    errors_path = root_dir / "errors_bin.txt"

    with errors_path.open("w") as file_err:
        file_err.write(
            "# 1: id | 2: maxMAE | 3: maxMSE | 4: overallMAE | 5: overallMSE\n"
        )

        for folder_idx in range(N_FOLDERS):
            folder = root_dir / f"{folder_idx:03d}"

            if not folder.is_dir():
                raise FileNotFoundError(f"La cartella {folder} non esiste.")

            max_mae, max_mse, overall_mae, overall_mse = process_folder(
                folder=folder,
                jump=JUMP,
                epsilon=EPSILON,
                dx=DX,
                dt=DT,
                steps_per_save=STEPS_PER_SAVE,
                starting_frame=STARTING_FRAME,
                delta_png=DELTA_PNG,
            )

            file_err.write(
                f"{folder_idx:03d} "
                f"{max_mae} {max_mse} {overall_mae} {overall_mse}\n"
            )


if __name__ == "__main__":
    main()