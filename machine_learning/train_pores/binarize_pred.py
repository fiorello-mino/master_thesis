from pathlib import Path
from types import SimpleNamespace
import sys
import numpy as np
from numba import njit

PROJECT_ROOT = Path("/home/fiorello/CRANE_bc")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import seq2png_treaded


GLOB_DIR = Path("/data/fiorello/pores/ext_test/ext_test_var_depth")
MODEL_DIR = "coeffE1e-3_coeffG3e-4_hl3_reload_random"
N_FOLDERS = 100
N_NPY = 201
THRESHOLD = 0.5
DELTA_PNG = 1

EPSILON = 0.024739583333333336
DX = 0.014960629921259842
DT = 0.1
STEPS_PER_SAVE = 1
STARTING_FRAME = 0
JUMP = 0


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


def bin_keep_half(arr: np.ndarray, threshold: float = THRESHOLD) -> np.ndarray:
    out = arr.copy()
    out[arr > threshold] = 1
    out[arr < threshold] = 0
    return out


def load_true_sequence_raw(npy_dir: Path, n_npy: int) -> np.ndarray:
    frames = []

    for i in range(n_npy):
        file_path = npy_dir / f"surf_{i / 10:.1f}.npy"
        arr = np.load(file_path)
        arr = np.squeeze(arr)

        if arr.ndim != 2:
            raise ValueError(
                f"Atteso array 2D dopo squeeze in {file_path}, trovato shape {arr.shape}"
            )

        frames.append(arr)

    return np.stack(frames, axis=0)   # (T, H, W)


def load_pred_sequence_bin(npy_dir: Path, n_npy: int) -> np.ndarray:
    frames = []

    for i in range(n_npy):
        file_path = npy_dir / f"snap_{i}.npy"
        arr = np.load(file_path)
        arr = np.squeeze(arr)
        arr = bin_keep_half(arr)

        if arr.ndim != 2:
            raise ValueError(
                f"Atteso array 2D dopo squeeze in {file_path}, trovato shape {arr.shape}"
            )

        frames.append(arr)

    return np.stack(frames, axis=0)   # (T, H, W)


def compute_timestep_metrics_numpy(
    true_sequence: np.ndarray,
    pred_sequence_bin: np.ndarray,
    jump: int,
) -> tuple[np.ndarray, np.ndarray]:
    if true_sequence.shape != pred_sequence_bin.shape:
        raise ValueError(
            f"Shape diversa tra true {true_sequence.shape} e pred_bin {pred_sequence_bin.shape}"
        )

    T = true_sequence.shape[0]
    mae = np.full(T, np.nan, dtype=np.float64)
    mse = np.full(T, np.nan, dtype=np.float64)

    for t in range(jump, T):
        diff = pred_sequence_bin[t] - true_sequence[t]
        mae[t] = np.abs(diff).mean()
        mse[t] = np.square(diff).mean()

    return mae, mse


def write_evo_file_bin(
    kk_path: Path,
    true_sequence: np.ndarray,
    pred_sequence_bin: np.ndarray,
    jump: int,
    epsilon: float,
    dx: float,
    dt: float,
    steps_per_save: int,
    starting_frame: int,
) -> None:
    if true_sequence.shape != pred_sequence_bin.shape:
        raise ValueError(
            f"Shape diversa tra true {true_sequence.shape} e pred_bin {pred_sequence_bin.shape}"
        )

    T = true_sequence.shape[0]

    evo_path = kk_path / "evo_bin.txt"
    with evo_path.open("w") as file_evo:
        file_evo.write(
            "# 1: time | 2: MAE | 3: MSE | 4: avg_True | 5: avg_PredBin | "
            "6: min_True | 7: min_PredBin | 8: max_True | 9: max_PredBin | "
            "10: E_True | 11: E_PredBin | 12: mass_True | 13: mass_PredBin\n"
        )

        for t in range(jump):
            true_2d = true_sequence[t]
            e_true = total_free_energy(true_2d, epsilon, dx)
            mass_true = compute_mass(true_2d, dx)
            time = (t + starting_frame) * dt * steps_per_save

            file_evo.write(
                f"{time}\tnan\tnan\t{true_2d.mean()}\tnan\t"
                f"{true_2d.min()}\tnan\t{true_2d.max()}\tnan\t"
                f"{e_true}\tnan\t{mass_true}\tnan\n"
            )

        for t in range(jump, T):
            true_2d = true_sequence[t]
            pred_2d_bin = pred_sequence_bin[t]
            diff = pred_2d_bin - true_2d

            e_true = total_free_energy(true_2d, epsilon, dx)
            e_pred = total_free_energy(pred_2d_bin, epsilon, dx)
            mass_true = compute_mass(true_2d, dx)
            mass_pred = compute_mass(pred_2d_bin, dx)
            time = (t + starting_frame) * dt * steps_per_save

            file_evo.write(
                f"{time}\t{np.abs(diff).mean()}\t{np.square(diff).mean()}\t"
                f"{true_2d.mean()}\t{pred_2d_bin.mean()}\t"
                f"{true_2d.min()}\t{pred_2d_bin.min()}\t"
                f"{true_2d.max()}\t{pred_2d_bin.max()}\t"
                f"{e_true}\t{e_pred}\t{mass_true}\t{mass_pred}\n"
            )


def save_pred_bin_npy(
    kk_path: Path,
    pred_sequence_bin: np.ndarray
) -> None:
    pred_bin_dir = kk_path / "pred_bin_npy"
    pred_bin_dir.mkdir(exist_ok=True)

    T = pred_sequence_bin.shape[0]
    for i in range(T):
        out_path = pred_bin_dir / f"snap_{i}.npy"
        np.save(out_path, pred_sequence_bin[i])


def save_png_outputs_bin(
    kk_path: Path,
    pred_sequence_bin: np.ndarray,
    delta_png: int
) -> None:
    pred_dir = kk_path / "pred_png_bin"
    pred_dir.mkdir(exist_ok=True)

    args_pred = SimpleNamespace(
        nproc=4,
        cmap="RdBu_r",
        paths={"png": str(pred_dir)},
        vmin=0.0,
        vmax=1.0,
    )

    seq2png_treaded(
        pred_sequence_bin[None, ::delta_png, None, :, :],
        name="snap",
        args=args_pred,
        delta=1
    )


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

            true_dir = folder / "true_npy"
            pred_dir = folder / "pred_npy"

            true_seq = load_true_sequence_raw(true_dir, N_NPY)
            pred_seq_bin = load_pred_sequence_bin(pred_dir, N_NPY)

            write_evo_file_bin(
                kk_path=folder,
                true_sequence=true_seq,
                pred_sequence_bin=pred_seq_bin,
                jump=JUMP,
                epsilon=EPSILON,
                dx=DX,
                dt=DT,
                steps_per_save=STEPS_PER_SAVE,
                starting_frame=STARTING_FRAME,
            )

            mae, mse = compute_timestep_metrics_numpy(
                true_sequence=true_seq,
                pred_sequence_bin=pred_seq_bin,
                jump=JUMP,
            )

            file_err.write(
                f"{folder_idx:03d} "
                f"{np.nanmax(mae)} {np.nanmax(mse)} "
                f"{np.nanmean(mae)} {np.nanmean(mse)}\n"
            )

            save_pred_bin_npy(
                kk_path=folder,
                pred_sequence_bin=pred_seq_bin,
            )

            save_png_outputs_bin(
                kk_path=folder,
                pred_sequence_bin=pred_seq_bin,
                delta_png=DELTA_PNG,
            )


if __name__ == "__main__":
    main()