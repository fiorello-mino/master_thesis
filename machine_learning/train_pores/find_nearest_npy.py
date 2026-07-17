from pathlib import Path
import numpy as np
import re


GLOB_DIR = Path("/data/fiorello/pores/ext_test/ext_test_var_depth")
MODEL_DIR = "coeffE1e-3_hl3_reload_random"
N_FOLDERS = 100

NPY_IDX = 190
DT = 0.1

output_txt = GLOB_DIR / MODEL_DIR / "nearest_npy.txt"


def find_pred_npy(path: Path, npy_idx: int) -> np.ndarray:
    pred_dir = path / "pred_npy"
    if not pred_dir.is_dir():
        raise FileNotFoundError(f"La cartella pred_npy non esiste: {pred_dir}")

    pred_npy = pred_dir / f"snap_{npy_idx}.npy"
    if not pred_npy.is_file():
        raise FileNotFoundError(f"Il file npy non esiste: {pred_npy}")

    return np.load(pred_npy)


def time_from_name(p: Path) -> float:
    return float(re.search(r"surf_(.+)\.npy$", p.name).group(1))


def find_nearest_npy(path: Path, pred_npy: np.ndarray, pred_npy_idx: int, dt: float):
    true_dir = path / "true_npy"
    if not true_dir.is_dir():
        raise FileNotFoundError(f"La cartella true_npy non esiste: {true_dir}")

    mae_min = float("inf")
    nearest_npy = None
    nearest_idx = -1
    nearest_file = None

    true_files = sorted(true_dir.glob("surf_*.npy"), key=time_from_name)

    for idx, file in enumerate(true_files):
        true_npy = np.load(file)
        mae = np.mean(np.abs(pred_npy - true_npy))

        if mae < mae_min:
            mae_min = mae
            nearest_npy = true_npy
            nearest_idx = idx
            nearest_file = file

    t_diff = (pred_npy_idx * dt) - time_from_name(nearest_file)
    return nearest_file, nearest_npy, mae_min, t_diff


def main() -> None:
    root_dir = GLOB_DIR / MODEL_DIR
    output_txt.parent.mkdir(parents=True, exist_ok=True)

    with open(output_txt, "w") as f:
        for folder_idx in range(N_FOLDERS):
            folder = root_dir / f"{folder_idx:03d}"

            if not folder.is_dir():
                raise FileNotFoundError(f"La cartella {folder} non esiste.")

            pred_npy = find_pred_npy(folder, NPY_IDX)
            nearest_file, nearest_npy, mae, t_diff = find_nearest_npy(folder, pred_npy, NPY_IDX, DT)

            f.write(
                f"{folder_idx:03d}\t{nearest_file}\t{mae}\t{t_diff}\n"
            )


if __name__ == "__main__":
    main()
