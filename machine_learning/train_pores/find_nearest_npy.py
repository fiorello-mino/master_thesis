from pathlib import Path
import numpy as np
import re


GLOB_DIR = Path("/data/fiorello/pores/ext_test/ext_test_var_depth")
MODEL_DIR = "coeffE1e-3_coeffG3e-4_hl3_reload_random"
N_FOLDERS = 100

NPY_IDX = 150  # -> surf_15.0.npy
DT = 0.1

output_txt = GLOB_DIR / MODEL_DIR / "nearest_npy.txt"


def time_from_name(p: Path) -> float:
    # funziona sia per surf_X.npy che per snap_X.npy
    return float(re.search(r"(surf|snap)_(.+)\.npy$", p.name).group(2))


def find_true_npy(path: Path, npy_idx: int) -> tuple[np.ndarray, Path]:
    true_dir = path / "true_npy"
    if not true_dir.is_dir():
        raise FileNotFoundError(f"La cartella true_npy non esiste: {true_dir}")

    true_npy = true_dir / f"surf_{npy_idx/10:.1f}.npy"  # es: surf_15.0.npy
    if not true_npy.is_file():
        raise FileNotFoundError(f"Il file npy non esiste: {true_npy}")

    arr = np.load(true_npy)   # <-- true NON flippato
    return arr, true_npy


def find_nearest_pred(path: Path, true_npy: np.ndarray) -> tuple[Path, np.ndarray, float]:
    pred_dir = path / "pred_bin_npy"
    if not pred_dir.is_dir():
        raise FileNotFoundError(f"La cartella pred_bin_npy non esiste: {pred_dir}")

    mae_min = float("inf")
    nearest_npy = None
    nearest_file = None

    pred_files = sorted(pred_dir.glob("snap_*.npy"), key=time_from_name)

    for file in pred_files:
        pred_arr = np.load(file)
        pred_arr = np.flipud(pred_arr)  # <-- flip SOLO i pred

        mae = np.mean(np.abs(pred_arr - true_npy))

        if mae < mae_min:
            mae_min = mae
            nearest_npy = pred_arr
            nearest_file = file

    if nearest_file is None:
        raise RuntimeError(f"Nessun file snap_*.npy trovato in {pred_dir}")

    return nearest_file, nearest_npy, mae_min


def main() -> None:
    root_dir = GLOB_DIR / MODEL_DIR
    output_txt.parent.mkdir(parents=True, exist_ok=True)

    with open(output_txt, "w") as f:
        for folder_idx in range(N_FOLDERS):
            folder = root_dir / f"{folder_idx:03d}"

            if not folder.is_dir():
                raise FileNotFoundError(f"La cartella {folder} non esiste.")

            true_arr, true_file = find_true_npy(folder, NPY_IDX)

            nearest_file, nearest_pred, mae = find_nearest_pred(folder, true_arr)

            # tempo true: NPY_IDX * DT (es 150 * 0.1 = 15.0)
            t_true = NPY_IDX * DT
            # tempo pred dal nome snap_X.npy
            t_pred = time_from_name(nearest_file)

            t_diff = t_pred/10 - t_true

            f.write(
                f"{folder_idx:03d}\t{true_file.name}\t{nearest_file.name}\t{mae}\t{t_diff}\n"
            )


if __name__ == "__main__":
    main()
