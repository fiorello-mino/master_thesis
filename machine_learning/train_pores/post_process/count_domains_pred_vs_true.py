from pathlib import Path
import numpy as np
from scipy import ndimage

PHI_THRESHOLD = 0.5


def count_blue_domains_array(phi: np.ndarray, threshold: float = PHI_THRESHOLD) -> int:
    """
    Conta le componenti connesse di phi<threshold su griglia 2D,
    escludendo le componenti che toccano il bordo superiore.
    """

    mask = phi < threshold

    structure = ndimage.generate_binary_structure(2, 1)
    labels, n_comp = ndimage.label(mask, structure=structure)

    count = 0
    for comp_id in range(1, n_comp + 1):
        comp = (labels == comp_id)

        # bordo superiore (indice 0)
        touches_top = comp[0, :].any()

        if not touches_top:
            count += 1

    return count


def main():
    base_dir = Path("/data/fiorello/pores/ext_test/ext_test_var_depth/coeffE1e-3_coeffG3e-4_hl3_reload_random")

    results = []

    for i in range(100):
        folder_name = f"{i:03d}"  # 000, 001, ..., 099
        run_dir = base_dir / folder_name

        pred_path = run_dir / "pred_bin_npy" / "snap_200.npy"
        true_path = run_dir / "true_npy" / "surf_20.0.npy"

        if not pred_path.is_file():
            raise FileNotFoundError(f"File non trovato: {pred_path}")
        if not true_path.is_file():
            raise FileNotFoundError(f"File non trovato: {true_path}")

        # Pred: snap_200.npy, flipud
        phi_pred = np.load(pred_path)
        phi_pred = np.flipud(phi_pred)

        # True: surf_20.0.npy
        phi_true = np.load(true_path)

        n_pred = count_blue_domains_array(phi_pred, PHI_THRESHOLD)
        n_true = count_blue_domains_array(phi_true, PHI_THRESHOLD)

        diff = n_pred - n_true

        results.append((i, n_pred, n_true, diff))

    out_path = base_dir / "domains_pred_vs_true.txt"
    with out_path.open("w") as f:
        for idx, n_pred, n_true, diff in results:
            f.write(f"{idx:03d} {n_pred} {n_true} {diff}\n")


if __name__ == "__main__":
    main()