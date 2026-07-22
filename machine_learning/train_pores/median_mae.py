from pathlib import Path
import numpy as np

BASE_DIR = Path("/data/fiorello/pores/ext_test/ext_test_var_depth/coeffE1e-3_coeffG3e-4_hl3_reload_random")
ERRORS_PATH = BASE_DIR / "errors.txt"
OUTPUT_PATH = BASE_DIR / "median_mae_error.txt"


def main() -> None:
    if not ERRORS_PATH.is_file():
        raise FileNotFoundError(f"File errors.txt non trovato: {ERRORS_PATH}")

    # leggi saltando l'header che inizia con '#'
    data = np.loadtxt(ERRORS_PATH, comments="#")

    # colonne: 0=id, 1=maxMAE, 2=maxMSE, 3=overallMAE, 4=overallMSE
    max_mae = data[:, 1]

    median = np.median(max_mae)
    q25 = np.percentile(max_mae, 25)
    q75 = np.percentile(max_mae, 75)

    # una sola riga con le tre statistiche
    row = np.array([[median, q25, q75]])

    header = "# median_maxMAE q25_maxMAE q75_maxMAE"

    np.savetxt(
        OUTPUT_PATH,
        row,
        fmt="%.9f",
        header=header,
        comments="",
    )


if __name__ == "__main__":
    main()