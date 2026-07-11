from pathlib import Path
import csv

from pore_evolution_classifier import count_blue_domains

INPUT_CSV = Path("/home/fiorello/init_files/test/pores_periodic_var_depth/summary.csv")
VTU_ROOT = Path("/scratch/fiorello/data_test/pores_periodic_var_depth_vtu")
FINAL_VTU_NAME = "surf_20.0.vtu"


def find_final_vtu_from_tag(tag: str) -> Path:
    tag = str(tag).strip()
    vtu_path = VTU_ROOT / tag / FINAL_VTU_NAME
    if not vtu_path.is_file():
        raise FileNotFoundError(f"VTU finale non trovato: {vtu_path}")
    return vtu_path


def main():
    if not INPUT_CSV.is_file():
        raise FileNotFoundError(f"CSV input non trovato: {INPUT_CSV}")

    # leggi tutte le righe
    with open(INPUT_CSV, "r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        original_fieldnames = reader.fieldnames or []

    if not rows:
        raise RuntimeError("Il CSV input è vuoto.")

    # aggiungi colonne se non ci sono già
    extra_cols = ["n_bubbles", "status"]
    fieldnames = original_fieldnames[:]
    for col in extra_cols:
        if col not in fieldnames:
            fieldnames.append(col)

    # calcola n_bubbles per ciascuna riga
    for row in rows:
        tag = row["tag"]
        try:
            vtu_path = find_final_vtu_from_tag(tag)
            n_bubbles = count_blue_domains(vtu_path)
            row["n_bubbles"] = int(n_bubbles)
            row["status"] = "ok"
        except Exception as e:
            row["n_bubbles"] = ""
            row["status"] = f"error: {e}"

    # sovrascrivi lo stesso file con le nuove colonne
    with open(INPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"CSV aggiornato con n_bubbles in {INPUT_CSV}")


if __name__ == "__main__":
    main()
