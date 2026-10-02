# medians.py

from pathlib import Path
import numpy as np

TEST_DIR = Path("/scratch/fiorello/test3D/square/train1/fv_clip_mse")
DT = 0.005
STEPS_PER_SAVE = 1
STARTING_FRAME = 1

def read_evo_file(path: Path):
    """
    Legge un file evo.txt e restituisce una lista di tuple:
      (e_true, e_pred, m_true, m_pred)
    Assunzione:
      - riga 0: intestazione (da saltare)
      - riga 1: da saltare
      - righe >= 2: dati
      - colonna 11 (indice 10) -> e_true
      - colonna 12 (indice 11) -> e_pred
      - colonna 13 (indice 12) -> m_true
      - colonna 14 (indice 13) -> m_pred
    """
    data = []
    
    if not path.is_file():
        raise FileNotFoundError(f"File evo.txt non trovato: {path}")
    
    with path.open() as f:
        lines = f.readlines()
    
    # parto dalla riga 2 (indice 2)
    for line in lines[2:]:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        
        parts = line.split()
        if len(parts) < 14:
            raise ValueError(f"Linea con meno di 14 colonne in {path}: {line}")
        
        e_true = float(parts[10])  # colonna 11
        e_pred = float(parts[11])  # colonna 12
        m_true = float(parts[12])  # colonna 13
        m_pred = float(parts[13])  # colonna 14
        
        data.append((e_true, e_pred, m_true, m_pred))
    
    return data
            
        
def main() -> None:
    root = TEST_DIR
    dt = DT
    steps_per_save = STEPS_PER_SAVE
    starting_frame = STARTING_FRAME
    
    errors_e_t = []  # errori relativi energia per ogni timestep
    errors_m_t = []  # errori relativi massa per ogni timestep
    
    out_path = root / "medians.txt"

    for evo_path in root.glob("*/evo.txt"):
        data = read_evo_file(evo_path)
        
        # allungo le liste se necessario
        if len(errors_e_t) < len(data):
            diff = len(data) - len(errors_e_t)
            for _ in range(diff):
                errors_e_t.append([])
                errors_m_t.append([])
        
        # calcolo errori relativi per ogni timestep
        for t, (e_true, e_pred, m_true, m_pred) in enumerate(data):
            # energia
            num_e = abs(e_true - e_pred)
            den_e = max(abs(e_true), 1e-12)
            err_rel_e = num_e / den_e
            errors_e_t[t].append(err_rel_e)
            
            # massa
            num_m = abs(m_true - m_pred)
            den_m = max(abs(m_true), 1e-12)
            err_rel_m = num_m / den_m
            errors_m_t[t].append(err_rel_m)
            
    # --- energia: mediana e percentili ---
    median_e = np.empty(len(errors_e_t), dtype=float)
    p25_e = np.empty(len(errors_e_t), dtype=float)
    p75_e = np.empty(len(errors_e_t), dtype=float)

    for t in range(len(errors_e_t)):
        median_e[t] = np.median(errors_e_t[t])
        p25_e[t] = np.percentile(errors_e_t[t], 25)
        p75_e[t] = np.percentile(errors_e_t[t], 75)
    
    # --- massa: mediana e percentili ---
    median_m = np.empty(len(errors_m_t), dtype=float)
    p25_m = np.empty(len(errors_m_t), dtype=float)
    p75_m = np.empty(len(errors_m_t), dtype=float)

    for t in range(len(errors_m_t)):
        median_m[t] = np.median(errors_m_t[t])
        p25_m[t] = np.percentile(errors_m_t[t], 25)
        p75_m[t] = np.percentile(errors_m_t[t], 75)
    
    # --- scrivo unico file ---
    with out_path.open("w") as f:
        f.write(
            "# t\t"
            "median_energy_error\t"
            "p25_energy_error\t"
            "p75_energy_error\t"
            "median_mass_error\t"
            "p25_mass_error\t"
            "p75_mass_error\n"
        )
        
        for t in range(len(median_e)):
            time = (starting_frame + t) * dt * steps_per_save
            line = (
                f"{time}\t"
                f"{median_e[t]}\t{p25_e[t]}\t{p75_e[t]}\t"
                f"{median_m[t]}\t{p25_m[t]}\t{p75_m[t]}\n"
            )
            f.write(line)
    
    
if __name__ == '__main__':
    main()
