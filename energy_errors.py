# energy_errors_median.py

from pathlib import Path
import numpy as np

dt = 1e-1

def read_evo_file(path: Path):
    
    data = []
    
    if not path.is_file():
        raise FileNotFoundError(f"File evo.txt non trovato: {path}")
    
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            
            parts = line.split()
            e_true = float(parts[8])
            e_pred = float(parts[9])
            
            data.append((e_true, e_pred))
    
    return data
            
        
root = Path("/data/fiorello/ext_test_64_2/lr1e-4_b4_k7_hl2_ch16_seq20_ramp5_wd2e-5_full")
out_path = Path(root / "median_energy_error.txt")

errors_by_t = []

for evo_path in root.glob("*/evo.txt"):
    
    data = read_evo_file(evo_path)
    
    if len(errors_by_t) < len(data):
        diff = len(data) - len(errors_by_t)
        for _ in range(diff):
            errors_by_t.append([])
    
    for t, (e_true, e_pred) in enumerate(data):
        num = abs(e_true - e_pred)
        den = max(abs(e_true), 1e-12)
        err_rel = num / den
        errors_by_t[t].append(err_rel)
        
median_t = np.empty(len(errors_by_t), dtype=float)

for t in range(len(errors_by_t)):
    median_t[t] = np.median(errors_by_t[t])
    
with out_path.open("w") as f:
    f.write("# t\tmedian_relative_energy_error\n")
    
    for t in range(len(median_t)):
        time = (10 + t) * dt
        line = f"{time}\t{median_t[t]}\n"
        f.write(line)
    
        