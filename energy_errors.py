# energy_errors_median.py

from pathlib import Path

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

errors_by_t = []

for evo_path in root.glob("*/evo.txt"):
    
    data = read_evo_file(evo_path)
    
    for t, (e_true, e_pred) in enumerate(data):
        errors_by_t[t].append((e_true, e_pred))
    