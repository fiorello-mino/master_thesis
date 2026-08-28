# energy_errors_median.py

from pathlib import Path
import numpy as np
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=Path, required=True)
    parser.add_argument('--dt', type=float, required=True)
    parser.add_argument('--steps_per_save', type=int, required=True)
    parser.add_argument('--starting_frame', type=int, required=True)
    return parser.parse_args()

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
            
        
def main() -> None:
    args = parse_args()
    root: Path = args.root_dir
    dt: float = args.dt
    steps_per_save: int = args.steps_per_save
    starting_frame: int = args.starting_frame
    errors_t = []
    
    out_path = root / "median_energy_error.txt"

    for evo_path in root.glob("*/evo.txt"):
        
        data = read_evo_file(evo_path)
        
        if len(errors_t) < len(data):
            diff = len(data) - len(errors_t)
            for _ in range(diff):
                errors_t.append([])
        
        for t, (e_true, e_pred) in enumerate(data):
            num = abs(e_true - e_pred)
            den = max(abs(e_true), 1e-12)
            err_rel = num / den
            errors_t[t].append(err_rel)
            
    median_t = np.empty(len(errors_t), dtype=float)
    p25_t = np.empty(len(errors_t), dtype=float)
    p75_t = np.empty(len(errors_t), dtype=float)

    for t in range(len(errors_t)):
        median_t[t] = np.median(errors_t[t])
        p25_t[t] = np.percentile(errors_t[t], 25)
        p75_t[t] = np.percentile(errors_t[t], 75)
        
    with out_path.open("w") as f:
        f.write("# t\tmedian_relative_energy_error\tpercentile25_relative_energy_error\tpercentile75_relative_energy_error\n")
        
        for t in range(len(median_t)):
            time = (starting_frame + t) * dt * steps_per_save
            line = f"{time}\t{median_t[t]}\t{p25_t[t]}\t{p75_t[t]}\n"
            f.write(line)    
    
    
if __name__ == '__main__':
    main()