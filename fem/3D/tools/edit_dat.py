#!/usr/bin/env python3
"""
Script per modificare i file .dat in /data/fiorello/pores3D/data_train/square/init/

Modifiche:
1. La riga che CONTIENE "surf->adapt->strategy:" diventa:
   surf->adapt->strategy: 3
   E sotto aggiunge:
   surf->adapt->time delta 1:                       0.5
   surf->adapt->time delta 2:                       2.0
   surf->adapt->relative energy tolerance: 1e-5
   surf->adapt->min timestep: 5e-6
   surf->adapt->max timestep: 1e-4

2. Modifica la riga che CONTIENE "surf->adapt->timestep:" in:
   surf->adapt->timestep: 1e-4

3. Modifica la riga che CONTIENE "output->directory:" in:
   output->directory: /scratch/fiorello/data_train3D/square/<sim_name>

4. Dopo "output->directory:" aggiunge:
   surf->output->write every delta: 0.005

5. Modifica la riga che CONTIENE "surf->output->write every i-th timestep:" in:
   surf->output->write every i-th timestep:         10000
"""

import os
import re
from pathlib import Path

INIT_DIR = Path("/data/fiorello/pores3D/data_train/square/init")

def modify_dat_file(dat_path, sim_name):
    """Modifica un file .dat secondo le specifiche"""
    
    with open(dat_path, 'r') as f:
        lines = f.readlines()
    
    new_lines = []
    i = 0
    
    while i < len(lines):
        line = lines[i]
        
        # 1. Cerca riga che CONTIENE "surf->adapt->strategy:" -> modifica e aggiungi 5 righe dopo
        if "surf->adapt->strategy:" in line:
            new_lines.append("surf->adapt->strategy: 3\n")
            # Aggiungi le cinque righe sotto
            new_lines.append("surf->adapt->time delta 1:                       0.5\n")
            new_lines.append("surf->adapt->time delta 2:                       2.0\n")
            new_lines.append("surf->adapt->relative energy tolerance: 1e-5\n")
            new_lines.append("surf->adapt->min timestep: 5e-6\n")
            new_lines.append("surf->adapt->max timestep: 1e-4\n")
            i += 1
            continue
        
        # 2. Modifica riga che CONTIENE "surf->adapt->timestep:"
        if "surf->adapt->timestep:" in line:
            new_lines.append("surf->adapt->timestep: 1e-4\n")
            i += 1
            continue
        
        # 3. Cerca riga che CONTIENE "output->directory:" e modifica con il percorso corretto
        if "output->directory:" in line:
            new_lines.append(f"output->directory: /scratch/fiorello/data_train3D/square/{sim_name}\n")
            # Aggiungi la riga "write every delta" (con spazio dopo i due punti)
            new_lines.append("surf->output->write every delta: 0.005\n")
            i += 1
            continue
        
        # 4. Salta la riga che CONTIENE "surf->adapt->min timestep:" (già inserita sopra)
        if "surf->adapt->min timestep:" in line:
            i += 1
            continue
        
        # 5. Modifica riga che CONTIENE "surf->output->write every i-th timestep:"
        if "surf->output->write every i-th timestep:" in line:
            new_lines.append("surf->output->write every i-th timestep:         10000\n")
            i += 1
            continue
        
        # Tutte le altre righe restano invariate
        new_lines.append(line)
        i += 1
    
    # Scrivi il file modificato
    with open(dat_path, 'w') as f:
        f.writelines(new_lines)


def main():
    if not INIT_DIR.exists():
        print(f"ERRORE: {INIT_DIR} non esiste o non è accessibile")
        return
    
    # Trova tutti i file .dat
    dat_files = sorted(list(INIT_DIR.glob("*.dat")))
    print(f"Trovati {len(dat_files)} file .dat in {INIT_DIR}\n")
    
    modified = 0
    errors = []
    
    for dat_path in dat_files:
        # Nome della simulazione (nome file senza estensione)
        sim_name = dat_path.stem
        
        try:
            modify_dat_file(dat_path, sim_name)
            print(f"Modificato: {dat_path.name}")
            modified += 1
        except Exception as e:
            errors.append(f"ERRORE su {dat_path.name}: {e}")
            print(f"ERRORE su {dat_path.name}: {e}")
    
    # Riepilogo
    print("\n" + "=" * 60)
    print("RIEPILOGO MODIFICHE")
    print("=" * 60)
    print(f"File .dat modificati: {modified}")
    
    if errors:
        print(f"\nERRORI ({len(errors)}):")
        for err in errors:
            print(f"  - {err}")
    
    # Salva log
    log_file = Path("modify_log.txt")
    with log_file.open("w") as f:
        f.write("LOG MODIFICA FILE .dat\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"File .dat modificati: {modified}\n\n")
        if errors:
            f.write(f"ERRORI ({len(errors)}):\n")
            for err in errors:
                f.write(f"  {err}\n")
    print(f"\nLog salvato in: {log_file.resolve()}")


if __name__ == "__main__":
    main()