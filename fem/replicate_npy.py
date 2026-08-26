import numpy as np
import os
from pathlib import Path

# Definizione delle cartelle sorgente e destinazione
folders_map = {
    '/scratch/fiorello/data_test/data_test_npy': range(0, 10),      
    '/scratch/fiorello/data_test/pores_periodic_npy': range(10, 20),
    '/scratch/fiorello/data_test/pores_periodic_var_depth_npy': range(20, 30) 
}

output_base = Path('/scratch/fiorello/data_test/pores_replicated_npy')
output_base.mkdir(exist_ok=True)

# Processa ogni cartella sorgente
for source_folder, indices in folders_map.items():
    for idx in indices:
        # Nome della cartella formattato (es. 000, 001, ..., 029)
        folder_name = f'{idx:03d}'
        
        # Percorso sorgente e destinazione
        src_path = Path(source_folder) / folder_name / 'surf_0.0.npy'
        dst_folder = output_base / folder_name
        dst_path = dst_folder / 'surf_0.0.npy'
        
        # Crea la cartella di destinazione
        dst_folder.mkdir(exist_ok=True)
        
        # Carica, replica e salva
        surf = np.load(src_path)
        surf_tiled = np.tile(surf, (1, 10))  # 10 repliche lungo x (asse 0)
        np.save(dst_path, surf_tiled)
        
        print(f"Processato: {src_path} -> {dst_path} (forma: {surf.shape} -> {surf_tiled.shape})")

print("\nFatto! Tutte le cartelle 000-029 create")
