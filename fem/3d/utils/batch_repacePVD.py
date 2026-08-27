#!/usr/bin/env python3
"""
Script per applicare repacePVD.py a tutte le sottocartelle di un root folder.
"""

import subprocess
import sys
import os
from pathlib import Path

# ===============================
#           PARAMETRI
# ===============================
root_folder = Path('/home/fiorello/prova_pp')
filename_root = 'surf' 
dt = 0.005
delete_files = 'y'  # 'y' per cancellare, 'n' per mantenere
# ===============================

subfolders = sorted([d for d in root_folder.iterdir() if d.is_dir()])

print(f"Root folder: {root_folder}")
print(f"Filename root: {filename_root}")
print(f"dt: {dt}")
print(f"Delete files: {delete_files}")
print(f"Sottocartelle da processare: {len(subfolders)}")
print()

for i, folder in enumerate(subfolders):
    print(f"\n[{i+1}/{len(subfolders)}] {folder.name}")
    
    os.chdir(folder)
    
    cmd = [sys.executable, 'repacePVD.py', filename_root]
    
    try:
        proc = subprocess.run(
            cmd,
            input=f'{dt}\n{delete_files}\n',
            text=True,
            capture_output=True,
            timeout=300
        )
        
        if proc.returncode == 0:
            print("OK")
        else:
            print(f"Errore")
    
    except Exception as e:
        print(f"{e}")
    
    os.chdir(root_folder)