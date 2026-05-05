from pathlib import Path
import numpy as np
base_dir = Path("/data/fiorello/dataset")
output_txt = "training_set.txt"

n_folders = 800          # 0000 ... 0999
n_files_total = 201       # 0000.npy ... 0200.npy
#n_selected = 50           # vogliamo 50 file per cartella

# Uno snap ogni 4
#indices = list(range(0, 200, 4))[:50]

with open(output_txt, "w") as f:
    for folder_idx in range(n_folders):
        folder = base_dir / f"{folder_idx:04d}"
        start_index = np.random.randint(1,150)
        indices = list(range(start_index, start_index + 50))
        paths = [str(folder / f"{file_idx:04d}.npy") for file_idx in indices]
        f.write(" ".join(paths) + "\n")

print(f"File creato: {output_txt}")
