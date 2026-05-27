from pathlib import Path
import numpy as np
base_dir = Path("/data/fiorello/dataset_64")
output_txt = "training_set_64_from_5.txt"

n_folders = 800

with open(output_txt, "w") as f:
    for folder_idx in range(n_folders):
        folder = base_dir / f"{folder_idx:04d}"
        start_index = 5
        indices = list(range(start_index, start_index + 50))
        paths = [str(folder / f"{file_idx:04d}.npy") for file_idx in indices]
        f.write(" ".join(paths) + "\n")

print(f"File creato: {output_txt}")
