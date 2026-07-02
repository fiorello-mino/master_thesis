from pathlib import Path
import numpy as np
base_dir = Path("/data/fiorello/random/dataset_64_2")
output_txt = "/home/fiorello/master_thesis/machine_learning/train_pores/train_set_64.txt"

n_folders = 1000

with open(output_txt, "w") as f:
    for folder_idx in range(800):
        folder = base_dir / f"{folder_idx:04d}"
        start_index = 10
        indices = list(range(start_index, start_index + 65))
        paths = [str(folder / f"{file_idx:04d}.npy") for file_idx in indices]
        f.write(" ".join(paths) + "\n")

print(f"File creato: {output_txt}")
