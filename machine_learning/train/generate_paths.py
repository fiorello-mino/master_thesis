from pathlib import Path
import numpy as np
base_dir = Path("/data/fiorello/random/dataset_64_2_ext_test")
output_txt = "/data/fiorello/random/ext_test_64_2/test_set_2_from_10.txt"
#output_txt = "/home/fiorello/master_thesis/machine_learning/train_pores/train_set_64.txt"

n_folders = 1000

with open(output_txt, "w") as f:
    for folder_idx in range(100):
        folder = base_dir / f"{folder_idx:04d}"
        start_index = 10
        indices = list(range(start_index, 201))
        paths = [str(folder / f"{file_idx:04d}.npy") for file_idx in indices]
        f.write(" ".join(paths) + "\n")

print(f"File creato: {output_txt}")
