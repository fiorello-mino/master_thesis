from pathlib import Path
import numpy as np
base_dir = Path("/data/fiorello/pores/dataset_pores_ext_test/data_test_npy")
output_txt = "/data/fiorello/pores/ext_test/test_set.txt"

n_folders = 100

with open(output_txt, "w") as f:
    for folder_idx in range(n_folders):
        folder = base_dir / f"{folder_idx:03d}"
        start_idx = 2
        # indice → tempo in step da 0.1
        times = [start_idx * 0.1 + k * 0.1 for k in range(199)]
        # formattazione 1 decimale: 0.0, 0.1, ..., 20.0
        file_names = [f"surf_{t:.1f}.npy" for t in times]

        paths = [str(folder / name) for name in file_names]
        f.write(" ".join(paths) + "\n")

print(f"File creato: {output_txt}")
