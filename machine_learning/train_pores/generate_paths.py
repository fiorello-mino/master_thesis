from pathlib import Path
import csv
import random

base_dir1 = Path("/data/fiorello/pores/dataset_pores_npy")
base_dir2 = Path("/data/fiorello/pores/dataset_pores_grid_npy")
train_txt = "train_set_from0.txt"
valid_txt  = "valid_set_from0.txt"

csv_path1 = Path("/home/fiorello/init_files/dataset_pores.csv")
csv_path2 = Path("/home/fiorello/master_thesis/fem/csv/grid_data.csv")

N_TOTAL   = 200
N_TRAIN   = 160

# probabilità che una sequenza parta forzatamente da frame 0
FORCE_ZERO_PROB = 0.1   # ~10% delle sequenze partiranno da t=0

# ── lettura CSV 1 ──────────────────────────────────────────────
rows1 = []
with csv_path1.open("r", newline="") as f:
    for row in csv.DictReader(f):
        rows1.append(row)

# ── lettura CSV 2 ──────────────────────────────────────────────
rows2 = []
with csv_path2.open("r", newline="") as f:
    for row in csv.DictReader(f):
        rows2.append(row)

tags_nonzero = [int(r["tag"]) for r in rows2 if int(r["n_bubbles"]) > 0]
tags_zero    = [int(r["tag"]) for r in rows2 if int(r["n_bubbles"]) == 0]

n_zero_needed = N_TOTAL - len(tags_nonzero)
extra_zeros = random.sample(tags_zero, n_zero_needed)

selected_tags = sorted(tags_nonzero + extra_zeros)
tag_to_nbubbles = {int(r["tag"]): int(r["n_bubbles"]) for r in rows2}

train_tags2 = selected_tags[:N_TRAIN]
valid_tags2  = selected_tags[N_TRAIN:]


def write_sequences(f, folder_idx_iter, base_dir, depth_lookup=None, nbubbles_lookup=None, force_zero_prob=FORCE_ZERO_PROB):
    for folder_idx in folder_idx_iter:
        folder = base_dir / f"{folder_idx:03d}"

        if depth_lookup is not None:
            key = depth_lookup[folder_idx]
            if key == "deep":
                max_start = 0
            elif key == "shallow":
                max_start = 0
            else:
                continue

        elif nbubbles_lookup is not None:
            n = nbubbles_lookup[folder_idx]
            max_start = 0 if n == 0 else 0
        else:
            max_start = 0

        # forza start_idx = 0 con una certa probabilità,
        # altrimenti campiona come prima
        if random.random() < force_zero_prob:
            start_idx = 0
        else:
            start_idx = random.randint(0, max_start)

        times = [start_idx * 0.1 + k * 0.1 for k in range(50)]
        file_names = [f"surf_{t:.1f}.npy" for t in times]

        paths = [str(folder / name) for name in file_names]
        f.write(" ".join(paths) + "\n")


depth_by_idx = {i: rows1[i]["depth"] for i in range(len(rows1))}

with open(train_txt, "w") as f:
    write_sequences(f, range(160), base_dir1, depth_lookup=depth_by_idx)
    write_sequences(f, train_tags2, base_dir2, nbubbles_lookup=tag_to_nbubbles)

print(f"Creato: {train_txt}")

with open(valid_txt, "w") as f:
    write_sequences(f, range(160, 200), base_dir1, depth_lookup=depth_by_idx)
    write_sequences(f, valid_tags2, base_dir2, nbubbles_lookup=tag_to_nbubbles)

print(f"Creato: {valid_txt}")
