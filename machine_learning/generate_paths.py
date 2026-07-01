from pathlib import Path
import csv
import random

base_dir1 = Path("/data/fiorello/dataset_pores_npy")
base_dir2 = Path("/data/fiorello/dataset_pores_grid_npy")
train_txt = "train_set.txt"
valid_txt  = "valid_set.txt"

csv_path1 = Path("/home/fiorello/init_files/dataset_pores.csv")
csv_path2 = Path("/home/fiorello/master_thesis/fem/grid_data.csv")

N_TOTAL   = 200   # tag totali da selezionare per dataset_pores_grid
N_TRAIN   = 160   # quanti usare per train
# → i restanti 40 vanno in valid

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

# ── costruzione lista tag per dataset_pores_grid ───────────────
tags_nonzero = [int(r["tag"]) for r in rows2 if int(r["n_bubbles"]) > 0]  # 99 tag
tags_zero    = [int(r["tag"]) for r in rows2 if int(r["n_bubbles"]) == 0]

n_zero_needed = N_TOTAL - len(tags_nonzero)   # 200 - 99 = 101
extra_zeros = random.sample(tags_zero, n_zero_needed)

selected_tags = sorted(tags_nonzero + extra_zeros)  # 200 tag ordinati
# dizionario tag → n_bubbles per lookup rapido
tag_to_nbubbles = {int(r["tag"]): int(r["n_bubbles"]) for r in rows2}

train_tags2 = selected_tags[:N_TRAIN]    # primi 160
valid_tags2  = selected_tags[N_TRAIN:]   # ultimi 40


def write_sequences(f, folder_idx_iter, base_dir, depth_lookup=None, nbubbles_lookup=None):
    """Scrive le righe di path nel file f, una sequenza per folder."""
    for folder_idx in folder_idx_iter:
        folder = base_dir / f"{folder_idx:03d}"

        if depth_lookup is not None:
            key = depth_lookup[folder_idx]
            if key == "deep":
                start_idx = random.randint(0, 65)
            elif key == "shallow":
                start_idx = random.randint(0, 149)
            else:
                continue

        elif nbubbles_lookup is not None:
            n = nbubbles_lookup[folder_idx]
            if n == 0:
                start_idx = random.randint(0, 65)
            else:   # n_bubbles > 0
                start_idx = random.randint(0, 149)

        indices = list(range(start_idx, start_idx + 50))
        paths = [str(folder / f"{file_idx:03d}.npy") for file_idx in indices]
        f.write(" ".join(paths) + "\n")


# lookup per dataset_pores (indicizzato per posizione nella lista)
depth_by_idx    = {i: rows1[i]["depth"] for i in range(len(rows1))}

# ── train_set.txt ──────────────────────────────────────────────
with open(train_txt, "w") as f:
    # dataset_pores: folder 000–159
    write_sequences(f, range(160), base_dir1, depth_lookup=depth_by_idx)
    # dataset_pores_grid: primi 160 tag selezionati
    write_sequences(f, train_tags2, base_dir2, nbubbles_lookup=tag_to_nbubbles)

print(f"Creato: {train_txt}")

# ── valid_set.txt ──────────────────────────────────────────────
with open(valid_txt, "w") as f:
    # dataset_pores: folder 160–199
    write_sequences(f, range(160, 200), base_dir1, depth_lookup=depth_by_idx)
    # dataset_pores_grid: ultimi 40 tag selezionati
    write_sequences(f, valid_tags2, base_dir2, nbubbles_lookup=tag_to_nbubbles)

print(f"Creato: {valid_txt}")