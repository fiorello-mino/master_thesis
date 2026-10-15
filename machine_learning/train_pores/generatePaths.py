from pathlib import Path
import csv
import random

N_CASES = 125
N_TRAIN = 100
SEQ_LEN = 50
FORCE_ZERO_PROB = 0.1

rng = random.Random(42)

train_txt = Path("train_set.txt")
valid_txt = Path("valid_set.txt")

datasets = [
    {
        "name": "Folder 1",
        "base_dir": Path("/data/fiorello/dataset2D/dataset_pores_npy"),
        "csv_path": Path("dataset_pores.csv"),
        "include_all": False,
        "bubble_column": "depth",
    },
    {
        "name": "Folder 2",
        "base_dir": Path("/data/fiorello/pores/dataset_pores_grid_npy"),
        "csv_path": Path("grid_data.csv"),
        "include_all": False,
        "bubble_column": "n_bubbles",
    },
    {
        "name": "Folder 3",
        "base_dir": Path("/data/fiorello/dataset2D/pores_periodic0.5_npy"),
        "csv_path": Path("pores_periodic0.5.csv"),
        "include_all": True,
        "bubble_column": "n_bubbles",
    },
    {
        "name": "Folder 4",
        "base_dir": Path(
            "/data/fiorello/dataset2D/pores_periodic0.5_var_depth_npy"
        ),
        "csv_path": Path("pores_periodic0.5_var_depth.csv"),
        "include_all": True,
        "bubble_column": "n_bubbles",
    },
]


def read_cases(csv_path, bubble_column):
    with csv_path.open("r", newline="") as f:
        rows = list(csv.DictReader(f))

    # Associa ogni tag alla presenza o assenza di bolle.
    cases = {}

    for idx, row in enumerate(rows):
        # Se manca "tag", usa l'indice della riga a partire da zero.
        tag = int(row["tag"]) if row.get("tag", "").strip() else idx

        if tag in cases:
            raise ValueError(f"{csv_path}: tag duplicato {tag}")

        if bubble_column == "depth":
            depth = row["depth"].strip().lower()

            if depth not in {"deep", "shallow"}:
                raise ValueError(
                    f"{csv_path}, tag {tag}: depth non valido: {depth!r}"
                )

            has_bubbles = depth == "deep"

        else:
            n_bubbles = int(row["n_bubbles"])

            if n_bubbles < 0:
                raise ValueError(
                    f"{csv_path}, tag {tag}: n_bubbles negativo."
                )

            has_bubbles = n_bubbles > 0

        cases[tag] = has_bubbles

    return cases


def select_cases(cases, include_all, name, bubble_column):
    nonzero = [tag for tag, has_bubbles in cases.items() if has_bubbles]
    zero = [tag for tag, has_bubbles in cases.items() if not has_bubbles]

    criterion = "deep" if bubble_column == "depth" else "n_bubbles > 0"

    print(f"\n{name}")
    print(
        f"  CSV completo: {len(cases)} casi, "
        f"{len(nonzero)} con {criterion}"
    )

    if include_all:
        if len(cases) != N_CASES:
            raise ValueError(
                f"{name}: attesi {N_CASES} casi, trovati {len(cases)}"
            )

        selected = list(cases)

    else:
        n_zero_needed = N_CASES - len(nonzero)

        if n_zero_needed < 0:
            raise ValueError(
                f"{name}: ci sono {len(nonzero)} casi con {criterion}. "
                f"Non possono essere inclusi tutti in {N_CASES} casi."
            )

        if n_zero_needed > len(zero):
            raise ValueError(
                f"{name}: non ci sono abbastanza casi senza bolle "
                f"per arrivare a {N_CASES}."
            )

        selected = nonzero + rng.sample(zero, n_zero_needed)

    print(
        f"  Selezionati: {len(selected)} casi, "
        f"{sum(cases[tag] for tag in selected)} con {criterion}"
    )

    rng.shuffle(selected)

    train_tags = selected[:N_TRAIN]
    valid_tags = selected[N_TRAIN:]

    print(
        f"  Train: {len(train_tags)} casi, "
        f"{sum(cases[tag] for tag in train_tags)} con {criterion}"
    )
    print(
        f"  Valid: {len(valid_tags)} casi, "
        f"{sum(cases[tag] for tag in valid_tags)} con {criterion}"
    )

    return train_tags, valid_tags


def sequence_paths(folder, start_idx):
    return [
        folder / f"surf_{(start_idx + k) / 10:.1f}.npy"
        for k in range(SEQ_LEN)
    ]


def available_starts(folder):
    frame_indices = set()

    for path in folder.glob("surf_*.npy"):
        try:
            time = float(path.stem.removeprefix("surf_"))
        except ValueError:
            continue

        idx = round(time * 10)

        if idx >= 0 and path.name == f"surf_{idx / 10:.1f}.npy":
            frame_indices.add(idx)

    return sorted(
        start
        for start in frame_indices
        if all(start + k in frame_indices for k in range(SEQ_LEN))
    )


def make_sequences(tags, base_dir):
    lines = []
    n_start_zero = 0

    for tag in tags:
        folder = base_dir / f"{tag:03d}"
        starts = available_starts(folder)

        if 0 not in starts:
            raise ValueError(
                f"{folder}: manca la sequenza completa "
                f"da 0.0 a {(SEQ_LEN - 1) / 10:.1f}."
            )

        nonzero_starts = [start for start in starts if start > 0]

        if not nonzero_starts:
            raise ValueError(
                f"{folder}: non esistono sequenze complete "
                f"di {SEQ_LEN} frame con partenza diversa da zero."
            )

        if rng.random() < FORCE_ZERO_PROB:
            start_idx = 0
            n_start_zero += 1
        else:
            start_idx = rng.choice(nonzero_starts)

        paths = sequence_paths(folder, start_idx)
        lines.append(" ".join(str(path) for path in paths))

    return lines, n_start_zero


train_lines = []
valid_lines = []
total_start_zero = 0

for dataset in datasets:
    cases = read_cases(
        dataset["csv_path"],
        dataset["bubble_column"],
    )

    train_tags, valid_tags = select_cases(
        cases,
        dataset["include_all"],
        dataset["name"],
        dataset["bubble_column"],
    )

    folder_train, train_zero = make_sequences(
        train_tags, dataset["base_dir"]
    )
    folder_valid, valid_zero = make_sequences(
        valid_tags, dataset["base_dir"]
    )

    train_lines.extend(folder_train)
    valid_lines.extend(folder_valid)
    total_start_zero += train_zero + valid_zero

    print(f"  Sequenze con partenza da zero: {train_zero + valid_zero}")

# Scrive i file solo dopo aver verificato tutti i dataset.
train_txt.write_text("\n".join(train_lines) + "\n")
valid_txt.write_text("\n".join(valid_lines) + "\n")

n_sequences = len(train_lines) + len(valid_lines)

print(f"\nCreato {train_txt}: {len(train_lines)} sequenze")
print(f"Creato {valid_txt}: {len(valid_lines)} sequenze")
print(
    f"Partenze da zero: {total_start_zero}/{n_sequences} "
    f"({100 * total_start_zero / n_sequences:.1f}%)"
)
