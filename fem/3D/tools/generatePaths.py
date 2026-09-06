from __future__ import annotations

import random
from decimal import Decimal, InvalidOperation
from pathlib import Path


# ============================================================
# CONFIGURAZIONE
# ============================================================

# Root contenente iso_P06, iso_P07, ..., iso_P10 e le cartelle simulazione.
ROOT_DIR = Path("/data/fiorello/poresAMDIS")

# Cartella in cui scrivere ESCLUSIVAMENTE train_set.txt e valid_set.txt.
OUTPUT_DIR = Path(
    "/home/fiorello/master_thesis/machine_learning/train3D"
)

# Una sequenza di N_SEQ path per ogni riga dei file TXT.
TRAIN_TXT = OUTPUT_DIR / "train_set.txt"
VALID_TXT = OUTPUT_DIR / "valid_set.txt"

# Numero di frame consecutivi in ogni sequenza.
N_SEQ = 20

# Circa 4/5 training e 1/5 validation.
TRAIN_FRACTION = 0.8

# Seed fisso: split e frame scelti sono riproducibili.
# Metti None se vuoi uno split diverso a ogni esecuzione.
RANDOM_SEED = 42

# Cerca solo i veri frame del campo phi.
FRAME_GLOB = "surf_*.npy"
VTK_MASK_SUFFIX = "_vtk_fallback_mask.npy"

# Se True, ogni sequenza deve avere frame temporalmente separati esattamente
# di EXPECTED_DT. Decimal evita problemi di confronto con floating point.
CHECK_TIME_SPACING = True
EXPECTED_DT = Decimal("0.005000")

# Se una simulazione contiene PIU' DI 50 frame, la sequenza selezionata può
# iniziare solo nei primi 30 frame. Il frame iniziale è quindi compreso fra:
# 1 e 30 in numerazione umana, oppure 0 e 29 in indice Python.
#
# Se una simulazione contiene 50 frame o meno, tutti gli start validi restano
# disponibili.
LIMIT_START_ONLY_IF_MORE_THAN_FRAMES = 50
MAX_START_FRAME_NUMBER = 30


# ============================================================
# FUNZIONI
# ============================================================


def is_data_frame(path: Path) -> bool:
    """Accetta surf_<tempo>.npy ed esclude le mask di fallback VTK."""
    return (
        path.is_file()
        and path.name.startswith("surf_")
        and path.suffix == ".npy"
        and not path.name.endswith(VTK_MASK_SUFFIX)
    )


def parse_time(frame_path: Path) -> Decimal:
    """Estrae il tempo da un nome nel formato surf_<tempo>.npy."""
    time_string = frame_path.stem.removeprefix("surf_")

    try:
        return Decimal(time_string)
    except InvalidOperation as exc:
        raise ValueError(
            f"Nome frame non valido: {frame_path.name}"
        ) from exc


def get_data_frames(folder: Path) -> list[Path]:
    """Restituisce i frame della simulazione ordinati per tempo."""
    frames = [
        path
        for path in folder.glob(FRAME_GLOB)
        if is_data_frame(path)
    ]

    return sorted(frames, key=parse_time)


def find_simulation_folders(root_dir: Path) -> list[Path]:
    """Trova ogni cartella che contiene direttamente almeno un frame NPY."""
    folders = {
        path.parent
        for path in root_dir.rglob(FRAME_GLOB)
        if is_data_frame(path)
    }

    return sorted(folders)


def valid_sequence_start_indices(
    frames: list[Path],
) -> list[int]:
    """Trova gli indici che consentono una sequenza valida di N_SEQ frame."""
    if len(frames) < N_SEQ:
        return []

    all_starts = list(range(len(frames) - N_SEQ + 1))

    if not CHECK_TIME_SPACING:
        return all_starts

    times = [parse_time(path) for path in frames]
    valid_starts: list[int] = []

    for start_index in all_starts:
        sequence_times = times[start_index : start_index + N_SEQ]

        time_spacing_is_valid = all(
            current_time - previous_time == EXPECTED_DT
            for previous_time, current_time in zip(
                sequence_times[:-1],
                sequence_times[1:],
            )
        )

        if time_spacing_is_valid:
            valid_starts.append(start_index)

    return valid_starts


def allowed_sequence_start_indices(
    frames: list[Path],
) -> list[int]:
    """Applica il controllo temporale e l'eventuale limite al frame iniziale."""
    valid_starts = valid_sequence_start_indices(frames)

    if len(frames) <= LIMIT_START_ONLY_IF_MORE_THAN_FRAMES:
        return valid_starts

    max_start_index = MAX_START_FRAME_NUMBER - 1

    return [
        start_index
        for start_index in valid_starts
        if start_index <= max_start_index
    ]


def split_simulations(
    folders: list[Path],
    rng: random.Random,
) -> tuple[list[Path], list[Path]]:
    """Divide le simulazioni in train e validation senza mescolare i frame."""
    shuffled_folders = folders.copy()
    rng.shuffle(shuffled_folders)

    n_train = int(TRAIN_FRACTION * len(shuffled_folders))

    train_folders = sorted(shuffled_folders[:n_train])
    valid_folders = sorted(shuffled_folders[n_train:])

    return train_folders, valid_folders


def write_sequences(
    folders: list[Path],
    output_txt: Path,
    rng: random.Random,
) -> tuple[int, int]:
    """
    Scrive una sequenza casuale per simulazione.

    Ogni riga contiene N_SEQ path assoluti, separati da spazi.
    Ritorna: (numero righe scritte, numero simulazioni saltate).
    """
    output_txt.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped = 0

    with output_txt.open("w", encoding="utf-8") as file:
        for folder in folders:
            frames = get_data_frames(folder)
            allowed_starts = allowed_sequence_start_indices(frames)

            if not allowed_starts:
                skipped += 1
                continue

            start_index = rng.choice(allowed_starts)
            sequence = frames[start_index : start_index + N_SEQ]

            file.write(" ".join(str(path) for path in sequence) + "\n")
            written += 1

    return written, skipped


# ============================================================
# MAIN
# ============================================================


def main() -> None:
    if not ROOT_DIR.is_dir():
        raise FileNotFoundError(
            f"Root directory non trovata: {ROOT_DIR}"
        )

    if N_SEQ <= 0:
        raise ValueError(f"N_SEQ deve essere positivo, trovato {N_SEQ}")

    if not 0.0 < TRAIN_FRACTION < 1.0:
        raise ValueError(
            "TRAIN_FRACTION deve stare strettamente tra 0 e 1, "
            f"trovato {TRAIN_FRACTION}"
        )

    if LIMIT_START_ONLY_IF_MORE_THAN_FRAMES < 0:
        raise ValueError(
            "LIMIT_START_ONLY_IF_MORE_THAN_FRAMES non può essere negativo"
        )

    if MAX_START_FRAME_NUMBER <= 0:
        raise ValueError(
            "MAX_START_FRAME_NUMBER deve essere positivo, "
            f"trovato {MAX_START_FRAME_NUMBER}"
        )

    rng = random.Random(RANDOM_SEED)

    all_folders = find_simulation_folders(ROOT_DIR)

    if not all_folders:
        raise FileNotFoundError(
            f"Nessun frame '{FRAME_GLOB}' trovato sotto {ROOT_DIR}"
        )

    # Mantiene solo le simulazioni da cui si può estrarre una sequenza di
    # N_SEQ frame consecutivi rispettando anche la regola sui primi 30 frame.
    eligible_folders: list[Path] = []

    for folder in all_folders:
        frames = get_data_frames(folder)

        if allowed_sequence_start_indices(frames):
            eligible_folders.append(folder)

    if not eligible_folders:
        raise RuntimeError(
            f"Nessuna simulazione contiene una sequenza valida di {N_SEQ} frame"
        )

    train_folders, valid_folders = split_simulations(eligible_folders, rng)

    train_written, train_skipped = write_sequences(
        folders=train_folders,
        output_txt=TRAIN_TXT,
        rng=rng,
    )

    valid_written, valid_skipped = write_sequences(
        folders=valid_folders,
        output_txt=VALID_TXT,
        rng=rng,
    )

    print("=" * 78)
    print("GENERAZIONE TRAIN/VALID SET 3D")
    print("=" * 78)
    print(f"Root directory              : {ROOT_DIR}")
    print(f"Directory output            : {OUTPUT_DIR}")
    print(f"N_SEQ                       : {N_SEQ}")
    print(f"TRAIN_FRACTION              : {TRAIN_FRACTION}")
    print(f"Random seed                 : {RANDOM_SEED}")
    print(f"Controllo dt                : {CHECK_TIME_SPACING}")

    if CHECK_TIME_SPACING:
        print(f"Passo temporale richiesto   : {EXPECTED_DT}")

    print(
        "Regola frame iniziale       : "
        f"se frame > {LIMIT_START_ONLY_IF_MORE_THAN_FRAMES}, "
        f"inizio entro frame {MAX_START_FRAME_NUMBER}"
    )

    print(f"\nSimulazioni trovate        : {len(all_folders)}")
    print(f"Simulazioni utilizzabili    : {len(eligible_folders)}")
    print(f"Simulazioni non utilizzabili: {len(all_folders) - len(eligible_folders)}")

    print(f"\nTrain simulazioni          : {len(train_folders)}")
    print(f"Train righe scritte         : {train_written}")
    print(f"Train simulazioni saltate   : {train_skipped}")
    print(f"Train file                  : {TRAIN_TXT}")

    print(f"\nValidation simulazioni     : {len(valid_folders)}")
    print(f"Validation righe scritte    : {valid_written}")
    print(f"Validation simulazioni saltate: {valid_skipped}")
    print(f"Validation file             : {VALID_TXT}")


if __name__ == "__main__":
    main()
