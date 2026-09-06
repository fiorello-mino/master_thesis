from __future__ import annotations

import random
from decimal import Decimal, InvalidOperation
from pathlib import Path


# ============================================================
# CONFIGURAZIONE
# ============================================================

# Root contenente iso_P06, iso_P07, ..., iso_P10 e le cartelle simulazione.
ROOT_DIR = Path("/data/fiorello/poresAMDIS")

# File TXT di output: una sequenza di N_SEQ path per riga.
TRAIN_TXT = ROOT_DIR / "/home/fiorello/master_thesis/machine_learning/train3D/train_set.txt"
VALID_TXT = ROOT_DIR / "/home/fiorello/master_thesis/machine_learning/train3D/valid_set.txt"

# Numero di frame consecutivi in ogni sequenza.
N_SEQ = 20

# Circa 4/5 training e 1/5 validation.
TRAIN_FRACTION = 0.8

# Seed fisso: split e frame scelti sono riproducibili.
# Metti None se vuoi uno split diverso a ogni esecuzione.
RANDOM_SEED = 42

# Cerca solo i veri frame phi.
FRAME_GLOB = "surf_*.npy"
VTK_MASK_SUFFIX = "_vtk_fallback_mask.npy"

# Se True, richiede anche che i tempi consecutivi differiscano esattamente di DT.
# Consigliato se hai già ripulito il dataset con lo script degli intervalli.
CHECK_TIME_SPACING = True
EXPECTED_DT = Decimal("0.005000")


# ============================================================
# FUNZIONI
# ============================================================

def is_data_frame(path: Path) -> bool:
    """Accetta solo surf_<time>.npy ed esclude le mask VTK."""
    return (
        path.is_file()
        and path.name.startswith("surf_")
        and path.suffix == ".npy"
        and not path.name.endswith(VTK_MASK_SUFFIX)
    )


def parse_time(frame_path: Path) -> Decimal:
    """Estrae il valore temporale da surf_<time>.npy usando Decimal."""
    time_string = frame_path.stem.removeprefix("surf_")

    try:
        return Decimal(time_string)
    except InvalidOperation as exc:
        raise ValueError(
            f"Nome frame non valido: {frame_path.name}"
        ) from exc


def get_data_frames(folder: Path) -> list[Path]:
    """Restituisce i frame ordinati per valore temporale, non lessicograficamente."""
    frames = [
        path
        for path in folder.glob(FRAME_GLOB)
        if is_data_frame(path)
    ]

    return sorted(frames, key=parse_time)


def find_simulation_folders(root_dir: Path) -> list[Path]:
    """Trova ogni cartella che contiene direttamente almeno un frame phi."""
    return sorted(
        {
            path.parent
            for path in root_dir.rglob(FRAME_GLOB)
            if is_data_frame(path)
        }
    )


def valid_sequence_start_indices(
    frames: list[Path],
    n_seq: int,
) -> list[int]:
    """
    Restituisce tutti gli indici iniziali che producono N_SEQ frame
    temporalmente consecutivi e, se richiesto, separati da EXPECTED_DT.
    """
    if len(frames) < n_seq:
        return []

    if not CHECK_TIME_SPACING:
        return list(range(len(frames) - n_seq + 1))

    times = [parse_time(path) for path in frames]
    valid_starts: list[int] = []

    for start_index in range(len(frames) - n_seq + 1):
        window_times = times[start_index : start_index + n_seq]

        is_valid = all(
            current_time - previous_time == EXPECTED_DT
            for previous_time, current_time in zip(
                window_times[:-1],
                window_times[1:],
            )
        )

        if is_valid:
            valid_starts.append(start_index)

    return valid_starts


def split_simulations(
    folders: list[Path],
    rng: random.Random,
) -> tuple[list[Path], list[Path]]:
    """
    Divide per simulazione, non per frame.

    Con 93 simulazioni e TRAIN_FRACTION=0.8:
        n_train = int(0.8 * 93) = 74
        n_valid = 19
    """
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
) -> tuple[int, list[tuple[Path, str]]]:
    """
    Scrive una riga per simulazione.

    Ogni riga contiene N_SEQ path consecutivi separati da uno spazio.
    Ritorna numero di righe scritte ed eventuali simulazioni saltate.
    """
    output_txt.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped: list[tuple[Path, str]] = []

    with output_txt.open("w", encoding="utf-8") as file:
        for folder in folders:
            frames = get_data_frames(folder)
            valid_starts = valid_sequence_start_indices(frames, N_SEQ)

            if not valid_starts:
                if len(frames) < N_SEQ:
                    reason = (
                        f"solo {len(frames)} frame; "
                        f"ne servono almeno {N_SEQ}"
                    )
                else:
                    reason = (
                        f"nessuna finestra di {N_SEQ} frame con "
                        f"dt={EXPECTED_DT}"
                    )

                skipped.append((folder, reason))
                continue

            start_index = rng.choice(valid_starts)
            sequence = frames[start_index : start_index + N_SEQ]

            file.write(" ".join(str(path) for path in sequence) + "\n")
            written += 1

    return written, skipped


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    if not ROOT_DIR.is_dir():
        raise FileNotFoundError(f"Root directory non trovata: {ROOT_DIR}")

    if N_SEQ <= 0:
        raise ValueError(f"N_SEQ deve essere positivo, trovato {N_SEQ}")

    if not 0.0 < TRAIN_FRACTION < 1.0:
        raise ValueError(
            f"TRAIN_FRACTION deve stare tra 0 e 1, trovato {TRAIN_FRACTION}"
        )

    rng = random.Random(RANDOM_SEED)

    all_folders = find_simulation_folders(ROOT_DIR)

    if not all_folders:
        raise FileNotFoundError(
            f"Nessun frame '{FRAME_GLOB}' trovato sotto {ROOT_DIR}"
        )

    # Tiene solo le simulazioni che possono effettivamente fornire
    # almeno una sequenza valida di N_SEQ frame.
    eligible_folders: list[Path] = []
    ineligible_folders: list[tuple[Path, str]] = []

    for folder in all_folders:
        frames = get_data_frames(folder)
        valid_starts = valid_sequence_start_indices(frames, N_SEQ)

        if valid_starts:
            eligible_folders.append(folder)
        elif len(frames) < N_SEQ:
            ineligible_folders.append(
                (folder, f"solo {len(frames)} frame")
            )
        else:
            ineligible_folders.append(
                (folder, f"nessuna finestra valida con dt={EXPECTED_DT}")
            )

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
    print(f"Root directory            : {ROOT_DIR}")
    print(f"N_SEQ                     : {N_SEQ}")
    print(f"TRAIN_FRACTION            : {TRAIN_FRACTION}")
    print(f"Random seed               : {RANDOM_SEED}")
    print(f"Controllo dt              : {CHECK_TIME_SPACING}")

    if CHECK_TIME_SPACING:
        print(f"Passo temporale richiesto : {EXPECTED_DT}")

    print(f"\nSimulazioni trovate      : {len(all_folders)}")
    print(f"Simulazioni utilizzabili  : {len(eligible_folders)}")
    print(f"Simulazioni non usate     : {len(ineligible_folders)}")

    print(f"\nTrain simulazioni        : {len(train_folders)}")
    print(f"Train righe scritte       : {train_written}")
    print(f"Train file                : {TRAIN_TXT}")

    print(f"\nValidation simulazioni   : {len(valid_folders)}")
    print(f"Validation righe scritte  : {valid_written}")
    print(f"Validation file           : {VALID_TXT}")

    all_skipped = ineligible_folders + train_skipped + valid_skipped

    if all_skipped:
        skipped_report = ROOT_DIR / "skipped_simulations_generate_paths.txt"

        skipped_report.write_text(
            "\n".join(
                f"{folder} -> {reason}"
                for folder, reason in all_skipped
            ) + "\n",
            encoding="utf-8",
        )

        print(f"\nReport simulazioni escluse: {skipped_report}")
    else:
        print("\nNessuna simulazione esclusa.")


if __name__ == "__main__":
    main()
