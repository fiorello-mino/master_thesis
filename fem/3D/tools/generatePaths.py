from pathlib import Path
import random


# ============================================================
# CONFIGURAZIONE
# ============================================================

# Root contenente le cartelle convertite in NPY.
BASE_DIR = Path("/data/fiorello/iso_P09")

# File di output: una sequenza per riga.
OUTPUT_TXT = Path("/home/fiorello/master_thesis/machine_learning/train3D/paths/iso_P09/paths.txt")

# Numero di frame consecutivi da prendere in ogni cartella.
N_SEQ = 20

# Per rendere l'estrazione riproducibile.
# Metti None se vuoi sequenze diverse a ogni esecuzione.
RANDOM_SEED = 42

# Se True, salva una sequenza casuale per ogni folder trovato.
# Se False, puoi eventualmente limitare i folder con MAX_FOLDERS.
USE_ALL_FOLDERS = True
MAX_FOLDERS = None

# Cerca solo i file di simulazione e non grid_metadata.npz.
NPY_GLOB = "surf_*.npy"


# ============================================================
# FUNZIONI
# ============================================================

def get_trajectory_folders(base_dir: Path) -> list[Path]:
    """
    Restituisce tutte le cartelle sotto base_dir che contengono
    almeno un file surf_*.npy.

    sorted(...) garantisce l'ordine stabile delle cartelle.
    """
    folders = sorted({
        npy_path.parent
        for npy_path in base_dir.rglob(NPY_GLOB)
    })

    return folders


def choose_random_sequence(
    npy_files: list[Path],
    n_seq: int,
    rng: random.Random,
) -> tuple[int, list[Path]]:
    """
    Sceglie N_SEQ file consecutivi a partire da un indice casuale.

    Se ci sono N file, l'indice iniziale massimo valido è:
        N - N_SEQ

    Il range di randint è inclusivo a entrambi gli estremi.
    """
    total_npy = len(npy_files)

    if total_npy < n_seq:
        raise ValueError(
            f"Frame insufficienti: trovati {total_npy}, "
            f"ma N_SEQ={n_seq}"
        )

    start_index = rng.randint(0, total_npy - n_seq)
    sequence_paths = npy_files[start_index : start_index + n_seq]

    return start_index, sequence_paths


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    if not BASE_DIR.is_dir():
        raise FileNotFoundError(
            f"BASE_DIR non trovata: {BASE_DIR}"
        )

    if N_SEQ <= 0:
        raise ValueError(f"N_SEQ deve essere positivo, trovato {N_SEQ}")

    OUTPUT_TXT.parent.mkdir(parents=True, exist_ok=True)

    rng = random.Random(RANDOM_SEED)

    folders = get_trajectory_folders(BASE_DIR)

    if not folders:
        raise FileNotFoundError(
            f"Nessun file '{NPY_GLOB}' trovato sotto {BASE_DIR}"
        )

    if not USE_ALL_FOLDERS and MAX_FOLDERS is not None:
        folders = folders[:MAX_FOLDERS]

    print(f"Root input : {BASE_DIR}")
    print(f"Output txt : {OUTPUT_TXT}")
    print(f"Cartelle   : {len(folders)}")
    print(f"N_SEQ      : {N_SEQ}")
    print(f"Seed       : {RANDOM_SEED}")

    written_sequences = 0
    skipped_folders = 0

    with OUTPUT_TXT.open("w", encoding="utf-8") as f:
        for folder_index, folder in enumerate(folders, start=1):
            # sorted è fondamentale: surf_0.000000.npy deve venire
            # prima di surf_0.010000.npy, ecc.
            npy_files = sorted(folder.glob(NPY_GLOB))

            total_npy = len(npy_files)

            if total_npy < N_SEQ:
                print(
                    f"[{folder_index:03d}/{len(folders):03d}] SKIP "
                    f"{folder.name}: {total_npy} frame, "
                    f"ma N_SEQ={N_SEQ}"
                )
                skipped_folders += 1
                continue

            start_index, sequence_paths = choose_random_sequence(
                npy_files=npy_files,
                n_seq=N_SEQ,
                rng=rng,
            )

            # Una riga = una traiettoria/sequenza da passare al dataset.
            f.write(" ".join(str(path) for path in sequence_paths) + "\n")

            end_index = start_index + N_SEQ - 1

            print(
                f"[{folder_index:03d}/{len(folders):03d}] OK   "
                f"{folder.name} | "
                f"tot={total_npy} | "
                f"indici={start_index}:{end_index} | "
                f"salvati={N_SEQ}"
            )

            written_sequences += 1

    print("\n" + "=" * 65)
    print(f"File creato        : {OUTPUT_TXT}")
    print(f"Sequenze scritte   : {written_sequences}")
    print(f"Cartelle saltate   : {skipped_folders}")


if __name__ == "__main__":
    main()