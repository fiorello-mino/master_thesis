import os
import re
import random
from pathlib import Path

BASE = Path("/data/fiorello/pores3D/data/square")
N_TRAIN_VAL = 20  # lunghezza sequenza per train/val
N_EXT = 41        # tutti i frame per external test

N_TRAIN_SIMS = 80
N_VAL_SIMS   = 20
N_EXT_SIMS   = 50

SEED = 42  # per riproducibilità

random.seed(SEED)

def extract_H(folder_name: str) -> float | None:
    """
    Estrae H da nomi tipo:
    iso2_R0.2_H1.0_P0.6
    iso2_R0.2_H3.5_P0.7
    """
    m = re.search(r"_H([0-9]+(?:\.[0-9]+)?)", folder_name)
    if not m:
        return None
    return float(m.group(1))

def get_start_range(H: float) -> tuple[int, int]:
    """
    Restituisce (start_min, start_max) in base a H.
    Regole:
      - 1.0 <= H <= 2.0 : start = 0 (fisso)
      - 2.0 <  H <= 3.0 : start in [0, 5]
      - 3.0 <  H <= 4.0 : start in [0, 10]
      - 4.0 <  H <= 5.0 : start in [0, 20]  # aggiornato
    """
    if H <= 2.0:
        return 0, 0
    elif H <= 3.0:
        return 0, 5
    elif H <= 4.0:
        return 0, 10
    else:
        return 0, 20

def find_sim_folders():
    """
    Scansiona data/square/iso_P*/ e raccoglie le simulazioni
    che hanno:
      - nome con H estraibile
      - almeno surf_000000.npy presente
    """
    sims = []
    for p_folder in sorted(BASE.iterdir()):
        if not p_folder.is_dir():
            continue
        if not p_folder.name.startswith("iso_P"):
            continue
        for sim_folder in sorted(p_folder.iterdir()):
            if not sim_folder.is_dir():
                continue
            H = extract_H(sim_folder.name)
            if H is None:
                continue
            first_file = sim_folder / "surf_000000.npy"
            if not first_file.exists():
                continue
            sims.append((sim_folder, H))
    return sims

def build_frame_paths(sim_path: Path, start: int, length: int) -> list[str]:
    """
    Costruisce la lista di path ai file .npy per una data simulazione,
    a partire da 'start' e per 'length' frame.
    I file sono del tipo surf_0.000000.npy, surf_0.005000.npy, ...
    con dt = 5e-3.
    """
    paths = []
    for i in range(start, start + length):
        t = i * 5e-3
        fname = f"surf_{t:.6f}.npy"
        paths.append(str(sim_path / fname))
    return paths

def main():
    sims = find_sim_folders()
    print(f"Trovate {len(sims)} simulazioni candidate.")

    total_needed = N_TRAIN_SIMS + N_VAL_SIMS + N_EXT_SIMS
    if len(sims) < total_needed:
        print(f"Attenzione: ho solo {len(sims)} simulazioni, meno delle {total_needed} richieste.")
        # procedo comunque con quelle disponibili, ridimensionando

    # Mischia e seleziona
    random.shuffle(sims)
    selected = sims[:total_needed]

    train_sims = selected[:N_TRAIN_SIMS]
    val_sims   = selected[N_TRAIN_SIMS:N_TRAIN_SIMS + N_VAL_SIMS]
    ext_sims   = selected[N_TRAIN_SIMS + N_VAL_SIMS:
                          N_TRAIN_SIMS + N_VAL_SIMS + N_EXT_SIMS]

    def write_sequences(sim_list, out_file: Path, length: int):
        lines = []
        for sim_path, H in sim_list:
            if length == N_TRAIN_VAL:
                # train/val: start random con vincoli, 20 frame
                start_min, start_max = get_start_range(H)
                start = random.randint(start_min, start_max)
                # sicurezza: non superare 41 - length
                start = min(start, 41 - length)
            else:
                # external test: tutti i frame da 0 a 40
                start = 0

            frame_paths = build_frame_paths(sim_path, start, length)
            line = " ".join(frame_paths)
            lines.append(line + "\n")

        out_file.write_text("".join(lines))
        print(f"Scritto {len(lines)} righe in {out_file}")

    write_sequences(train_sims, Path("train_set.txt"), N_TRAIN_VAL)
    write_sequences(val_sims,   Path("test_set.txt"),   N_TRAIN_VAL)
    write_sequences(ext_sims,   Path("ext_test.txt"),   N_EXT)

if __name__ == "__main__":
    main()