import re
import random
from pathlib import Path

BASE = Path("/data/fiorello/pores3D_2/data/square")

N_SEQ = 40
N_ALL_FRAMES = 101

N_TRAIN = 80
N_VAL = 20
N_EXT = 50
N_REQUIRED = N_TRAIN + N_VAL + N_EXT

DT = 2e-3
SEED = 42

rng = random.Random(SEED)


def extract_H(folder_name: str) -> float | None:
    """
    Esempio:
    iso2_R0.2_H1.0_P0.6 -> 1.0
    """
    match = re.search(r"_H([0-9]+(?:\.[0-9]+)?)(?:_|$)", folder_name)
    return float(match.group(1)) if match else None


def get_start_range(H: float) -> tuple[int, int]:
    """
    Vincoli scelti:

    1.0 <= H <= 2.0  -> start = 0
    2.0 <  H <= 3.0  -> start casuale in [0, 5]
    3.0 <  H <= 4.0  -> start casuale in [0, 10]
    4.0 <  H <= 5.0  -> start casuale in [0, 20]
    """
    if 1.0 <= H <= 2.0:
        return 0, 0
    elif 2.0 < H <= 3.0:
        return 0, 5
    elif 3.0 < H <= 4.0:
        return 0, 10
    elif 4.0 < H <= 5.0:
        return 0, 20

    raise ValueError(f"H={H} fuori dall'intervallo [1.0, 5.0]")


def frame_path(sim_path: Path, frame_idx: int) -> Path:
    """
    0  -> surf_0.000000.npy
    1  -> surf_0.005000.npy
    2  -> surf_0.010000.npy
    ...
    40 -> surf_0.200000.npy
    """
    t = frame_idx * DT
    return sim_path / f"surf_{t:.6f}.npy"


def has_all_frames(sim_path: Path) -> bool:
    """
    Una simulazione è valida se possiede tutti i frame da 0 a 40.
    """
    return all(
        frame_path(sim_path, frame_idx).is_file()
        for frame_idx in range(N_ALL_FRAMES)
    )


def find_simulations() -> list[tuple[Path, float]]:
    if not BASE.is_dir():
        raise FileNotFoundError(
            f"Directory non trovata o non accessibile:\n{BASE}"
        )

    p_folders = sorted(
        p for p in BASE.iterdir()
        if p.is_dir() and p.name.startswith("iso_P")
    )

    print(f"Cartelle iso_P trovate: {len(p_folders)}")
    print(" ".join(p.name for p in p_folders))

    simulations = []
    n_no_H = 0
    n_missing_frames = 0

    for p_folder in p_folders:
        sim_folders = sorted(
            p for p in p_folder.iterdir()
            if p.is_dir()
        )

        print(f"{p_folder.name}: {len(sim_folders)} sottocartelle")

        for sim_path in sim_folders:
            H = extract_H(sim_path.name)

            if H is None:
                n_no_H += 1
                print(f"[H non riconosciuto] {sim_path}")
                continue

            if not has_all_frames(sim_path):
                n_missing_frames += 1
                print(f"[frame mancanti] {sim_path}")
                continue

            simulations.append((sim_path, H))

    print(f"\nValide: {len(simulations)}")
    print(f"Scartate per H non riconosciuto: {n_no_H}")
    print(f"Scartate per frame mancanti: {n_missing_frames}")

    return simulations


def build_sequence(
    sim_path: Path,
    start: int,
    n_frames: int,
) -> list[str]:
    paths = [
        str(frame_path(sim_path, i))
        for i in range(start, start + n_frames)
    ]

    missing = [path for path in paths if not Path(path).is_file()]

    if missing:
        raise FileNotFoundError(
            "Frame non trovati:\n" + "\n".join(missing)
        )

    return paths


def write_train_or_val(
    simulations: list[tuple[Path, float]],
    output_path: Path,
) -> None:
    """
    Scrive una riga per simulazione: 20 path per riga.
    """
    lines = []

    for sim_path, H in simulations:
        low, high = get_start_range(H)

        # Limite assoluto: con 41 frame e N_SEQ = 20,
        # lo start massimo ammesso è 21.
        high = min(high, N_ALL_FRAMES - N_SEQ)

        start = rng.randint(low, high)

        paths = build_sequence(
            sim_path=sim_path,
            start=start,
            n_frames=N_SEQ,
        )

        lines.append(" ".join(paths))

    output_path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print(f"{output_path}: {len(lines)} righe, {N_SEQ} path per riga")


def write_ext_test(
    simulations: list[tuple[Path, float]],
    output_path: Path,
) -> None:
    """
    Scrive una riga per simulazione: tutti i 41 frame, da 0 a 40.
    """
    lines = []

    for sim_path, _ in simulations:
        paths = build_sequence(
            sim_path=sim_path,
            start=0,
            n_frames=N_ALL_FRAMES,
        )

        lines.append(" ".join(paths))

    output_path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print(f"{output_path}: {len(lines)} righe, {N_ALL_FRAMES} path per riga")


def main():
    simulations = find_simulations()

    if len(simulations) < N_REQUIRED:
        raise RuntimeError(
            f"Servono {N_REQUIRED} simulazioni valide "
            f"(80 train + 20 validation + 50 external test), "
            f"ma ne hai {len(simulations)}."
        )

    # Estrazione casuale senza ripetizioni:
    # train, validation ed external test sono disgiunti.
    selected = rng.sample(simulations, N_REQUIRED)

    train_sims = selected[:N_TRAIN]
    val_sims = selected[N_TRAIN:N_TRAIN + N_VAL]
    ext_sims = selected[N_TRAIN + N_VAL:]

    write_train_or_val(train_sims, Path("/scratch/fiorello/train3D_2/train_set.txt"))
    write_train_or_val(val_sims, Path("/scratch/fiorello/train3D_2/valid_set.txt"))
    write_ext_test(ext_sims, Path("/scratch/fiorello/test3D_2/square/test_set.txt"))

    print("\nGenerazione completata.")
    print("train_set.txt: 80 righe × 40 frame")
    print("test_set.txt:  20 righe × 40 frame")
    print("ext_test.txt:  50 righe × 101 frame")


if __name__ == "__main__":
    main()
