from __future__ import annotations

from decimal import Decimal, InvalidOperation
from pathlib import Path


# ============================================================
# CONFIGURAZIONE
# ============================================================

# Root contenente le cartelle con i frame surf_<time>.npy.
ROOT_DIR = Path("/data/fiorello/iso_P06")

# Passo temporale richiesto, confrontato in modo esatto con Decimal.
EXPECTED_DT = Decimal("0.005000")

# Frame di dati da controllare.
# Le mask PyVista/VTK vengono escluse automaticamente.
FRAME_GLOB = "surf_*.npy"
VTK_MASK_SUFFIX = "_vtk_fallback_mask.npy"

# Sicurezza: prima esegui con False.
# Lo script crea solo il report e NON cancella nulla.
DELETE_INVALID_FOLDERS = False

# Se True, anche una cartella con meno di due frame viene considerata invalida.
REQUIRE_AT_LEAST_TWO_FRAMES = True

REPORT_PATH = ROOT_DIR / "invalid_time_spacing_folders.txt"


# ============================================================
# FUNZIONI
# ============================================================

def is_data_frame(path: Path) -> bool:
    """Accetta surf_<time>.npy ed esclude le mask *_vtk_fallback_mask.npy."""
    return (
        path.is_file()
        and path.name.startswith("surf_")
        and path.suffix == ".npy"
        and not path.name.endswith(VTK_MASK_SUFFIX)
    )


def parse_time_from_frame(frame_path: Path) -> Decimal:
    """
    Estrae il tempo dal nome surf_<time>.npy usando Decimal.

    Esempio:
        surf_0.005000.npy -> Decimal('0.005000')
    """
    prefix = "surf_"

    if not frame_path.name.startswith(prefix) or frame_path.suffix != ".npy":
        raise ValueError(f"Nome frame non valido: {frame_path.name}")

    time_string = frame_path.stem[len(prefix):]

    try:
        return Decimal(time_string)
    except InvalidOperation as exc:
        raise ValueError(
            f"Impossibile estrarre il tempo da: {frame_path.name}"
        ) from exc


def get_trajectory_folders(root_dir: Path) -> list[Path]:
    """Trova tutte le cartelle che contengono almeno un vero frame phi."""
    return sorted(
        {
            path.parent
            for path in root_dir.rglob(FRAME_GLOB)
            if is_data_frame(path)
        }
    )


def validate_folder(folder: Path) -> tuple[bool, str, list[tuple[Decimal, Decimal, Decimal]]]:
    """
    Controlla la sequenza temporale della cartella.

    Restituisce:
        valid, message, invalid_steps

    invalid_steps contiene tuple:
        (t_precedente, t_corrente, dt_trovato)
    """
    frame_paths = sorted(
        (path for path in folder.glob(FRAME_GLOB) if is_data_frame(path)),
        key=parse_time_from_frame,
    )

    if REQUIRE_AT_LEAST_TWO_FRAMES and len(frame_paths) < 2:
        return (
            False,
            f"solo {len(frame_paths)} frame, richiesti almeno 2",
            [],
        )

    if len(frame_paths) < 2:
        return True, f"{len(frame_paths)} frame", []

    times = [parse_time_from_frame(path) for path in frame_paths]
    invalid_steps: list[tuple[Decimal, Decimal, Decimal]] = []

    for previous_time, current_time in zip(times[:-1], times[1:]):
        dt = current_time - previous_time

        if dt != EXPECTED_DT:
            invalid_steps.append((previous_time, current_time, dt))

    if invalid_steps:
        return (
            False,
            f"{len(invalid_steps)} intervalli diversi da dt={EXPECTED_DT}",
            invalid_steps,
        )

    return True, f"{len(frame_paths)} frame, dt={EXPECTED_DT}", []


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    if not ROOT_DIR.is_dir():
        raise FileNotFoundError(f"Root directory non trovata: {ROOT_DIR}")

    folders = get_trajectory_folders(ROOT_DIR)

    if not folders:
        raise FileNotFoundError(
            f"Nessun frame '{FRAME_GLOB}' trovato sotto {ROOT_DIR}"
        )

    print(f"Root directory          : {ROOT_DIR}")
    print(f"Cartelle da controllare : {len(folders)}")
    print(f"Passo temporale atteso  : {EXPECTED_DT}")
    print(f"Eliminazione attiva     : {DELETE_INVALID_FOLDERS}")

    valid_folders = 0
    invalid_folders: list[tuple[Path, str, list[tuple[Decimal, Decimal, Decimal]]]] = []

    for index, folder in enumerate(folders, start=1):
        valid, message, invalid_steps = validate_folder(folder)
        relative_folder = folder.relative_to(ROOT_DIR)

        if valid:
            valid_folders += 1
            print(
                f"[{index:03d}/{len(folders):03d}] OK     "
                f"{relative_folder} | {message}"
            )
        else:
            invalid_folders.append((folder, message, invalid_steps))
            print(
                f"[{index:03d}/{len(folders):03d}] INVALID "
                f"{relative_folder} | {message}"
            )

            for previous_time, current_time, dt in invalid_steps[:10]:
                print(
                    f"    {previous_time} -> {current_time} "
                    f"| dt={dt}"
                )

            if len(invalid_steps) > 10:
                print(f"    ... altri {len(invalid_steps) - 10} intervalli errati")

    # Report dettagliato: viene sempre scritto prima di ogni eventuale delete.
    report_lines = [
        f"ROOT_DIR = {ROOT_DIR}",
        f"EXPECTED_DT = {EXPECTED_DT}",
        f"FOLDER TOTALI = {len(folders)}",
        f"FOLDER VALIDE = {valid_folders}",
        f"FOLDER INVALIDE = {len(invalid_folders)}",
        "",
    ]

    for folder, message, invalid_steps in invalid_folders:
        report_lines.append(str(folder))
        report_lines.append(f"  Motivo: {message}")

        for previous_time, current_time, dt in invalid_steps:
            report_lines.append(
                f"  {previous_time} -> {current_time} | dt={dt}"
            )

        report_lines.append("")

    REPORT_PATH.write_text("\n".join(report_lines), encoding="utf-8")

    print("\n" + "=" * 78)
    print("RIEPILOGO")
    print(f"Cartelle valide   : {valid_folders}")
    print(f"Cartelle invalide : {len(invalid_folders)}")
    print(f"Report            : {REPORT_PATH}")

    # Eliminazione irreversibile, attiva solo se impostata esplicitamente.
    if DELETE_INVALID_FOLDERS:
        if not invalid_folders:
            print("Nessuna cartella invalida da eliminare.")
            return

        print("\nELIMINAZIONE CARTELLE INVALIDE")

        for folder, message, _ in invalid_folders:
            print(f"Elimino: {folder} | {message}")

            # Elimina ricorsivamente la cartella e tutti i file contenuti.
            import shutil
            shutil.rmtree(folder)

        print(f"Eliminate {len(invalid_folders)} cartelle invalide.")
    else:
        print(
            "\nModalità sicura: nessuna cartella è stata eliminata.\n"
            "Controlla il report; se è corretto, imposta:\n"
            "DELETE_INVALID_FOLDERS = True\n"
            "e rilancia lo script."
        )


if __name__ == "__main__":
    main()
