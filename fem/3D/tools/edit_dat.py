#!/usr/bin/env python3
"""
Modifica tutti i file .dat nella cartella init di square.

Esempio:
    iso2_R0.2_H1.0_P0.6.dat

diventa:
    output->directory: /scratch/fiorello/data_train3D/square/iso_P06/iso2_R0.2_H1.0_P0.6
"""

from pathlib import Path
import re
import shutil

INIT_DIR = Path("/data/fiorello/pores3D/data_train/square/init")
OUTPUT_BASE = Path("/scratch/fiorello/data_train3D/square")


def extract_pitch(sim_name: str) -> str | None:
    """
    Estrae il pitch dal nome della simulazione.

    Esempi:
        iso2_R0.2_H1.0_P0.6 -> iso_P06
        iso2_R0.2_H1.0_P0.8 -> iso_P08
        iso2_R0.2_H5.0_P1.0 -> iso_P10
    """
    match = re.search(r"_P(\d+)\.(\d+)$", sim_name)

    if match is None:
        return None

    integer_part = match.group(1)
    decimal_part = match.group(2)

    return f"iso_P{integer_part}{decimal_part}"


def is_adapt_parameter(line: str) -> bool:
    """
    Riconosce le righe che vengono rigenerate immediatamente dopo strategy.
    Serve a poter rieseguire lo script senza duplicare le righe.
    """
    stripped = line.strip()

    return any(key in stripped for key in [
        "surf->adapt->time delta 1:",
        "surf->adapt->time delta 2:",
        "surf->adapt->relative energy tolerance:",
        "surf->adapt->min timestep:",
        "surf->adapt->max timestep:",
    ])


def modify_dat_file(dat_path: Path) -> tuple[bool, str]:
    """
    Modifica il file .dat e restituisce:
        (True/False, messaggio)
    """
    sim_name = dat_path.stem
    pitch = extract_pitch(sim_name)

    if pitch is None:
        return False, f"Pitch non riconosciuto: {dat_path.name}"

    # Esempio:
    # /scratch/fiorello/data_train3D/square/iso_P06/iso2_R0.2_H1.0_P0.6
    output_dir = OUTPUT_BASE / pitch / sim_name

    lines = dat_path.read_text(encoding="utf-8").splitlines(keepends=True)
    new_lines = []

    found = {
        "strategy": False,
        "timestep": False,
        "end_time": False,
        "output_directory": False,
        "write_every_step": False,
    }

    i = 0
    while i < len(lines):
        line = lines[i]

        # strategy + blocco completo dei parametri di adaptive timestep
        if "surf->adapt->strategy:" in line:
            found["strategy"] = True

            new_lines.append("surf->adapt->strategy: 3\n")
            new_lines.append("surf->adapt->time delta 1: 0.5\n")
            new_lines.append("surf->adapt->time delta 2: 2.0\n")
            new_lines.append("surf->adapt->relative energy tolerance: 1e-5\n")
            new_lines.append("surf->adapt->min timestep: 5e-6\n")
            new_lines.append("surf->adapt->max timestep: 1e-4\n")

            # Salta eventuali righe già inserite in precedenza.
            i += 1
            while i < len(lines) and is_adapt_parameter(lines[i]):
                i += 1
            continue

        # timestep iniziale
        if "surf->adapt->timestep:" in line:
            found["timestep"] = True
            new_lines.append("surf->adapt->timestep: 1e-4\n")
            i += 1
            continue

        # end time
        if "surf->adapt->end time:" in line:
            found["end_time"] = True
            new_lines.append("surf->adapt->end time: 0.2\n")
            i += 1
            continue

        # directory output specifica per pitch + nome simulazione
        if "output->directory:" in line:
            found["output_directory"] = True

            new_lines.append(f"output->directory: {output_dir}\n")
            new_lines.append("surf->output->write every delta: 0.005\n")

            # Se già presente, evita di aggiungere più righe duplicate.
            i += 1
            while (
                i < len(lines)
                and "surf->output->write every delta:" in lines[i]
            ):
                i += 1
            continue

        # frequenza output in timestep
        if "surf->output->write every i-th timestep:" in line:
            found["write_every_step"] = True
            new_lines.append(
                "surf->output->write every i-th timestep:         10000\n"
            )
            i += 1
            continue

        # Rimuove eventuali min/max timestep presenti in altri punti del file:
        # le uniche righe min/max restano quelle subito dopo strategy.
        if (
            "surf->adapt->min timestep:" in line
            or "surf->adapt->max timestep:" in line
        ):
            i += 1
            continue

        new_lines.append(line)
        i += 1

    dat_path.write_text("".join(new_lines), encoding="utf-8")

    missing = [key for key, was_found in found.items() if not was_found]

    if missing:
        return True, (
            f"{dat_path.name} -> {pitch} | "
            f"ATTENZIONE, righe non trovate: {', '.join(missing)}"
        )

    return True, f"{dat_path.name} -> {pitch}"


def main():
    if not INIT_DIR.is_dir():
        raise SystemExit(f"ERRORE: directory non trovata: {INIT_DIR}")

    dat_files = sorted(INIT_DIR.glob("*.dat"))

    if not dat_files:
        raise SystemExit(f"ERRORE: nessun file .dat trovato in {INIT_DIR}")

    print(f"Trovati {len(dat_files)} file .dat in {INIT_DIR}")

    # Backup di tutti i file prima della prima modifica.
    backup_dir = INIT_DIR.with_name(f"{INIT_DIR.name}_backup")

    if not backup_dir.exists():
        shutil.copytree(INIT_DIR, backup_dir)
        print(f"Backup creato: {backup_dir}")
    else:
        print(f"Backup già esistente, non sovrascritto: {backup_dir}")

    print()

    success = 0
    errors = []

    for dat_path in dat_files:
        try:
            ok, message = modify_dat_file(dat_path)
            print(message)

            if ok:
                success += 1
            else:
                errors.append(message)

        except Exception as exc:
            message = f"{dat_path.name}: ERRORE: {exc}"
            print(message)
            errors.append(message)

    print("\n" + "=" * 65)
    print("RIEPILOGO")
    print("=" * 65)
    print(f"File modificati: {success}/{len(dat_files)}")

    if errors:
        print(f"\nProblemi trovati: {len(errors)}")
        for message in errors:
            print(f"  - {message}")


if __name__ == "__main__":
    main()