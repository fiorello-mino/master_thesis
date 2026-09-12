#!/usr/bin/env python3
"""
Modifica tutti i file .dat nella cartella:

    /data/fiorello/pores3D/data_train/hexagon/init/

Esempio input:
    iso2_R0.2_H1.0_P0.8H.dat

Output directory:
    /scratch/fiorello/data_train3D/hexagon/iso_P08/isoHex_R0.2_H1.0_P0.8
"""

from pathlib import Path
import re
import shutil

INIT_DIR = Path("/data/fiorello/pores3D/data_train/hexagon/init")
OUTPUT_BASE = Path("/scratch/fiorello/data_train3D/hexagon")


def extract_pitch(sim_name: str) -> str | None:
    """
    Esempi:
        iso2_R0.2_H1.0_P0.8H -> iso_P08
        iso2_R0.2_H1.0_P1.0H -> iso_P10
    """
    match = re.search(r"_P(\d+)\.(\d+)H?$", sim_name)

    if match is None:
        return None

    return f"iso_P{match.group(1)}{match.group(2)}"


def output_sim_name(init_sim_name: str) -> str | None:
    """
    Trasforma il nome dell'init nel nome originale della cartella simulazione.

    Esempio:
        iso2_R0.2_H1.0_P0.8H
        -> isoHex_R0.2_H1.0_P0.8
    """
    if not init_sim_name.startswith("iso2"):
        return None

    # iso2... -> isoHex...
    sim_name = "isoHex" + init_sim_name[len("iso2"):]

    # Rimuove esclusivamente la H finale nel nome dell'init.
    return sim_name.removesuffix("H")


def is_adapt_parameter(line: str) -> bool:
    """Righe rigenerate sotto 'surf->adapt->strategy:'."""
    return any(key in line for key in (
        "surf->adapt->time delta 1:",
        "surf->adapt->time delta 2:",
        "surf->adapt->relative energy tolerance:",
        "surf->adapt->min timestep:",
        "surf->adapt->max timestep:",
    ))


def modify_dat_file(dat_path: Path) -> tuple[bool, str]:
    init_sim_name = dat_path.stem
    pitch = extract_pitch(init_sim_name)
    original_sim_name = output_sim_name(init_sim_name)

    if pitch is None:
        return False, f"Pitch non riconosciuto: {dat_path.name}"

    if original_sim_name is None:
        return False, (
            f"Prefisso init non riconosciuto (atteso 'iso2'): {dat_path.name}"
        )

    # Esempio:
    # /scratch/fiorello/data_train3D/hexagon/iso_P08/isoHex_R0.2_H1.0_P0.8
    output_dir = OUTPUT_BASE / pitch / original_sim_name

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

        # Riscrive strategy e inserisce tutti i parametri adaptive sotto di essa.
        if "surf->adapt->strategy:" in line:
            found["strategy"] = True

            new_lines.append("surf->adapt->strategy: 3\n")
            new_lines.append("surf->adapt->time delta 1: 0.7071\n")
            new_lines.append("surf->adapt->time delta 2: 1.4142\n")
            new_lines.append("surf->adapt->relative energy tolerance: 1e-5\n")
            new_lines.append("surf->adapt->min timestep: 5e-6\n")
            new_lines.append("surf->adapt->max timestep: 1e-4\n")

            i += 1
            while i < len(lines) and is_adapt_parameter(lines[i]):
                i += 1
            continue

        # Imposta timestep iniziale.
        if "surf->adapt->timestep:" in line:
            found["timestep"] = True
            new_lines.append("surf->adapt->timestep: 1e-4\n")
            i += 1
            continue

        # Imposta end time.
        if "surf->adapt->end time:" in line:
            found["end_time"] = True
            new_lines.append("surf->adapt->end time: 0.2\n")
            i += 1
            continue

        # Imposta la directory output.
        if "output->directory:" in line:
            found["output_directory"] = True

            new_lines.append(f"output->directory: {output_dir}\n")
            new_lines.append("surf->output->write every delta: 0.005\n")

            i += 1
            while (
                i < len(lines)
                and "surf->output->write every delta:" in lines[i]
            ):
                i += 1
            continue

        # Imposta ogni quanti timestep scrivere.
        if "surf->output->write every i-th timestep:" in line:
            found["write_every_step"] = True
            new_lines.append(
                "surf->output->write every i-th timestep:         10000\n"
            )
            i += 1
            continue

        # Rimuove min/max eventualmente presenti altrove nel file.
        if (
            "surf->adapt->min timestep:" in line
            or "surf->adapt->max timestep:" in line
        ):
            i += 1
            continue

        new_lines.append(line)
        i += 1

    dat_path.write_text("".join(new_lines), encoding="utf-8")

    missing = [key for key, value in found.items() if not value]

    if missing:
        return True, (
            f"{dat_path.name} -> {output_dir} | "
            f"ATTENZIONE: stringhe non trovate: {', '.join(missing)}"
        )

    return True, f"{dat_path.name} -> {output_dir}"


def main():
    if not INIT_DIR.is_dir():
        raise SystemExit(f"ERRORE: directory non trovata: {INIT_DIR}")

    dat_files = sorted(INIT_DIR.glob("*.dat"))

    if not dat_files:
        raise SystemExit(f"ERRORE: nessun file .dat trovato in {INIT_DIR}")

    print(f"Trovati {len(dat_files)} file .dat in {INIT_DIR}")

    # Backup eseguito una volta sola.
    backup_dir = INIT_DIR.with_name(f"{INIT_DIR.name}_backup")

    if not backup_dir.exists():
        shutil.copytree(INIT_DIR, backup_dir)
        print(f"Backup creato: {backup_dir}")
    else:
        print(f"Backup già esistente: {backup_dir}")

    print()

    success = 0
    problems = []

    for dat_path in dat_files:
        try:
            ok, message = modify_dat_file(dat_path)
            print(message)

            if ok:
                success += 1
            else:
                problems.append(message)

        except Exception as exc:
            message = f"{dat_path.name}: ERRORE: {exc}"
            print(message)
            problems.append(message)

    print("\n" + "=" * 70)
    print("RIEPILOGO")
    print("=" * 70)
    print(f"File modificati: {success}/{len(dat_files)}")

    if problems:
        print(f"\nProblemi trovati: {len(problems)}")
        for message in problems:
            print(f"  - {message}")


if __name__ == "__main__":
    main()