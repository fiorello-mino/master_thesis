#!/usr/bin/env python3
"""
Modifica tutti i file .dat in:
    /data/fiorello/pores3D/data_train/square/init/

Dal nome della simulazione estrae il pitch:
    ..._P0.6.dat  -> iso_P06
    ..._P0.7.dat  -> iso_P07
    ..._P0.8.dat  -> iso_P08
    ..._P0.9.dat  -> iso_P09
    ..._P1.0.dat  -> iso_P10

e imposta:
    output->directory: /scratch/fiorello/data_train3D/square/iso_P0X/

Le altre modifiche sono:
    surf->adapt->strategy: 3
    surf->adapt->time delta 1: 0.5
    surf->adapt->time delta 2: 2.0
    surf->adapt->relative energy tolerance: 1e-5
    surf->adapt->min timestep: 5e-6
    surf->adapt->max timestep: 1e-4
    surf->adapt->timestep: 1e-4
    surf->adapt->end time: 0.2
    surf->output->write every delta: 0.005
    surf->output->write every i-th timestep: 10000
"""

from pathlib import Path
import re
import shutil

INIT_DIR = Path("/data/fiorello/pores3D/data_train/square/init")
OUTPUT_BASE = "/scratch/fiorello/data_train3D/square"


def extract_pitch(sim_name: str) -> str | None:
    """
    Esempi:
      iso2_R0.2_H1.0_P0.8 -> iso_P08
      iso2_R0.2_H1.0_P1.0 -> iso_P10
    """
    match = re.search(r"_P(\d+)\.(\d+)(?:_|$)", sim_name)

    if match is None:
        return None

    integer_part = match.group(1)
    decimal_part = match.group(2)

    # P0.8 -> P08; P1.0 -> P10
    return f"iso_P{integer_part}{decimal_part}"


def is_generated_adapt_line(line: str) -> bool:
    """True per righe che vogliamo rigenerare sotto strategy."""
    stripped = line.strip()

    return (
        "surf->adapt->time delta 1:" in stripped
        or "surf->adapt->time delta 2:" in stripped
        or "surf->adapt->relative energy tolerance:" in stripped
        or "surf->adapt->min timestep:" in stripped
        or "surf->adapt->max timestep:" in stripped
    )


def modify_dat_file(dat_path: Path) -> tuple[bool, str]:
    """
    Modifica un file .dat.
    Restituisce (successo, messaggio).
    """
    sim_name = dat_path.stem
    pitch = extract_pitch(sim_name)

    if pitch is None:
        return False, (
            f"{dat_path.name}: pitch non riconosciuto nel nome della simulazione"
        )

    output_dir = f"{OUTPUT_BASE}/{pitch}/"

    original_text = dat_path.read_text(encoding="utf-8")
    lines = original_text.splitlines(keepends=True)

    new_lines = []

    found_strategy = False
    found_timestep = False
    found_end_time = False
    found_output_dir = False
    found_write_every_step = False

    i = 0
    while i < len(lines):
        line = lines[i]

        # Strategia: sostituisce la riga e ricrea il blocco di parametri sotto.
        if "surf->adapt->strategy:" in line:
            found_strategy = True

            new_lines.append("surf->adapt->strategy: 3\n")
            new_lines.append("surf->adapt->time delta 1: 0.5\n")
            new_lines.append("surf->adapt->time delta 2: 2.0\n")
            new_lines.append("surf->adapt->relative energy tolerance: 1e-5\n")
            new_lines.append("surf->adapt->min timestep: 5e-6\n")
            new_lines.append("surf->adapt->max timestep: 1e-4\n")

            # Se lo script era già stato eseguito, salta il vecchio blocco inserito.
            i += 1
            while i < len(lines) and is_generated_adapt_line(lines[i]):
                i += 1

            continue

        # Modifica timestep iniziale.
        if "surf->adapt->timestep:" in line:
            found_timestep = True
            new_lines.append("surf->adapt->timestep: 1e-4\n")
            i += 1
            continue

        # Modifica tempo finale.
        if "surf->adapt->end time:" in line:
            found_end_time = True
            new_lines.append("surf->adapt->end time: 0.2\n")
            i += 1
            continue

        # Imposta la directory di output in base al pitch.
        if "output->directory:" in line:
            found_output_dir = True

            new_lines.append(f"output->directory: {output_dir}\n")
            new_lines.append("surf->output->write every delta: 0.005\n")

            # Se già presente da una vecchia esecuzione, rimuove la riga duplicata.
            i += 1
            while (
                i < len(lines)
                and "surf->output->write every delta:" in lines[i]
            ):
                i += 1

            continue

        # Modifica frequenza output in numero di timestep.
        if "surf->output->write every i-th timestep:" in line:
            found_write_every_step = True
            new_lines.append(
                "surf->output->write every i-th timestep:         10000\n"
            )
            i += 1
            continue

        new_lines.append(line)
        i += 1

    dat_path.write_text("".join(new_lines), encoding="utf-8")

    missing_keys = []
    if not found_strategy:
        missing_keys.append("strategy")
    if not found_timestep:
        missing_keys.append("timestep")
    if not found_end_time:
        missing_keys.append("end time")
    if not found_output_dir:
        missing_keys.append("output->directory")
    if not found_write_every_step:
        missing_keys.append("write every i-th timestep")

    if missing_keys:
        return True, (
            f"{dat_path.name} -> {pitch}; ATTENZIONE: stringhe non trovate: "
            + ", ".join(missing_keys)
        )

    return True, f"{dat_path.name} -> {pitch}"


def main():
    if not INIT_DIR.is_dir():
        raise SystemExit(f"ERRORE: cartella non trovata: {INIT_DIR}")

    dat_files = sorted(INIT_DIR.glob("*.dat"))

    if not dat_files:
        raise SystemExit(f"ERRORE: nessun file .dat in {INIT_DIR}")

    print(f"Trovati {len(dat_files)} file .dat")
    print(f"Output base: {OUTPUT_BASE}\n")

    # Backup unico dell'intera cartella, prima di modificare.
    backup_dir = INIT_DIR.with_name(f"{INIT_DIR.name}_backup")
    if not backup_dir.exists():
        shutil.copytree(INIT_DIR, backup_dir)
        print(f"Backup creato in: {backup_dir}\n")
    else:
        print(f"Backup già presente: {backup_dir}\n")

    success_count = 0
    failed = []

    for dat_path in dat_files:
        try:
            ok, message = modify_dat_file(dat_path)
            print(message)

            if ok:
                success_count += 1
            else:
                failed.append(message)

        except Exception as exc:
            failed.append(f"{dat_path.name}: ERRORE: {exc}")

    print("\n" + "=" * 60)
    print("RIEPILOGO")
    print("=" * 60)
    print(f"File elaborati correttamente: {success_count}/{len(dat_files)}")

    if failed:
        print(f"\nProblemi: {len(failed)}")
        for message in failed:
            print(f"  - {message}")


if __name__ == "__main__":
    main()