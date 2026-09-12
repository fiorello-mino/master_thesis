#!/usr/bin/env python3
"""
Copia e modifica gli init delle simulazioni cone.

Sorgente:
    /archive/roberto/poresAMDIS/cones/

Processa solo:
    iso_con_...

Ignora automaticamente:
    iso_cyl_...
    old/
    immagini e altri file

Esempio:
    cartella simulazione: iso_con_a86_H2.0_P0.8
    init interno:         iso_con2_a86_H2.0_P0.8.dat
    init copiato:         .../cone/init/iso_con2_a86_H2.0_P0.8.dat
    output directory:     /scratch/fiorello/data_train3D/cone/iso_P08/iso_con_a86_H2.0_P0.8
"""

from pathlib import Path
import re
import shutil

SOURCE_DIR = Path("/archive/roberto/poresAMDIS/cones")
DEST_DIR = Path("/data/fiorello/pores3D/data_train/cone/init")
OUTPUT_BASE = Path("/scratch/fiorello/data_train3D/cone")

SIM_PREFIX = "iso_con_"
INIT_PREFIX = "iso_con2_"


def extract_pitch(sim_name: str) -> str | None:
    """
    Estrae il pitch dalla cartella della simulazione.

    Esempi:
        iso_con_a86_H2.0_P0.8 -> iso_P08
        iso_con_a86_H2.0_P1.0 -> iso_P10
    """
    match = re.search(r"_P(\d+)\.(\d+)$", sim_name)

    if match is None:
        return None

    return f"iso_P{match.group(1)}{match.group(2)}"


def init_name_from_simulation_folder(sim_name: str) -> str:
    """
    Esempio:
        iso_con_a86_H2.0_P0.8
        -> iso_con2_a86_H2.0_P0.8.dat
    """
    if not sim_name.startswith(SIM_PREFIX):
        raise ValueError(f"Cartella non valida: {sim_name}")

    return INIT_PREFIX + sim_name[len(SIM_PREFIX):] + ".dat"


def is_adapt_parameter(line: str) -> bool:
    """
    Riconosce le righe rigenerate sotto 'surf->adapt->strategy:'.
    Evita duplicazioni quando lo script viene rilanciato.
    """
    return any(key in line for key in (
        "surf->adapt->time delta 1:",
        "surf->adapt->time delta 2:",
        "surf->adapt->relative energy tolerance:",
        "surf->adapt->min timestep:",
        "surf->adapt->max timestep:",
    ))


def modify_dat_file(dat_path: Path, output_dir: Path) -> list[str]:
    """Modifica un singolo file .dat già copiato in DEST_DIR."""
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

        # Riscrive strategy e il blocco adaptive sottostante.
        if "surf->adapt->strategy:" in line:
            found["strategy"] = True

            new_lines.append("surf->adapt->strategy: 3\n")
            new_lines.append("surf->adapt->time delta 1: 0.5\n")
            new_lines.append("surf->adapt->time delta 2: 2.0\n")
            new_lines.append("surf->adapt->relative energy tolerance: 1e-5\n")
            new_lines.append("surf->adapt->min timestep: 5e-6\n")
            new_lines.append("surf->adapt->max timestep: 1e-4\n")

            i += 1
            while i < len(lines) and is_adapt_parameter(lines[i]):
                i += 1

            continue

        # Imposta il timestep iniziale.
        if "surf->adapt->timestep:" in line:
            found["timestep"] = True
            new_lines.append("surf->adapt->timestep: 1e-4\n")
            i += 1
            continue

        # Imposta il tempo finale.
        if "surf->adapt->end time:" in line:
            found["end_time"] = True
            new_lines.append("surf->adapt->end time: 0.2\n")
            i += 1
            continue

        # Imposta la directory di output e la frequenza temporale dell'output.
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

        # Imposta frequenza output in numero di timestep.
        if "surf->output->write every i-th timestep:" in line:
            found["write_every_step"] = True
            new_lines.append(
                "surf->output->write every i-th timestep:         10000\n"
            )
            i += 1
            continue

        # Rimuove eventuali min/max timestep presenti altrove nel file.
        if (
            "surf->adapt->min timestep:" in line
            or "surf->adapt->max timestep:" in line
        ):
            i += 1
            continue

        new_lines.append(line)
        i += 1

    dat_path.write_text("".join(new_lines), encoding="utf-8")

    return [name for name, was_found in found.items() if not was_found]


def main():
    if not SOURCE_DIR.is_dir():
        raise SystemExit(f"ERRORE: directory sorgente non trovata: {SOURCE_DIR}")

    DEST_DIR.mkdir(parents=True, exist_ok=True)

    # Prende esclusivamente le directory iso_con_...
    simulation_dirs = sorted(
        path for path in SOURCE_DIR.iterdir()
        if path.is_dir() and path.name.startswith(SIM_PREFIX)
    )

    if not simulation_dirs:
        raise SystemExit(
            f"ERRORE: nessuna directory '{SIM_PREFIX}...' trovata in {SOURCE_DIR}"
        )

    print(f"Cartelle cone da processare: {len(simulation_dirs)}")
    print(f"Destinazione init: {DEST_DIR}\n")

    copied = 0
    overwritten = 0
    missing_init = []
    invalid_pitch = []
    warnings = []

    for sim_dir in simulation_dirs:
        sim_name = sim_dir.name
        pitch = extract_pitch(sim_name)

        if pitch is None:
            invalid_pitch.append(sim_name)
            print(f"[ERRORE] Pitch non riconosciuto: {sim_name}")
            continue

        init_filename = init_name_from_simulation_folder(sim_name)
        source_dat = sim_dir / init_filename
        dest_dat = DEST_DIR / init_filename

        if not source_dat.is_file():
            missing_init.append(source_dat)
            print(f"[MANCANTE] {source_dat}")
            continue

        if dest_dat.exists():
            overwritten += 1

        # Copia l'init nella directory piatta.
        shutil.copy2(source_dat, dest_dat)
        copied += 1

        # Output: .../cone/iso_P08/iso_con_a86_H2.0_P0.8
        output_dir = OUTPUT_BASE / pitch / sim_name

        missing_keys = modify_dat_file(dest_dat, output_dir)

        print(f"[OK] {source_dat.name}")
        print(f"     output->directory: {output_dir}")

        if missing_keys:
            warning = (
                f"{source_dat.name}: righe non trovate: "
                + ", ".join(missing_keys)
            )
            warnings.append(warning)
            print(f"     ATTENZIONE: {warning}")

    print("\n" + "=" * 70)
    print("RIEPILOGO COPIA E MODIFICA CONE")
    print("=" * 70)
    print(f"Cartelle iso_con trovate: {len(simulation_dirs)}")
    print(f"Init copiati/modificati:  {copied}")
    print(f"Init sovrascritti:        {overwritten}")
    print(f"Init mancanti:            {len(missing_init)}")
    print(f"Pitch non riconosciuti:   {len(invalid_pitch)}")
    print(f"Avvisi:                   {len(warnings)}")

    if missing_init:
        print("\nFile init non trovati:")
        for path in missing_init:
            print(f"  - {path}")

    if invalid_pitch:
        print("\nCartelle con pitch non riconosciuto:")
        for name in invalid_pitch:
            print(f"  - {name}")


if __name__ == "__main__":
    main()