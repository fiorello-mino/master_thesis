from pathlib import Path
import numpy as np

# ============================================================
# CONFIGURAZIONE
# ============================================================

# File txt:
# ogni riga contiene i path ai surf_*.npy di UNA simulazione.
INPUT_TXT = Path("/scratch/fiorello/test3D/cone/ext_test.txt")

# Root del nuovo dataset di pori completi.
# I file originali non vengono mai modificati.
OUTPUT_ROOT = Path(
    "/data/fiorello/pores3D/data/cone_full_pores"
)

# False: salta i file già presenti nell'output.
# True: sovrascrive i file già presenti.
OVERWRITE = False


# ============================================================
# RICOSTRUZIONE DEL VOLUME
# ============================================================

def make_full_pore_bottom_right(a: np.ndarray) -> np.ndarray:
    """
    Input:
        a.shape = (nx, ny, nz)

    Output:
        full.shape = (2*nx, 2*ny, nz)

    Riflette solo:
        x -> axis 0
        y -> axis 1

    Non modifica:
        z -> axis 2

    Layout nel piano x-y:

        x < nx, y < ny      | x < nx, y >= ny
        flip(x,y)           | flip(x)
        --------------------+--------------------
        x >= nx, y < ny     | x >= nx, y >= ny
        flip(y)             | originale

    Quindi:
        full[nx:, ny:, :] == a
    """

    if a.ndim != 3:
        raise ValueError(
            "Ogni file npy deve contenere un array 3D "
            f"con shape (nx, ny, nz). Shape ricevuta: {a.shape}"
        )

    # Quadranti superiori: riflessione lungo x.
    top = np.concatenate(
        (
            np.flip(a, axis=(0, 1)),  # top-left: flip x e y
            np.flip(a, axis=0),       # top-right: flip x
        ),
        axis=1,
    )

    # Quadranti inferiori: x invariato.
    bottom = np.concatenate(
        (
            np.flip(a, axis=1),  # bottom-left: flip y
            a,                   # bottom-right: originale
        ),
        axis=1,
    )

    # Unisce top e bottom lungo x.
    full = np.concatenate(
        (top, bottom),
        axis=0,
    )

    return full


# ============================================================
# COSTRUZIONE PATH DI OUTPUT
# ============================================================

def output_path_from_input(input_path: Path) -> Path:
    """
    Input esempio:

    /data/fiorello/pores3D/data/hexagon/iso_P08/
    isoHex_R0.2_H1.6_P0.8/surf_0.050000.npy

    Output esempio:

    /data/fiorello/pores3D/data/hexagon_full_pores/iso_P08/
    isoHex_R0.2_H1.6_P0.8/surf_0.050000.npy
    """

    simulation_folder = input_path.parent
    iso_p_folder = simulation_folder.parent

    return (
        OUTPUT_ROOT
        / iso_p_folder.name
        / simulation_folder.name
        / input_path.name
    )


# ============================================================
# LETTURA TXT
# ============================================================

def read_simulations(txt_path: Path) -> list[list[Path]]:
    """
    Legge il txt.

    Formato atteso:

    path_0.npy path_1.npy ... path_N.npy
    path_0.npy path_1.npy ... path_N.npy
    ...

    Una riga = una simulazione.
    """

    if not txt_path.is_file():
        raise FileNotFoundError(
            f"File txt non trovato:\n{txt_path.resolve()}"
        )

    simulations = []

    with txt_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            frame_paths = [Path(path_string) for path_string in line.split()]

            missing = [
                path
                for path in frame_paths
                if not path.is_file()
            ]

            if missing:
                preview = "\n".join(str(path) for path in missing[:5])

                raise FileNotFoundError(
                    f"Riga {line_number}: trovati {len(missing)} file mancanti.\n"
                    f"Primi file mancanti:\n{preview}"
                )

            simulations.append(frame_paths)

    return simulations


# ============================================================
# CONVERSIONE
# ============================================================

def main():
    simulations = read_simulations(INPUT_TXT)

    if not simulations:
        raise RuntimeError(
            f"Nessuna simulazione letta da:\n{INPUT_TXT.resolve()}"
        )

    n_total_frames = sum(len(sim) for sim in simulations)

    print(f"Input txt: {INPUT_TXT.resolve()}")
    print(f"Numero simulazioni: {len(simulations)}")
    print(f"Numero totale frame: {n_total_frames}")
    print(f"Output root: {OUTPUT_ROOT.resolve()}")

    n_saved = 0
    n_skipped = 0

    first_input_shape = None
    first_output_shape = None

    for sim_index, frame_paths in enumerate(simulations, start=1):
        simulation_name = frame_paths[0].parent.name

        print(
            f"[{sim_index:03d}/{len(simulations):03d}] "
            f"{simulation_name} | "
            f"{len(frame_paths)} frame"
        )

        for input_path in frame_paths:
            output_path = output_path_from_input(input_path)

            if output_path.exists() and not OVERWRITE:
                n_skipped += 1
                continue

            quarter = np.load(input_path)
            full = make_full_pore_bottom_right(quarter)

            # Controllo una sola volta sulle shape e sul blocco originale.
            if first_input_shape is None:
                first_input_shape = quarter.shape
                first_output_shape = full.shape

                nx, ny, nz = quarter.shape

                assert full.shape == (2 * nx, 2 * ny, nz), (
                    f"Shape output errata: {full.shape}, "
                    f"attesa {(2 * nx, 2 * ny, nz)}"
                )

                assert np.array_equal(
                    full[nx:, ny:, :],
                    quarter,
                ), "Il blocco bottom-right non coincide con l'array originale."

                print(f"  Shape originale: {first_input_shape}")
                print(f"  Shape completa:  {first_output_shape}")
                print("  Verifica bottom-right: OK")

            output_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            np.save(output_path, full)
            n_saved += 1

    print("\nConversione completata.")
    print(f"File creati: {n_saved}")
    print(f"File saltati: {n_skipped}")

    if first_input_shape is not None:
        print(f"Shape input:  {first_input_shape}")
        print(f"Shape output: {first_output_shape}")


if __name__ == "__main__":
    main()
