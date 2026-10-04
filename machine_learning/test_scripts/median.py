from pathlib import Path
import sys

import numpy as np


# ============================================================
# SCRIPT VARIABLES
# ============================================================

TEST_DIR = Path(
    "/scratch/fiorello/test3D/square/train1/fv_c24_mse"
)

ERRORS_FILE = TEST_DIR / "errors.txt"
MEDIANS_FILE = TEST_DIR / "medians.txt"


# ============================================================
# TIME PARAMETERS
# ============================================================

DT = 0.005
STEPS_PER_SAVE = 1
STARTING_FRAME = 1


# ============================================================
# EVO.TXT COLUMNS
# ============================================================
#
# evo.txt prodotto da testAnalysis.py:
#
#  1-10 : colonne originali
#
# 11 : E_true
# 12 : E_pred
# 13 : mass_true
# 14 : mass_pred
#
# 15 : phi_min_sequence
# 16 : phi_max_sequence
# 17 : domains_final_true
# 18 : domains_final_pred
#
# Indici Python:
#
# 10 -> E_true
# 11 -> E_pred
# 12 -> mass_true
# 13 -> mass_pred
#
# 14 -> phi_min_sequence
# 15 -> phi_max_sequence
# 16 -> domains_final_true
# 17 -> domains_final_pred
#

COL_E_TRUE = 10
COL_E_PRED = 11

COL_MASS_TRUE = 12
COL_MASS_PRED = 13

COL_PHI_MIN_SEQUENCE = 14
COL_PHI_MAX_SEQUENCE = 15

COL_DOMAINS_TRUE = 16
COL_DOMAINS_PRED = 17

N_REQUIRED_COLUMNS = 18


# ============================================================
# READ EVO
# ============================================================

def read_evo_file(evo_path: Path):

    if not evo_path.is_file():

        raise FileNotFoundError(
            f"File evo.txt non trovato:\n{evo_path}"
        )


    with evo_path.open("r") as f:
        lines = f.readlines()


    temporal_data = []
    sequence_data = None


    # riga 0 -> header
    # riga 1 -> frame iniziale
    # righe >= 2 -> timestep predetti

    for line in lines[2:]:

        line = line.strip()

        if not line:
            continue

        if line.startswith("#"):
            continue


        parts = line.split()


        if len(parts) < N_REQUIRED_COLUMNS:

            raise ValueError(
                f"\n{evo_path}\n"
                f"ha {len(parts)} colonne, "
                f"ma ne servono almeno {N_REQUIRED_COLUMNS}.\n"
                f"Esegui prima testAnalysis.py."
            )


        # ====================================================
        # TEMPORAL DATA
        # ====================================================

        e_true = float(
            parts[COL_E_TRUE]
        )

        e_pred = float(
            parts[COL_E_PRED]
        )

        mass_true = float(
            parts[COL_MASS_TRUE]
        )

        mass_pred = float(
            parts[COL_MASS_PRED]
        )


        temporal_data.append(
            (
                e_true,
                e_pred,
                mass_true,
                mass_pred,
            )
        )


        # ====================================================
        # GLOBAL SEQUENCE DATA
        #
        # queste quantità sono ripetute in tutte le righe,
        # quindi basta leggerle una volta
        # ====================================================

        if sequence_data is None:

            phi_min_sequence = float(
                parts[COL_PHI_MIN_SEQUENCE]
            )

            phi_max_sequence = float(
                parts[COL_PHI_MAX_SEQUENCE]
            )

            domains_true = int(
                float(
                    parts[COL_DOMAINS_TRUE]
                )
            )

            domains_pred = int(
                float(
                    parts[COL_DOMAINS_PRED]
                )
            )


            sequence_data = (
                phi_min_sequence,
                phi_max_sequence,
                domains_true,
                domains_pred,
            )


    if len(temporal_data) == 0:

        raise ValueError(
            f"Nessuna riga dati valida in:\n{evo_path}"
        )


    return (
        temporal_data,
        sequence_data,
    )


# ============================================================
# READ PREDICTED FRAMES
# ============================================================

def get_pred_frames(sim_dir: Path):

    pred_dir = sim_dir / "pred_npy"


    if not pred_dir.is_dir():

        raise FileNotFoundError(
            f"Cartella pred_npy non trovata:\n{pred_dir}"
        )


    frames = [
        f
        for f in pred_dir.iterdir()
        if (
            f.is_file()
            and f.name.startswith("phi_")
            and f.name.endswith(".npy")
        )
    ]


    if len(frames) == 0:

        raise ValueError(
            f"Nessun frame predetto trovato in:\n{pred_dir}"
        )


    frames.sort(
        key=lambda p: float(
            p.name
            .removeprefix("phi_")
            .removesuffix(".npy")
        )
    )


    return frames


# ============================================================
# MIN / MAX PER TIMESTEP
# ============================================================

def read_pred_min_max(frames):
    """
    Per ogni timestep calcola:

        min_xyz phi_pred(t)
        max_xyz phi_pred(t)

    Restituisce due array con lunghezza = numero di frame.
    """

    phi_min_t = np.empty(
        len(frames),
        dtype=float
    )

    phi_max_t = np.empty(
        len(frames),
        dtype=float
    )


    for i, frame_path in enumerate(frames):

        phi = np.load(
            str(frame_path)
        )


        phi_min_t[i] = np.min(
            phi
        )

        phi_max_t[i] = np.max(
            phi
        )


    return (
        phi_min_t,
        phi_max_t,
    )


# ============================================================
# MEDIAN + QUARTILES
# ============================================================

def median_quartiles(values):

    values = np.asarray(
        values,
        dtype=float
    )


    median = float(
        np.median(values)
    )

    p25 = float(
        np.percentile(
            values,
            25
        )
    )

    p75 = float(
        np.percentile(
            values,
            75
        )
    )


    return (
        median,
        p25,
        p75,
    )


# ============================================================
# UPDATE ERRORS.TXT
# ============================================================

def update_errors_file(sequence_results):
    """
    errors.txt originale:

      1 id
      2 maxMAE
      3 maxMSE
      4 overallMAE
      5 overallMSE
      6 max(symDiff)
      7 avg(symDiff)

    Nuovo errors.txt:

      1  id
      2  maxMAE
      3  maxMSE
      4  overallMAE
      5  overallMSE
      6  max(symDiff)
      7  avg(symDiff)
      8  phi_min_sequence
      9  phi_max_sequence
      10 domains_final_true
      11 domains_final_pred
      12 delta_domains
    """

    if not ERRORS_FILE.is_file():

        raise FileNotFoundError(
            f"errors.txt non trovato:\n{ERRORS_FILE}"
        )


    with ERRORS_FILE.open("r") as f:
        lines = f.readlines()


    new_lines = []


    # ========================================================
    # HEADER
    # ========================================================

    new_lines.append(
        "# 1: id | "
        "2: maxMAE | "
        "3: maxMSE | "
        "4: overallMAE | "
        "5: overallMSE | "
        "6: max(symDiff) | "
        "7: avg(symDiff) | "
        "8: phi_min_sequence | "
        "9: phi_max_sequence | "
        "10: domains_final_true | "
        "11: domains_final_pred | "
        "12: delta_domains\n"
    )


    # ========================================================
    # DATA
    # ========================================================

    for line in lines:

        line = line.strip()


        if not line:
            continue

        if line.startswith("#"):
            continue


        parts = line.split()


        if len(parts) < 7:

            raise ValueError(
                f"Riga errors.txt con meno di 7 colonne:\n"
                f"{line}"
            )


        # Mantengo sempre soltanto le prime 7 colonne
        # originali, così rilanciare median.py non duplica
        # le colonne nuove.

        base = parts[:7]

        sim_name = base[0]


        if sim_name not in sequence_results:

            raise ValueError(
                f"Simulazione '{sim_name}' presente in "
                f"errors.txt ma senza evo.txt corrispondente."
            )


        (
            phi_min_sequence,
            phi_max_sequence,
            domains_true,
            domains_pred,
            delta_domains,
        ) = sequence_results[sim_name]


        new_lines.append(
            " ".join(base)
            + f" {phi_min_sequence:.6e}"
            + f" {phi_max_sequence:.6e}"
            + f" {domains_true:d}"
            + f" {domains_pred:d}"
            + f" {delta_domains:d}"
            + "\n"
        )


    with ERRORS_FILE.open("w") as f:

        f.writelines(
            new_lines
        )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    # ========================================================
    # CHECK DIRECTORY
    # ========================================================

    if not TEST_DIR.is_dir():

        sys.exit(
            f"Directory non trovata:\n{TEST_DIR}"
        )


    # ========================================================
    # EVO FILES
    # ========================================================

    evo_files = sorted(
        TEST_DIR.glob("*/evo.txt"),
        key=lambda p: p.parent.name
    )


    if len(evo_files) == 0:

        sys.exit(
            f"Nessun evo.txt trovato in:\n{TEST_DIR}"
        )


    print(
        f"Trovate {len(evo_files)} simulazioni."
    )


    # ========================================================
    # STORAGE PER TIMESTEP
    # ========================================================

    energy_errors_t = []
    mass_errors_t = []

    phi_min_t = []
    phi_max_t = []


    # ========================================================
    # STORAGE PER-SEQUENCE PER ERRORS.TXT
    # ========================================================

    sequence_results = {}


    # ========================================================
    # LOOP SIMULAZIONI
    # ========================================================

    for evo_path in evo_files:

        sim_dir = evo_path.parent
        sim_name = sim_dir.name


        # ----------------------------------------------------
        # EVO
        # ----------------------------------------------------

        (
            temporal_data,
            sequence_data,
        ) = read_evo_file(
            evo_path
        )


        # ----------------------------------------------------
        # PRED NPY
        # ----------------------------------------------------

        frames_pred = get_pred_frames(
            sim_dir
        )


        if len(frames_pred) != len(temporal_data):

            raise ValueError(
                f"Mismatch numero timestep in {sim_name}:\n"
                f"evo.txt = {len(temporal_data)}\n"
                f"pred_npy = {len(frames_pred)}"
            )


        (
            sim_phi_min_t,
            sim_phi_max_t,
        ) = read_pred_min_max(
            frames_pred
        )


        # ====================================================
        # PREPARO STORAGE TEMPORALE
        # ====================================================

        while len(energy_errors_t) < len(temporal_data):

            energy_errors_t.append([])
            mass_errors_t.append([])

            phi_min_t.append([])
            phi_max_t.append([])


        # ====================================================
        # STATISTICHE PER TIMESTEP
        # ====================================================

        for t, (
            e_true,
            e_pred,
            mass_true,
            mass_pred,
        ) in enumerate(
            temporal_data
        ):


            # ------------------------------------------------
            # relative energy error
            # ------------------------------------------------

            err_energy = (
                abs(e_pred - e_true)
                / max(
                    abs(e_true),
                    1e-12
                )
            )


            # ------------------------------------------------
            # relative mass error
            # ------------------------------------------------

            err_mass = (
                abs(mass_pred - mass_true)
                / max(
                    abs(mass_true),
                    1e-12
                )
            )


            energy_errors_t[t].append(
                err_energy
            )

            mass_errors_t[t].append(
                err_mass
            )


            # ------------------------------------------------
            # min/max spaziale a questo timestep
            # ------------------------------------------------

            phi_min_t[t].append(
                sim_phi_min_t[t]
            )

            phi_max_t[t].append(
                sim_phi_max_t[t]
            )


        # ====================================================
        # DATI GLOBALI DELLA SEQUENZA
        # ====================================================

        (
            phi_min_sequence,
            phi_max_sequence,
            domains_true,
            domains_pred,
        ) = sequence_data


        delta_domains = (
            domains_pred
            - domains_true
        )


        sequence_results[sim_name] = (
            phi_min_sequence,
            phi_max_sequence,
            domains_true,
            domains_pred,
            delta_domains,
        )


        print(
            f"{sim_name}: "
            f"global min={phi_min_sequence:.6e}, "
            f"global max={phi_max_sequence:.6e}, "
            f"N_true={domains_true}, "
            f"N_pred={domains_pred}, "
            f"DeltaN={delta_domains:+d}"
        )


    # ========================================================
    # UPDATE ERRORS.TXT
    # ========================================================

    update_errors_file(
        sequence_results
    )


    # ========================================================
    # NUMBER OF TIMESTEPS
    # ========================================================

    n_timesteps = len(
        energy_errors_t
    )


    # ========================================================
    # OUTPUT ARRAYS
    # ========================================================

    median_energy = np.empty(n_timesteps)
    p25_energy = np.empty(n_timesteps)
    p75_energy = np.empty(n_timesteps)

    median_mass = np.empty(n_timesteps)
    p25_mass = np.empty(n_timesteps)
    p75_mass = np.empty(n_timesteps)

    median_phi_min = np.empty(n_timesteps)
    p25_phi_min = np.empty(n_timesteps)
    p75_phi_min = np.empty(n_timesteps)

    median_phi_max = np.empty(n_timesteps)
    p25_phi_max = np.empty(n_timesteps)
    p75_phi_max = np.empty(n_timesteps)


    # ========================================================
    # CALCOLO MEDIANE + QUARTILI PER OGNI TIMESTEP
    # ========================================================

    for t in range(n_timesteps):

        # ----------------------------------------------------
        # ENERGY ERROR
        # ----------------------------------------------------

        (
            median_energy[t],
            p25_energy[t],
            p75_energy[t],
        ) = median_quartiles(
            energy_errors_t[t]
        )


        # ----------------------------------------------------
        # MASS ERROR
        # ----------------------------------------------------

        (
            median_mass[t],
            p25_mass[t],
            p75_mass[t],
        ) = median_quartiles(
            mass_errors_t[t]
        )


        # ----------------------------------------------------
        # PHI MIN
        #
        # per ogni simulazione:
        #
        # min_xyz phi_pred(t)
        #
        # poi mediana/quartili sulle simulazioni
        # ----------------------------------------------------

        (
            median_phi_min[t],
            p25_phi_min[t],
            p75_phi_min[t],
        ) = median_quartiles(
            phi_min_t[t]
        )


        # ----------------------------------------------------
        # PHI MAX
        # ----------------------------------------------------

        (
            median_phi_max[t],
            p25_phi_max[t],
            p75_phi_max[t],
        ) = median_quartiles(
            phi_max_t[t]
        )


    # ========================================================
    # WRITE MEDIANS.TXT
    # ========================================================
    #
    # Una sola tabella.
    #
    # Una riga per timestep:
    #
    #  1 time
    #
    #  2 median relative energy error
    #  3 p25
    #  4 p75
    #
    #  5 median relative mass error
    #  6 p25
    #  7 p75
    #
    #  8 median min(phi)
    #  9 p25
    # 10 p75
    #
    # 11 median max(phi)
    # 12 p25
    # 13 p75
    #
    # ========================================================

    with MEDIANS_FILE.open("w") as f:


        f.write(
            "# "
            "1:time\t"
            "2:median_energy_error\t"
            "3:p25_energy_error\t"
            "4:p75_energy_error\t"
            "5:median_mass_error\t"
            "6:p25_mass_error\t"
            "7:p75_mass_error\t"
            "8:median_phi_min\t"
            "9:p25_phi_min\t"
            "10:p75_phi_min\t"
            "11:median_phi_max\t"
            "12:p25_phi_max\t"
            "13:p75_phi_max\n"
        )


        for t in range(n_timesteps):

            time = (
                (STARTING_FRAME + t)
                * DT
                * STEPS_PER_SAVE
            )


            f.write(
                f"{time:.6e}\t"

                f"{median_energy[t]:.6e}\t"
                f"{p25_energy[t]:.6e}\t"
                f"{p75_energy[t]:.6e}\t"

                f"{median_mass[t]:.6e}\t"
                f"{p25_mass[t]:.6e}\t"
                f"{p75_mass[t]:.6e}\t"

                f"{median_phi_min[t]:.6e}\t"
                f"{p25_phi_min[t]:.6e}\t"
                f"{p75_phi_min[t]:.6e}\t"

                f"{median_phi_max[t]:.6e}\t"
                f"{p25_phi_max[t]:.6e}\t"
                f"{p75_phi_max[t]:.6e}\n"
            )


    # ========================================================
    # DONE
    # ========================================================

    print()

    print(
        f"Aggiornato:\n{ERRORS_FILE}"
    )

    print()

    print(
        f"Scritto:\n{MEDIANS_FILE}"
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()
