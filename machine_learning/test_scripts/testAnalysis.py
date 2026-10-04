from pathlib import Path
import os
import sys

import numpy as np
from numba import njit
from scipy import ndimage


# ========== SCRIPT VARIABLES ==========

TEST_DIR = Path("/scratch/fiorello/test3D/square/train1/fv_c24_mse")

# --------------------------------------


# ========== PARAMETERS ===========

DX = 0.025
DY = 0.025
DZ = 0.025

EPS = 0.1

DOMAIN_THRESHOLD = 0.5

# =================================


# N26 connectivity:
#
# un voxel è connesso a tutti i 26 vicini:
# facce + spigoli + vertici
STRUCTURE_26 = ndimage.generate_binary_structure(3, 3)


def time_key(entry: os.DirEntry):
    name = entry.name  # es. surf_0.000000.npy
    numero_str = name.removeprefix("surf_").removesuffix(".npy")
    return float(numero_str)


# ============================================================
# FREE ENERGY
# ============================================================

@njit(fastmath=True)
def w_field_3D(
    phi: np.ndarray,
    epsilon: float,
    w: np.ndarray
):
    nx, ny, nz = phi.shape

    factor = 18.0 / epsilon

    for x in range(nx):
        for y in range(ny):
            for z in range(nz):

                phi_xyz = phi[x, y, z]

                w[x, y, z] = (
                    factor
                    * phi_xyz
                    * phi_xyz
                    * (1.0 - phi_xyz)
                    * (1.0 - phi_xyz)
                )


# ============================================================
# GRADIENT
# ============================================================

@njit(fastmath=True)
def grad_3D_neumann(
    phi: np.ndarray,
    dx: float,
    dy: float,
    dz: float,
    grad_x: np.ndarray,
    grad_y: np.ndarray,
    grad_z: np.ndarray
):

    nx, ny, nz = phi.shape

    dx2_inv = 1.0 / (2.0 * dx)
    dy2_inv = 1.0 / (2.0 * dy)
    dz2_inv = 1.0 / (2.0 * dz)


    # --------------------------------------------------------
    # gradiente in x
    # --------------------------------------------------------

    for y in range(ny):
        for z in range(nz):

            for x in range(1, nx - 1):

                grad_x[x, y, z] = (
                    phi[x + 1, y, z]
                    - phi[x - 1, y, z]
                ) * dx2_inv

            grad_x[0, y, z] = (
                phi[1, y, z]
                - phi[0, y, z]
            ) * dx2_inv

            grad_x[nx - 1, y, z] = (
                phi[nx - 1, y, z]
                - phi[nx - 2, y, z]
            ) * dx2_inv


    # --------------------------------------------------------
    # gradiente in y
    # --------------------------------------------------------

    for x in range(nx):
        for z in range(nz):

            for y in range(1, ny - 1):

                grad_y[x, y, z] = (
                    phi[x, y + 1, z]
                    - phi[x, y - 1, z]
                ) * dy2_inv

            grad_y[x, 0, z] = (
                phi[x, 1, z]
                - phi[x, 0, z]
            ) * dy2_inv

            grad_y[x, ny - 1, z] = (
                phi[x, ny - 1, z]
                - phi[x, ny - 2, z]
            ) * dy2_inv


    # --------------------------------------------------------
    # gradiente in z
    # --------------------------------------------------------

    for x in range(nx):
        for y in range(ny):

            for z in range(1, nz - 1):

                grad_z[x, y, z] = (
                    phi[x, y, z + 1]
                    - phi[x, y, z - 1]
                ) * dz2_inv

            grad_z[x, y, 0] = (
                phi[x, y, 1]
                - phi[x, y, 0]
            ) * dz2_inv

            grad_z[x, y, nz - 1] = (
                phi[x, y, nz - 1]
                - phi[x, y, nz - 2]
            ) * dz2_inv


# ============================================================
# ENERGY
# ============================================================

@njit(fastmath=True)
def compute_e_3D(
    phi: np.ndarray,
    epsilon: float,
    dx: float,
    dy: float,
    dz: float
) -> float:

    nx, ny, nz = phi.shape

    eps2 = 0.5 * epsilon

    w_local = np.empty_like(phi)

    gx = np.empty_like(phi)
    gy = np.empty_like(phi)
    gz = np.empty_like(phi)

    w_field_3D(
        phi,
        epsilon,
        w_local
    )

    grad_3D_neumann(
        phi,
        dx,
        dy,
        dz,
        gx,
        gy,
        gz
    )

    total_E = 0.0

    for x in range(nx):
        for y in range(ny):
            for z in range(nz):

                grad2 = (
                    gx[x, y, z] * gx[x, y, z]
                    + gy[x, y, z] * gy[x, y, z]
                    + gz[x, y, z] * gz[x, y, z]
                )

                f_xyz = (
                    w_local[x, y, z]
                    + eps2 * grad2
                )

                total_E += f_xyz

    return total_E * dx * dy * dz


# ============================================================
# CONNECTED DOMAINS
# ============================================================

def count_connected_domains_3D(
    phi: np.ndarray,
    threshold: float = DOMAIN_THRESHOLD
) -> int:
    """
    Conta i domini connessi della regione:

        phi >= threshold

    usando connettività N26.
    """

    mask = phi >= threshold

    _, n_domains = ndimage.label(
        mask,
        structure=STRUCTURE_26
    )

    return int(n_domains)


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    # --------------------------------------------------------
    # controllo directory principale
    # --------------------------------------------------------

    if not TEST_DIR.is_dir():

        sys.exit(
            f"{TEST_DIR} directory non trovata."
        )


    # --------------------------------------------------------
    # prendo directory simulazioni
    # --------------------------------------------------------

    sim_folders = sorted(
        [
            d
            for d in TEST_DIR.iterdir()
            if d.is_dir()
        ],
        key=lambda p: p.name
    )

    if len(sim_folders) == 0:

        sys.exit(
            "0 simulazioni trovate."
        )


    print(
        f"Trovate {len(sim_folders)} simulazioni."
    )


    # ========================================================
    # LOOP SIMULAZIONI
    # ========================================================

    for sim in sim_folders:

        print(
            f"Analizzo {sim.name}"
        )


        # ----------------------------------------------------
        # EVO FILE
        # ----------------------------------------------------

        evo_file = sim / "evo.txt"

        if not evo_file.is_file():

            sys.exit(
                f"File evo non trovato in {sim}"
            )


        with open(
            evo_file,
            "r"
        ) as f:

            stats = f.readlines()


        if len(stats) < 3:

            sys.exit(
                f"evo.txt troppo corto in {sim}"
            )


        # ====================================================
        # PREDICTED NPY
        # ====================================================

        pred_dir = sim / "pred_npy"

        if not pred_dir.is_dir():

            sys.exit(
                f"Cartella pred_npy non trovata in {sim}"
            )


        frames_pred = [
            f
            for f in pred_dir.iterdir()
            if (
                f.is_file()
                and f.name.startswith("phi_")
                and f.name.endswith(".npy")
            )
        ]


        if len(frames_pred) == 0:

            sys.exit(
                f"0 frames pred nella cartella {pred_dir}"
            )


        frames_pred.sort(
            key=lambda p: float(
                p.name
                .removeprefix("phi_")
                .removesuffix(".npy")
            )
        )


        phi_pred = [
            np.load(str(f))
            for f in frames_pred
        ]


        # ====================================================
        # TRUE NPY
        # ====================================================

        true_dir = sim / "true_npy"

        if not true_dir.is_dir():

            sys.exit(
                f"Cartella true_npy non trovata in {sim}"
            )


        frames_true = [
            f
            for f in true_dir.iterdir()
            if (
                f.is_file()
                and f.name.startswith("surf_")
                and f.name.endswith(".npy")
            )
        ]


        if len(frames_true) == 0:

            sys.exit(
                f"0 frames true nella cartella {true_dir}"
            )


        # ----------------------------------------------------
        # ordino temporalmente
        # ----------------------------------------------------

        frames_true.sort(
            key=lambda p: float(
                p.name
                .removeprefix("surf_")
                .removesuffix(".npy")
            )
        )


        # ----------------------------------------------------
        # elimino t = 0
        # ----------------------------------------------------

        frames_true = [
            f
            for f in frames_true
            if float(
                f.name
                .removeprefix("surf_")
                .removesuffix(".npy")
            ) > 0.0
        ]


        if len(frames_true) == 0:

            sys.exit(
                f"Nessun frame true (escluso t=0) "
                f"nella cartella {true_dir}"
            )


        phi_true = [
            np.load(str(f))
            for f in frames_true
        ]


        # ====================================================
        # CONTROLLO COERENZA FRAME
        # ====================================================

        n_frames = len(phi_pred)


        if len(phi_true) != n_frames:

            sys.exit(
                f"Mismatch numero frame pred/true in {sim}: "
                f"pred={len(phi_pred)}, "
                f"true={len(phi_true)}"
            )


        # evo:
        #
        # riga 0 -> header
        # riga 1 -> frame iniziale
        # righe da 2 -> frame predetti
        #
        n_evo_frames = len(stats) - 2


        if n_evo_frames != n_frames:

            sys.exit(
                f"Mismatch evo/npy in {sim}: "
                f"evo={n_evo_frames}, "
                f"npy={n_frames}"
            )


        # ====================================================
        # ENERGY + MASS PER TIMESTEP
        # ====================================================

        e_pred = np.zeros(
            n_frames,
            dtype=float
        )

        e_true = np.zeros(
            n_frames,
            dtype=float
        )

        m_pred = np.zeros(
            n_frames,
            dtype=float
        )

        m_true = np.zeros(
            n_frames,
            dtype=float
        )


        for i in range(n_frames):

            e_pred[i] = compute_e_3D(
                phi_pred[i],
                EPS,
                DX,
                DY,
                DZ
            )

            e_true[i] = compute_e_3D(
                phi_true[i],
                EPS,
                DX,
                DY,
                DZ
            )

            m_pred[i] = (
                np.sum(phi_pred[i])
                * DX
                * DY
                * DZ
            )

            m_true[i] = (
                np.sum(phi_true[i])
                * DX
                * DY
                * DZ
            )


        # ====================================================
        # GLOBAL MIN / MAX DELLA SEQUENZA PRED
        # ====================================================

        phi_min_sequence = np.inf
        phi_max_sequence = -np.inf


        for phi in phi_pred:

            current_min = float(
                np.min(phi)
            )

            current_max = float(
                np.max(phi)
            )

            if current_min < phi_min_sequence:
                phi_min_sequence = current_min

            if current_max > phi_max_sequence:
                phi_max_sequence = current_max


        # ====================================================
        # CONNECTED DOMAINS AL FRAME FINALE
        # ====================================================

        domains_final_true = count_connected_domains_3D(
            phi_true[-1]
        )

        domains_final_pred = count_connected_domains_3D(
            phi_pred[-1]
        )


        print(
            f"  global min pred = "
            f"{phi_min_sequence:.6e}"
        )

        print(
            f"  global max pred = "
            f"{phi_max_sequence:.6e}"
        )

        print(
            f"  domains final true = "
            f"{domains_final_true}"
        )

        print(
            f"  domains final pred = "
            f"{domains_final_pred}"
        )


        # ====================================================
        # SCRITTURA EVO.TXT
        # ====================================================
        #
        # IMPORTANTE:
        #
        # evo.txt originale ha 10 colonne.
        #
        # Ricostruiamo SEMPRE le righe prendendo solo le prime
        # 10 colonne originali.
        #
        # In questo modo, se lo script viene rilanciato,
        # NON vengono duplicate le colonne aggiunte.
        #
        # Output finale:
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
        # Le colonne 15-18 sono proprietà dell'intera
        # simulazione, quindi vengono ripetute su tutte
        # le righe temporali.
        # ====================================================


        # ----------------------------------------------------
        # header
        # ----------------------------------------------------

        header = stats[0].rstrip("\n")


        # se era già stato analizzato in precedenza,
        # elimino eventuali colonne aggiunte
        if "\t11:" in header:

            header = header.split(
                "\t11:",
                1
            )[0]


        stats[0] = (
            header
            + "\t11: E_true"
            + "\t12: E_pred"
            + "\t13: mass_true"
            + "\t14: mass_pred"
            + "\t15: phi_min_sequence"
            + "\t16: phi_max_sequence"
            + "\t17: domains_final_true"
            + "\t18: domains_final_pred\n"
        )


        # ----------------------------------------------------
        # righe temporali
        # ----------------------------------------------------

        for i in range(2, len(stats)):

            # prendo SOLO le 10 colonne originali
            base_columns = stats[i].split()[:10]


            if len(base_columns) < 10:

                sys.exit(
                    f"Riga evo con meno di 10 colonne "
                    f"in {sim}:\n"
                    f"{stats[i]}"
                )


            base_line = "\t".join(
                base_columns
            )


            j = i - 2


            stats[i] = (
                base_line
                + f"\t{e_true[j]:.6e}"
                + f"\t{e_pred[j]:.6e}"
                + f"\t{m_true[j]:.6e}"
                + f"\t{m_pred[j]:.6e}"
                + f"\t{phi_min_sequence:.6e}"
                + f"\t{phi_max_sequence:.6e}"
                + f"\t{domains_final_true:d}"
                + f"\t{domains_final_pred:d}"
                + "\n"
            )


        # ====================================================
        # WRITE FILE
        # ====================================================

        with open(
            evo_file,
            "w"
        ) as f:

            f.writelines(
                stats
            )


    print(
        "\nAnalisi completata."
    )


if __name__ == "__main__":
    main()
