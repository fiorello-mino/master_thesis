"""Analisi 2D con condizioni periodiche in x e y negli operatori.

Gli array hanno forma (nx, ny): asse 0 = x, asse 1 = y.
Configurare TEST_DIR, DX, DY ed EPS prima dell'esecuzione.
Il conteggio dei domini usa N8 senza collegamenti tra bordi opposti,
come nell'analisi originale: le BC riguardano solo gli operatori.
"""
from pathlib import Path
import sys

import numpy as np
from numba import njit
from scipy import ndimage


# ========== SCRIPT VARIABLES ==========
TEST_DIR = Path("/scratch/fiorello/test2D/square/train1/fv_c24_mse")

# ========== PARAMETERS ===========
DX = 0.025
DY = 0.025
EPS = 0.1
DOMAIN_THRESHOLD = 0.5

# N8: ogni pixel è connesso ai vicini di lato e di vertice.
STRUCTURE_8 = ndimage.generate_binary_structure(2, 2)


def load_phi_2D(path: Path) -> np.ndarray:
    phi = np.asarray(np.load(str(path)), dtype=np.float64)
    if phi.ndim != 2 or min(phi.shape) < 1:
        sys.exit(f"Atteso un campo 2D non vuoto (nx, ny) in {path}, ricevuto {phi.shape}")
    return phi


# ============================================================
# FREE ENERGY
# ============================================================

@njit(fastmath=True)
def w_field_2D(phi: np.ndarray, epsilon: float, w: np.ndarray):
    nx, ny = phi.shape
    factor = 18.0 / epsilon
    for x in range(nx):
        for y in range(ny):
            value = phi[x, y]
            w[x, y] = factor * value * value * (1.0 - value) * (1.0 - value)


# ============================================================
# GRADIENT
# ============================================================

@njit(fastmath=True)
def grad_2D_pbc(
    phi: np.ndarray,
    dx: float,
    dy: float,
    grad_x: np.ndarray,
    grad_y: np.ndarray,
):
    """Differenze centrate, con BC periodiche in x e y."""
    nx, ny = phi.shape
    dx2_inv = 1.0 / (2.0 * dx)
    dy2_inv = 1.0 / (2.0 * dy)

    for x in range(nx):
        # PBC in x: gli indici attraversano il bordo del dominio.
        xm = (x - 1) % nx
        xp = (x + 1) % nx
        for y in range(ny):
            # PBC in y.
            ym = (y - 1) % ny
            yp = (y + 1) % ny
            grad_x[x, y] = (phi[xp, y] - phi[xm, y]) * dx2_inv
            grad_y[x, y] = (phi[x, yp] - phi[x, ym]) * dy2_inv


# ============================================================
# ENERGY
# ============================================================

@njit(fastmath=True)
def compute_e_2D(
    phi: np.ndarray,
    epsilon: float,
    dx: float,
    dy: float,
) -> float:
    nx, ny = phi.shape
    w_local = np.empty_like(phi)
    gx = np.empty_like(phi)
    gy = np.empty_like(phi)
    w_field_2D(phi, epsilon, w_local)
    grad_2D_pbc(phi, dx, dy, gx, gy)

    total_E = 0.0
    for x in range(nx):
        for y in range(ny):
            grad2 = gx[x, y] * gx[x, y] + gy[x, y] * gy[x, y]
            total_E += w_local[x, y] + 0.5 * epsilon * grad2
    return total_E * dx * dy


# ============================================================
# CONNECTED DOMAINS
# ============================================================

def count_connected_domains_2D(
    phi: np.ndarray,
    threshold: float = DOMAIN_THRESHOLD,
) -> int:
    """Conta le componenti di phi >= threshold con N8, senza wrapping."""
    _, n_domains = ndimage.label(phi >= threshold, structure=STRUCTURE_8)
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
            load_phi_2D(f)
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
            load_phi_2D(f)
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

            if phi_pred[i].shape != phi_true[i].shape:
                sys.exit(
                    f"Mismatch forma pred/true in {sim}, frame {i}: "
                    f"{phi_pred[i].shape} != {phi_true[i].shape}"
                )

            e_pred[i] = compute_e_2D(
                phi_pred[i],
                EPS,
                DX,
                DY
            )

            e_true[i] = compute_e_2D(
                phi_true[i],
                EPS,
                DX,
                DY
            )

            m_pred[i] = (
                np.sum(phi_pred[i])
                * DX
                * DY
            )

            m_true[i] = (
                np.sum(phi_true[i])
                * DX
                * DY
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

        domains_final_true = count_connected_domains_2D(
            phi_true[-1]
        )

        domains_final_pred = count_connected_domains_2D(
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
