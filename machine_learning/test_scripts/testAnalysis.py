from pathlib import Path
import os
import sys
import numpy as np
from numba import njit


# ========== SCRIPT VARIABLES ==========
TEST_DIR = Path("/scratch/fiorello/test3D/square/E1e-1")
ERRORS_FILE = TEST_DIR / "errors.txt"
# --------------------------------------


# ========== PARAMETERS ===========
DX = 0.025
DY = 0.025
DZ = 0.025
EPS = 0.1
# =================================


def time_key(entry: os.DirEntry):
    name = entry.name  # es. surf_0.000000.npy
    numero_str = name.removeprefix("surf_").removesuffix(".npy")
    return float(numero_str)


@njit(fastmath=True)
def w_field_3D(phi: np.ndarray, epsilon: float, w: np.ndarray):
    nx, ny, nz = phi.shape
    factor = 18.0 / epsilon
            
    for x in range(nx):
        for y in range(ny):
            for z in range(nz):
                phi_xyz = phi[x, y, z]
                w[x, y, z] = factor * phi_xyz * phi_xyz * (1 - phi_xyz) * (1 - phi_xyz)


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
    
    # --- gradiente in x ---
    for y in range(ny):
        for z in range(nz):
            for x in range(1, nx - 1):
                grad_x[x, y, z] = (phi[x + 1, y, z] - phi[x - 1, y, z]) * dx2_inv
            
            grad_x[0, y, z] = (phi[1, y, z] - phi[0, y, z]) * dx2_inv
            grad_x[nx - 1, y, z] = (phi[nx - 1, y, z] - phi[nx - 2, y, z]) * dx2_inv

    # --- gradiente in y ---
    for x in range(nx):
        for z in range(nz):
            for y in range(1, ny - 1):
                grad_y[x, y, z] = (phi[x, y + 1, z] - phi[x, y - 1, z]) * dy2_inv
            
            grad_y[x, 0, z] = (phi[x, 1, z] - phi[x, 0, z]) * dy2_inv
            grad_y[x, ny - 1, z] = (phi[x, ny - 1, z] - phi[x, ny - 2, z]) * dy2_inv

    # --- gradiente in z ---
    for x in range(nx):
        for y in range(ny):
            for z in range(1, nz - 1):
                grad_z[x, y, z] = (phi[x, y, z + 1] - phi[x, y, z - 1]) * dz2_inv
            
            grad_z[x, y, 0] = (phi[x, y, 1] - phi[x, y, 0]) * dz2_inv
            grad_z[x, y, nz - 1] = (phi[x, y, nz - 1] - phi[x, y, nz - 2]) * dz2_inv
            
            
@njit(fastmath=True)
def compute_e_3D(phi: np.ndarray, epsilon: float, dx: float, dy: float, dz: float) -> float:
    nx, ny, nz = phi.shape
    eps2 = 0.5 * epsilon
    
    w_local = np.empty_like(phi)
    gx = np.empty_like(phi)
    gy = np.empty_like(phi)
    gz = np.empty_like(phi)
    
    w_field_3D(phi, epsilon, w_local)
    grad_3D_neumann(phi, dx, dy, dz, gx, gy, gz)
    
    total_E = 0.0
    for x in range(nx):
        for y in range(ny):
            for z in range(nz):
                grad2 = (
                    gx[x, y, z] * gx[x, y, z] +
                    gy[x, y, z] * gy[x, y, z] +
                    gz[x, y, z] * gz[x, y, z]
                )
                f_xyz = w_local[x, y, z] + eps2 * grad2
                total_E += f_xyz
    
    return total_E * dx * dy * dz
    


def main() -> None:
    
    # ----- prendo le directory delle simulazioni -----
    if not TEST_DIR.is_dir():
        sys.exit(f"{TEST_DIR} directory non trovata.")
        
    sim_folders = [ d for d in TEST_DIR.iterdir() if d.is_dir() ]
    if len(sim_folders) == 0:
        sys.exit("0 simulazioni trovate.")
    print(f"Trovate {len(sim_folders)} simulazioni.")
    
    # ----- per ogni simulazione calcolo energia e massa -----
    for sim in sim_folders:
        evo_file = sim / "evo.txt"
        if not evo_file.is_file():
            sys.exit(f"File evo non trovato in {sim}")
        
        with open(evo_file, "r") as f:
            stats = f.readlines()
        
        # ----- carico npy pred -----
        pred_dir = sim / "pred_npy"
        if not pred_dir.is_dir():
            sys.exit(f"Cartella pred_npy non trovata in {sim}")
        
        frames_pred = [
            f for f in pred_dir.iterdir()
            if f.is_file() and f.name.endswith(".npy")
        ]
        if len(frames_pred) == 0:
            sys.exit(f"0 frames pred nella cartella {pred_dir}")
            
        frames_pred.sort(key=lambda p: float(p.name.removeprefix("surf_").removesuffix(".npy")))
        phi_pred = [np.load(str(f)) for f in frames_pred]
        
        # ----- carico npy true -----
        true_dir = sim / "true_npy"
        if not true_dir.is_dir():
            sys.exit(f"Cartella true_npy non trovata in {sim}")
        
        frames_true = [
            f for f in true_dir.iterdir()
            if f.is_file() and f.name.endswith(".npy")
        ]
        if len(frames_true) == 0:
            sys.exit(f"0 frames true nella cartella {true_dir}")
            
        frames_true.sort(key=lambda p: float(p.name.removeprefix("surf_").removesuffix(".npy")))
        phi_true = [np.load(str(f)) for f in frames_true]
        
        # ----- controllo coerenza -----
        n_frames = len(phi_pred)
        if len(phi_true) != n_frames:
            sys.exit(f"Mismatch numero frame pred/true in {sim}")
        
        # ----- calcolo energia e massa -----
        e_pred = np.zeros(n_frames)
        e_true = np.zeros(n_frames)
        m_pred = np.zeros(n_frames)
        m_true = np.zeros(n_frames)

        for i in range(n_frames):
            e_pred[i] = compute_e_3D(phi_pred[i], EPS, DX, DY, DZ)
            e_true[i] = compute_e_3D(phi_true[i], EPS, DX, DY, DZ)
            m_pred[i] = np.sum(phi_pred[i]) * DX * DY * DZ
            m_true[i] = np.sum(phi_true[i]) * DX * DY * DZ
        
        # ----- scrivo su file -----
        # riga 0: commento
        stats[0] = stats[0].rstrip("\n")
        stats[0] += "\t11: E_pred\t12: E_true\t13: mass_pred\t14: mass_true\n"
        
        # righe 1..N: unisco i nuovi dati alla riga esistente
        for i in range(1, len(stats)):
            stats[i] = stats[i].rstrip("\n")
            stats[i] += (
                f"\t{e_pred[i-1]:.6e}"
                f"\t{e_true[i-1]:.6e}"
                f"\t{m_pred[i-1]:.6e}"
                f"\t{m_true[i-1]:.6e}\n"
            )
        
        with open(evo_file, "w") as f:
            f.writelines(stats)


if __name__ == "__main__":
    main()