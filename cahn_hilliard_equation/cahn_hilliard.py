# cahn_hilliard.py

from numba import njit
import numpy as np
import os
import shutil
from free_energy import w_field, dw_dphi, mu_field
from operators import lapl_2D_neumann_along_y


@njit
def evolve_cahn_hilliard_const_mobility(
    phi_init: np.ndarray, 
    dt: float, 
    n_steps: int, 
    epsilon: float, 
    M: float, 
    dx: float
) -> np.ndarray:
    
    """
    Evolve il campo iniziale phi_init secondo l’equazione di Cahn-Hilliard
    con mobilità costante M per n_steps passi temporali di ampiezza dt.
    """
    ny, nx = phi_init.shape
    
    # Allocazioni
    phi = np.copy(phi_init)
    phi_new = np.empty_like(phi_init)
    lapl_phi = np.empty_like(phi_init)
    w_prime = np.empty_like(phi_init)
    mu = np.empty_like(phi_init)
    lapl_mu = np.empty_like(phi)
    
    for step in range(n_steps):
        
        # Calcolo mu
        lapl_phi[:] = lapl_2D_neumann_along_y(phi, dx)
        w_prime[:] = dw_dphi(phi, epsilon)
        mu[:] = mu_field(lapl_phi, w_prime, epsilon)
        
        # Step esplicito per phi
        lapl_mu[:] = lapl_2D_neumann_along_y(mu, dx)
        phi[:] += M * dt * lapl_mu
    
    return phi


# Evolve Cahn-Hilliard con mobilità costante e salva il campo phi in formato .npy
def evolve_ch_const_mob_with_snapshots(
    phi_init: np.ndarray,
    dt: float,
    n_steps: int,
    steps_per_save: int,
    epsilon: float,
    M: float,
    dx: float,
    out_dir: str = "snapshots"
) -> np.ndarray:
    
    out_dir = "snapshots"

    # se esiste, svuota; se non esiste, crea
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)    # elimina tutta la cartella
    os.makedirs(out_dir, exist_ok=True)

    
    phi = phi_init.copy()
    
    times = []  # lista dei tempi corrispondenti agli snapshot
    idx = 0
    filename = str(idx).zfill(4)
    
    # snapshot iniziale t=0
    np.save(f"{out_dir}/{filename}", phi)
    times.append(0.0)  # tempo iniziale
    
    block_size = steps_per_save
    
    # Alla fine di ogni blocco di steps salva phi.npy
    for block in range(0, n_steps, block_size):
        n_block = min(block_size, n_steps - block)
        phi = evolve_cahn_hilliard_const_mobility(phi, dt, n_block, epsilon, M, dx)
        step = block + n_block
        
        idx += 1
        filename = str(idx).zfill(4)
        np.save(f"{out_dir}/{filename}", phi)
        
        # tempo fisico di questo snapshot
        t = (block + n_block) * dt
        times.append(t)
    
    # salva i tempi in un file separato
    np.save(f"{out_dir}/times.npy", np.array(times, dtype=float))
    
    return phi        
    