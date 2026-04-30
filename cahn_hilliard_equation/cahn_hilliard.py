# cahn_hilliard.py

from numba import njit
import numpy as np
import os
import matplotlib.pyplot as plt
import shutil
from free_energy import w_field, dw_dphi, mu_field, weighted_mu_field, M_field
from operators import (
    lapl_2D,
    grad_2D,
    div_2D,
    lapl_2D_neumann_along_y, 
    grad_2D_neumann_along_y, 
    divergence_2D_neumann_along_y
)


@njit
def evolve_cahn_hilliard_const_mobility(
    phi_init: np.ndarray, 
    dt: float, 
    n_steps: int, 
    epsilon: float, 
    M: float, 
    dx: float,
    j_left: np.ndarray,
    j_right: np.ndarray
) -> np.ndarray:
    
    """
    Evolve il campo iniziale phi_init secondo l’equazione di Cahn-Hilliard
    con mobilità costante M per n_steps passi temporali di ampiezza dt.
    """
    ny, nx = phi_init.shape
    
    # Allocazioni
    phi = np.copy(phi_init)
    lapl_phi = np.empty_like(phi_init)
    w_prime = np.empty_like(phi_init)
    mu = np.empty_like(phi_init)
    lapl_mu = np.empty_like(phi)
    
    for step in range(n_steps):
        
        # Calcolo mu
        lapl_2D_neumann_along_y(phi, dx, lapl_phi, j_left, j_right)
        dw_dphi(phi, epsilon, w_prime)
        mu_field(lapl_phi, w_prime, epsilon, mu)
        
        # Step esplicito per phi
        lapl_2D_neumann_along_y(mu, dx, lapl_mu, j_left, j_right)
        phi[:] += M * dt * lapl_mu
    
    return phi


@njit(fastmath=True)
def evolve_cahn_hilliard_surf_mobility(
    phi: np.ndarray,
    dt: float,
    n_steps: int,
    epsilon: float,
    M0: float,
    dx: float,
    lapl_phi: np.ndarray,
    w_prime: np.ndarray,
    mu: np.ndarray,
    grad_mu_x: np.ndarray,
    grad_mu_y: np.ndarray,
    mobility: np.ndarray,
    J_x: np.ndarray,
    J_y: np.ndarray,
    div_J: np.ndarray,
    j_left: np.ndarray,
    j_right: np.ndarray,
    i_up: np.ndarray,
    i_down: np.ndarray
):
    
    """
    Evolve il campo iniziale phi_init secondo l’equazione di Cahn-Hilliard
    con mobilità variabile per n_steps passi temporali di ampiezza dt.
    """
    ny, nx = phi.shape
    
    for step in range(n_steps):
        
        # Calcolo mu
        lapl_2D(phi, dx, lapl_phi, j_left, j_right, i_up, i_down)
        dw_dphi(phi, epsilon, w_prime)
        mu_field(lapl_phi, w_prime, epsilon, mu)
        
        # Step esplicito per phi
        grad_2D(mu, dx, grad_mu_x, grad_mu_y, j_left, j_right, i_up, i_down)
        M_field(phi, M0, epsilon, mobility)
        
        for i in range(ny):
            for j in range(nx):
                J_x[i, j] = mobility[i, j] * grad_mu_x[i, j]
                J_y[i, j] = mobility[i, j] * grad_mu_y[i, j]
        
        div_2D(J_x, J_y, dx, div_J, j_left, j_right, i_up, i_down) 
        for i in range(ny):
            for j in range(nx):
                phi[i, j] += dt * div_J[i, j]   


@njit(fastmath=True)
def evolve_cahn_hilliard_surf_mobility_g_phi(
    phi,
    dt,
    n_steps,
    epsilon,
    M0,
    dx,
    lapl_phi,
    w_prime,
    mu_weighted,
    grad_mu_x,
    grad_mu_y,
    mobility,
    J_x,
    J_y,
    div_J,
    j_left,
    j_right
):
    
    """
    Evolve il campo iniziale phi_init secondo l’equazione di Cahn-Hilliard
    con mobilità variabile per n_steps passi temporali di ampiezza dt.
    """
    ny, nx = phi.shape
    
    for step in range(n_steps):
        
        # Calcolo mu pesato g(phi) * mu
        lapl_2D_neumann_along_y(phi, dx, lapl_phi, j_left, j_right)
        dw_dphi(phi, epsilon, w_prime)
        weighted_mu_field(lapl_phi, w_prime, phi, epsilon, mu_weighted)
        #mu_field(lapl_phi, w_prime, phi, epsilon, mu_weighted)
        
        # Step esplicito per phi
        grad_2D_neumann_along_y(mu_weighted, dx, grad_mu_x, grad_mu_y, j_left, j_right)
        M_field(phi, M0, epsilon, mobility)
        
        for i in range(ny):
            for j in range(nx):
                J_x[i, j] = mobility[i, j] * grad_mu_x[i, j]
                J_y[i, j] = mobility[i, j] * grad_mu_y[i, j]
        
        divergence_2D_neumann_along_y(J_x, J_y, dx, div_J, j_left, j_right) 
        for i in range(ny):
            for j in range(nx):
                phi[i, j] += dt * div_J[i, j]   

def evolve_ch_const_mob_with_snapshots(
    phi_init: np.ndarray,
    dt: float,
    n_steps: int,
    steps_per_save: int,
    epsilon: float,
    M: float,
    dx: float,
    out_dir,
    live_plot: bool = True,
    cmap: str = "RdBu_r"
) -> np.ndarray:
    """
    Evolve Cahn-Hilliard con mobilità costante e salva il campo phi in formato .npy
    """

    # se esiste, svuota; se non esiste, crea
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)  
    os.makedirs(out_dir, exist_ok=True)

    
    phi = phi_init.copy()
    
    times = []  # lista dei tempi corrispondenti agli snapshot
    idx = 0
    filename = str(idx).zfill(4)
    
    # snapshot iniziale t=0
    # Plot live condizione iniziale
    if live_plot:
        plt.ion()
        fig, ax = plt.subplots(figsize=(6, 6))
        im = ax.imshow(phi, cmap=cmap, origin="lower")
        cbar = plt.colorbar(im, ax=ax)
        title = ax.set_title("step = 0, t = 0.0")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        plt.show(block=False)
    
    np.save(f"{out_dir}/{filename}", phi)
    times.append(0.0)  # tempo iniziale
    
    block_size = steps_per_save
    
    # Alla fine di ogni blocco di steps salva phi.npy
    for block in range(0, n_steps, block_size):
        n_block = min(block_size, n_steps - block)
        phi = evolve_cahn_hilliard_const_mobility(phi, dt, n_block, epsilon, M, dx, j_left, j_right)
        step = block + n_block
        
        idx += 1
        filename = str(idx).zfill(4)
        np.save(f"{out_dir}/{filename}", phi)
        
        # tempo fisico di questo snapshot
        t = (block + n_block) * dt
        times.append(t)
        
        # Plot live
        if live_plot:
            im.set_data(phi)
            title.set_text(f"step = {step}, t = {t:.4e}")
            fig.canvas.draw_idle()
            fig.canvas.flush_events()
            plt.pause(0.001)
    
    # salva i tempi in un file separato
    np.save(f"{out_dir}/times.npy", np.array(times, dtype=float))
    
    if live_plot:
        plt.pause(2)
        plt.ioff()
        plt.close(fig)
    
    return phi        
    

def evolve_ch_surf_mob_with_snapshots(
    phi_init: np.ndarray,
    phi: np.ndarray,
    dt: float,
    n_steps: int,
    steps_per_save: int,
    epsilon: float,
    M0: float,
    dx: float,
    out_dir,
    live_plot: bool = True,
    cmap: str = "RdBu_r",
):
    """
    Evolve Cahn-Hilliard con mobilità superficiale e salva il campo phi in formato .npy
    """

    # se esiste, svuota; se non esiste, crea
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)    # elimina tutta la cartella
    os.makedirs(out_dir, exist_ok=True)

    # inizializza phi con phi_init
    phi[:] = phi_init
    
    times = []  
    idx = 0
    filename = str(idx).zfill(4)
    
    if live_plot:
        plt.ion()
        fig, ax = plt.subplots(figsize=(8, 8))
        im = ax.imshow(phi, cmap=cmap, origin="lower", vmin = 0,vmax = 1)
        cbar = plt.colorbar(im, ax=ax)
        title = ax.set_title("step = 0, t = 0.0")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        plt.show(block=False)

    np.save(f"{out_dir}/{filename}", phi)
    times.append(0.0)

    block_size = steps_per_save
    
    # Alla fine di ogni blocco di steps salva phi.npy
    lapl_phi    = np.empty_like(phi)
    w_prime     = np.empty_like(phi)
    mu          = np.empty_like(phi)
    grad_mu_x   = np.empty_like(phi)
    grad_mu_y   = np.empty_like(phi)
    mobility    = np.empty_like(phi)
    J_x         = np.empty_like(phi)
    J_y         = np.empty_like(phi)
    div_J       = np.empty_like(phi)
    
    ny, nx = phi_init.shape
    j_left = np.empty(nx, dtype=np.int64)
    j_right = np.empty(nx, dtype=np.int64)
    i_up = np.empty(ny, dtype=np.int64)
    i_down = np.empty(ny, dtype=np.int64)
    
    for j in range(nx):
        j_left[j] = (j - 1) % nx
        j_right[j] = (j + 1) % nx
    
    for i in range(ny):
        i_up[i] = (i - 1) % ny
        i_down[i] = (i + 1) % ny
    
    for block in range(0, n_steps, block_size):
        n_block = min(block_size, n_steps - block)
        
        # evolve modifica phi in-place
        evolve_cahn_hilliard_surf_mobility(
        phi, dt, n_block, epsilon, M0, dx,
        lapl_phi, w_prime, mu,
        grad_mu_x, grad_mu_y, mobility,
        J_x, J_y, div_J, j_left, j_right, i_up, i_down)
        
        step = block + n_block
        idx += 1
        filename = str(idx).zfill(4)
        np.save(f"{out_dir}/{filename}", phi)
        
        # tempo fisico di questo snapshot
        t = (block + n_block) * dt
        times.append(t)
        
        # Plot live
        if live_plot:
            im.set_data(phi)
            title.set_text(f"step = {step}, t = {t:.4e}")
            fig.canvas.draw_idle()
            fig.canvas.flush_events()
            plt.pause(0.001)
    
    # salva i tempi in un file separato
    #np.save(f"{out_dir}/times.npy", np.array(times, dtype=float))
    
    if live_plot:
        plt.pause(2)
        plt.ioff()
        plt.close(fig)
    