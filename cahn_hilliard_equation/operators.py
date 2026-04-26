# operators.py

import numpy as np
from numba import njit

@njit(fastmath=True)
def lapl_2D_neumann_along_y(phi: np.ndarray, dx: float, lapl: np.ndarray, j_left: np.ndarray, j_right: np.ndarray):
    """
    Calcola il laplaciano 2D su grigilia uniforme con BC di Neumann lungo y
    e periodicità in x usando schema a croce a 4 punti.
    """
    ny, nx = phi.shape
    dx2_inv = 1.0 / (dx * dx)
    
    for i in range(ny):
        for j in range(nx):
            jl = j_left[j]
            jr = j_right[j]
            
            # Bordo superiore
            if i == 0:
                lapl[i, j] = (phi[i, jr] + phi[i, jl] + 2*phi[i+1, j] - 4*phi[i, j]) * dx2_inv
            # Bordo inferiore
            elif i == ny - 1:
                lapl[i, j] = (phi[i, jr] + phi[i, jl] + 2*phi[i-1, j] - 4*phi[i, j]) * dx2_inv
            # Punti interni
            else:
                lapl[i, j] = (phi[i, jr] + phi[i, jl] + phi[i-1, j] + phi[i+1, j] - 4*phi[i, j]) * dx2_inv


@njit(fastmath=True)
def grad_2D_neumann_along_y(phi, dx, grad_x, grad_y, j_left: np.ndarray, j_right: np.ndarray):
    """
    Calcola il gradiente del campo scalare 2D su griglia uniforme con BC di Neumann lungo y
    e periodicità in x usando schema delle differenze centrate.
    """
    
    ny, nx = phi.shape
    dx2_inv = 1.0 / (2.0 * dx)
    
    for i in range(ny):
        for j in range(nx):
            jl = j_left[j]
            jr = j_right[j]
            
            grad_x[i, j] = (phi[i, jr] - phi[i, jl]) * dx2_inv
            
            # Bordi superiore e inferiore
            if i == 0 or i == ny - 1:
                grad_y[i, j] = 0.0
            # Punti interni
            else:
                grad_y[i, j] = (phi[i-1, j] - phi[i+1, j]) * dx2_inv


@njit(fastmath=True)
def divergence_2D_neumann_along_y(v_x, v_y, dx, div, j_left: np.ndarray, j_right: np.ndarray):
    """
    Calcola la divergenza di un campo vettoriale 2D (v_x, v_y) con BC di Neumann lungo y usando
    schema delle differenze centrate su griglia uniforme.
    """
    
    ny, nx = v_x.shape
    dx2_inv = 1.0 / (2.0 * dx)
    
    for i in range(ny):
        for j in range(nx):
            jl = j_left[j]
            jr = j_right[j]
            
            div_x = (v_x[i, jr] - v_x[i, jl]) * dx2_inv
            
            # v_y solo per punti interni
            if i == 0 or i == ny - 1:
                div[i, j] = div_x
            else:
                div_y = (v_y[i-1, j] - v_y[i+1, j]) * dx2_inv
                div[i, j] = div_x + div_y
            