# operators.py

import numpy as np
from numba import njit
from typing import Tuple

@njit
def lapl_2D_neumann_along_y(phi: np.ndarray, dx: float) -> np.ndarray:
    """
    Calcola il laplaciano 2D su grigilia uniforme con BC di Neumann lungo y
    e periodicità in x
    """
    ny, nx = phi.shape
    lapl = np.empty_like(phi)
    dx2 = dx * dx
    
    # Bordi y=0 e y=ny-1 (tutti j)
    for j in range(nx):
        j_right = (j + 1) % nx
        j_left = (j - 1) % nx
        
        lapl[0, j]    = (phi[0, j_right]     + phi[0, j_left]     + 2*phi[1, j]     - 4*phi[0, j])     / dx2
        lapl[ny-1, j] = (phi[ny-1, j_right]  + phi[ny-1, j_left]  + 2*phi[ny-2, j]  - 4*phi[ny-1, j])  / dx2
    
    # Punti interni y=1..ny-2
    for i in range(1, ny-1):
        i_up = i - 1
        i_down = i + 1
        
        for j in range(nx):
            j_right = (j + 1) % nx
            j_left = (j - 1) % nx
            lapl[i, j] = (phi[i, j_right] + phi[i, j_left] + phi[i_up, j] + phi[i_down, j] - 4*phi[i, j]) / dx2
    
    return lapl


@njit
def grad_2D_neumann_along_y(phi: np.ndarray, dx:float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calcola il gradiente del campo scalare 2D su griglia uniforme con BC di Neumann lungo y
    e periodicità in x
    """
    
    ny, nx = phi.shape
    grad_x = np.empty_like(phi)
    grad_y = np.empty_like(phi)
    dx2 = 2*dx
    
    # Bordi y=0 e y=ny-1 (tutti j)
    for j in range(nx):
        j_right = (j + 1) % nx
        j_left = (j - 1) % nx
        
        grad_x[0,j]     = (phi[0, j_right]     - phi[0, j_left])    / dx2
        grad_x[ny-1,j]  = (phi[ny-1, j_right]  - phi[ny-1, j_left]) / dx2
        grad_y[0,j]     = 0
        grad_y[ny-1,j]  = 0
    
    # Punti interni
    for i in range(1, ny-1):
        i_up = i - 1
        i_down = i + 1
        
        for j in range(nx):
            j_right = (j + 1) % nx
            j_left = (j - 1) % nx
            
            grad_x[i,j] = (phi[i, j_right]   - phi[i, j_left])   / dx2
            grad_y[i,j] = (phi[i_up, j]      - phi[i_down, j])  / dx2
    
    return grad_x, grad_y