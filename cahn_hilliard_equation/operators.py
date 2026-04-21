# operators.py

import numpy as np
from numba import njit

@njit
def lapl_2D_neumann_along_y(phi: np.ndarray, dx: float, lapl: np.ndarray):
    """
    Calcola il laplaciano 2D su grigilia uniforme con BC di Neumann lungo y
    e periodicità in x usando schema a croce a 4 punti.
    """
    ny, nx = phi.shape
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


@njit
def grad_2D_neumann_along_y(phi: np.ndarray, dx:float, grad_x: np.ndarray, grad_y: np.ndarray):
    """
    Calcola il gradiente del campo scalare 2D su griglia uniforme con BC di Neumann lungo y
    e periodicità in x usando schema delle differenze centrate.
    """
    
    ny, nx = phi.shape
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


@njit
def divergence_2D_neumann_along_y(v_x: np.ndarray, v_y: np.ndarray, dx: float, div: np.ndarray):
    """
    Calcola la divergenza di un campo vettoriale 2D (v_x, v_y) con BC di Neumann lungo y usando
    schema delle differenze centrate su griglia uniforme.
    """
    
    ny, nx = v_x.shape
    dx2 = 2*dx
    
    # Bordi y=0 e y=ny-1 (tutte le j)
    for j in range(nx):
        j_right =  (j+1) % nx
        j_left  =  (j-1) % nx
        div[0, j]     =   (v_x[0, j_right]    - v_x[0, j_left])     / dx2
        div[ny-1, j]  =   (v_x[ny-1, j_right] - v_x[ny-1, j_left])  / dx2
        
    #Punti interni
    for i in range(1, ny-1):
        i_up = i-1
        i_down = i+1
        
        for j in range(nx):
            j_right =  (j+1) % nx
            j_left  =  (j-1) % nx
            
            div[i,j] = (v_x[i, j_right] - v_x[i, j_left] + v_y[i_up, j] - v_y[i_down, j]) / dx2
            