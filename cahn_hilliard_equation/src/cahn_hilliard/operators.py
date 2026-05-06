# operators.py

import numpy as np
from numba import njit


@njit(fastmath=True)
def lapl_2D(
    phi: np.ndarray, 
    dx: float, 
    lapl: np.ndarray, 
    j_left: np.ndarray, 
    j_right: np.ndarray,
    i_up: np.ndarray,
    i_down: np.ndarray
):
    """
    Calcola il laplaciano 2D su grigilia uniforme con PBC in x e y usando schema a croce a 4 punti.
    """
    ny, nx = phi.shape
    dx2_inv = 1.0 / (dx * dx)
    
    for x in range(ny):
        i_u = i_up[x]
        i_d = i_down[x]
        for y in range(nx):
            jl = j_left[y]
            jr = j_right[y]
            lapl[x, y] = (phi[x, jr] + phi[x, jl] + phi[i_u, y] + phi[i_d, y] - 4*phi[x, y]) * dx2_inv


@njit(fastmath=True)
def lapl_2D_neumann_along_y(phi: np.ndarray, dx: float, lapl: np.ndarray, j_left: np.ndarray, j_right: np.ndarray):
    """
    Calcola il laplaciano 2D su grigilia uniforme con BC di Neumann lungo y
    e periodicità in x usando schema a croce a 4 punti.
    """
    ny, nx = phi.shape
    dx2_inv = 1.0 / (dx * dx)
    
    for x in range(ny):
        for y in range(nx):
            jl = j_left[y]
            jr = j_right[y]
            
            # Bordo superiore
            if x == 0:
                lapl[x, y] = (phi[x, jr] + phi[x, jl] + 2*phi[x+1, y] - 4*phi[x, y]) * dx2_inv
            # Bordo inferiore
            elif x == ny - 1:
                lapl[x, y] = (phi[x, jr] + phi[x, jl] + 2*phi[x-1, y] - 4*phi[x, y]) * dx2_inv
            # Punti interni
            else:
                lapl[x, y] = (phi[x, jr] + phi[x, jl] + phi[x-1, y] + phi[x+1, y] - 4*phi[x, y]) * dx2_inv


@njit(fastmath=True)
def grad_2D(
    phi: np.ndarray, 
    dx: float, 
    grad_x: np.ndarray, 
    grad_y: np.ndarray, 
    j_left: np.ndarray, 
    j_right: np.ndarray,
    i_up: np.ndarray,
    i_down: np.ndarray
):
    """
    Calcola il gradiente del campo scalare 2D su griglia uniforme con PBC in x e y
    usando schema delle differenze centrate.
    """
    
    ny, nx = phi.shape
    dx2_inv = 1.0 / (2.0 * dx)
    
    for x in range(ny):
        i_u = i_up[x]
        i_d = i_down[x]
        for y in range(nx):
            jl = j_left[y]
            jr = j_right[y]
            
            grad_x[x, y] = (phi[x, jr] - phi[x, jl]) * dx2_inv
            grad_y[x, y] = (phi[i_u, y] - phi[i_d, y]) * dx2_inv
            
            
@njit(fastmath=True)
def grad_2D_neumann_along_y(phi, dx, grad_x, grad_y, j_left: np.ndarray, j_right: np.ndarray):
    """
    Calcola il gradiente del campo scalare 2D su griglia uniforme con BC di Neumann lungo y
    e periodicità in x usando schema delle differenze centrate.
    """
    
    ny, nx = phi.shape
    dx2_inv = 1.0 / (2.0 * dx)
    
    for x in range(ny):
        for y in range(nx):
            jl = j_left[y]
            jr = j_right[y]
            
            grad_x[x, y] = (phi[x, jr] - phi[x, jl]) * dx2_inv
            
            # Bordi superiore e inferiore
            if x == 0 or x == ny - 1:
                grad_y[x, y] = 0.0
            # Punti interni
            else:
                grad_y[x, y] = (phi[x-1, y] - phi[x+1, y]) * dx2_inv


@njit(fastmath=True)
def div_2D(
    v_x: np.ndarray, 
    v_y: np.ndarray, 
    dx: float, 
    div: np.ndarray, 
    j_left: np.ndarray, 
    j_right: np.ndarray,
    i_up: np.ndarray,
    i_down: np.ndarray
):
    """
    Calcola la divergenza di un campo vettoriale 2D (v_x, v_y) con PBC in x e y usando
    schema delle differenze centrate su griglia uniforme.
    """
    
    ny, nx = v_x.shape
    dx2_inv = 1.0 / (2.0 * dx)
    
    for x in range(ny):
        i_u = i_up[x]
        i_d = i_down[x]
        for y in range(nx):
            jl = j_left[y]
            jr = j_right[y]
            
            div_x = (v_x[x, jr] - v_x[x, jl]) * dx2_inv
            div_y = (v_y[i_u, y] - v_y[i_d, y]) * dx2_inv
            div[x, y] = div_x + div_y
                
                
@njit(fastmath=True)
def divergence_2D_neumann_along_y(v_x, v_y, dx, div, j_left: np.ndarray, j_right: np.ndarray):
    """
    Calcola la divergenza di un campo vettoriale 2D (v_x, v_y) con BC di Neumann lungo y usando
    schema delle differenze centrate su griglia uniforme.
    """
    
    ny, nx = v_x.shape
    dx2_inv = 1.0 / (2.0 * dx)
    
    for x in range(ny):
        for y in range(nx):
            jl = j_left[y]
            jr = j_right[y]
            
            div_x = (v_x[x, jr] - v_x[x, jl]) * dx2_inv
            
            # v_y solo per punti interni
            if x == 0 or x == ny - 1:
                div[x, y] = div_x
            else:
                div_y = (v_y[x-1, y] - v_y[x+1, y]) * dx2_inv
                div[x, y] = div_x + div_y


# @njit(fastmath=True)
# def lapl_3D(
#     phi: np.ndarray, 
#     dx: float, 
#     lapl: np.ndarray 
# ):
#     """
#     Calcola il laplaciano 3D su grigilia uniforme con PBC in x, y, z usando schema a croce a 6 punti.
#     """
#     nz, ny, nx = phi.shape
#     dx2_inv = 1.0 / (dx * dx)
    
#     for z in range(nz):
#         z_up = (z + 1) % nz
#         z_down = (z - 1) % nz
#         for y in range(ny):
#             y_front = (y - 1) % ny
#             y_back = (y + 1) % ny
#             for x in range(nx):
#                 x_left = (x - 1) % nx
#                 x_right = (x + 1) % nx
                
#                 lapl[z, y, x] = (phi[z, y, x_left] + phi[z, y, x_right]
#                                  + phi[z, y_back, x] + phi[z, y_front, x]
#                                  + phi[z_up, y, x] + phi[z_down, y, x]
#                                  - 6*phi[z, y, x]) * dx2_inv)
                                
                                
@njit(fastmath=True)
def grad_3D(
    phi: np.ndarray, 
    dx: float, 
    grad_x: np.ndarray, 
    grad_y: np.ndarray,
    grad_z: np.ndarray
):
    """
    Calcola il gradiente del campo scalare 3D su griglia uniforme con PBC in x, y, z
    usando schema delle differenze centrate.
    """
    
    nz, ny, nx = phi.shape
    dx2_inv = 1.0 / (2.0 * dx)
    
        
    for z in range(nz):
        z_up = (z + 1) % nz
        z_down = (z - 1) % nz
        for y in range(ny):
            y_front = (y - 1) % ny
            y_back = (y + 1) % ny
            for x in range(nx):
                x_left = (x - 1) % nx
                x_right = (x + 1) % nx
                
                grad_x[z,y,x] = (phi[z, y, x_right] - phi[z, y, x_left]) * dx2_inv
                grad_y[z,y,x] = (phi[z, y_back, x] - phi[z, y_front, x]) * dx2_inv
                grad_z[z,y,x] = (phi[z_up, y, x] - phi[z_down, y, x]) * dx2_inv
                

@njit(fastmath=True)
def div_3D(
    v_x: np.ndarray, 
    v_y: np.ndarray,
    v_z: np.ndarray, 
    dx: float, 
    div: np.ndarray
):
    """
    Calcola la divergenza di un campo vettoriale 3D (v_x, v_y, v_z) con PBC in x, y, z usando
    schema delle differenze centrate su griglia uniforme.
    """
    
    nz, ny, nx = v_x.shape
    dx2_inv = 1.0 / (2.0 * dx)
         
    for z in range(nz):
        z_up = (z + 1) % nz
        z_down = (z - 1) % nz
        for y in range(ny):
            y_front = (y - 1) % ny
            y_back = (y + 1) % ny
            for x in range(nx):
                x_left = (x - 1) % nx
                x_right = (x + 1) % nx
                
                div_x = (v_x[z, y, x_right] - v_x[z, y, x_left]) * dx2_inv
                div_y = (v_y[z, y_back, x] - v_y[z, y_front, x]) * dx2_inv
                div_z = (v_z[z_up, y, x] - v_z[z_down, y, x]) * dx2_inv
                div[z, y, x] = div_x + div_y + div_z
                