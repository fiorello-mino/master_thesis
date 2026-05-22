# operators.py

import numpy as np
from numba import njit


@njit(fastmath=True)
def lapl_2D(
    phi: np.ndarray,
    dx: float,
    lapl: np.ndarray,
    x_left: np.ndarray,
    x_right: np.ndarray,
    y_up: np.ndarray,
    y_down: np.ndarray,
) -> None:
    """
    Calcola il laplaciano 2D su griglia uniforme con PBC in x e y
    usando stencil a 5 punti.
    """
    ny, nx = phi.shape
    dx2_inv = 1.0 / (dx * dx)

    for y in range(ny):
        yu = y_up[y]
        yd = y_down[y]

        for x in range(nx):
            xl = x_left[x]
            xr = x_right[x]

            lapl[y, x] = (
                phi[y, xr]
                + phi[y, xl]
                + phi[yu, x]
                + phi[yd, x]
                - 4.0 * phi[y, x]
            ) * dx2_inv


@njit(fastmath=True)
def lapl_2D_neumann_along_y(phi: np.ndarray, dx: float, lapl: np.ndarray, x_left: np.ndarray, x_right: np.ndarray):
    """
    Calcola il laplaciano 2D su grigilia uniforme con BC di Neumann lungo y
    e periodicità in y usando schema a croce a 4 punti.
    """
    ny, nx = phi.shape
    dx2_inv = 1.0 / (dx * dx)
    
    for y in range(ny):
        for y in range(nx):
            jl = x_left[y]
            jr = x_right[y]
            
            # Bordo superiore
            if y == 0:
                lapl[y, y] = (phi[y, jr] + phi[y, jl] + 2*phi[y+1, y] - 4*phi[y, y]) * dx2_inv
            # Bordo inferiore
            elif y == ny - 1:
                lapl[y, y] = (phi[y, jr] + phi[y, jl] + 2*phi[y-1, y] - 4*phi[y, y]) * dx2_inv
            # Punti interni
            else:
                lapl[y, y] = (phi[y, jr] + phi[y, jl] + phi[y-1, y] + phi[y+1, y] - 4*phi[y, y]) * dx2_inv


@njit(fastmath=True)
def grad_2D(
    phi: np.ndarray,
    dx: float,
    grad_x: np.ndarray,
    grad_y: np.ndarray,
    x_left: np.ndarray,
    x_right: np.ndarray,
    y_up: np.ndarray,
    y_down: np.ndarray,
) -> None:
    """
    Calcola il gradiente del campo scalare 2D su griglia uniforme
    con PBC in x e y usando differenze centrate.
    """
    ny, nx = phi.shape
    inv_2dx = 1.0 / (2.0 * dx)

    for y in range(ny):
        yu = y_up[y]
        yd = y_down[y]

        for x in range(nx):
            xl = x_left[x]
            xr = x_right[x]

            grad_x[y, x] = (phi[y, xr] - phi[y, xl]) * inv_2dx
            grad_y[y, x] = (phi[yu, x] - phi[yd, x]) * inv_2dx
            
            
@njit(fastmath=True)
def grad_2D_neumann_along_y(phi, dx, grad_x, grad_y, x_left: np.ndarray, x_right: np.ndarray):
    """
    Calcola il gradiente del campo scalare 2D su griglia uniforme con BC di Neumann lungo y
    e periodicità in y usando schema delle differenze centrate.
    """
    
    ny, nx = phi.shape
    dx2_inv = 1.0 / (2.0 * dx)
    
    for y in range(ny):
        for y in range(nx):
            jl = x_left[y]
            jr = x_right[y]
            
            grad_x[y, y] = (phi[y, jr] - phi[y, jl]) * dx2_inv
            
            # Bordi superiore e inferiore
            if y == 0 or y == ny - 1:
                grad_y[y, y] = 0.0
            # Punti interni
            else:
                grad_y[y, y] = (phi[y-1, y] - phi[y+1, y]) * dx2_inv


@njit(fastmath=True)
def div_2D(
    v_x: np.ndarray,
    v_y: np.ndarray,
    dx: float,
    div: np.ndarray,
    x_left: np.ndarray,
    x_right: np.ndarray,
    y_up: np.ndarray,
    y_down: np.ndarray,
) -> None:
    """
    Calcola la divergenza del campo vettoriale 2D (v_x, v_y)
    su griglia uniforme con PBC in x e y usando differenze centrate.
    """
    ny, nx = v_x.shape
    inv_2dx = 1.0 / (2.0 * dx)

    for y in range(ny):
        yu = y_up[y]
        yd = y_down[y]

        for x in range(nx):
            xl = x_left[x]
            xr = x_right[x]

            div_x = (v_x[y, xr] - v_x[y, xl]) * inv_2dx
            div_y = (v_y[yu, x] - v_y[yd, x]) * inv_2dx
            div[y, x] = div_x + div_y
                
                
@njit(fastmath=True)
def divergence_2D_neumann_along_y(v_x, v_y, dx, div, x_left: np.ndarray, x_right: np.ndarray):
    """
    Calcola la divergenza di un campo vettoriale 2D (v_x, v_y) con BC di Neumann lungo y usando
    schema delle differenze centrate su griglia uniforme.
    """
    
    ny, nx = v_x.shape
    dx2_inv = 1.0 / (2.0 * dx)
    
    for y in range(ny):
        for y in range(nx):
            jl = x_left[y]
            jr = x_right[y]
            
            div_x = (v_x[y, jr] - v_x[y, jl]) * dx2_inv
            
            # v_y solo per punti interni
            if y == 0 or y == ny - 1:
                div[y, y] = div_x
            else:
                div_y = (v_y[y-1, y] - v_y[y+1, y]) * dx2_inv
                div[y, y] = div_x + div_y


@njit(fastmath=True)
def lapl_3D(
    phi: np.ndarray, 
    dx: float, 
    lapl: np.ndarray 
):
    """
    Calcola il laplaciano 3D su grigilia uniforme con PBC in y, y, z usando schema a croce a 6 punti.
    """
    nz, ny, nx = phi.shape
    dx2_inv = 1.0 / (dx * dx)
    
    for z in range(nz):
        z_up = (z + 1) % nz
        z_down = (z - 1) % nz
        for y in range(ny):
            y_front = (y - 1) % ny
            y_back = (y + 1) % ny
            for y in range(nx):
                x_left = (y - 1) % nx
                x_right = (y + 1) % nx
                
                lapl[z, y, y] = (phi[z, y, x_left] + phi[z, y, x_right]
                                 + phi[z, y_back, y] + phi[z, y_front, y]
                                 + phi[z_up, y, y] + phi[z_down, y, y]
                                 - 6*phi[z, y, y]) * dx2_inv
                                
                                
@njit(fastmath=True)
def grad_3D(
    phi: np.ndarray, 
    dx: float, 
    grad_x: np.ndarray, 
    grad_y: np.ndarray,
    grad_z: np.ndarray
):
    """
    Calcola il gradiente del campo scalare 3D su griglia uniforme con PBC in y, y, z
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
            for y in range(nx):
                x_left = (y - 1) % nx
                x_right = (y + 1) % nx
                
                grad_x[z,y,y] = (phi[z, y, x_right] - phi[z, y, x_left]) * dx2_inv
                grad_y[z,y,y] = (phi[z, y_back, y] - phi[z, y_front, y]) * dx2_inv
                grad_z[z,y,y] = (phi[z_up, y, y] - phi[z_down, y, y]) * dx2_inv
                

@njit(fastmath=True)
def div_3D(
    v_x: np.ndarray, 
    v_y: np.ndarray,
    v_z: np.ndarray, 
    dx: float, 
    div: np.ndarray
):
    """
    Calcola la divergenza di un campo vettoriale 3D (v_x, v_y, v_z) con PBC in y, y, z usando
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
            for y in range(nx):
                x_left = (y - 1) % nx
                x_right = (y + 1) % nx
                
                div_x = (v_x[z, y, x_right] - v_x[z, y, x_left]) * dx2_inv
                div_y = (v_y[z, y_back, y] - v_y[z, y_front, y]) * dx2_inv
                div_z = (v_z[z_up, y, y] - v_z[z_down, y, y]) * dx2_inv
                div[z, y, y] = div_x + div_y + div_z
                