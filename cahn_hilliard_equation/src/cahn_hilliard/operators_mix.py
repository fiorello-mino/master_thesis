# operators_mix.py

import numpy as np
from numba import njit


@njit(fastmath=True)
def lapl_2D(
    phi: np.ndarray,
    dx: float,
    dy: float,
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
    dy2_inv = 1.0 / (dy * dy)

    for y in range(ny):
        yu = y_up[y]
        yd = y_down[y]

        for x in range(nx):
            xl = x_left[x]
            xr = x_right[x]

            lapl[y, x] = (
                (phi[y, xr] - 2.0 * phi[y, x] + phi[y, xl]) * dx2_inv
                + (phi[yu, x] - 2.0 * phi[y, x] + phi[yd, x]) * dy2_inv
            )
            
            
@njit(fastmath=True)
def grad_2D(
    phi: np.ndarray,
    dx: float,
    dy: float,
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
    inv_2dy = 1.0 / (2.0 * dy)

    for y in range(ny):
        yu = y_up[y]
        yd = y_down[y]

        for x in range(nx):
            xl = x_left[x]
            xr = x_right[x]

            grad_x[y, x] = (phi[y, xr] - phi[y, xl]) * inv_2dx
            grad_y[y, x] = (phi[yu, x] - phi[yd, x]) * inv_2dy
            
            
@njit(fastmath=True)
def div_2D(
    v_x: np.ndarray,
    v_y: np.ndarray,
    dx: float,
    dy: float,
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
    inv_2dy = 1.0 / (2.0 * dy)

    for y in range(ny):
        yu = y_up[y]
        yd = y_down[y]

        for x in range(nx):
            xl = x_left[x]
            xr = x_right[x]

            div_x = (v_x[y, xr] - v_x[y, xl]) * inv_2dx
            div_y = (v_y[yu, x] - v_y[yd, x]) * inv_2dy
            div[y, x] = div_x + div_y