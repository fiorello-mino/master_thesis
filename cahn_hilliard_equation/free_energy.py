# free_energy.py

import numpy as np
from numba import njit
from operators import grad_2D_neumann_along_y


@njit(fastmath=True)
def w_field(phi: np.ndarray, epsilon: float, w: np.ndarray):
    """
    Calcola il potenziale doppia buca
    """
    ny, nx = phi.shape
    factor = 18.0 / epsilon
    
    for i in range(ny):
        for j in range(nx):
            phi_ij = phi[i, j]
            w[i, j] = factor * phi_ij * phi_ij * (1 - phi_ij) * (1 - phi_ij)

    
@njit(fastmath=True)
def dw_dphi(phi: np.ndarray, epsilon: float, w_prime:np.ndarray):
    """
    Calcola la derivata del potenziale doppia buca
    """
    ny, nx = phi.shape
    factor = 36.0 / epsilon
    
    for i in range(ny):
        for j in range(nx):
            phi_ij = phi[i,j]
            phi2 = phi_ij * phi_ij
            w_prime[i,j] = factor * phi_ij * (1.0 + 2.0 * phi2 - 3.0 * phi_ij)


@njit(fastmath=True)
def mu_field(lapl_phi: np.ndarray, dw_dphi: np.ndarray, epsilon: float, mu: np.ndarray):
    """
    Calcola il potenziale chimico
    """
    ny, nx = dw_dphi.shape
    
    for i in range(ny):
        for j in range(nx):
            mu[i,j] = - epsilon * lapl_phi[i,j] + dw_dphi[i,j]


@njit(fastmath=True)
def weighted_mu_field(
    lapl_phi: np.ndarray, 
    dw_dphi: np.ndarray, 
    phi: np.ndarray, 
    epsilon: float,
    mu_field: np.ndarray
):
    """
    Calcola g(phi) * mu, con mu = -epsilon*lapl_phi + dw_dphi
    e g(phi) = 6*abs(phi)*abs(1-phi)
    """
    ny, nx = phi.shape
    eps_neg = -epsilon
    
    for i in range(ny):
        for j in range(nx):
            phi_ij = phi[i, j]
            g_inv = 1.0 / (6.0 * abs(phi_ij) * abs(1.0 - phi_ij) + 1e-6)
            mu_field[i, j] = (eps_neg * lapl_phi[i, j] + dw_dphi[i, j]) * g_inv


@njit(fastmath=True)
def total_mass(phi, dx):
    return np.sum(phi) * dx * dx


@njit(fastmath=True)
def total_free_energy(phi: np.ndarray, epsilon: float, dx: float) -> float:
    ny, nx = phi.shape
    eps2 = 0.5 * epsilon
    dx2 = dx * dx
    
    # Preallocazioni locali
    w_local = np.empty_like(phi)
    gx = np.empty_like(phi)
    gy = np.empty_like(phi)
    j_left = np.empty(nx, dtype=np.int64)
    j_right = np.empty(nx, dtype=np.int64)
    for j in range(nx):
        j_left[j] = (j - 1) % nx
        j_right[j] = (j + 1) % nx
    
    w_field(phi, epsilon, w_local)
    grad_2D_neumann_along_y(phi, dx, gx, gy, j_left, j_right)  # Assumendo fix della funzione grad
    
    total_E = 0.0
    for i in range(ny):
        for j in range(nx):
            grad2 = gx[i, j] * gx[i, j] + gy[i, j] * gy[i, j]
            f_ij = w_local[i, j] + eps2 * grad2
            total_E += f_ij
    
    return total_E * dx2


@njit(fastmath=True)
def M_field(phi: np.ndarray, M0: float, epsilon: float, M: np.ndarray):
    """
    Calcola il campo scalare di mobilità
    """
    ny, nx = phi.shape
    factor = M0 * 36.0 / epsilon
    
    for i in range(ny):
        for j in range(nx):
            phi_ij = phi[i, j]
            one_minus = 1.0 - phi_ij
            M[i,j] =  factor * phi_ij*phi_ij * one_minus*one_minus
