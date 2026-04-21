# free_energy.py

import numpy as np
from numba import njit
from operators import grad_2D_neumann_along_y


@njit
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

    
@njit
def dw_dphi(phi: np.ndarray, epsilon: float, w_prime:np.ndarray):
    """
    Calcola la derivata del potenziale doppia buca
    """
    ny, nx = phi.shape
    factor = 36.0 / epsilon
    
    for i in range(ny):
        for j in range(nx):
            phi_ij = phi[i,j]
            w_prime[i,j] = factor * phi_ij * (1 + 2 * phi_ij * phi_ij - 3 * phi_ij)


@njit
def mu_field(lapl_phi: np.ndarray, dw_dphi: np.ndarray, epsilon: float, mu: np.ndarray):
    """
    Calcola il potenziale chimico
    """
    ny, nx = dw_dphi.shape
    
    for i in range(ny):
        for j in range(nx):
            mu[i,j] = - epsilon * lapl_phi[i,j] + dw_dphi[i,j]


@njit
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
    
    for i in range(ny):
        for j in range(nx):
            mu = -epsilon * lapl_phi[i, j] + dw_dphi[i, j]
            g = 6.0 * abs(phi[i, j]) * abs(1.0 - phi[i, j])
            mu_field[i, j] = g * mu


@njit
def total_mass(phi: np.ndarray, dx: float) -> float:
    total = 0.0
    for i in range(phi.shape[0]):
        for j in range(phi.shape[1]):
            total += phi[i,j]
    return total * dx * dx


@njit
def total_free_energy(phi: np.ndarray, epsilon: float, dx: float) -> float:
    ny, nx = phi.shape
    eps2 = 0.5 * epsilon
    dx2 = dx * dx
    
    # Preallocazioni locali
    w_local = np.empty_like(phi)
    gx = np.empty_like(phi)
    gy = np.empty_like(phi)
    
    w_field(phi, epsilon, w_local)
    grad_2D_neumann_along_y(phi, dx, gx, gy)  # Assumendo fix della funzione grad
    
    total_E = 0.0
    for i in range(ny):
        for j in range(nx):
            grad2 = gx[i, j] * gx[i, j] + gy[i, j] * gy[i, j]
            f_ij = w_local[i, j] + eps2 * grad2
            total_E += f_ij
    
    return total_E * dx2


@njit
def M_field(phi: np.ndarray, M0: float, epsilon: float, M: np.ndarray):
    """
    Calcola il campo scalare di mobilità
    """
    ny, nx = phi.shape
    factor = M0 * 36.0 / epsilon
    
    for i in range(ny):
        for j in range(nx):
            phi_ij = phi[i, j]
            one_minus = 1 - phi_ij
            M[i,j] =  factor * phi_ij*phi_ij * one_minus*one_minus + 1e-6
