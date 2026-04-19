# free_energy.py

import numpy as np
from numba import njit
from operators import grad_2D_neumann_along_y


@njit
def w_field(phi: np.ndarray, epsilon: float) -> np.ndarray:
    """
    Calcola il potenziale doppia buca
    """
    ny, nx = phi.shape
    w = np.empty_like(phi)
    factor = 18.0 / epsilon
    
    for i in range(ny):
        for j in range(nx):
            phi_ij = phi[i, j]
            w[i, j] = factor * phi_ij * phi_ij * (1 - phi_ij) * (1 - phi_ij)
    
    return w

    
@njit
def dw_dphi(phi: np.ndarray, epsilon: float) -> np.ndarray:
    """
    Calcola la derivata del potenziale doppia buca
    """
    ny, nx = phi.shape
    dw_dphi = np.empty_like(phi)
    factor = 36.0 / epsilon
    
    for i in range(ny):
        for j in range(nx):
            phi_ij = phi[i,j]
            dw_dphi[i,j] = factor * phi_ij * (1 + 2 * phi_ij * phi_ij - 3 * phi_ij)
            
    return dw_dphi


@njit
def mu_field(lapl_phi: np.ndarray, dw_dphi: np.ndarray, epsilon: float) -> np.ndarray:
    """
    Calcola il potenziale chimico
    """
    ny, nx = dw_dphi.shape
    mu = np.empty_like(dw_dphi)
    
    for i in range(ny):
        for j in range(nx):
            mu[i,j] = - epsilon * lapl_phi[i,j] + dw_dphi[i,j]
    
    return mu

@njit
def mu_field_g_phi(lapl_phi: np.ndarray, dw_dphi: np.ndarray, phi: np.ndarray, epsilon: float) -> np.ndarray:
    """
    Calcola il potenziale chimico con stabilizzazione numerica.
    """
    ny, nx = phi.shape
    mu = np.empty_like(dw_dphi)
    
    for i in range(ny):
        for j in range(nx):
            phi_ij = np.abs(phi[i,j])
            one_minus = np.abs(1 - phi[i,j])
            factor = 6*phi_ij*one_minus 
            mu[i,j] = factor * (- epsilon * lapl_phi[i,j] + dw_dphi[i,j])
    
    return mu


@njit
def total_mass(phi: np.ndarray, dx:float) -> float:
    """
    Calcola la massa totale conservata
    """
    return np.sum(phi) * dx * dx


@njit
def total_free_energy(phi: np.ndarray, epsilon: float, dx: float) -> float:
    """
    Calcola l'energia libera totale del sistema
    """
    ny, nx = phi.shape
    eps2 = 0.5 * epsilon
    dx2 = dx * dx
    
    w = w_field(phi, epsilon)               
    gx, gy = grad_2D_neumann_along_y(phi, dx)
    
    total_E = 0.0
    for i in range(ny):
        for j in range(nx):
            grad2 = gx[i, j] * gx[i, j] + gy[i, j] * gy[i, j]
            f_ij = w[i, j] + eps2 * grad2
            total_E += f_ij

    return total_E * dx2


@njit
def M_field(phi: np.ndarray, M0: float, epsilon: float) -> np.ndarray:
    """
    Calcola il campo scalare di mobilità
    """
    ny, nx = phi.shape
    M = np.empty_like(phi)
    factor = M0 * 36.0 / epsilon
    
    for i in range(ny):
        for j in range(nx):
            phi_ij = phi[i, j]
            one_minus = 1 - phi_ij
            M[i,j] =  factor * phi_ij*phi_ij * one_minus*one_minus + 1e-6
    
    return M
