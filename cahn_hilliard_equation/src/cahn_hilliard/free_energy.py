# free_energy.py

import numpy as np
from numba import njit
from .operators import grad_2D


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
    
    w_field(phi, epsilon, w_local)
    grad_2D(phi, dx, gx, gy, x_left, x_right, y_up, y_down)
    
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
            

@njit(fastmath=True)
def w_field_3D(phi: np.ndarray, epsilon: float, w: np.ndarray):
    """
    Calcola il potenziale doppia buca su griglia 3D uniforme
    """
    nz, ny, nx = phi.shape
    factor = 18.0 / epsilon
            
    for z in range(nz):
        for y in range(ny):
            for x in range(nx):
                phi_zyx = phi[z,y,x]
                w[z,y,x] = factor * phi_zyx * phi_zyx * (1 - phi_zyx) * (1 - phi_zyx)
                
                
@njit(fastmath=True)
def dw_dphi_3D(phi: np.ndarray, epsilon: float, w_prime:np.ndarray):
    """
    Calcola la derivata del potenziale doppia buca su griglia uniforme 3D
    """
    nz, ny, nx = phi.shape
    factor = 36.0 / epsilon
            
    for z in range(nz):
        for y in range(ny):
            for x in range(nx):
                phi_zyx = phi[z,y,x]
                phi2 = phi_zyx * phi_zyx
                w_prime[z,y,x] = factor * phi_zyx * (1.0 + 2.0 * phi2 - 3.0 * phi_zyx)


@njit(fastmath=True)
def mu_field_3D(lapl_phi: np.ndarray, dw_dphi: np.ndarray, epsilon: float, mu: np.ndarray):
    """
    Calcola il potenziale chimico su griglia uniforme 3D
    """
    nz, ny, nx = dw_dphi.shape
    eps_minus = - epsilon
            
    for z in range(nz):
        for y in range(ny):
            for x in range(nx):
                mu[z,y,x] = eps_minus * lapl_phi[z,y,x] + dw_dphi[z,y,x]
                
                
@njit(fastmath=True)
def M_field_3D(phi: np.ndarray, M0: float, epsilon: float, M: np.ndarray):
    """
    Calcola il campo scalare di mobilità su griglia uniforme 3D
    """
    nz, ny, nx = phi.shape
    factor = M0 * 36.0 / epsilon
            
    for z in range(nz):
        for y in range(ny):
            for x in range(nx):
                phi_zyx = phi[z,y,x]
                one_minus = 1.0 - phi_zyx
                M[z,y,x] = factor * phi_zyx*phi_zyx * one_minus*one_minus