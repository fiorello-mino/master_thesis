import numpy as np
import matplotlib.pyplot as plt
from numba import njit

# Parameters
L = 1.0
N = 64
dx = L / N

dt = 1e-9
steps = 5000000
eps = 10*dx

# Initial condition
c = 0.5 + 0.05 * (np.random.rand(N, N) - 0.5)
plt.clf()
plt.imshow(c, cmap='RdBu', origin='lower',vmin=0,vmax=1)
plt.colorbar()
plt.title(f"Step 0")
plt.pause(1)


@njit
def laplacian(u, dx):
    N = u.shape[0]
    lap = np.zeros_like(u)
    
    for i in range(N):
        ip = (i + 1) % N
        im = (i - 1) % N
        for j in range(N):
            jp = (j + 1) % N
            jm = (j - 1) % N
            
            lap[i, j] = (
                u[ip, j] + u[im, j] +
                u[i, jp] + u[i, jm] -
                4.0 * u[i, j]
            ) / (dx * dx)
    
    return lap

@njit
def grad(u,dx):
    N = u.shape[0]
    grad = np.zeros((N,N,2))
    
    for i in range(N):
        ip = (i + 1) % N
        im = (i - 1) % N
        
        for j in range(N):
            jp = (j + 1) % N
            jm = (j - 1) % N
 
            grad[i,j,0] = ( u[i, jp] - u[i, jm] ) / (2*dx)
            grad[i,j,1] = ( u[ip, j] - u[im, j] ) / (2*dx)

    return grad

@njit
def div(f,vec,dx):
    N = vec.shape[0]
    d = np.zeros((N,N))

    for i in range(N):
        ip = (i + 1) % N
        im = (i - 1) % N
 
        for j in range(N):
            jp = (j + 1) % N
            jm = (j - 1) % N
     
            d[i,j] = ( (f[i,jp]*vec[i,jp,0] - f[i,jm]*vec[i,jm,0]) + (f[ip,j]*vec[ip,j,1] - f[im,j]*vec[im,j,1]) ) / (2*dx)

    return d


@njit
def df_dc(c):
    return 2.0 * c * (1.0 - c) * (1.0 - 2.0 * c)

@njit
def step(c, dx, dt, kappa):
    mu = 18/eps*df_dc(c) - eps * laplacian(c, dx)
    mobi=36./eps * c**2 * (1-c)**2
    return c + dt * div(mobi, grad(mu,dx), dx)

for step_id in range(steps+1):
    c = step(c, dx, dt, eps)

    if step_id % 100000 == 0:
        plt.clf()
        plt.imshow(c, cmap='RdBu', origin='lower',vmin=0,vmax=1)
        plt.colorbar()
        plt.title(f"Step {step_id}")
        plt.pause(0.01)

plt.show()
