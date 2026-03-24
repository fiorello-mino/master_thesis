import numpy as np
import matplotlib.pyplot as plt

# Imposto i parametri dell'equazione
beta = 10
T = 3600
L = 1
N = 100
dx = L / (N - 1)
dy = L / (N - 1)
delta = dx
epsilon = 5 * delta
dt = 0.2 * delta**2 / (beta * epsilon)
# per la seconda simulazione T = 7200, epsilon = 3*delta, dt = 0.1 * (...)

# Imposto la condizione iniziale: valori random tra 0 e 1
phi = np.random.rand(N, N)
phi_new = np.empty_like(phi)
phi_initial = np.copy(phi)

# Energia del sistema
energy = np.zeros(T)


# Snapshots per plot
snapshots = {}
steps = [0, T//3, 2*T//3, T-1]

# Integro l'equazione di Allen-Cahn
for n in range(T):
    
    if n in steps:
        snapshots[n] = np.copy(phi)
    
    energy[n] = (2.0 * 18.0 / epsilon) * np.sum(phi**2 * (1 - phi**2))
    
    laplacian = (
            
        np.roll(phi, 1, axis=0) 
        + np.roll(phi, -1, axis=0) 
        + np.roll(phi, 1, axis=1) 
        + np.roll(phi, -1, axis=1) 
        - 4 * phi
        
    ) / delta**2
    
    w_prime = (36.0 / epsilon) * phi * (1 + 2 * phi**2 - 3 * phi)
    
    phi_new = phi + dt * beta * (epsilon * laplacian - w_prime)
    
    phi, phi_new = phi_new, phi # swap tra phi e phi_new in modo che phi al passo n+1 sia phi_new


# Plot sistema
fig, axes = plt.subplots(2, 2, figsize=(10, 8))

for ax, n in zip(axes.ravel(), steps):
    im = ax.imshow(snapshots[n], origin="lower", extent=[0, L, 0, L])
    t_fin = n * dt
    ax.set_title(fr"$\phi(x,y,t={t_fin:.3f})$")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.colorbar(im, ax=ax)

plt.tight_layout()
plt.show()

#Plot energia
n_steps = np.arange(len(energy))
t = n_steps * dt
plt.figure(figsize=(6,4))
plt.plot(t, energy, '-o', markersize=2)
plt.xlabel('t')
plt.ylabel('E(t)')
plt.title('Energy')
plt.tight_layout()
plt.show()
